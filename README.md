# scale-transformer

Qwen3-30B-A3B pretraining / continued pretraining on [torchtitan](https://github.com/pytorch/torchtitan),
with PG-19 as the corpus, YaRN long-context scaling and a one-rank Nsight Systems
capture. Everything project-specific lives in `src/scale_transformer` and plugs
into torchtitan's registries at import time; the installed package is never
patched on disk.

## Layout

    src/scale_transformer/
      train.py          entry point: `torchrun ... -m scale_transformer.train`
      pg19.py           registers the "pg19" / "pg19_validation" datasets
      smoke.py          registers the "smoke-moe" flavor (1-GPU wiring test)
      yarn.py           registers "30B-A3B-yarn" + YaRN rope cache; --check / --stamp
      nsys_profile.py   ProfiledTrainer + the [nsys] config section
      fetch_assets.py   downloads what a config refers to (tokenizer, weights, PG-19)
    configs/
      qwen3_30b_a3b.toml        real run, 8x80GB node
      qwen3_30b_a3b_yarn.toml   32k -> 262k YaRN continued pretraining
      qwen3_smoke_1gpu.toml     1-GPU smoke test
    train.sh            one node: fetch what is missing, then torchrun (the container's job)
    run_nsys.sh         torchrun under `nsys profile`, single- or multi-node
    tests/              CPU unit tests (loss, capture window, fetch plan, launchers)
    assets/             tokenizer.json + config.json (+ weights) -- fetched, gitignored
    data/pg19/          PG-19 parquet mirror -- fetched, gitignored
    outputs/            logs, tensorboard, profiler traces, nsys, checkpoints (gitignored)

## 0. Install

    uv sync --extra dev            # installs the package editable + pytest/ruff
    uv run pytest                  # CPU tests; the causality test needs a GPU

The Docker image (see `Dockerfile`) does the same `uv sync` into `/opt/venv`;
bind-mount or sync the checkout over `/workspace` and the editable install
picks it up.

## 1. Assets and data

    python -m scale_transformer.fetch_assets --config configs/qwen3_30b_a3b.toml

reads the config and downloads whatever it refers to that is not there yet:
`model.hf_assets_path` (tokenizer + config.json), the safetensors too when
`checkpoint.initial_load_path` points at it with `initial_load_in_hf = true`
(~61 GB, the YaRN config), and the PG-19 parquet shards into
`training.dataset_path`. Each directory gets a `.fetched` marker, so a second
call makes no network requests; `--dry-run` prints the plan. The Hub repo comes
from the directory name (`qwen3-30b-a3b` -> `Qwen/Qwen3-30B-A3B`,
`qwen3-30b-a3b-base` -> `Qwen/Qwen3-30B-A3B-Base`); pass `--repo DIR=ORG/NAME`
for anything else. `train.sh` runs this before every launch.

The YaRN config uses the Base repo (not the instruct one: its eos is
`<|endoftext|>`, which is what the dataloader appends between PG-19 documents).

## 2. Smoke test (1 GPU)

    torchrun --nproc_per_node=1 -m scale_transformer.train --job.config_file configs/qwen3_smoke_1gpu.toml

## 3. Real run (8 x 80GB), by hand

    torchrun --nproc_per_node=8 -m scale_transformer.train --job.config_file configs/qwen3_30b_a3b.toml

Multi-node:

    torchrun --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=8 \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
      -m scale_transformer.train --job.config_file configs/qwen3_30b_a3b.toml

Always launch `scale_transformer.train`, not `torchtitan.train`: importing it is
what registers the datasets and flavors the configs refer to. Any field is
overridable on the CLI: `--training.seq_len 8192 --optimizer.lr 1e-5`.

## 4. In the container: `train.sh` (2 x 8 H100)

`train.sh` is what the image runs. On each node it fetches whatever the config
needs into `/workspace` (step 1, serialised on a lock file in case the volume is
shared), then launches torchrun for that node. Everything the run touches lives
under `/workspace`, the volume: `assets/`, `data/pg19/`, the HF cache, and
`outputs/` with checkpoints, tensorboard, torch-profiler traces
(`outputs/profile_traces`, one window every 100 steps), nsys reports and the
per-node logs in `outputs/logs/`.

    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --network host -v scale-workspace:/workspace \
      -e NNODES=2 -e NODE_RANK=0 -e MASTER_ADDR=<node 0 address> \
      ghcr.io/<owner>/scale-transformer train.sh configs/qwen3_30b_a3b.toml

Run the same on node 1 with `NODE_RANK=1`. The job starts when both nodes reach
the c10d rendezvous on `MASTER_ADDR:29500`. `NPROC_PER_NODE` defaults to what
`nvidia-smi` lists. Extra arguments are torchtitan overrides
(`--training.steps 200`). `NSYS=1` runs the job under `run_nsys.sh` instead
(a capture run: it ends after the `[nsys]` window).

On RunPod, put the same variables in the pod template plus `AUTO_TRAIN=1` and
your `PUBLIC_KEY`: the entrypoint starts `train.sh` in the background on deploy,
logs to `/workspace/outputs/logs/auto_train.out`, and keeps sshd in the
foreground so the pod stays up after the job finishes. `TRAIN_CONFIG` and
`TRAIN_ARGS` select the config and overrides. Set NCCL's interface variables
(`NCCL_SOCKET_IFNAME`, `NCCL_IB_*`) for your fabric as usual.

The code is baked into `/opt/scale-transformer`, not `/workspace`, so the volume
does not hide it. A checkout synced over `/workspace` (mutagen, bind mount) still
wins: `train.sh` puts its `src/` on `PYTHONPATH` and uses its `configs/`.

## Memory budget (30.5B total / 3.3B active)

    bf16 weights                         57 GiB
    + fp32 master + AdamW m,v + grads   455 GiB

Divide by the FSDP shard degree. 8x80GB = 640 GiB is the practical floor; leave
headroom for activations (full AC, seq_len 4096, local_batch_size 1).
A single 48GB GPU cannot hold the model: smoke config only.

## Parallelism knobs

- `data_parallel_shard_degree = -1`  FSDP over whatever world size is left.
- `expert_parallel_degree = 8`       128 experts sharded across ranks (all-to-all).
- `expert_tensor_parallel_degree`    raise with TP if experts still don't fit.
- `tensor_parallel_degree`           requires `seq_len % (tp * 2*cp) == 0`.
- `context_parallel_degree`          shards the sequence; the YaRN run needs it above 32k.
- `pipeline_parallel_degree`         for >= 4 nodes; 48 layers split evenly.

## Continued pretraining from released weights

In `[checkpoint]`: set `initial_load_path = "./assets/qwen3-30b-a3b"`,
`initial_load_in_hf = true`, `initial_load_model_only = true`, and drop the LR to
~1e-5 with a short warmup. The `Qwen3StateDictAdapter` maps HF <-> torchtitan FQNs;
`last_save_in_hf = true` writes checkpoints back in HF format.
`configs/qwen3_30b_a3b_yarn.toml` is wired this way already.

## YaRN long context (32k -> 262k)

    torchrun --nproc_per_node=8 -m scale_transformer.train --job.config_file configs/qwen3_30b_a3b_yarn.toml

`model.flavor = "30B-A3B-yarn"` is the stock `30B-A3B` args plus YaRN knobs
(`factor = 8`, `original_max_seq_len = 32768`). The rope cache is a
non-persistent buffer, so the checkpoint format does not change, but the
scaling lives in the config: torchtitan's HF export writes safetensors only, so
copy `config.json` and the tokenizer from `assets/qwen3-30b-a3b-base` into the
export dir and stamp `rope_scaling` into it, or inference will run unscaled:

    python -m scale_transformer.yarn --stamp outputs/yarn/checkpoint/step-500 --factor 8
    python -m scale_transformer.yarn --check     # assert the math matches transformers

`training.seq_len` may not exceed `original_max_seq_len * factor`; raise the
factor rather than training past the scaled window.

## Profiling one rank with Nsight Systems

    ./run_nsys.sh 1 configs/qwen3_smoke_1gpu.toml --nsys.start_step 3 --nsys.stop_step 5
    ./run_nsys.sh 8 configs/qwen3_30b_a3b.toml
    NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0> ./run_nsys.sh 8 configs/qwen3_30b_a3b.toml

`nsys` is armed with `--capture-range=cudaProfilerApi`; only `NSYS_RANK`
(default 0) calls `cudaProfilerStart/Stop`, around `[nsys.start_step,
nsys.stop_step]`, and only the node hosting that rank runs under `nsys` at all.
Reports land in `outputs/nsys/`. The `[nsys]` section comes from
`job.custom_config_module = "scale_transformer.nsys_profile"`, which the script
passes on the CLI; with `nsys.enable = false` (the default) the trainer is the
stock loop. Run the container with `--cap-add=SYS_ADMIN` for the sampling counters.

## Data

`training.dataset = "pg19"` is `emozilla/pg19` (~28k Project Gutenberg books,
~11B tokens), registered in `src/scale_transformer/pg19.py`, which inserts a
`DatasetConfig` into torchtitan's `DATASETS` registry at import time. The two
real configs set `training.dataset_path = "./data/pg19"`, a local mirror of the
parquet shards that `fetch_assets` downloads; drop the line to stream from the
Hub instead (the smoke config does). `"c4"` and `"c4_validation"` still work.

`pg19_validation` is registered but inert: the validator is only built and called
when `validation.enable = true` (it is `false` here). When you turn it on, also
set `validation.dataset` -- it defaults to `"c4_validation"`, so leaving it out
means validating on C4 while training on PG-19. The configs are pre-wired for that.

Add another corpus by copying the `DATASETS[...] = DatasetConfig(...)` block
(loader + sample_processor). For local files, keep the registered name and point
`training.dataset_path` at the path -- it overrides `DatasetConfig.path`.

Notes on PG-19 vs C4:
- Documents are whole books, not web pages, so one sample fills many sequences;
  the loader concatenates and chunks to `seq_len` either way.
- The train split has 23 parquet shards. `split_dataset_by_node` shards cleanly
  only when `num_shards % dp_degree == 0`; at dp=8 or 16 it falls back to
  round-robin, where every rank reads the whole set and keeps 1/dp of the rows.
  Reading from the local mirror makes that cheap; reshard the mirror to a
  multiple of dp_degree if it still shows up in the step time.
- No `name="en"` config and no `validation` config quirk: splits are
  `train` / `validation` / `test`.
