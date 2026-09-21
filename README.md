# scale-transformer

Supervised finetuning (SFT) baseline for **Qwen3-30B-A3B** built directly on Hugging Face
`transformers` + `peft`, driven by YAML configs.

This is deliberately a **single-device** baseline: one process, one accelerator, the stock
`Trainer` loop. There is no FSDP, DeepSpeed, tensor/pipeline parallelism or multi-node
launcher here. The point is to have a correct, readable reference run to measure any
distributed version against.

## The model

`Qwen/Qwen3-30B-A3B` is a Mixture-of-Experts causal LM:

| | |
|---|---|
| Total parameters | ~30.5B |
| Active per token | ~3.3B |
| Layers | 48 |
| Experts | 128, top-8 routed |
| Attention | GQA, 32 query / 4 KV heads |
| Context | 40,960 positions in `config.json` (Qwen documents 32,768 native, 131,072 with YaRN) |
| License | Apache-2.0 |

Two consequences drive the design of this repo:

1. **All 30B parameters must be resident** even though only 3.3B are active per token.
   bf16 weights alone are ~61 GB, so full finetuning on one device is off the table —
   LoRA (or 4-bit QLoRA) is the default.
2. **The router needs its load-balancing loss.** Finetuning an MoE without
   `output_router_logits` lets routing collapse onto a few experts. It is on by default
   (`model.router_aux_loss_coef`) and is automatically ignored on dense checkpoints.

## Setup

```bash
uv sync                      # core deps (torch, transformers, datasets, peft)
uv sync --extra quant        # + bitsandbytes for 4-bit QLoRA (Linux/CUDA only)
uv sync --extra track        # + tensorboard / wandb
uv sync --extra dev          # + ruff, pytest
```

FlashAttention-2 is not a declared dependency because it needs a compiler and the CUDA
toolkit. On a GPU box:

```bash
uv pip install flash-attn --no-build-isolation
# then set model.attn_implementation: flash_attention_2
```

## Quickstart

Verify the pipeline end to end on a 0.6B model — runs on a laptop, CPU or MPS, ~20 steps:

```bash
./scripts/smoke_test.sh
```

Then the real run:

```bash
uv run scale-train configs/qwen3_30b_a3b_lora.yaml          # 80 GB card
uv run scale-train configs/qwen3_30b_a3b_qlora.yaml         # 24-48 GB card
```

Inspect a resolved config without training, and override any value from the CLI:

```bash
uv run scale-train configs/qwen3_30b_a3b_lora.yaml --print-config
uv run scale-train configs/qwen3_30b_a3b_lora.yaml \
  --set train.learning_rate=5e-5 \
  --set data.max_seq_len=4096 \
  --set train.output_dir=outputs/exp-002
```

Training writes the resolved config to `<output_dir>/experiment_config.json` and the
adapter to `<output_dir>/final`. To get a standalone model for serving:

```bash
uv run scale-merge outputs/qwen3-30b-a3b-lora/final outputs/qwen3-30b-a3b-merged
```

## Memory budget (single device, `max_seq_len=2048`, batch 1)

| Setup | Weights | Fits on |
|---|---|---|
| LoRA, bf16 | ~61 GB | H100 80GB, A100 80GB |
| QLoRA, nf4 | ~18 GB | A100 40GB, L40S, RTX 4090 (short sequences) |
| Full finetune | ~61 GB + ~366 GB optimizer/grads | not on one device — that's the distributed work |

Gradient checkpointing is on by default; turning it off roughly doubles activation memory.

## Layout

```
configs/                     YAML experiment configs
  qwen3_30b_a3b_lora.yaml      bf16 LoRA, 80 GB card
  qwen3_30b_a3b_qlora.yaml     4-bit QLoRA, smaller cards
  qwen3_0.6b_smoke.yaml        laptop-sized pipeline check
src/scale_transformer/
  config.py                  dataclass schema, YAML loading, --set overrides
  data.py                    chat-template rendering, completion-only masking, collation
  modeling.py                dtype/attention selection, MoE knobs, LoRA wrapping
  train.py                   Trainer wiring and the run itself
  merge.py                   adapter -> merged weights
  cli.py                     `scale-train`
tests/                       masking, collation and config-schema tests
scripts/smoke_test.sh
```

Run the tests with `uv run pytest` (they exercise the real Qwen3 tokenizer and chat
template, and skip themselves if the hub is unreachable).

## Data

Three input shapes are supported via `data.format`:

- `messages` — a column of `[{"role": ..., "content": ...}]`, the usual chat SFT layout.
- `alpaca` — `instruction` / `input` / `output` columns (the default config's dataset).
- `prompt_completion` — two plain text columns.

All three are normalised to chat messages and rendered with the model's own chat template,
so the finetuned model sees exactly the format it will see at inference. Local files work
through the `json` loader:

```yaml
data:
  dataset_name: json
  data_files: { train: data/train.jsonl, validation: data/val.jsonl }
  train_split: train
  eval_split: validation
  format: messages
```

By default loss is taken **only on assistant turns** (`train_on_completions_only`).
Assistant spans are located by searching the *final* rendered text for each message's
content and mapping those character spans to tokens via the fast tokenizer's offset
mapping. This matters more than it sounds: Qwen3's template rewrites history as a
conversation grows — it emits an empty `<think>` block only for the *last* assistant turn —
so the usual trick of diffing incremental renders silently breaks on multi-turn data and
trains on the user's prompts. Working from the final render sidesteps that entirely. The
end-of-turn token is supervised too, so the model learns to stop. If a turn genuinely
cannot be located the example is dropped rather than trained with a wrong mask, and the run
logs how often that happened.

`data.enable_thinking` controls Qwen3's hybrid thinking template. It defaults to `false`,
which trains the model in non-thinking mode.

## Scope

Intentionally **not** here: FSDP/DeepSpeed configs, `accelerate launch` wrappers, multi-node
orchestration, expert parallelism, sequence parallelism, custom kernels. Those are the next
milestone; this repo is the correctness and throughput reference they get compared to.
