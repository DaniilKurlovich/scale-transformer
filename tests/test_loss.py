"""Loss-correctness tests for the training loop.

What "correct" means here, top to bottom:

1. `cross_entropy_loss` is a *sum* over non-ignored tokens, computed in fp32
   even when the logits are bf16, and ignored (-100) tokens contribute nothing.
2. With gradient accumulation the trainer divides every microbatch's loss sum
   by the *global* number of valid tokens, so (a) the summed loss equals the
   token-mean loss over the whole global batch and (b) the accumulated
   gradients equal the gradients of that single token-mean loss. Averaging
   per-microbatch means would be wrong whenever microbatches hold different
   numbers of valid tokens; the tests use uneven token counts on purpose.
3. The logged metrics are reconstructed correctly from those per-rank
   partial losses (`global_avg_loss = sum over ranks`, `local_avg_loss =
   loss * global_tokens / local_tokens`).
4. The dataloader feeds next-token labels (`label[t] == input[t + 1]`), with
   the eos boundary between PG-19 books, so the loss is a true LM loss.
5. The real Qwen3-MoE model is causal: the loss at position t does not depend
   on tokens after t (no label leakage through attention). GPU only.

Everything but (5) runs on CPU.
"""

import contextlib
import copy
import types

import pytest
import torch
import torch.nn.functional as F
from torchtitan.components.loss import IGNORE_INDEX, cross_entropy_loss
from torchtitan.train import Trainer

torch.manual_seed(0)


def _reference_token_mean_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-mean CE the way a human would write it: fp32, mask out -100."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    valid = labels != IGNORE_INDEX
    safe_labels = labels.clamp(min=0)
    nll = -logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (nll * valid).sum() / valid.sum()


# ---------------------------------------------------------------- 1. loss fn ---


class TestCrossEntropyLoss:
    def test_sum_reduction_matches_manual(self):
        B, T, V = 2, 7, 11
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[0, :3] = IGNORE_INDEX
        expected = _reference_token_mean_ce(logits, labels) * (labels != IGNORE_INDEX).sum()
        torch.testing.assert_close(cross_entropy_loss(logits, labels), expected)

    def test_is_a_sum_not_a_mean(self):
        B, T, V = 3, 5, 9
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        per_token_mean = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        torch.testing.assert_close(cross_entropy_loss(logits, labels), per_token_mean * (B * T))

    def test_ignored_tokens_contribute_nothing(self):
        B, T, V = 2, 6, 13
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        base = cross_entropy_loss(logits, labels)

        # Append ignored positions with wild logits: loss must be unchanged.
        extra_logits = torch.randn(B, 4, V) * 100
        extra_labels = torch.full((B, 4), IGNORE_INDEX)
        padded = cross_entropy_loss(
            torch.cat([logits, extra_logits], 1), torch.cat([labels, extra_labels], 1)
        )
        torch.testing.assert_close(padded, base)

    def test_all_ignored_gives_zero_not_nan(self):
        logits = torch.randn(1, 4, 5)
        labels = torch.full((1, 4), IGNORE_INDEX)
        assert cross_entropy_loss(logits, labels).item() == 0.0

    def test_bf16_logits_are_upcast_before_softmax(self):
        B, T, V = 2, 8, 1024
        logits = (torch.randn(B, T, V) * 20).to(torch.bfloat16)
        labels = torch.randint(0, V, (B, T))
        loss = cross_entropy_loss(logits, labels)
        assert loss.dtype == torch.float32
        # exactly what you get by upcasting first (no bf16 log-softmax).
        expected = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction="sum")
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)

    def test_gradient_flows_to_logits(self):
        logits = torch.randn(1, 3, 4, requires_grad=True)
        labels = torch.tensor([[0, 2, IGNORE_INDEX]])
        cross_entropy_loss(logits, labels).backward()
        # d(sum CE)/d logits = softmax - onehot on valid rows, 0 on ignored rows
        expected = torch.softmax(logits.detach(), -1)
        expected[0, 0, 0] -= 1
        expected[0, 1, 2] -= 1
        expected[0, 2] = 0
        torch.testing.assert_close(logits.grad, expected)


# ---------------------------------------------- 2. accumulation normalization ---


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=17, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.out = torch.nn.Linear(dim, vocab)

    def forward(self, tokens, **kwargs):
        return self.out(torch.tanh(self.emb(tokens)))


def _uneven_microbatches(n=3, B=2, T=6, vocab=17):
    """Microbatches with different numbers of valid tokens."""
    mbs = []
    for i in range(n):
        x = torch.randint(0, vocab, (B, T))
        y = torch.randint(0, vocab, (B, T))
        y[:, : i + 1] = IGNORE_INDEX  # 1, 2, 3 ... ignored prefix tokens per row
        mbs.append(({"input": x}, y))
    return mbs


def _accumulate_like_trainer(model, microbatches):
    """Mirror Trainer.train_step's arithmetic without the distributed plumbing."""
    global_valid_tokens = sum((y != IGNORE_INDEX).sum() for _, y in microbatches).float()
    model.zero_grad()
    partial_losses = []
    for inp, y in microbatches:
        loss = cross_entropy_loss(model(inp["input"]), y) / global_valid_tokens
        loss.backward()
        partial_losses.append(loss.detach())
    return torch.sum(torch.stack(partial_losses)), [p.grad.clone() for p in model.parameters()]


class TestGradientAccumulation:
    def test_summed_partial_losses_equal_global_token_mean(self):
        model = TinyLM()
        mbs = _uneven_microbatches()
        loss, _ = _accumulate_like_trainer(model, mbs)

        with torch.no_grad():
            logits = torch.cat([model(inp["input"]) for inp, _ in mbs], 0)
            labels = torch.cat([y for _, y in mbs], 0)
        torch.testing.assert_close(loss, _reference_token_mean_ce(logits, labels))

    def test_accumulated_grads_equal_single_batch_token_mean_grads(self):
        model = TinyLM()
        mbs = _uneven_microbatches()
        _, acc_grads = _accumulate_like_trainer(model, mbs)

        ref = copy.deepcopy(model)
        ref.zero_grad()
        logits = torch.cat([ref(inp["input"]) for inp, _ in mbs], 0)
        labels = torch.cat([y for _, y in mbs], 0)
        _reference_token_mean_ce(logits, labels).backward()
        for g, r in zip(acc_grads, ref.parameters(), strict=True):
            torch.testing.assert_close(g, r.grad)

    def test_mean_of_microbatch_means_would_be_wrong(self):
        """Guards the *reason* for global-token normalization: with uneven
        microbatches, averaging per-microbatch means is not the token mean."""
        model = TinyLM()
        mbs = _uneven_microbatches()
        loss, _ = _accumulate_like_trainer(model, mbs)
        with torch.no_grad():
            naive = torch.stack(
                [
                    cross_entropy_loss(model(inp["input"]), y) / (y != IGNORE_INDEX).sum()
                    for inp, y in mbs
                ]
            ).mean()
        assert not torch.isclose(loss, naive, rtol=1e-3, atol=1e-3)

    def test_gradient_scale_independent_of_microbatch_count(self):
        """Splitting the same global batch into 1, 2 or 3 microbatches must give identical grads."""
        model = TinyLM()
        mbs = _uneven_microbatches(n=6)
        xs = torch.cat([inp["input"] for inp, _ in mbs])
        ys = torch.cat([y for _, y in mbs])
        results = []
        for n_splits in (1, 2, 3, 6):
            split = [
                ({"input": x}, y)
                for x, y in zip(xs.chunk(n_splits), ys.chunk(n_splits), strict=True)
            ]
            loss, grads = _accumulate_like_trainer(copy.deepcopy(model), split)
            results.append((loss, grads))
        for loss, grads in results[1:]:
            torch.testing.assert_close(loss, results[0][0])
            for g, g0 in zip(grads, results[0][1], strict=True):
                torch.testing.assert_close(g, g0)


# --------------------------------------------- 2b. the actual Trainer methods ---


class _Recorder:
    """Captures what the trainer would log / hand to the optimizer."""

    def __init__(self, model):
        self.model = model
        self.logged = []
        self.grads_at_step = None
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.ntokens_since_last_log = 0
        self.data_loading_times = []

    # OptimizersContainer API used by train_step
    def zero_grad(self):
        self.zero_grad_calls += 1
        self.model.zero_grad()

    def step(self):
        self.step_calls += 1
        self.grads_at_step = [p.grad.clone() for p in self.model.parameters()]

    # MetricsProcessor API used by train_step
    def should_log(self, step):
        return True

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        self.logged.append(
            dict(
                step=step,
                loss=global_avg_loss,
                max_loss=global_max_loss,
                grad_norm=grad_norm,
                **(extra_metrics or {}),
            )
        )


def _bare_trainer(model, grad_accum: int, max_norm: float = 1e9) -> Trainer:
    """A Trainer without __init__: just enough state to run train_step on CPU
    with no parallelism (mirrors the dp=1 / no-PP / no-CP code path)."""
    t = Trainer.__new__(Trainer)
    rec = _Recorder(model)
    t.model_parts = [model]
    t.model_args = types.SimpleNamespace(attn_type="sdpa")
    t.tokenizer = None
    t.device = torch.device("cpu")
    t.step = 0
    t.ntokens_seen = 0
    t.gradient_accumulation_steps = grad_accum
    t.loss_fn = cross_entropy_loss
    t.train_context = contextlib.nullcontext
    t.maybe_enable_amp = contextlib.nullcontext()
    t.parallel_dims = types.SimpleNamespace(
        pp_enabled=False,
        cp_enabled=False,
        dp_enabled=False,
        dp_cp_enabled=False,
        ep_enabled=False,
        get_optional_mesh=lambda name: None,
    )
    t.optimizers = rec
    t.lr_schedulers = types.SimpleNamespace(
        schedulers=[types.SimpleNamespace(get_last_lr=lambda: [3e-4])], step=lambda: None
    )
    t.checkpointer = types.SimpleNamespace(maybe_wait_for_staging=lambda: None)
    t.metrics_processor = rec
    t.job_config = types.SimpleNamespace(
        training=types.SimpleNamespace(max_norm=max_norm),
        parallelism=types.SimpleNamespace(context_parallel_load_balancer=None),
    )
    t._recorder = rec
    return t


class TestTrainerLossPath:
    def test_forward_backward_step_returns_partial_normalized_loss(self):
        model = TinyLM()
        t = _bare_trainer(model, grad_accum=1)
        inp, y = _uneven_microbatches(n=1)[0]
        gvt = torch.tensor(float((y != IGNORE_INDEX).sum()) * 4)  # pretend 3 other microbatches
        loss = t.forward_backward_step(input_dict=inp, labels=y, global_valid_tokens=gvt)
        with torch.no_grad():
            expected = cross_entropy_loss(model(inp["input"]), y) / gvt
        torch.testing.assert_close(loss.detach(), expected)
        assert all(p.grad is not None for p in model.parameters())

    def test_train_step_logs_global_token_mean_loss_and_correct_grads(self):
        model = TinyLM()
        mbs = _uneven_microbatches(n=3)
        t = _bare_trainer(model, grad_accum=3)
        t.step = 1
        ref_model = copy.deepcopy(model)  # loss/grad reference uses pre-step weights

        t.train_step(iter(mbs))

        rec = t._recorder
        assert rec.zero_grad_calls == 1 and rec.step_calls == 1
        assert len(rec.logged) == 1
        logged = rec.logged[0]

        logits = torch.cat([ref_model(inp["input"]) for inp, _ in mbs], 0)
        labels = torch.cat([y for _, y in mbs], 0)
        expected_loss = _reference_token_mean_ce(logits, labels)
        expected_loss.backward()

        assert logged["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)
        # single rank: max loss is the same number
        assert logged["max_loss"] == pytest.approx(expected_loss.item(), rel=1e-5)
        for g, r in zip(rec.grads_at_step, ref_model.parameters(), strict=True):
            torch.testing.assert_close(g, r.grad)
        ref_norm = torch.norm(torch.stack([p.grad.norm() for p in ref_model.parameters()]))
        assert logged["grad_norm"] == pytest.approx(ref_norm.item(), rel=1e-5)

    def test_train_step_clips_but_loss_is_unaffected(self):
        model = TinyLM()
        mbs = _uneven_microbatches(n=2)
        unclipped = _bare_trainer(copy.deepcopy(model), grad_accum=2)
        clipped = _bare_trainer(copy.deepcopy(model), grad_accum=2, max_norm=1e-3)
        unclipped.train_step(iter(copy.deepcopy(mbs)))
        clipped.train_step(iter(copy.deepcopy(mbs)))
        a, b = unclipped._recorder.logged[0], clipped._recorder.logged[0]
        assert a["loss"] == pytest.approx(b["loss"])  # loss is measured before clipping
        assert a["grad_norm"] == pytest.approx(b["grad_norm"])  # reported norm is pre-clip
        total = torch.norm(torch.stack([g.norm() for g in clipped._recorder.grads_at_step]))
        assert total.item() <= 1e-3 * (1 + 1e-4)


# ----------------------------------------------- 3. multi-rank reconstruction ---


class TestGlobalLossReconstruction:
    """Pure arithmetic of train_step's logging block, simulated over DP ranks."""

    def test_sum_of_rank_partials_is_global_mean_and_local_avg_recovers_rank_mean(self):
        V = 11
        rank_batches = []
        for r in range(4):
            logits = torch.randn(2, 5 + r, V)
            labels = torch.randint(0, V, (2, 5 + r))
            labels[0, :r] = IGNORE_INDEX
            rank_batches.append((logits, labels))

        local_tokens = [(y != IGNORE_INDEX).sum().float() for _, y in rank_batches]
        global_tokens = sum(local_tokens)
        rank_loss = [cross_entropy_loss(logits, y) / global_tokens for logits, y in rank_batches]

        global_avg = sum(rank_loss)  # dist_sum(loss)
        all_logits = torch.cat([logits.flatten(0, 1) for logits, _ in rank_batches])
        all_labels = torch.cat([y.flatten() for _, y in rank_batches])
        torch.testing.assert_close(global_avg, _reference_token_mean_ce(all_logits, all_labels))

        for (logits, y), loss, n in zip(rank_batches, rank_loss, local_tokens, strict=True):
            local_avg = loss * global_tokens / n
            torch.testing.assert_close(local_avg, _reference_token_mean_ce(logits, y))


# ---------------------------------------------------------- 4. next-token labels ---


class _StubTokenizer:
    BOS, EOS = 1, 2

    def encode(self, text, add_bos=False, add_eos=False):
        toks = [ord(c) for c in text]
        if add_bos:
            toks = [self.BOS] + toks
        if add_eos:
            toks = toks + [self.EOS]
        return toks


class TestDataloaderLabels:
    def test_labels_are_inputs_shifted_by_one_across_book_boundaries(self):
        from datasets import Dataset
        from torchtitan.hf_datasets import DatasetConfig
        from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataset

        from scale_transformer import pg19  # noqa: F401  registers "pg19" (reads ["text"])

        books = ["abcdefgh", "ijk", "lmnopqrstuvwxyz"]
        DATASETS["pg19_unit"] = DatasetConfig(
            path="in-memory",
            loader=lambda path: Dataset.from_dict({"text": books, "short_book_title": books}),
            sample_processor=DATASETS["pg19"].sample_processor,
        )
        seq_len = 5
        ds = HuggingFaceTextDataset(
            "pg19_unit", None, _StubTokenizer(), seq_len=seq_len, infinite=False
        )
        samples = list(ds)
        assert samples, "dataset yielded nothing"

        tok = _StubTokenizer()
        stream = sum((tok.encode(b, add_bos=True, add_eos=True) for b in books), [])
        pos = 0
        for inp, label in samples:
            x = inp["input"]
            assert x.shape == label.shape == (seq_len,)
            # label[t] is the token that follows input[t] ...
            torch.testing.assert_close(label[:-1], x[1:])
            # ... including at the chunk boundary and across eos/bos joins.
            assert x.tolist() == stream[pos : pos + seq_len]
            assert label.tolist() == stream[pos + 1 : pos + seq_len + 1]
            pos += seq_len + 1  # torchtitan consumes seq_len + 1 tokens per sample (no overlap)
        # every eos is a label somewhere (the model is trained to end books)
        all_labels = torch.cat([label for _, label in samples]).tolist()
        assert all_labels.count(tok.EOS) >= len(books) - 1


# -------------------------------------------------- 5. causality of Qwen3-MoE ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the grouped-mm experts")
class TestQwen3MoECausality:
    def _tiny_model(self):
        from torchtitan.models.moe import MoEArgs
        from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
        from torchtitan.models.qwen3.model.model import Qwen3Model

        args = Qwen3ModelArgs(
            vocab_size=256,
            max_seq_len=64,
            head_dim=16,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            qk_norm=True,
            hidden_dim=128,
            rope_theta=10000,
            moe_enabled=True,
            moe_inter_dim=64,
            moe_args=MoEArgs(
                num_experts=4,
                num_shared_experts=0,
                top_k=2,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
        )
        torch.manual_seed(0)
        with torch.device("cuda"):
            model = Qwen3Model(args)
        with torch.no_grad():
            model.init_weights(buffer_device=torch.device("cuda"))
        return model.eval()

    def _per_token_nll(self, model, tokens, labels):
        logits = model(tokens).float()
        return F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none", ignore_index=IGNORE_INDEX
        ).view_as(labels)

    def test_loss_at_t_ignores_future_tokens(self):
        model = self._tiny_model()
        B, T, cut = 2, 32, 20
        torch.manual_seed(1)
        tokens = torch.randint(0, 256, (B, T), device="cuda")
        labels = torch.randint(0, 256, (B, T), device="cuda")
        with torch.no_grad():
            base = self._per_token_nll(model, tokens, labels)
            perturbed = tokens.clone()
            perturbed[:, cut:] = torch.randint(0, 256, (B, T - cut), device="cuda")
            other = self._per_token_nll(model, perturbed, labels)
        torch.testing.assert_close(base[:, :cut], other[:, :cut], rtol=1e-4, atol=1e-4)
        assert not torch.allclose(base[:, cut:], other[:, cut:]), (
            "future tokens had no effect at all?"
        )

    def test_model_loss_matches_trainer_loss_fn(self):
        model = self._tiny_model()
        tokens = torch.randint(0, 256, (2, 16), device="cuda")
        labels = torch.randint(0, 256, (2, 16), device="cuda")
        labels[0, :4] = IGNORE_INDEX
        with torch.no_grad():
            logits = model(tokens)
            loss = cross_entropy_loss(logits, labels)
            expected = _reference_token_mean_ce(logits, labels) * (labels != IGNORE_INDEX).sum()
        torch.testing.assert_close(loss, expected)
