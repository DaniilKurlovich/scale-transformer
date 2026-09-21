import pytest

TOKENIZER_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def tokenizer():
    """The real Qwen3 tokenizer + chat template. Skips when the hub is unreachable."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as exc:  # offline, rate-limited, gated...
        pytest.skip(f"cannot download {TOKENIZER_ID}: {exc}")
