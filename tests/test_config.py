from pathlib import Path

import pytest

from scale_transformer.config import ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    """Catches typos and keys that drifted away from the dataclasses."""
    cfg = ExperimentConfig.load(path)
    assert cfg.model.name_or_path


def test_dotted_overrides_are_parsed_as_yaml(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("train:\n  learning_rate: 1.0e-4\n")
    cfg = ExperimentConfig.load(
        config,
        [
            "train.learning_rate=5e-5",
            "lora.target_modules=[q_proj]",
            "model.trust_remote_code=true",
        ],
    )
    assert cfg.train.learning_rate == 5e-5
    assert cfg.lora.target_modules == ["q_proj"]
    assert cfg.model.trust_remote_code is True


def test_unknown_keys_are_rejected(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("train:\n  lerning_rate: 1.0e-4\n")
    with pytest.raises(ValueError, match="lerning_rate"):
        ExperimentConfig.load(config)


def test_malformed_override_is_rejected(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("{}\n")
    with pytest.raises(ValueError, match="train.learning_rate"):
        ExperimentConfig.load(config, ["train.learning_rate"])
