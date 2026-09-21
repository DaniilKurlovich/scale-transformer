"""Training entry point: single-device SFT with the Hugging Face Trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import transformers
from transformers import Trainer, TrainingArguments, set_seed

from .config import ExperimentConfig
from .data import CausalCollator, build_datasets
from .modeling import build_model, build_tokenizer

logger = logging.getLogger(__name__)


def _describe_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda:0 ({name}, {total:.0f} GiB)"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_training_arguments(cfg: ExperimentConfig, has_eval: bool) -> TrainingArguments:
    train_cfg = cfg.train
    return TrainingArguments(
        output_dir=train_cfg.output_dir,
        seed=train_cfg.seed,
        data_seed=train_cfg.seed,
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=train_cfg.warmup_steps,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        optim=train_cfg.optim,
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        # Reentrant checkpointing does not compose with frozen inputs / LoRA.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        eval_strategy=train_cfg.eval_strategy if has_eval else "no",
        eval_steps=train_cfg.eval_steps,
        report_to=train_cfg.report_to,
        run_name=train_cfg.run_name,
        # Our dataset carries exactly the columns the collator needs.
        remove_unused_columns=False,
        # Keeps the loss comparable across gradient-accumulation settings.
        average_tokens_across_devices=False,
        logging_first_step=True,
    )


def run(cfg: ExperimentConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    transformers.utils.logging.set_verbosity_info()

    logger.info("device: %s | torch %s | transformers %s",
                _describe_device(), torch.__version__, transformers.__version__)
    set_seed(cfg.train.seed)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tokenizer = build_tokenizer(cfg.model)
    train_dataset, eval_dataset = build_datasets(tokenizer, cfg.data)
    logger.info("train examples: %d | eval examples: %s",
                len(train_dataset), len(eval_dataset) if eval_dataset else "none")

    model = build_model(cfg)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(cfg, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
        processing_class=tokenizer,
    )

    result = trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    # With LoRA this writes the adapter only; merge separately for deployment.
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("done; artifacts in %s", output_dir)
