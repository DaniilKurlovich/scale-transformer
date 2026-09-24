"""Register PG-19 (emozilla/pg19) in torchtitan's dataset registry.

torchtitan resolves `training.dataset` against the DATASETS dict when the
dataloader is built, so importing this module before the Trainer starts is
enough -- the installed package stays untouched. `scale_transformer.train`
does that import.
"""

from functools import partial
from typing import Any

from datasets import load_dataset
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS


def _load_pg19_dataset(dataset_path: str, split: str):
    """Load PG-19: a parquet mirror, so no config name, unlike c4's name="en"."""
    return load_dataset(dataset_path, split=split, streaming=True)


def _process_pg19_text(sample: dict[str, Any]) -> str:
    """PG-19 rows are {short_book_title, publication_date, url, text}."""
    return sample["text"]


DATASETS["pg19"] = DatasetConfig(
    path="emozilla/pg19",
    loader=partial(_load_pg19_dataset, split="train"),
    sample_processor=_process_pg19_text,
)

DATASETS["pg19_validation"] = DatasetConfig(
    path="emozilla/pg19",
    loader=partial(_load_pg19_dataset, split="validation"),
    sample_processor=_process_pg19_text,
)
