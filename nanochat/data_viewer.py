"""
Dataset visualization helpers.

Provides stage-aware sampling utilities so the web UI can display random
examples from the different training phases (base, mid, sft, rl).
"""

import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files

try:
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K, extract_answer
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk
    _TASK_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    ARC = None
    GSM8K = None
    MMLU = None
    SmolTalk = None
    _TASK_IMPORT_ERROR = exc

    def extract_answer(_: str) -> Optional[str]:
        return None


class DataUnavailable(RuntimeError):
    """Raised when a dataset for a stage cannot be sampled."""



_OPTIONAL_DATASETS_MESSAGE = (
    "Optional dependency 'datasets' is required to load instruction datasets."
    " Install it via `pip install datasets`."
)




@dataclass
class DatasetConfig:
    """Metadata and sampler callback for a dataset within a stage."""

    key: str
    label: str
    description: str
    sampler: Callable[[random.Random], Dict[str, Any]]


@dataclass
class StageConfig:
    """Description of a training stage and its constituent datasets."""

    key: str
    label: str
    description: str
    datasets: List[DatasetConfig]


def _make_json_safe(value: Any) -> Any:
    """Convert common python containers so they are JSON serializable."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    return value


def _serialize_conversation(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize conversation structure for the front-end."""
    messages = []
    raw_messages = conversation.get("messages", [])
    for message in raw_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "text")
                    text = part.get("text", "")
                else:
                    part_type = "text"
                    text = str(part)
                parts.append({"type": part_type, "text": text})
        else:
            parts = [{"type": "text", "text": str(content)}]
        messages.append({"role": role, "parts": parts})

    metadata = {
        k: _make_json_safe(v)
        for k, v in conversation.items()
        if k != "messages"
    }
    return {
        "kind": "conversation",
        "messages": messages,
        "metadata": metadata,
    }


class _BaseShardSampler:
    """Randomly sample raw text rows from FineWeb-Edu parquet shards."""

    def __init__(self):
        self._lock = threading.Lock()
        self._parquet_paths: Optional[List[str]] = None

    def _ensure_paths(self) -> None:
        """Refresh the cached shard list."""
        self._parquet_paths = list_parquet_files()

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        with self._lock:
            if not self._parquet_paths:
                self._ensure_paths()
            paths = self._parquet_paths
        if not paths:
            data_dir = get_base_dir() / "base_data"
            raise DataUnavailable(
                f"No parquet shards found in {data_dir}. Download base data before viewing examples."
            )

        shard_path = Path(rng.choice(paths))
        parquet = pq.ParquetFile(shard_path)
        num_row_groups = parquet.num_row_groups
        if num_row_groups == 0:
            raise DataUnavailable(f"Shard {shard_path} has no row groups to sample.")
        row_group_index = rng.randrange(num_row_groups)
        table = parquet.read_row_group(row_group_index, columns=["text"])
        column = table.column("text")
        texts = column.to_pylist()
        if not texts:
            raise DataUnavailable(f"Row group {row_group_index} in {shard_path} is empty.")
        row_index = rng.randrange(len(texts))
        text = texts[row_index]
        if text is None:
            raise DataUnavailable(f"Encountered empty text in {shard_path}.")
        return {
            "kind": "text",
            "text": text,
            "metadata": {
                "file": shard_path.name,
                "row_group": row_group_index,
                "row": row_index,
            },
        }


def _make_task_sampler(factory: Callable[[], Any]) -> Callable[[random.Random], Dict[str, Any]]:
    """Create a sampler that lazily instantiates a Task dataset."""
    dataset_lock = threading.Lock()
    dataset_ref: Dict[str, Any] = {"dataset": None}

    def get_dataset():
        dataset = dataset_ref["dataset"]
        if dataset is None:
            with dataset_lock:
                dataset = dataset_ref["dataset"]
                if dataset is None:
                    dataset = factory()
                    dataset_ref["dataset"] = dataset
        return dataset

    def sampler(rng: random.Random) -> Dict[str, Any]:
        dataset = get_dataset()
        length = len(dataset)
        if length == 0:
            raise DataUnavailable("Dataset is empty, cannot sample.")
        index = rng.randrange(length)
        conversation = dataset[index]
        example = _serialize_conversation(conversation)
        metadata = example.setdefault("metadata", {})
        metadata.update({
            "index": index,
            "total_examples": length,
        })
        return example

    return sampler


def _make_gsm8k_sampler(subset: str, split: str, annotate_answer: bool = False) -> Callable[[random.Random], Dict[str, Any]]:
    if GSM8K is None:
        return _make_missing_sampler(_OPTIONAL_DATASETS_MESSAGE)

    base_sampler = _make_task_sampler(lambda: GSM8K(subset=subset, split=split))

    def sampler(rng: random.Random) -> Dict[str, Any]:
        example = base_sampler(rng)
        if annotate_answer and extract_answer is not None:
            assistant_messages = [m for m in example["messages"] if m["role"] == "assistant"]
            if assistant_messages:
                parts = assistant_messages[-1]["parts"]
                # The final textual part usually contains the #### marker in GSM8K.
                text_parts = [p["text"] for p in parts if p["type"] == "text"]
                final_snippet = text_parts[-1] if text_parts else ""
                answer = extract_answer(final_snippet or "")
                example.setdefault("metadata", {})
                example["metadata"].update({
                    "expected_answer": answer,
                    "notes": "Answer parsed from #### marker in GSM8K solution.",
                })
        return example

    return sampler


def _make_missing_sampler(message: str) -> Callable[[random.Random], Dict[str, Any]]:
    def sampler(_rng: random.Random) -> Dict[str, Any]:
        raise DataUnavailable(message)
    return sampler


class StageDataProvider:
    """High-level helper that knows about the available stages and datasets."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        base_sampler = _BaseShardSampler()

        if _TASK_IMPORT_ERROR is None:
            mid_datasets = [
                DatasetConfig(
                    key="smoltalk_train",
                    label="SmolTalk (train)",
                    description="General multi-turn assistant conversations.",
                    sampler=_make_task_sampler(lambda: SmolTalk(split="train")),
                ),
                DatasetConfig(
                    key="mmlu_auxiliary",
                    label="MMLU Auxiliary (train)",
                    description="Multiple choice questions spanning STEM and humanities subjects.",
                    sampler=_make_task_sampler(lambda: MMLU(subset="auxiliary_train", split="train")),
                ),
                DatasetConfig(
                    key="gsm8k_mid",
                    label="GSM8K (train)",
                    description="Grade school math problems with step-by-step tool traces.",
                    sampler=_make_gsm8k_sampler(subset="main", split="train"),
                ),
            ]
            sft_datasets = [
                DatasetConfig(
                    key="smoltalk_sft",
                    label="SmolTalk (subset)",
                    description="10K-example subset of SmolTalk used during SFT.",
                    sampler=_make_task_sampler(lambda: SmolTalk(split="train", stop=10_000)),
                ),
                DatasetConfig(
                    key="arc_easy",
                    label="ARC-Easy",
                    description="Multiple-choice grade-school science questions (easy split).",
                    sampler=_make_task_sampler(lambda: ARC(subset="ARC-Easy", split="train")),
                ),
                DatasetConfig(
                    key="arc_challenge",
                    label="ARC-Challenge",
                    description="Harder multiple-choice ARC questions (challenge split).",
                    sampler=_make_task_sampler(lambda: ARC(subset="ARC-Challenge", split="train")),
                ),
                DatasetConfig(
                    key="gsm8k_sft",
                    label="GSM8K (train)",
                    description="Math tutoring traces reused during SFT stage.",
                    sampler=_make_gsm8k_sampler(subset="main", split="train"),
                ),
            ]
            rl_datasets = [
                DatasetConfig(
                    key="gsm8k_rl",
                    label="GSM8K (train)",
                    description="Prompts with solutions; metadata includes the extracted gold answer.",
                    sampler=_make_gsm8k_sampler(subset="main", split="train", annotate_answer=True),
                ),
            ]
        else:
            missing_sampler = _make_missing_sampler(_OPTIONAL_DATASETS_MESSAGE)
            mid_datasets = [
                DatasetConfig(
                    key="smoltalk_train",
                    label="SmolTalk (train)",
                    description="General multi-turn assistant conversations.",
                    sampler=missing_sampler,
                ),
                DatasetConfig(
                    key="mmlu_auxiliary",
                    label="MMLU Auxiliary (train)",
                    description="Multiple choice questions spanning STEM and humanities subjects.",
                    sampler=missing_sampler,
                ),
                DatasetConfig(
                    key="gsm8k_mid",
                    label="GSM8K (train)",
                    description="Grade school math problems with step-by-step tool traces.",
                    sampler=missing_sampler,
                ),
            ]
            sft_datasets = [
                DatasetConfig(
                    key="smoltalk_sft",
                    label="SmolTalk (subset)",
                    description="10K-example subset of SmolTalk used during SFT.",
                    sampler=missing_sampler,
                ),
                DatasetConfig(
                    key="arc_easy",
                    label="ARC-Easy",
                    description="Multiple-choice grade-school science questions (easy split).",
                    sampler=missing_sampler,
                ),
                DatasetConfig(
                    key="arc_challenge",
                    label="ARC-Challenge",
                    description="Harder multiple-choice ARC questions (challenge split).",
                    sampler=missing_sampler,
                ),
                DatasetConfig(
                    key="gsm8k_sft",
                    label="GSM8K (train)",
                    description="Math tutoring traces reused during SFT stage.",
                    sampler=missing_sampler,
                ),
            ]
            rl_datasets = [
                DatasetConfig(
                    key="gsm8k_rl",
                    label="GSM8K (train)",
                    description="Prompts with solutions; metadata includes the extracted gold answer.",
                    sampler=missing_sampler,
                ),
            ]

        self._stages: Dict[str, StageConfig] = {
            "base": StageConfig(
                key="base",
                label="Base Pretraining",
                description="Unsupervised text drawn from FineWeb-Edu shards used during base training.",
                datasets=[
                    DatasetConfig(
                        key="fineweb_edu",
                        label="FineWeb-Edu 100B",
                        description="Raw shuffled web text shards prepared for base training.",
                        sampler=base_sampler.sample,
                    ),
                ],
            ),
            "mid": StageConfig(
                key="mid",
                label="Mid Training",
                description="Mixture of instruction-style tasks used for mid-training curriculum.",
                datasets=mid_datasets,
            ),
            "sft": StageConfig(
                key="sft",
                label="Supervised Finetuning",
                description="Curated tasks for supervised chat finetuning covering conversations and exams.",
                datasets=sft_datasets,
            ),
            "rl": StageConfig(
                key="rl",
                label="Reinforcement Learning",
                description="Rollout prompts for GSM8K reward optimization.",
                datasets=rl_datasets,
            ),
        }


    def list_stages(self) -> List[Dict[str, Any]]:
        """Return metadata about available stages and datasets."""
        stages = []
        for stage in self._stages.values():
            datasets = [
                {
                    "dataset": dataset.key,
                    "label": dataset.label,
                    "description": dataset.description,
                }
                for dataset in stage.datasets
            ]
            stages.append({
                "stage": stage.key,
                "label": stage.label,
                "description": stage.description,
                "datasets": datasets,
            })
        return stages

    def sample(self, stage_key: str, dataset_key: Optional[str] = None) -> Dict[str, Any]:
        """Return a random example for the requested stage and dataset."""
        if stage_key not in self._stages:
            raise DataUnavailable(f"Unknown stage '{stage_key}'.")
        stage = self._stages[stage_key]
        dataset: Optional[DatasetConfig] = None
        if dataset_key:
            dataset = next((d for d in stage.datasets if d.key == dataset_key), None)
            if dataset is None:
                raise DataUnavailable(f"Stage '{stage_key}' has no dataset '{dataset_key}'.")
        else:
            dataset = self._rng.choice(stage.datasets)

        example = dataset.sampler(self._rng)
        return {
            "stage": stage.key,
            "stage_label": stage.label,
            "stage_description": stage.description,
            "dataset": dataset.key,
            "dataset_label": dataset.label,
            "dataset_description": dataset.description,
            "example": example,
        }


__all__ = [
    "DataUnavailable",
    "DatasetConfig",
    "StageConfig",
    "StageDataProvider",
]
