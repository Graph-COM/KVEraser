from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

from .schema import (
    DATA_PREFIX, DATASETS, data_path, parquet_schema, validate_record,
)

DEFAULT_DATASET_REPO = "Graph-COM/KVEraser"


class DatasetSource:
    """Load Qwen3-8B data from a local root or one immutable HF snapshot."""

    def __init__(
        self, *, data_root: str | Path | None = None,
        repo_id: str = DEFAULT_DATASET_REPO, revision: str = "main",
        cache_dir: str | Path | None = None, local_files_only: bool = False,
    ):
        self._hub = dict(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir,
                         local_files_only=local_files_only)
        manifest_path = f"{DATA_PREFIX}/manifest.json"
        self.revision = None
        if data_root is not None:
            self.root = Path(data_root).expanduser()
        else:
            self.root = Path(snapshot_download(
                revision=revision, allow_patterns=[manifest_path], **self._hub))
            self.revision = self.root.name
        path = self.root / manifest_path
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        try:
            self._datasets = manifest["datasets"]
            self.finetune_order = tuple(manifest["finetune_order"])
            finetune = {key for key in DATASETS if key.startswith("finetune/")}
            if (manifest["format"] != DATA_PREFIX or set(self._datasets) != set(DATASETS)
                    or set(self.finetune_order) != finetune
                    or len(self.finetune_order) != len(finetune)):
                raise ValueError("format, datasets, or finetune order mismatch")
            for dataset, spec in DATASETS.items():
                splits = self._datasets[dataset]
                if set(splits) != set(spec.splits):
                    raise ValueError(f"split mismatch: {dataset}")
                for split, count in spec.splits.items():
                    files = sorted(splits[split], key=lambda entry: entry["path"])
                    if not files or sum(entry["rows"] for entry in files) != count:
                        raise ValueError(f"row count mismatch: {dataset}/{split}")
                    for index, entry in enumerate(files):
                        expected = (f"{DATA_PREFIX}/{dataset}/{split}-"
                                    f"{index:05d}-of-{len(files):05d}.parquet")
                        if (entry["path"] != expected
                                or type(entry["rows"]) is not int or entry["rows"] <= 0
                                or type(entry["bytes"]) is not int or entry["bytes"] <= 0
                                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                            raise ValueError(f"invalid shard: {entry['path']}")
                    splits[split] = files
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid data manifest {path}: {error}") from error

    def load(self, dataset: str, split: str) -> list[dict[str, Any]]:
        """Verify and read a split in shard order, preserving every row and field."""
        data_path(dataset, split)
        rows, seen = [], set()
        for entry in self._datasets[dataset][split]:
            path = self.root / entry["path"]
            if self.revision is not None:
                path = Path(hf_hub_download(
                    filename=entry["path"], revision=self.revision, **self._hub))
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"File size mismatch: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path}")
            with pq.ParquetFile(path) as parquet:
                if not parquet.schema_arrow.equals(parquet_schema(), check_metadata=True):
                    raise ValueError(f"Parquet schema mismatch: {path}")
                if parquet.metadata.num_rows != entry["rows"]:
                    raise ValueError(f"Parquet row count mismatch: {path}")
                for batch in parquet.iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        validate_record(row, dataset)
                        if row["id"] in seen:
                            raise ValueError(f"Duplicate ID in {dataset}/{split}: {row['id']}")
                        seen.add(row["id"])
                        rows.append(row)
        return rows


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def _span_to_token_range(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int]:
    token_start = None
    token_end = None
    for idx, (start_offset, end_offset) in enumerate(offsets):
        if int(end_offset) <= start_char or int(start_offset) >= end_char:
            continue
        if token_start is None:
            token_start = idx
        token_end = idx
    if token_start is None or token_end is None:
        raise ValueError("Erase span was not found in token offsets.")
    return token_start, token_end + 1


def map_prompt_span_from_offsets(
    offsets: list[tuple[int, int]],
    prompt_char_span: list[int],
    *,
    prompt_text: str | None = None,
    span_text: str | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    start_char, end_char = [int(value) for value in prompt_char_span]
    if start_char < 0 or end_char <= start_char:
        raise ValueError("Invalid erase span.")
    if prompt_text is not None:
        if end_char > len(prompt_text):
            raise ValueError("Erase span exceeds cached prompt length.")
        if span_text is not None and prompt_text[start_char:end_char] != span_text:
            raise ValueError("Erase span text does not match cached prompt.")
    token_span = _span_to_token_range(offsets, start_char, end_char)
    aligned_char_span = (
        int(offsets[token_span[0]][0]),
        int(offsets[token_span[1] - 1][1]),
    )
    return token_span, aligned_char_span


def resolve_erase_token_span(cached_offsets: list[tuple[int, int]], sample: dict[str, Any]) -> tuple[int, int]:
    erase_span = sample["erase_span"]
    (erase_start, erase_end), _ = map_prompt_span_from_offsets(
        cached_offsets,
        erase_span["cached_prompt_char_span"],
        prompt_text=sample["cached_prompt"],
        span_text=erase_span["text"],
    )
    return int(erase_start), int(erase_end)


def erase_cached_prefix(
    model: Any,
    old_context_kv: Any,
    cached_input_ids: Any,
    cached_offsets: list[tuple[int, int]],
    sample: dict[str, Any],
    *,
    erase_token_span: tuple[int, int] | None = None,
) -> Any:
    if erase_token_span is None:
        erase_start, erase_end = resolve_erase_token_span(cached_offsets, sample)
    else:
        erase_start, erase_end = [int(value) for value in erase_token_span]
    needle_ids = cached_input_ids[:, erase_start:erase_end]
    return model.erase_tokens(old_context_kv, erase_start, needle_ids)


def parse_answer_numbers(response_text: str) -> list[str]:
    return re.findall(r"\d+", response_text or "")


def score_erased_prediction(prediction: list[str], sample: dict[str, Any]) -> bool:
    return list(prediction) == _list_value(sample["targets"]["erased_answers"])


def target_text(sample: dict[str, Any], tokenizer: Any) -> str:
    eos_token = tokenizer.eos_token or ""
    return str(sample["targets"]["erased_train_text"]) + eos_token
