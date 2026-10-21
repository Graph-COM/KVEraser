from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

from .schema import DATA_PREFIX, DATASETS, data_path, parquet_schema, validate_record

CONTEXT_SIZES = (1000, 2000, 4000, 8000, 16000, 32000)
TOKENS_TO_GENERATE = 32
TRAIN_ROWS_PER_CONTEXT = 200
VALIDATION_ROWS_PER_CONTEXT = 100
TEST_ROWS_PER_CONTEXT = 100
DEFAULT_DATASET_REPO = "Graph-COM/KVEraser"
PROMPT_FORMAT = "qwen3_nonthinking_delayed_query_prompt_span"
DATASET_SEED = 42
RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DATA_ROOT = RELEASE_ROOT / "data"


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


@dataclass(frozen=True)
class ContextSplits:
    context_tokens: int
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]
    tokens_to_generate: int = TOKENS_TO_GENERATE


def config_name(context_tokens: int, tokens_to_generate: int = TOKENS_TO_GENERATE) -> str:
    return f"niah_ctx{int(context_tokens)}_t{int(tokens_to_generate)}"


def context_dir_name(context_tokens: int, tokens_to_generate: int = TOKENS_TO_GENERATE) -> str:
    return f"ctx{int(context_tokens)}_t{int(tokens_to_generate)}"


def parse_contexts(value: str | None) -> tuple[int, ...]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return CONTEXT_SIZES
    contexts = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    unknown = [ctx for ctx in contexts if ctx not in CONTEXT_SIZES]
    if unknown:
        raise ValueError(f"Unsupported context size(s): {unknown}. Expected one of {CONTEXT_SIZES}.")
    if len(set(contexts)) != len(contexts):
        raise ValueError("--contexts contains duplicate values.")
    return contexts


def normalize_split_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized == "val":
        return "validation"
    if normalized not in {"train", "validation", "test"}:
        raise ValueError("split must be one of train, validation, val, test.")
    return normalized


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def remove_prompt_char_span(prompt_text: str, prompt_char_span: list[int], *, span_text: str | None = None) -> str:
    start_char, end_char = [int(value) for value in prompt_char_span]
    if start_char < 0 or end_char <= start_char or end_char > len(prompt_text):
        raise ValueError("Invalid cached-prompt erase character span.")
    if span_text is not None and prompt_text[start_char:end_char] != span_text:
        raise ValueError("Erase span text does not match cached prompt.")
    return prompt_text[:start_char] + prompt_text[end_char:]


def sample_to_row(
    sample: dict[str, Any],
    *,
    context_tokens: int,
    split: str,
    source_id: str | None = None,
    seed: int = DATASET_SEED,
) -> dict[str, Any]:
    split = normalize_split_name(split)
    erase_span = sample["erase_span"]
    targets = sample["targets"]
    prompt_char_span = [int(x) for x in erase_span["cached_prompt_char_span"]]
    edited_cached_prompt = sample.get("variants", {}).get("edited_cached_prompt")
    if edited_cached_prompt is None:
        edited_cached_prompt = remove_prompt_char_span(
            sample["cached_prompt"],
            prompt_char_span,
            span_text=erase_span["text"],
        )
    erased_answers = _list_value(targets["erased_answers"])
    original_answers = _list_value(targets["original_answers"])
    return {
        "id": str(sample["id"]),
        "source_id": str(source_id if source_id is not None else sample.get("source_id", sample["id"])),
        "split": split,
        "context_tokens": int(context_tokens),
        "tokens_to_generate": int(sample.get("tokens_to_generate", TOKENS_TO_GENERATE)),
        "seed": int(seed),
        "prompt_format": str(sample.get("prompt_format", PROMPT_FORMAT)),
        "query": str(sample.get("query", "")),
        "cached_prompt": str(sample["cached_prompt"]),
        "query_suffix": str(sample["query_suffix"]),
        "erase_span_text": str(erase_span["text"]),
        "erase_span_start": int(prompt_char_span[0]),
        "erase_span_end": int(prompt_char_span[1]),
        "original_answers": original_answers,
        "erased_answers": erased_answers,
        "erased_train_text": str(targets.get("erased_train_text", erased_answers[0])),
        "edited_cached_prompt": edited_cached_prompt,
    }


def row_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    if "erase_span" in row and "targets" in row:
        return dict(row)
    erase_span = {
        "text": str(row["erase_span_text"]),
        "cached_prompt_char_span": [int(row["erase_span_start"]), int(row["erase_span_end"])],
    }
    original_answers = _list_value(row["original_answers"])
    erased_answers = _list_value(row["erased_answers"])
    sample = {
        "id": str(row["id"]),
        "source_id": str(row.get("source_id", row["id"])),
        "split": normalize_split_name(str(row.get("split", "test"))),
        "context_tokens": int(row.get("context_tokens", 0) or 0),
        "tokens_to_generate": int(row.get("tokens_to_generate", TOKENS_TO_GENERATE)),
        "prompt_format": str(row.get("prompt_format", PROMPT_FORMAT)),
        "query": str(row.get("query", "")),
        "cached_prompt": str(row["cached_prompt"]),
        "query_suffix": str(row["query_suffix"]),
        "erase_span": erase_span,
        "targets": {
            "original_answers": original_answers,
            "erased_answers": erased_answers,
            "erased_train_text": str(row.get("erased_train_text", erased_answers[0] if erased_answers else "")),
        },
        "variants": {
            "edited_cached_prompt": str(row.get("edited_cached_prompt") or remove_prompt_char_span(
                str(row["cached_prompt"]),
                erase_span["cached_prompt_char_span"],
                span_text=erase_span["text"],
            )),
        },
    }
    validate_sample(sample)
    return sample


def validate_sample(sample: dict[str, Any]) -> None:
    for key in ("id", "cached_prompt", "query_suffix", "erase_span", "targets"):
        if key not in sample:
            raise ValueError(f"Sample is missing required key '{key}'.")
    if not sample["cached_prompt"]:
        raise ValueError("cached_prompt must be non-empty.")
    if not sample["query_suffix"]:
        raise ValueError("query_suffix must be non-empty.")
    erase_span = sample["erase_span"]
    span = erase_span.get("cached_prompt_char_span")
    text = erase_span.get("text")
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError("erase_span.cached_prompt_char_span must be a two-element list.")
    start_char, end_char = [int(x) for x in span]
    if sample["cached_prompt"][start_char:end_char] != text:
        raise ValueError(f"Erase span mismatch in sample {sample['id']}.")
    targets = sample["targets"]
    if not _list_value(targets.get("original_answers")):
        raise ValueError("targets.original_answers must be non-empty.")
    if not _list_value(targets.get("erased_answers")):
        raise ValueError("targets.erased_answers must be non-empty.")
    if not targets.get("erased_train_text"):
        raise ValueError("targets.erased_train_text must be non-empty.")


def schema_features() -> Any:
    from datasets import Features, Sequence, Value

    return Features(
        {
            "id": Value("string"),
            "source_id": Value("string"),
            "split": Value("string"),
            "context_tokens": Value("int32"),
            "tokens_to_generate": Value("int32"),
            "seed": Value("int32"),
            "prompt_format": Value("string"),
            "query": Value("string"),
            "cached_prompt": Value("string"),
            "query_suffix": Value("string"),
            "erase_span_text": Value("string"),
            "erase_span_start": Value("int32"),
            "erase_span_end": Value("int32"),
            "original_answers": Sequence(Value("string")),
            "erased_answers": Sequence(Value("string")),
            "erased_train_text": Value("string"),
            "edited_cached_prompt": Value("string"),
        }
    )


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


def _local_data_files(context_tokens: int, data_root: Path | None = None) -> dict[str, str] | None:
    if data_root is None:
        data_root = DEFAULT_LOCAL_DATA_ROOT

    context_dir = data_root / "niah" / context_dir_name(context_tokens)
    data_files: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        data_files[split] = str(context_dir / filename)
    if not all(Path(path).exists() for path in data_files.values()):
        return None
    return data_files


def load_context_splits(
    *,
    context_tokens: int,
) -> ContextSplits:
    from datasets import load_dataset

    local_data_files = _local_data_files(context_tokens)
    if local_data_files is not None:
        dataset_dict = load_dataset(
            "parquet",
            data_files=local_data_files,
        )
    else:
        cfg = config_name(context_tokens)
        try:
            dataset_dict = load_dataset(DEFAULT_DATASET_REPO, cfg)
        except Exception:
            data_files = {
                split: f"niah/{context_dir_name(context_tokens)}/{split}-00000-of-00001.parquet"
                for split in ("train", "validation", "test")
            }
            dataset_dict = load_dataset(DEFAULT_DATASET_REPO, data_files=data_files)
    splits = {
        split: [row_to_sample(row) for row in dataset_dict[split]]
        for split in ("train", "validation", "test")
    }
    return ContextSplits(
        context_tokens=int(context_tokens),
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
    )


def load_all_context_splits(
    *,
    contexts: tuple[int, ...] = CONTEXT_SIZES,
) -> dict[int, ContextSplits]:
    return {
        int(ctx): load_context_splits(
            context_tokens=int(ctx),
        )
        for ctx in contexts
    }


def validate_split_counts(context_splits: ContextSplits) -> None:
    expected = {
        "train": TRAIN_ROWS_PER_CONTEXT,
        "validation": VALIDATION_ROWS_PER_CONTEXT,
        "test": TEST_ROWS_PER_CONTEXT,
    }
    actual = {
        "train": len(context_splits.train),
        "validation": len(context_splits.validation),
        "test": len(context_splits.test),
    }
    for split, expected_count in expected.items():
        if actual[split] != expected_count:
            raise ValueError(
                f"context={context_splits.context_tokens} split={split} has "
                f"{actual[split]} rows; expected {expected_count}."
            )
    split_ids = {
        "train": {sample["id"] for sample in context_splits.train},
        "validation": {sample["id"] for sample in context_splits.validation},
        "test": {sample["id"] for sample in context_splits.test},
    }
    if split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"] or split_ids["validation"] & split_ids["test"]:
        raise ValueError(f"Split ID leakage detected for context={context_splits.context_tokens}.")


def build_joint_train_sweep(
    splits_by_context: dict[int, ContextSplits],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    train_samples: list[dict[str, Any]] = []
    for ctx in sorted(splits_by_context):
        for sample in splits_by_context[ctx].train:
            tagged = dict(sample)
            tagged["_context_tokens"] = int(ctx)
            train_samples.append(tagged)
    rng = random.Random(int(seed))
    rng.shuffle(train_samples)
    return train_samples
