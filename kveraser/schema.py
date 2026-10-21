"""Qwen3-8B dataset names, Parquet fields, and record validation."""
from dataclasses import dataclass

import pyarrow as pa

DATA_PREFIX = "qwen3_8b/v1"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CONTEXT_SIZES = (1000, 2000, 4000, 8000, 16000, 32000)


@dataclass(frozen=True)
class DatasetSpec:
    train: int = 0
    validation: int = 0
    test: int = 0
    tokens_to_generate: int = 32

    @property
    def splits(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in ("train", "validation", "test")
                if getattr(self, name)}


DATASETS = {
    "pretrain/wiki/span": DatasetSpec(train=72000, validation=8000, tokens_to_generate=24),
    **{f"finetune/niah/ctx{ctx}": DatasetSpec(train=200, validation=100)
       for ctx in CONTEXT_SIZES},
    "finetune/nq/default": DatasetSpec(train=1793, validation=224),
    "finetune/triviaqa/default": DatasetSpec(train=2010, validation=251),
    "finetune/hotpotqa/default": DatasetSpec(train=2488, validation=311),
    "finetune/toolbench/default": DatasetSpec(train=4000, validation=256),
    **{f"eval/niah/ctx{ctx}": DatasetSpec(test=100) for ctx in CONTEXT_SIZES},
    "eval/2wiki/default": DatasetSpec(test=344),
    "eval/musique/default": DatasetSpec(test=300),
    "eval/iirc/default": DatasetSpec(test=178),
    "eval/bfcl/default": DatasetSpec(test=100),
}


def data_path(dataset: str, split: str) -> str:
    """Return the shard glob relative to either a local data root or HF repo."""
    if dataset not in DATASETS or split not in DATASETS[dataset].splits:
        raise ValueError(f"Unsupported dataset/split: {dataset}/{split}")
    return f"{DATA_PREFIX}/{dataset}/{split}-*.parquet"


def parquet_schema() -> pa.Schema:
    def field(name, dtype):
        return pa.field(name, dtype, nullable=False)

    strings = pa.list_(field("element", pa.string()))
    return pa.schema([
        field("id", pa.string()),
        field("cached_prompt", pa.string()),
        field("query_suffix", pa.string()),
        field("erase_span", pa.struct([
            field("text", pa.string()),
            field("cached_prompt_char_span", pa.list_(field("element", pa.int32()))),
        ])),
        field("targets", pa.struct([
            field("original_answers", strings),
            field("erased_answers", strings),
            field("erased_train_text", pa.string()),
        ])),
        field("variants", pa.struct([
            field("edited_cached_prompt", pa.string()),
            pa.field("prompt_forget_query_suffix", pa.string()),
        ])),
        field("candidate_tool_names", strings),
    ], metadata={"kveraser_format": DATA_PREFIX,
                 "tokenizer": "Qwen/Qwen3-8B", "tokenizer_revision": TOKENIZER_REVISION})


def validate_record(record: dict, dataset: str) -> None:
    """Validate one row; spans use Python Unicode character indices [start, end)."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")
    blocks = {"record": (record, parquet_schema().names),
              "erase_span": (record.get("erase_span"), ("text", "cached_prompt_char_span")),
              "targets": (record.get("targets"),
                          ("original_answers", "erased_answers", "erased_train_text")),
              "variants": (record.get("variants"),
                           ("edited_cached_prompt", "prompt_forget_query_suffix"))}
    for name, (value, keys) in blocks.items():
        if not isinstance(value, dict) or set(value) != set(keys):
            raise ValueError(f"{name} must contain exactly {list(keys)}")
    span, targets, variants = record["erase_span"], record["targets"], record["variants"]
    for name, value in [(key, record[key]) for key in ("id", "cached_prompt", "query_suffix")] + [
        ("erase_span.text", span["text"]), ("erased_train_text", targets["erased_train_text"]),
    ]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    bounds = span["cached_prompt_char_span"]
    if not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not int for v in bounds):
        raise ValueError("erase span must contain two integer character indices")
    start, end = bounds
    prompt = record["cached_prompt"]
    if not 0 <= start < end <= len(prompt):
        raise ValueError("erase span is empty or out of bounds")
    if prompt[start:end] != span["text"]:
        raise ValueError("erase span text does not match cached_prompt")
    if variants["edited_cached_prompt"] != prompt[:start] + prompt[end:]:
        raise ValueError("edited_cached_prompt must equal cached_prompt with the span removed")
    suffix = variants["prompt_forget_query_suffix"]
    if suffix is not None and (not isinstance(suffix, str) or not suffix.strip()):
        raise ValueError("prompt_forget_query_suffix must be a non-empty string or null")
    if dataset.startswith("eval/") and suffix is None:
        raise ValueError("evaluation requires prompt_forget_query_suffix")
    for name, values in [(key, targets[key]) for key in ("original_answers", "erased_answers")] + [
        ("candidate_tool_names", record["candidate_tool_names"]),
    ]:
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{name} must be a list of non-empty strings")
        if name != "candidate_tool_names" and not values:
            raise ValueError(f"{name} must not be empty")
    candidates = record["candidate_tool_names"]
    if dataset.split("/")[1] in {"toolbench", "bfcl"}:
        answers = targets["original_answers"] + targets["erased_answers"]
        if not candidates or not set(answers).issubset(candidates):
            raise ValueError("tool answers must appear in candidate_tool_names")
        if targets["erased_train_text"] not in targets["erased_answers"]:
            raise ValueError("tool training target must match an erased answer")
    elif candidates:
        raise ValueError("candidate_tool_names must be empty for non-tool tasks")
