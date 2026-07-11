from __future__ import annotations

import argparse
import os
import time
from typing import Any

import torch
from tqdm.auto import tqdm

from kveraser.data import (
    TOKENS_TO_GENERATE,
    erase_cached_prefix,
    load_all_context_splits,
    parse_answer_numbers,
    parse_contexts,
    score_erased_prediction,
    validate_split_counts,
)
from kveraser.model import KVEraser
from kveraser.runtime import (
    SUPPORTED_PLACEMENTS,
    dtype_from_name,
    dtype_name,
    load_eraser_checkpoint,
    load_generator_model,
    load_json,
    load_tokenizer,
    print_runtime_report,
    require_cuda,
    save_json,
    set_seed,
    sync_for_timing,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained KVEraser on all NIAH test contexts.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contexts", default="all")
    parser.add_argument("--placement", choices=SUPPORTED_PLACEMENTS, default="single_gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_backend", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test cap per context.")
    return parser


def _time_sync() -> float:
    sync_for_timing()
    return time.perf_counter()


@torch.no_grad()
def warmup_context(model: KVEraser, sample: dict[str, Any], tokens_to_generate: int) -> None:
    old_context_kv, cached_input_ids, cached_offsets = model.prefill(sample["cached_prompt"])
    updated_kv = erase_cached_prefix(
        model,
        old_context_kv,
        cached_input_ids,
        cached_offsets,
        sample,
    )
    last_prompt_token, prompt_seq_len, prompt_kv = model.prime_query(updated_kv, sample["query_suffix"])
    model.decode_with_count(last_prompt_token, prompt_seq_len, prompt_kv, tokens_to_generate)
    sync_for_timing()


@torch.no_grad()
def evaluate_context(
    *,
    model: KVEraser,
    context_tokens: int,
    samples: list[dict[str, Any]],
    tokens_to_generate: int,
) -> dict[str, float | int]:
    if not samples:
        raise ValueError(f"context={context_tokens} has no test samples.")
    warmup_context(model, samples[0], tokens_to_generate)

    correct = 0
    generated_tokens_total = 0
    total_latency_s = 0.0
    edit_latency_s = 0.0
    query_latency_s = 0.0
    decode_latency_s = 0.0

    progress = tqdm(samples, desc=f"KVEraser ctx{context_tokens}", unit="sample")
    for sample_index, sample in enumerate(progress, start=1):
        old_context_kv, cached_input_ids, cached_offsets = model.prefill(sample["cached_prompt"])

        total_start = _time_sync()
        edit_start = time.perf_counter()
        updated_kv = erase_cached_prefix(
            model,
            old_context_kv,
            cached_input_ids,
            cached_offsets,
            sample,
        )
        edit_latency_s += _time_sync() - edit_start

        query_start = time.perf_counter()
        last_prompt_token, prompt_seq_len, prompt_kv = model.prime_query(updated_kv, sample["query_suffix"])
        query_latency_s += _time_sync() - query_start

        decode_start = time.perf_counter()
        generated_text, generated_tokens = model.decode_with_count(
            last_prompt_token,
            prompt_seq_len,
            prompt_kv,
            tokens_to_generate,
        )
        decode_latency_s += _time_sync() - decode_start
        total_latency_s += time.perf_counter() - total_start

        generated_tokens_total += int(generated_tokens)
        correct += int(score_erased_prediction(parse_answer_numbers(generated_text), sample))
        progress.set_postfix(em=f"{correct / float(sample_index):.3f}")

    sample_count = len(samples)
    return {
        "context_tokens": int(context_tokens),
        "samples": int(sample_count),
        "em": correct / float(sample_count),
        "total_latency_s": float(total_latency_s),
        "avg_latency_s": float(total_latency_s) / float(sample_count),
        "avg_generated_tokens": float(generated_tokens_total) / float(sample_count),
        "edit_latency_s": float(edit_latency_s),
        "query_latency_s": float(query_latency_s),
        "decode_latency_s": float(decode_latency_s),
    }


def main(args: argparse.Namespace) -> None:
    require_cuda()
    set_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    contexts = parse_contexts(args.contexts)

    checkpoint_metadata = None
    metadata_path = os.path.join(args.checkpoint, "run_config.json")
    if os.path.exists(metadata_path):
        checkpoint_metadata = load_json(metadata_path)
        print(f"Checkpoint metadata={metadata_path}")

    tokenizer = load_tokenizer(args.model)
    generator = load_generator_model(
        args.model,
        placement=args.placement,
        dtype=dtype,
        attn_backend=args.attn_backend,
    )
    print_runtime_report(
        generator=generator,
        placement=args.placement,
        dtype=dtype,
        attn_backend=args.attn_backend,
    )
    eraser = load_eraser_checkpoint(
        args.checkpoint,
        placement=args.placement,
        dtype=dtype,
        attn_backend=args.attn_backend,
    )
    model = KVEraser(tokenizer=tokenizer, generator=generator, eraser=eraser)

    splits_by_context = load_all_context_splits(contexts=contexts)
    for context_splits in splits_by_context.values():
        validate_split_counts(context_splits)

    per_context = []
    for ctx in contexts:
        samples = list(splits_by_context[ctx].test)
        if args.max_samples is not None:
            samples = samples[: int(args.max_samples)]
        result = evaluate_context(
            model=model,
            context_tokens=ctx,
            samples=samples,
            tokens_to_generate=TOKENS_TO_GENERATE,
        )
        per_context.append(result)
        print(
            f"context={ctx} samples={result['samples']} em={result['em']:.3f} "
            f"total_latency_s={result['total_latency_s']:.3f} "
            f"avg_latency_s={result['avg_latency_s']:.3f} "
            f"avg_generated_tokens={result['avg_generated_tokens']:.2f} "
            f"edit_s={result['edit_latency_s']:.3f} "
            f"query_s={result['query_latency_s']:.3f} "
            f"decode_s={result['decode_latency_s']:.3f}"
        )

    macro_em = sum(float(item["em"]) for item in per_context) / float(len(per_context))
    print(f"macro_em={macro_em:.3f}")
    if args.output_json:
        save_json(
            args.output_json,
            {
                "approach": "kveraser",
                "model": args.model,
                "checkpoint": args.checkpoint,
                "checkpoint_metadata": checkpoint_metadata,
                "placement": args.placement,
                "dtype": dtype_name(dtype),
                "attn_backend": args.attn_backend,
                "contexts": per_context,
                "macro_em": macro_em,
            },
        )


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
