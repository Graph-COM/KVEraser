"""Evaluate Qwen3-8B KVEraser adapters and context-erasing baselines."""
import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from kveraser.adapters import BASE_MODEL, BASE_REVISION, DEFAULT_CHECKPOINT_REPO, AdapterCheckpoint
from kveraser.data import DEFAULT_DATASET_REPO, DatasetSource, erase_cached_prefix, resolve_erase_token_span
from kveraser.metrics import EVALUATION_SUITE, aggregate_scores, score_prediction
from kveraser.model import KVEraser, PARTIAL_RECOMPUTE_FRACTION
from kveraser.performance import OnlineMeasurement, summarize_performance
from kveraser.runtime import load_evaluation_models, require_cuda, save_json, set_seed, use_local_files_only
from kveraser.schema import DATASETS


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_REPO, help="HF model ID or local adapter directory.")
    parser.add_argument("--checkpoint_revision", default="main")
    parser.add_argument("--method", choices=("kveraser", "full-recompute", "discard", "instruction",
                                           "partial-erase", "partial-query"),
                        default="kveraser")
    parser.add_argument("--datasets", nargs="+", choices=EVALUATION_SUITE, default=EVALUATION_SUITE)
    parser.add_argument("--data_repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--data_revision", default="main")
    parser.add_argument("--data_root", help="Local directory containing qwen3_8b/v1/manifest.json.")
    parser.add_argument("--cache_dir")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--placement", choices=("single_gpu", "balanced"), default="single_gpu")
    parser.add_argument("--max_samples", type=int, help="Positive sample limit per dataset.")
    parser.add_argument("--measure_performance", action="store_true", help="Record online time and CUDA memory.")
    parser.add_argument("--warmup_samples", type=int, default=3, help="Warmup examples per dataset when measuring.")
    parser.add_argument("--output_dir", default="outputs/evaluation", help="An empty output directory.")
    return parser


@torch.inference_mode()
def predict(model, sample, method, max_new_tokens, measurement=None):
    query = sample["query_suffix"]
    if measurement is not None:
        measurement.prepare()
    cache, ids, offsets = model.prefill(sample["cached_prompt"])
    if measurement is not None:
        measurement.start()
    if method in ("kveraser", "discard", "instruction", "partial-erase", "partial-query"):
        if method == "kveraser":
            cache = erase_cached_prefix(model, cache, ids, offsets, sample)
        elif method in ("discard", "partial-erase", "partial-query"):
            start, end = resolve_erase_token_span(offsets, sample)
            if method == "discard":
                cache = model.discard_span_cache(cache, start, end)
            else:
                cache = model.partial_recompute_dropped_cache(cache, ids, start, end, method.removeprefix("partial-"))
        else:
            query = sample["variants"]["prompt_forget_query_suffix"]
    elif method == "full-recompute":
        cache, _ = model.recompute_edited_cached_prompt(
            cache, ids, sample["variants"]["edited_cached_prompt"], consume_cache=True)
    else:
        raise ValueError(f"Unsupported method: {method}")
    last, length, cache = model.prime_query(cache, query)
    result = model.decode_with_count(last, length, cache, max_new_tokens)
    if measurement is not None:
        measurement.stop()
    return result


def evaluate_dataset(model, dataset, samples, method, output_dir, measurement=None, warmup_samples=3):
    records = []
    if measurement is not None:
        for sample in samples[:warmup_samples]:
            predict(model, sample, method, DATASETS[dataset].tokens_to_generate, measurement)
    path = output_dir / f"{dataset.replace('/', '__')}.jsonl"
    with path.open("x", encoding="utf-8", buffering=1) as stream:
        for sample in tqdm(samples, desc=dataset, unit="sample"):
            text, count = predict(model, sample, method, DATASETS[dataset].tokens_to_generate, measurement)
            record = dict(id=sample["id"], dataset=dataset, generated_text=text, generated_tokens=count,
                          **score_prediction(dataset, text, sample))
            if measurement is not None:
                record.update(measurement.stats)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
    performance = summarize_performance(records) if measurement is not None else {}
    return dict(aggregate_scores(records), tokens_to_generate=DATASETS[dataset].tokens_to_generate,
                predictions=path.name, **performance)


def main(args):
    if args.warmup_samples < 0:
        raise ValueError("--warmup_samples must be nonnegative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("--datasets must not contain duplicates.")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    require_cuda()
    set_seed(42)
    offline = args.local_files_only or use_local_files_only()
    source = DatasetSource(data_root=args.data_root, repo_id=args.data_repo,
                           revision=args.data_revision, cache_dir=args.cache_dir, local_files_only=offline)
    adapter = None
    if args.method == "kveraser":
        adapter = AdapterCheckpoint(args.checkpoint, revision=args.checkpoint_revision,
                                    cache_dir=args.cache_dir, local_files_only=offline)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
                                             cache_dir=args.cache_dir, local_files_only=offline)
    generator, eraser = load_evaluation_models(adapter, placement=args.placement,
                                              cache_dir=args.cache_dir, local_files_only=offline)
    model = KVEraser(tokenizer, generator, eraser).eval()
    measurement = OnlineMeasurement(generator) if args.measure_performance else None
    config = dict(vars(args), seed=42, base_model=BASE_MODEL, base_revision=BASE_REVISION,
                  tokenizer_revision=BASE_REVISION, dtype="bf16", attn_backend="sdpa",
                  local_files_only=offline, data_commit=source.revision,
                  checkpoint_commit=adapter.revision if adapter is not None else None)
    if adapter is None:
        config.update(checkpoint=None, checkpoint_revision=None)
    if args.method.startswith("partial-"):
        config["recompute_fraction"] = PARTIAL_RECOMPUTE_FRACTION
    if measurement is not None:
        config["performance_environment"] = measurement.environment()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset in args.datasets:
        samples = source.load(dataset, "test")
        if args.max_samples is not None:
            samples = samples[:args.max_samples]
        results[dataset] = evaluate_dataset(model, dataset, samples, args.method, output, measurement, args.warmup_samples)
        print(json.dumps({"dataset": dataset, **results[dataset]}), flush=True)
    summary = {"config": config, "results": results}
    save_json(str(output / "summary.json"), summary)
    return summary


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
