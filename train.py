from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from tqdm.auto import tqdm

from kveraser.data import (
    DEFAULT_DATASET_REPO,
    TOKENS_TO_GENERATE,
    build_joint_train_sweep,
    erase_cached_prefix,
    load_all_context_splits,
    parse_answer_numbers,
    parse_contexts,
    score_erased_prediction,
    target_text,
    validate_split_counts,
)
from kveraser.model import KVEraser, answer_loss
from kveraser.runtime import (
    SUPPORTED_PLACEMENTS,
    cleanup_cuda,
    cuda_environment,
    dtype_from_name,
    dtype_name,
    load_eraser_backbone,
    load_generator_model,
    load_tokenizer,
    parameter_counts,
    print_runtime_report,
    require_cuda,
    resolved_device_map,
    save_eraser_checkpoint,
    set_seed,
    setup_wandb,
    utc_now,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a full-model KVEraser checkpoint on the six-context NIAH sweep.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--contexts", default="all", help="Comma-separated context sizes, or 'all'.")
    parser.add_argument("--placement", choices=SUPPORTED_PLACEMENTS, default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=os.path.join("checkpoints", "kveraser_niah_joint"))
    parser.add_argument("--attn_backend", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--wandb_project", default="kveraser-niah")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def _input_device(model: KVEraser) -> torch.device:
    return KVEraser._model_input_device(model.generator)


def _query_ids(model: KVEraser, sample: dict[str, Any]) -> torch.Tensor:
    return model.tokenizer(
        sample["query_suffix"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(_input_device(model))


def _target_ids(model: KVEraser, sample: dict[str, Any]) -> torch.Tensor:
    return model.tokenizer(
        target_text(sample, model.tokenizer),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(_input_device(model))


def train_epoch(
    *,
    model: KVEraser,
    train_samples: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    epoch: int,
    global_step: int,
    wandb_run: Any,
) -> int:
    model.eraser.train()
    trainable_params = [param for param in model.eraser.parameters() if param.requires_grad]
    progress = tqdm(train_samples, desc=f"Train epoch {epoch}", unit="sample")
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for sample_index, sample in enumerate(progress, start=1):
        old_context_kv, cached_input_ids, cached_offsets = model.prefill(sample["cached_prompt"])
        updated_kv = erase_cached_prefix(
            model,
            old_context_kv,
            cached_input_ids,
            cached_offsets,
            sample,
        )
        query_ids = _query_ids(model, sample)
        target_ids = _target_ids(model, sample)

        loss = answer_loss(model.generator, updated_kv, query_ids, target_ids)
        loss.backward()
        raw_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        loss_value = float(loss.detach().item())
        running_loss += loss_value
        lr = float(optimizer.param_groups[0]["lr"])
        progress.set_postfix(loss=f"{running_loss / sample_index:.4f}", grad=f"{float(raw_grad_norm):.2f}")
        wandb_run.log(
            {
                "train/loss": loss_value,
                "train/lr": lr,
                "train/grad_norm": float(raw_grad_norm),
                "train/epoch": int(epoch),
                "train/context_tokens": int(sample["_context_tokens"]),
            },
            step=global_step,
        )

        del old_context_kv, cached_input_ids, cached_offsets, updated_kv, query_ids, target_ids, loss
    return global_step


@torch.no_grad()
def evaluate_validation(model: KVEraser, splits_by_context: dict[int, Any]) -> dict[str, float]:
    model.eraser.eval()
    metrics: dict[str, float] = {}
    context_ems: list[float] = []
    for ctx in sorted(splits_by_context):
        subset = splits_by_context[ctx].validation
        correct = 0
        progress = tqdm(subset, desc=f"Validate {ctx}", unit="sample")
        for sample in progress:
            old_context_kv, cached_input_ids, cached_offsets = model.prefill(sample["cached_prompt"])
            updated_kv = erase_cached_prefix(
                model,
                old_context_kv,
                cached_input_ids,
                cached_offsets,
                sample,
            )
            last_prompt_token, prompt_seq_len, prompt_kv = model.prime_query(
                updated_kv,
                sample["query_suffix"],
            )
            generated_text, _ = model.decode_with_count(
                last_prompt_token,
                prompt_seq_len,
                prompt_kv,
                TOKENS_TO_GENERATE,
            )
            correct += int(score_erased_prediction(parse_answer_numbers(generated_text), sample))
        em = correct / float(len(subset))
        metrics[f"eval/ctx{ctx}/validation_em"] = float(em)
        context_ems.append(float(em))
    metrics["eval/macro_validation_em"] = sum(context_ems) / float(len(context_ems))
    model.eraser.train()
    return metrics


def build_metadata(args: argparse.Namespace, *, train_examples: int, generator: Any, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "model": args.model,
        "dataset_repo": DEFAULT_DATASET_REPO,
        "contexts": list(parse_contexts(args.contexts)),
        "tokens_to_generate": TOKENS_TO_GENERATE,
        "train_examples": int(train_examples),
        "train_rows_per_context": 200,
        "validation_rows_per_context": 100,
        "test_rows_per_context": 100,
        "epochs": int(args.epochs),
        "batch_size": 1,
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "seed": int(args.seed),
        "placement": args.placement,
        "dtype": dtype_name(dtype),
        "attn_backend": args.attn_backend,
        "generator_device_map": resolved_device_map(generator),
        "runtime": cuda_environment(),
    }


def main(args: argparse.Namespace) -> None:
    if int(args.epochs) != 1:
        raise ValueError("This NIAH training entry point requires --epochs 1.")
    require_cuda()
    set_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    contexts = parse_contexts(args.contexts)

    splits_by_context = load_all_context_splits(contexts=contexts)
    for context_splits in splits_by_context.values():
        validate_split_counts(context_splits)
    train_samples = build_joint_train_sweep(splits_by_context, seed=args.seed)

    print(f"Model={args.model}")
    print(f"Contexts={','.join(str(ctx) for ctx in contexts)}")
    print(f"Train examples={len(train_samples)}")
    print(f"Output directory={args.output_dir}")

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
    eraser = load_eraser_backbone(
        args.model,
        placement=args.placement,
        dtype=dtype,
        attn_backend=args.attn_backend,
    )
    eraser.train()
    trainable_params, total_params = parameter_counts(eraser)
    if trainable_params == 0:
        raise RuntimeError("Eraser backbone has no trainable parameters.")
    print(f"Trainable eraser params={trainable_params} / {total_params}")

    model = KVEraser(tokenizer=tokenizer, generator=generator, eraser=eraser)
    optimizer = torch.optim.AdamW(
        [param for param in eraser.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    wandb_config = vars(args).copy()
    wandb_config.update({"contexts": list(contexts), "train_examples": len(train_samples), "batch_size": 1})
    wandb_run = setup_wandb(
        project=args.wandb_project,
        config=wandb_config,
        mode=args.wandb_mode,
        run_name=args.wandb_run_name,
    )

    global_step = 0
    try:
        for epoch in range(1, int(args.epochs) + 1):
            global_step = train_epoch(
                model=model,
                train_samples=train_samples,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
                epoch=epoch,
                global_step=global_step,
                wandb_run=wandb_run,
            )
            validation_metrics = evaluate_validation(model, splits_by_context)
            wandb_run.log(validation_metrics, step=global_step)
            for name, value in sorted(validation_metrics.items()):
                print(f"{name}={value:.3f}")

        metadata = build_metadata(
            args,
            train_examples=len(train_samples),
            generator=generator,
            dtype=dtype,
        )
        save_eraser_checkpoint(eraser, args.output_dir, metadata)
        wandb_run.log({"checkpoint/path": os.path.abspath(args.output_dir)}, step=global_step)
        print(f"Saved eraser checkpoint to {args.output_dir}")
    finally:
        wandb_run.finish()
        del optimizer, model, eraser, generator, tokenizer
        cleanup_cuda()


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
