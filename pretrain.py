"""Pretrain a Qwen3-8B LoRA KVEraser on Wikipedia span-erasure data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
from tqdm.auto import tqdm

from kveraser.adapters import BASE_MODEL, BASE_REVISION, save_adapter
from kveraser.data import DEFAULT_DATASET_REPO, DatasetSource
from kveraser.model import KVEraser
from kveraser.runtime import (
    load_tokenizer, load_training_models, require_cuda, save_json, set_seed, setup_wandb, use_local_files_only,
)
from kveraser.training import TrainingConfig, create_optimizer, sample_loss, train_epoch


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--data_revision", default="main")
    parser.add_argument("--data_root", help="Local directory containing qwen3_8b/v1/manifest.json.")
    parser.add_argument("--cache_dir")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--placement", choices=("single_gpu", "balanced"), default="single_gpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_samples", type=int, help="Limit each epoch to this many samples after shuffling.")
    parser.add_argument("--max_validation_samples", type=int, help="Limit final validation to this many samples.")
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb_project", default="kv-eraser")
    parser.add_argument("--wandb_name")
    parser.add_argument("--output_dir", default="checkpoints/pretrain", help="An empty output directory.")
    return parser


def epoch_sample_order(sample_count, *, seed, epoch):
    """Shuffle row indices with a deterministic seed for each epoch."""
    digest = hashlib.sha1(f"{seed}:{epoch}".encode("utf-8")).digest()
    indices = list(range(sample_count))
    random.Random(int.from_bytes(digest[:8], byteorder="big")).shuffle(indices)
    return indices


def main(args):
    for name in ("epochs", "max_samples", "max_validation_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive.")
    training = TrainingConfig(args.lr, args.weight_decay, args.max_grad_norm, args.grad_accum_steps)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    require_cuda()
    set_seed(args.seed)
    offline = args.local_files_only or use_local_files_only()
    source = DatasetSource(data_root=args.data_root, repo_id=args.data_repo, revision=args.data_revision,
                           cache_dir=args.cache_dir, local_files_only=offline)
    samples = source.load("pretrain/wiki/span", "train")
    validation = source.load("pretrain/wiki/span", "validation")[:args.max_validation_samples]
    tokenizer = load_tokenizer(BASE_MODEL, revision=BASE_REVISION,
                               cache_dir=args.cache_dir, local_files_only=offline)
    generator, eraser = load_training_models(placement=args.placement, cache_dir=args.cache_dir,
                                              local_files_only=offline)
    model = KVEraser(tokenizer, generator, eraser)
    optimizer = create_optimizer(model, training)
    config = dict(vars(args), **asdict(training), base_model=BASE_MODEL, base_revision=BASE_REVISION,
                  tokenizer_revision=BASE_REVISION, dtype="bf16", attn_backend="sdpa",
                  local_files_only=offline, data_commit=source.revision,
                  train_samples=min(len(samples), args.max_samples or len(samples)), validation_samples=len(validation))
    save_json(str(output / "run_config.json"), config)
    run = setup_wandb(project=args.wandb_project, mode=args.wandb_mode, run_name=args.wandb_name, config=config)
    epochs, global_step = [], 0
    try:
        for epoch in range(1, args.epochs + 1):
            order = epoch_sample_order(len(samples), seed=args.seed, epoch=epoch)[:args.max_samples]
            metrics = train_epoch(model, [samples[i] for i in order], optimizer, training,
                                  wandb_run=run, epoch=epoch, global_step=global_step)
            epochs.append(dict(epoch=epoch, **metrics))
            global_step = metrics["global_step"]
        save_adapter(eraser, output)
        model.eval()
        with torch.no_grad():
            losses = [float(sample_loss(model, sample)) for sample in tqdm(validation, desc="Validation", unit="sample")]
        validation_loss = sum(losses) / len(losses)
        run.log({"validation/loss": validation_loss, "validation/samples": len(validation)}, step=global_step + 1)
        summary = dict(epochs=epochs, validation_loss=validation_loss, validation_samples=len(validation))
        save_json(str(output / "metrics.json"), summary)
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        run.finish()


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
