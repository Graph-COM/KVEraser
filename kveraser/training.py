"""Shared teacher-forced training for pretraining and task-specific LoRA adapters."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import math

import torch
from tqdm.auto import tqdm

from .adapters import save_adapter
from .data import DEFAULT_DATASET_REPO, erase_cached_prefix, target_text
from .model import answer_loss
from .runtime import DisabledWandbRun, save_json, setup_wandb


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1

    def __post_init__(self):
        if type(self.grad_accum_steps) is not int or self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be a positive integer.")
        if (not all(math.isfinite(v) for v in (self.lr, self.weight_decay, self.max_grad_norm))
                or self.lr <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0):
            raise ValueError("lr and max_grad_norm must be positive; weight_decay must be nonnegative.")


def create_optimizer(model, config: TrainingConfig):
    """Create AdamW over the separate eraser's trainable LoRA parameters only."""
    if model.eraser is None or any(p.requires_grad for p in model.generator.parameters()):
        raise ValueError("Training requires a frozen generator and a LoRA eraser.")
    parameters = [(name, p) for name, p in model.eraser.named_parameters() if p.requires_grad]
    if not parameters or any(".lora_" not in name for name, _ in parameters):
        raise ValueError("Only LoRA adapter parameters may be trainable.")
    return torch.optim.AdamW([p for _, p in parameters], lr=config.lr, weight_decay=config.weight_decay)


def sample_loss(model, sample):
    """Compute answer-token cross entropy, including EOS, through the edited KV cache."""
    cache, ids, offsets = model.prefill(sample["cached_prompt"])
    updated = erase_cached_prefix(model, cache, ids, offsets, sample)
    device = model._model_input_device(model.generator)
    query_ids = model.tokenizer(sample["query_suffix"], add_special_tokens=False,
                                return_tensors="pt")["input_ids"].to(device)
    target_ids = model.tokenizer(target_text(sample, model.tokenizer), add_special_tokens=False,
                                 return_tensors="pt")["input_ids"].to(device)
    return answer_loss(model.generator, updated, query_ids, target_ids)


def train_epoch(model, samples, optimizer, config: TrainingConfig, *,
                wandb_run=None, epoch=1, global_step=0):
    """Train in the supplied order; average each accumulation group by its actual size.

    global_step counts samples. The returned optimizer_steps counts this epoch's updates.
    """
    if not samples:
        raise ValueError("Training samples must not be empty.")
    model.train()
    run = DisabledWandbRun() if wandb_run is None else wandb_run
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    optimizer.zero_grad(set_to_none=True)
    loss_sum, optimizer_steps = 0.0, 0
    progress = tqdm(samples, desc=f"Train epoch {epoch}", unit="sample")
    for index, sample in enumerate(progress):
        group_start = index // config.grad_accum_steps * config.grad_accum_steps
        group_end = min(group_start + config.grad_accum_steps, len(samples))
        loss = sample_loss(model, sample)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss for sample {sample['id']}.")
        (loss / (group_end - group_start)).backward()
        loss_sum += loss_value
        metrics = {"train/loss": loss_sum / (index + 1), "train/loss_sample": loss_value,
                   "train/lr": optimizer.param_groups[0]["lr"], "train/epoch": epoch,
                   "train/step_in_epoch": index + 1}
        if index + 1 == group_end:
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm,
                                                       error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            metrics.update({"train/raw_grad_norm": norm, "train/grad_norm": min(norm, config.max_grad_norm)})
        metrics["train/optimizer_step"] = optimizer_steps
        run.log(metrics, step=global_step + index + 1)
        progress.set_postfix(loss=f"{loss_sum / (index + 1):.4f}")
        del loss
    return dict(loss=loss_sum / len(samples), samples=len(samples), optimizer_steps=optimizer_steps,
                global_step=global_step + len(samples))


def build_training_parser(description, *, grad_accum_steps=8, output_dir="checkpoints/pretrain"):
    parser = argparse.ArgumentParser(description=description)
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
    parser.add_argument("--grad_accum_steps", type=int, default=grad_accum_steps)
    parser.add_argument("--max_samples", type=int, help="Limit each epoch to this many samples after shuffling.")
    parser.add_argument("--max_validation_samples", type=int, help="Limit final validation to this many samples.")
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb_project", default="kv-eraser")
    parser.add_argument("--wandb_name")
    parser.add_argument("--output_dir", default=output_dir, help="An empty output directory.")
    return parser


def run_training(model, samples, validation, training, config, epoch_sample_order):
    """Train from the supplied adapter with a fresh optimizer, then save and validate."""
    output = Path(config["output_dir"])
    optimizer = create_optimizer(model, training)
    save_json(str(output / "run_config.json"), config)
    run = setup_wandb(project=config["wandb_project"], mode=config["wandb_mode"], run_name=config["wandb_name"], config=config)
    epochs, global_step = [], 0
    try:
        for epoch in range(1, config["epochs"] + 1):
            order = epoch_sample_order(len(samples), seed=config["seed"], epoch=epoch)[:config["max_samples"]]
            metrics = train_epoch(model, [samples[i] for i in order], optimizer, training,
                                  wandb_run=run, epoch=epoch, global_step=global_step)
            epochs.append(dict(epoch=epoch, **metrics))
            global_step = metrics["global_step"]
        save_adapter(model.eraser, output)
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
