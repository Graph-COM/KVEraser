"""Shared teacher-forced training for pretraining and task-specific LoRA adapters."""
from dataclasses import dataclass
import math

import torch
from tqdm.auto import tqdm

from .data import erase_cached_prefix, target_text
from .model import answer_loss
from .runtime import DisabledWandbRun


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
