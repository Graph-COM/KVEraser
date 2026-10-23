from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


SUPPORTED_PLACEMENTS = ("single_gpu", "balanced")
SUPPORTED_ATTN_BACKENDS = ("sdpa", "flash_attention_2")


def use_local_files_only() -> bool:
    return os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this GPU-only release script. "
            "Run in an environment where PyTorch can see the intended GPU(s)."
        )


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _native_qwen3_max_length(model_config: Any) -> int | None:
    if getattr(model_config, "model_type", None) != "qwen3":
        return None
    if getattr(model_config, "rope_scaling", None) is not None:
        return None
    try:
        max_pos = int(getattr(model_config, "max_position_embeddings", 0))
    except Exception:
        return None
    if max_pos == 40960:
        return 32768
    return None


def effective_model_max_length(tokenizer: Any, model_config: Any = None) -> int | None:
    candidates: list[int] = []
    for value in (
        getattr(tokenizer, "model_max_length", None),
        getattr(model_config, "max_position_embeddings", None),
        _native_qwen3_max_length(model_config),
    ):
        if value is None:
            continue
        try:
            value = int(value)
        except Exception:
            continue
        if 0 < value < 10**9:
            candidates.append(value)
    return min(candidates) if candidates else None


def validate_attn_backend(attn_backend: str, dtype: torch.dtype) -> None:
    if attn_backend not in SUPPORTED_ATTN_BACKENDS:
        raise ValueError(f"--attn_backend must be one of: {', '.join(SUPPORTED_ATTN_BACKENDS)}.")
    if attn_backend == "flash_attention_2" and dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("flash_attention_2 requires fp16 or bf16 weights.")


def resolve_device_map(placement: str) -> dict[str, int] | str:
    if placement not in SUPPORTED_PLACEMENTS:
        raise ValueError(f"--placement must be one of: {', '.join(SUPPORTED_PLACEMENTS)}.")
    if placement == "single_gpu":
        require_cuda()
        if torch.cuda.device_count() < 1:
            raise RuntimeError("Placement single_gpu requires one visible CUDA device.")
        return {"": 0}
    require_cuda()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Placement balanced requires at least two visible CUDA devices.")
    return "balanced"


def _from_pretrained(model_cls: Any, model_name_or_path: str, **kwargs: Any) -> Any:
    try:
        return model_cls.from_pretrained(model_name_or_path, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return model_cls.from_pretrained(model_name_or_path, **kwargs)


def model_load_kwargs(
    placement: str,
    *,
    dtype: torch.dtype,
    attn_backend: str,
    local_files_only: bool,
    use_safetensors: bool = True,
) -> dict[str, Any]:
    validate_attn_backend(attn_backend, dtype)
    return {
        "attn_implementation": attn_backend,
        "dtype": dtype,
        "trust_remote_code": True,
        "device_map": resolve_device_map(placement),
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
        "use_safetensors": use_safetensors,
    }


def load_tokenizer(model_name_or_path: str, *, local_files_only: bool | None = None,
                   revision: str | None = None, cache_dir: str | None = None) -> Any:
    if local_files_only is None:
        local_files_only = use_local_files_only()
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        model_name_or_path = str(local_path)
    elif local_files_only:
        # A local snapshot also prevents tokenizer-specific Hub metadata lookups.
        model_name_or_path = snapshot_download(model_name_or_path, revision=revision,
                                               cache_dir=cache_dir, local_files_only=True)
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def freeze_model(model: Any) -> Any:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_training_models(
    *, checkpoint=None, checkpoint_revision="main", placement="single_gpu",
    cache_dir=None, local_files_only=None,
):
    """Load a frozen generator and a separate backbone with a trainable LoRA adapter."""
    from .adapters import BASE_MODEL, BASE_REVISION, AdapterCheckpoint, initialize_adapter

    require_cuda()
    if local_files_only is None:
        local_files_only = use_local_files_only()
    adapter = checkpoint
    if checkpoint is not None and not isinstance(checkpoint, AdapterCheckpoint):
        adapter = AdapterCheckpoint(checkpoint, revision=checkpoint_revision, cache_dir=cache_dir,
                                    local_files_only=local_files_only)
    kwargs = model_load_kwargs(placement, dtype=torch.bfloat16, attn_backend="sdpa",
                               local_files_only=local_files_only)
    kwargs.update(revision=BASE_REVISION, cache_dir=cache_dir)
    generator = freeze_model(_from_pretrained(AutoModelForCausalLM, BASE_MODEL, **kwargs))
    backbone = _from_pretrained(AutoModel, BASE_MODEL, **kwargs)
    eraser = (initialize_adapter(backbone) if adapter is None
              else adapter.load(backbone, trainable=True))
    return generator, eraser


def load_evaluation_models(
    checkpoint="Graph-COM/KVEraser", *, checkpoint_revision="main", placement="single_gpu",
    cache_dir=None, local_files_only=None,
):
    """Load a frozen BF16 backbone (None omits the adapter); route through KVEraser."""
    from .adapters import BASE_MODEL, BASE_REVISION, AdapterCheckpoint

    require_cuda()
    if local_files_only is None:
        local_files_only = use_local_files_only()
    adapter = checkpoint
    if checkpoint is not None and not isinstance(checkpoint, AdapterCheckpoint):
        adapter = AdapterCheckpoint(checkpoint, revision=checkpoint_revision, cache_dir=cache_dir,
                                    local_files_only=local_files_only)
    kwargs = model_load_kwargs(placement, dtype=torch.bfloat16, attn_backend="sdpa",
                               local_files_only=local_files_only)
    kwargs.update(revision=BASE_REVISION, cache_dir=cache_dir)
    generator = freeze_model(_from_pretrained(AutoModelForCausalLM, BASE_MODEL, **kwargs))
    eraser = None if adapter is None else adapter.load(generator.base_model)
    return generator, eraser


def save_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


class DisabledWandbRun:
    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        del payload, step

    def finish(self) -> None:
        return None


def setup_wandb(
    *,
    project: str,
    config: dict[str, Any],
    mode: str,
    run_name: str | None = None,
) -> Any:
    if mode == "disabled":
        return DisabledWandbRun()
    import wandb

    wandb_dir = os.path.join(os.getcwd(), "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", wandb_dir)
    init_kwargs: dict[str, Any] = {
        "project": project,
        "config": config,
        "mode": mode,
    }
    if run_name:
        init_kwargs["name"] = run_name
    return wandb.init(**init_kwargs)
