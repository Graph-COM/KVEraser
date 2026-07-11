from __future__ import annotations

import gc
import json
import os
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


SUPPORTED_PLACEMENTS = ("single_gpu", "balanced")
SUPPORTED_ATTN_BACKENDS = ("sdpa", "flash_attention_2")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def sync_for_timing() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError("--dtype must be one of: bf16, fp16, fp32.")


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


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


def load_tokenizer(model_name_or_path: str, *, local_files_only: bool | None = None) -> Any:
    if local_files_only is None:
        local_files_only = use_local_files_only()
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def freeze_model(model: Any) -> Any:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_generator_model(
    model_name_or_path: str,
    *,
    placement: str,
    dtype: torch.dtype,
    attn_backend: str,
    local_files_only: bool | None = None,
) -> Any:
    require_cuda()
    if local_files_only is None:
        local_files_only = use_local_files_only()
    return freeze_model(
        _from_pretrained(
            AutoModelForCausalLM,
            model_name_or_path,
            **model_load_kwargs(
                placement,
                dtype=dtype,
                attn_backend=attn_backend,
                local_files_only=local_files_only,
            ),
        )
    )


def load_eraser_backbone(
    model_name_or_path: str,
    *,
    placement: str,
    dtype: torch.dtype,
    attn_backend: str,
    local_files_only: bool | None = None,
) -> Any:
    require_cuda()
    if local_files_only is None:
        local_files_only = use_local_files_only()
    return _from_pretrained(
        AutoModel,
        model_name_or_path,
        **model_load_kwargs(
            placement,
            dtype=dtype,
            attn_backend=attn_backend,
            local_files_only=local_files_only,
        ),
    )


def load_eraser_checkpoint(
    checkpoint_dir: str,
    *,
    placement: str,
    dtype: torch.dtype,
    attn_backend: str,
    local_files_only: bool | None = None,
) -> Any:
    eraser = load_eraser_backbone(
        checkpoint_dir,
        placement=placement,
        dtype=dtype,
        attn_backend=attn_backend,
        local_files_only=local_files_only,
    )
    eraser.eval()
    return eraser


def parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return int(trainable), int(total)


def save_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def save_eraser_checkpoint(eraser: Any, checkpoint_dir: str, metadata: dict[str, Any]) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    eraser.save_pretrained(checkpoint_dir, safe_serialization=True)
    save_json(os.path.join(checkpoint_dir, "run_config.json"), metadata)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def resolved_device_map(model: Any) -> Any:
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        return _jsonable(device_map)
    try:
        return {"": str(next(model.parameters()).device)}
    except StopIteration:
        return {}


def cuda_environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": [
            {
                "index": int(idx),
                "name": torch.cuda.get_device_name(idx),
                "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
            }
            for idx in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def print_runtime_report(
    *,
    generator: Any,
    placement: str,
    dtype: torch.dtype,
    attn_backend: str,
) -> None:
    env = cuda_environment()
    print(f"GPU count={env['gpu_count']}")
    for gpu in env["gpus"]:
        print(f"GPU {gpu['index']}={gpu['name']} capability={gpu['capability']}")
    print(f"Torch={env['torch']}, CUDA={env['cuda']}")
    print(f"dtype={dtype_name(dtype)}")
    print(f"attention_backend={attn_backend}")
    print(f"placement={placement}")
    print(f"generator_device_map={json.dumps(resolved_device_map(generator), sort_keys=True)}")


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
