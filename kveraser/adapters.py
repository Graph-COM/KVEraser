"""Qwen3-8B LoRA initialization and local or Hugging Face checkpoints."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import load_file, save_file
from transformers import Qwen3Model

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_CHECKPOINT_REPO = "Graph-COM/KVEraser"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _lora_config():
    return LoraConfig(task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32,
                      lora_dropout=0.0, bias="none", target_modules=list(TARGET_MODULES))


def initialize_adapter(backbone: Qwen3Model, *, trainable: bool = True):
    """Attach the fixed LoRA adapter to a bare Qwen3 transformer."""
    if not isinstance(backbone, Qwen3Model) or hasattr(backbone, "peft_config"):
        raise ValueError("Expected a Qwen3Model without existing adapters.")
    model = get_peft_model(backbone, _lora_config(), autocast_adapter_dtype=trainable)
    config = model.peft_config["default"]
    config.base_model_name_or_path, config.revision = BASE_MODEL, BASE_REVISION
    config.inference_mode = not trainable
    if not trainable:
        model.requires_grad_(False)
    return model.train(trainable)


class AdapterCheckpoint:
    """Resolve adapter files once; ``revision`` records the immutable HF commit."""

    def __init__(
        self, checkpoint: str | Path = DEFAULT_CHECKPOINT_REPO, *, revision: str = "main",
        cache_dir: str | Path | None = None, local_files_only: bool = False,
    ):
        self.path = Path(checkpoint).expanduser()
        self.revision = None
        local = (isinstance(checkpoint, Path) or self.path.exists()
                 or str(checkpoint).startswith((".", "/", "~")))
        if not local:
            self.path = Path(snapshot_download(
                repo_id=str(checkpoint), repo_type="model", revision=revision,
                allow_patterns=list(ADAPTER_FILES), cache_dir=cache_dir,
                local_files_only=local_files_only))
            self.revision = self.path.name
        for filename in ADAPTER_FILES:
            if not (self.path / filename).is_file():
                raise FileNotFoundError(self.path / filename)
        config = LoraConfig.from_pretrained(self.path)
        expected, actual = _lora_config().to_dict(), config.to_dict()
        metadata = {"base_model_name_or_path", "revision", "inference_mode"}
        mismatches = [key for key in expected if key not in metadata and actual[key] != expected[key]]
        if config.base_model_name_or_path != BASE_MODEL:
            mismatches.append("base_model_name_or_path")
        if config.revision not in (None, BASE_REVISION):
            mismatches.append("revision")
        if mismatches:
            raise ValueError(f"Incompatible adapter configuration: {', '.join(mismatches)}")

    def load(self, backbone: Qwen3Model, *, trainable: bool = False):
        """Load all adapter tensors; training uses FP32 adapters on BF16 backbones."""
        weights = load_file(str(self.path / "adapter_model.safetensors"))
        model = initialize_adapter(backbone, trainable=trainable)
        expected = get_peft_model_state_dict(model, save_embedding_layers=False)
        missing, unexpected = expected.keys() - weights.keys(), weights.keys() - expected.keys()
        if missing or unexpected:
            raise ValueError(f"Adapter tensor keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for name, tensor in weights.items():
            if tensor.shape != expected[name].shape or not tensor.is_floating_point():
                raise ValueError(f"Invalid adapter tensor: {name}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        set_peft_model_state_dict(model, weights)
        return model


def save_adapter(model, output_dir: str | Path) -> None:
    """Save the standard PEFT config and safetensors adapter weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = get_peft_model_state_dict(model, save_embedding_layers=False)
    save_file({key: value.detach().cpu().contiguous() for key, value in weights.items()},
              str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = deepcopy(model.peft_config["default"])
    config.inference_mode = True
    config.save_pretrained(output_dir)


class SharedBackboneAdapterRouter:
    """Route frozen inference through resident LoRA or its exact base layers.

    Bindings are cached once. Contexts serialize calls and restore the previous
    route, including nested calls and exceptions. No weights are copied/merged.
    Use routed KVEraser methods for inference on the shared backbone.
    """

    def __init__(self, backbone):
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("Shared adapter routing requires a frozen backbone and adapter.")
        self._bindings = [(parent, name, child) for parent in backbone.modules()
                          for name, child in parent.named_children() if isinstance(child, LoraLayer)]
        if not self._bindings or any(layer.merged or layer.disable_adapters
                                     for _, _, layer in self._bindings):
            raise ValueError("Expected enabled, unmerged LoRA layers.")
        self._enabled = True
        self._lock = RLock()

    def _switch(self, enabled):
        for parent, name, layer in self._bindings:
            parent._modules[name] = layer if enabled else layer.base_layer
        self._enabled = enabled

    @contextmanager
    def use_adapter(self, enabled):
        with self._lock:
            previous = self._enabled
            if enabled != previous:
                self._switch(enabled)
            try:
                yield
            finally:
                if self._enabled != previous:
                    self._switch(previous)
