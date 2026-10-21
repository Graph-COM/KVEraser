"""Qwen3 inference cache updates and RoPE position correction."""
from contextlib import contextmanager

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def shift_key_positions(keys, rotary_emb, old_start, new_start):
    """Undo the old RoPE rotation and rotate a contiguous key span at its new positions."""
    positions = torch.arange(keys.shape[2], device=keys.device)
    old_cos, old_sin = rotary_emb(keys, (positions + old_start).unsqueeze(0))
    new_cos, new_sin = rotary_emb(keys, (positions + new_start).unsqueeze(0))
    old_cos, old_sin = old_cos.to(keys), old_sin.to(keys)
    new_cos, new_sin = new_cos.to(keys), new_sin.to(keys)
    unrotated = keys * old_cos.unsqueeze(1) - rotate_half(keys) * old_sin.unsqueeze(1)
    return unrotated * new_cos.unsqueeze(1) + rotate_half(unrotated) * new_sin.unsqueeze(1)


@torch.no_grad()
def discard_span(cache, start, end, rotary_emb):
    """Consume a dynamic cache, dropping a span and correcting suffix key positions."""
    if (type(cache) not in (Cache, DynamicCache) or cache.offloading or not cache.layers
            or any(type(layer) is not DynamicLayer or not layer.is_initialized for layer in cache.layers)):
        raise ValueError("Discard requires an initialized, non-offloaded dynamic cache.")
    length = cache.get_seq_length()
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
        raise ValueError("Discard span is empty or out of bounds.")
    for layer in cache.layers:
        key, value = layer.keys, layer.values
        if (key.ndim != 4 or key.shape != value.shape or key.shape != cache.layers[0].keys.shape
                or key.shape[2] != length
                or key.device != value.device or key.dtype != value.dtype):
            raise ValueError("Inconsistent KV shapes, devices or dtypes.")
    for layer in cache.layers:
        key, value = layer.keys, layer.values
        suffix = key[:, :, end:]
        if end < length:
            suffix = shift_key_positions(suffix, rotary_emb, end, start)
        layer.keys = torch.cat((key[:, :, :start], suffix), dim=2).contiguous()
        layer.values = torch.cat((value[:, :, :start], value[:, :, end:]), dim=2).contiguous()
    return cache


@contextmanager
def restore_span_on_error(cache, start, end):
    """Back up only the overwritten span so a failed forward can be retried."""
    spans = [tensor[:, :, start:end] for pair in cache for tensor in pair]
    backups = [span.clone() for span in spans]
    try:
        yield
    except BaseException:
        for span, backup in zip(spans, backups):
            span.copy_(backup)
        raise


def can_write_span(owner, cache, batch_size):
    """Check every layer before permitting writes into an existing KV cache."""
    if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        return False
    if type(cache) not in (Cache, DynamicCache) or cache.offloading or not cache.layers:
        return False
    for model in (owner.generator, owner.eraser):
        config = getattr(model, "config", None)
        if (model is None or model.training or getattr(config, "model_type", None) != "qwen3"
                or getattr(config, "sliding_window", None) or getattr(config, "use_sliding_window", False)):
            return False
    config = owner.eraser.config
    layers = owner._model_layers(owner.eraser)
    if len(layers) != len(cache.layers):
        return False
    expected = (batch_size, config.num_key_value_heads, cache.get_seq_length(), config.head_dim)
    for layer, cached in zip(layers, cache.layers):
        if type(cached) is not DynamicLayer or not cached.is_initialized:
            return False
        parameters = tuple(layer.parameters())
        for tensor in (cached.keys, cached.values):
            if (tensor.shape != expected or tensor.requires_grad or tensor.grad_fn is not None
                    or not tensor.is_contiguous()
                    or (torch.is_inference(tensor) and not torch.is_inference_mode_enabled())):
                return False
            if any(p.device != tensor.device or (p.is_floating_point() and p.dtype != tensor.dtype)
                   for p in parameters):
                return False
    return True


class _SpanWriteLayer(DynamicLayer):
    """Expose the causal prefix, then write one span without concatenating it."""

    def __init__(self, key, value, start, length):
        super().__init__()
        self.original_key, self.original_value = key, value
        self.start, self.length, self.written = start, length, False
        self.keys, self.values = key[:, :, :start], value[:, :, :start]
        self.device, self.dtype, self.is_initialized = key.device, key.dtype, True

    def update(self, key_states, value_states, cache_kwargs=None):
        if torch.is_grad_enabled() or self.written:
            raise RuntimeError("Span writes require inference and exactly one update per layer.")
        end = self.start + self.length
        if not 0 <= self.start < end <= self.original_key.shape[2]:
            raise ValueError("Invalid span write boundary.")
        expected = (*self.original_key.shape[:2], self.length, self.original_key.shape[3])
        for states in (key_states, value_states):
            if states.shape != expected or states.device != self.device or states.dtype != self.dtype:
                raise ValueError("Span write shape, device or dtype mismatch.")
        self.original_key[:, :, self.start:end].copy_(key_states)
        self.original_value[:, :, self.start:end].copy_(value_states)
        self.keys, self.values = self.original_key[:, :, :end], self.original_value[:, :, :end]
        self.written = True
        return self.keys, self.values


def span_write_cache(cache, start, length):
    return Cache(layers=[_SpanWriteLayer(key, value, start, length) for key, value in cache])
