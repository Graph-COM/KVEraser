from __future__ import annotations

from contextlib import nullcontext
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .runtime import effective_model_max_length


def _adapter_mode(enabled):
    def decorate(method):
        @wraps(method)
        def routed(self, *args, **kwargs):
            context = (nullcontext() if self.adapter_router is None
                       else self.adapter_router.use_adapter(enabled))
            with context:
                return method(self, *args, **kwargs)
        return routed
    return decorate


class KVEraser(nn.Module):
    def __init__(self, tokenizer, generator, eraser=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.generator = generator
        self.eraser = eraser
        self.model_max_len = effective_model_max_length(tokenizer, getattr(generator, "config", None))
        self.stop_token_ids = self._stop_token_ids(generator, tokenizer)
        self.adapter_router = None
        if (eraser is not None and hasattr(eraser, "get_base_model")
                and eraser.get_base_model() is self._base_transformer(generator)):
            from .adapters import SharedBackboneAdapterRouter
            self.adapter_router = SharedBackboneAdapterRouter(eraser.get_base_model())

    def train(self, mode=True):
        super().train(mode)
        self.generator.eval()
        return self

    @staticmethod
    def _model_input_device(model):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings.weight.device
        return next(model.parameters()).device

    @staticmethod
    def _base_transformer(model):
        return getattr(model, "base_model", model)

    @staticmethod
    def _model_layers(model):
        base_model = KVEraser._base_transformer(model)
        if hasattr(base_model, "layers"):
            return base_model.layers
        inner = getattr(base_model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
        raise AttributeError("Could not locate transformer layers for KV cache placement.")

    @staticmethod
    def _layer_device(layers, layer_idx, fallback):
        try:
            return next(layers[layer_idx].parameters()).device
        except Exception:
            return fallback

    @staticmethod
    def _cache_seq_len(cache):
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        if len(cache) == 0:
            return 0
        key_states, _ = cache[0]
        return int(key_states.shape[2])

    @staticmethod
    def _shared_prefix_len(old_input_ids, new_input_ids):
        old_tokens = old_input_ids[0].tolist()
        new_tokens = new_input_ids[0].tolist()
        prefix_len = 0
        max_shared = min(len(old_tokens), len(new_tokens))
        while prefix_len < max_shared and old_tokens[prefix_len] == new_tokens[prefix_len]:
            prefix_len += 1
        return prefix_len

    @staticmethod
    def _stop_token_ids(generator, tokenizer):
        eos_token_id = getattr(getattr(generator, "generation_config", None), "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {int(eos_token_id)}
        return {int(value) for value in eos_token_id}

    @staticmethod
    def _clone_prefix_cache(cache, cache_len, layers):
        prefix_cache = DynamicCache()
        for layer_idx in range(len(cache)):
            key_states, value_states = cache[layer_idx]
            layer_device = KVEraser._layer_device(layers, layer_idx, key_states.device)
            prefix_cache.update(
                key_states[:, :, :cache_len, :].to(layer_device),
                value_states[:, :, :cache_len, :].to(layer_device),
                layer_idx,
            )
        return prefix_cache

    @_adapter_mode(False)
    def prefill(self, cached_prompt):
        enc = self.tokenizer(
            cached_prompt,
            add_special_tokens=False,
            truncation=(self.model_max_len is not None),
            max_length=self.model_max_len,
            return_offsets_mapping=True,
        )
        input_device = self._model_input_device(self.generator)
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=input_device)
        attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=input_device)
        if int(input_ids.shape[1]) < 1:
            raise ValueError("cached_prompt tokenized to an empty sequence.")
        with torch.no_grad():
            out = self._base_transformer(self.generator)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        return out.past_key_values, input_ids, enc["offset_mapping"]

    @torch.no_grad()
    @_adapter_mode(False)
    def prime_query(self, old_context_kv, query_suffix):
        input_device = self._model_input_device(self.generator)
        enc = self.tokenizer(
            query_suffix,
            add_special_tokens=False,
            truncation=(self.model_max_len is not None),
            max_length=self.model_max_len,
            return_tensors="pt",
        )
        query_ids = enc["input_ids"].to(input_device)
        if int(query_ids.shape[1]) < 1:
            raise ValueError("query_suffix tokenized to an empty sequence.")

        old_context_len = self._cache_seq_len(old_context_kv)
        query_len = int(query_ids.shape[1])
        prompt_seq_len = old_context_len + query_len
        last_prompt_token = query_ids[:, -1:]
        if query_len == 1:
            return last_prompt_token, prompt_seq_len, old_context_kv

        attention_mask = torch.ones(
            (int(query_ids.shape[0]), prompt_seq_len - 1),
            dtype=torch.long,
            device=input_device,
        )
        out = self._base_transformer(self.generator)(
            input_ids=query_ids[:, :-1],
            attention_mask=attention_mask,
            past_key_values=old_context_kv,
            use_cache=True,
            return_dict=True,
        )
        return last_prompt_token, prompt_seq_len, out.past_key_values

    @_adapter_mode(True)
    def erase_tokens(self, old_context_kv, erase_start, needle_ids):
        if self.eraser is None:
            raise RuntimeError("KVEraser.erase_tokens requires an eraser backbone.")
        erase_start = int(erase_start)
        needle_len = int(needle_ids.shape[1])
        if needle_len < 1:
            raise ValueError("Erase span must contain at least one token.")
        erase_end = erase_start + needle_len
        cache_len = self._cache_seq_len(old_context_kv)
        if not (0 <= erase_start < erase_end <= cache_len):
            raise ValueError("Erase token span is invalid for the cached sequence length.")

        eraser_input_device = self._model_input_device(self.eraser)
        eraser_layers = self._model_layers(self.eraser)
        needle_ids = needle_ids.to(eraser_input_device)
        prefix_kv = self._clone_prefix_cache(old_context_kv, erase_start, eraser_layers)
        attention_mask = torch.ones(
            (int(needle_ids.shape[0]), erase_end),
            dtype=torch.long,
            device=eraser_input_device,
        )
        step = self.eraser(
            input_ids=needle_ids,
            attention_mask=attention_mask,
            past_key_values=prefix_kv,
            use_cache=True,
            return_dict=True,
        )

        for layer_idx in range(len(old_context_kv)):
            old_k, old_v = old_context_kv[layer_idx]
            step_k, step_v = step.past_key_values[layer_idx]
            replacement_k = step_k[:, :, erase_start:erase_end, :].to(old_k.device)
            replacement_v = step_v[:, :, erase_start:erase_end, :].to(old_v.device)
            old_k[:, :, erase_start:erase_end, :] = replacement_k
            old_v[:, :, erase_start:erase_end, :] = replacement_v
        return old_context_kv

    def forward(self, old_context_kv, erase_start, needle_ids):
        return self.erase_tokens(old_context_kv, erase_start, needle_ids)

    @torch.no_grad()
    @_adapter_mode(False)
    def recompute_edited_cached_prompt(self, old_context_kv, cached_input_ids, edited_cached_prompt):
        input_device = self._model_input_device(self.generator)
        generator_layers = self._model_layers(self.generator)
        enc = self.tokenizer(
            edited_cached_prompt,
            add_special_tokens=False,
            truncation=(self.model_max_len is not None),
            max_length=self.model_max_len,
            return_tensors="pt",
        )
        edited_input_ids = enc["input_ids"].to(input_device)
        if int(edited_input_ids.shape[1]) < 1:
            raise ValueError("edited_cached_prompt tokenized to an empty sequence.")

        shared_prefix_len = self._shared_prefix_len(cached_input_ids, edited_input_ids)
        if shared_prefix_len == int(edited_input_ids.shape[1]):
            return self._clone_prefix_cache(old_context_kv, shared_prefix_len, generator_layers), edited_input_ids

        attention_mask = torch.ones(
            (1, int(edited_input_ids.shape[1])),
            dtype=torch.long,
            device=input_device,
        )
        if shared_prefix_len == 0:
            out = self._base_transformer(self.generator)(
                input_ids=edited_input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            return out.past_key_values, edited_input_ids

        prefix_kv = self._clone_prefix_cache(old_context_kv, shared_prefix_len, generator_layers)
        out = self._base_transformer(self.generator)(
            input_ids=edited_input_ids[:, shared_prefix_len:],
            attention_mask=attention_mask,
            past_key_values=prefix_kv,
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values, edited_input_ids

    @torch.no_grad()
    @_adapter_mode(False)
    def decode_with_count(self, last_prompt_token, prompt_seq_len, past_kv, max_new_tokens):
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return "", 0
        inp = last_prompt_token
        attention_buffer = torch.ones(
            (1, int(prompt_seq_len) + max_new_tokens - 1),
            dtype=torch.long,
            device=last_prompt_token.device,
        )
        generated = []
        for idx in range(max_new_tokens):
            step = self.generator(
                input_ids=inp,
                attention_mask=attention_buffer[:, : int(prompt_seq_len) + idx],
                past_key_values=past_kv,
                use_cache=True,
            )
            nxt = torch.argmax(step.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(nxt.item())
            generated.append(token_id)
            inp = nxt
            past_kv = step.past_key_values
            if token_id in self.stop_token_ids:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=True), len(generated)

    @torch.no_grad()
    def decode(self, last_prompt_token, prompt_seq_len, past_kv, max_new_tokens):
        generated_text, _ = self.decode_with_count(
            last_prompt_token,
            prompt_seq_len,
            past_kv,
            max_new_tokens,
        )
        return generated_text


def answer_loss(generator, updated_kv, query_ids, target_ids):
    old_context_len = KVEraser._cache_seq_len(updated_kv)
    query_len = int(query_ids.shape[1])
    target_len = int(target_ids.shape[1])
    if query_len < 1:
        raise ValueError("query_ids must contain at least one token.")
    if target_len < 1:
        raise ValueError("target_ids must contain at least one token.")

    prompt_kv = updated_kv
    if query_len > 1:
        query_attention = torch.ones(
            (1, old_context_len + query_len - 1),
            dtype=torch.long,
            device=query_ids.device,
        )
        query_step = KVEraser._base_transformer(generator)(
            input_ids=query_ids[:, :-1],
            attention_mask=query_attention,
            past_key_values=updated_kv,
            use_cache=True,
            return_dict=True,
        )
        prompt_kv = query_step.past_key_values

    teacher_ids = torch.cat((query_ids[:, -1:], target_ids[:, :-1]), dim=1)
    attention_mask = torch.ones(
        (1, old_context_len + query_len + target_len - 1),
        dtype=torch.long,
        device=query_ids.device,
    )
    outputs = generator(
        input_ids=teacher_ids,
        attention_mask=attention_mask,
        past_key_values=prompt_kv,
        use_cache=True,
    )
    return F.cross_entropy(
        outputs.logits.reshape(-1, outputs.logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="mean",
    )
