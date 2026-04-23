"""
Hybrid loader — map HuggingFace pretrained weights into OpenMythos Prelude/Coda.

Supports standard Llama/Qwen/Mistral weight naming with SwiGLU MLP.
Also supports Qwen3.5 hybrid attention (linear + full) by loading only
full-attention layers into OpenMythos GQA blocks and FFN weights from
linear-attention layers.

Usage
-----
    from open_mythos.hybrid_loader import load_hf_weights, freeze_pretrained_layers

    model = OpenMythos(cfg)
    load_hf_weights(model, "sulpikar2/Qwen3.6-27B-heretic")
    freeze_pretrained_layers(model)

Design
------
* embed, final norm, lm_head  →  loaded directly
* Prelude / Coda blocks  →  mapped from HF ``model.layers.*``
* RecurrentBlock (MoE, LTI, ACT, LoRA)  →  kept random (the trainable novel part)

Compatibility
-------------
OpenMythos must be configured with ``attn_type="gqa"`` because standard HF
models (including Qwen3.5) do not implement MLA.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM


def _get_weight(state_dict: dict, key: str) -> torch.Tensor:
    if key not in state_dict:
        raise KeyError(f"Missing expected weight in HF checkpoint: {key}")
    return state_dict[key]


def _map_block_ffn(hf_state: dict, prefix: str, block, loaded: dict):
    """Copy only FFN weights from a HF layer into an OpenMythos TransformerBlock."""
    block.ffn.gate.weight.copy_(_get_weight(hf_state, f"{prefix}.mlp.gate_proj.weight"))
    block.ffn.up.weight.copy_(_get_weight(hf_state, f"{prefix}.mlp.up_proj.weight"))
    block.ffn.down.weight.copy_(_get_weight(hf_state, f"{prefix}.mlp.down_proj.weight"))
    block.attn_norm.weight.copy_(_get_weight(hf_state, f"{prefix}.input_layernorm.weight"))
    block.ffn_norm.weight.copy_(_get_weight(hf_state, f"{prefix}.post_attention_layernorm.weight"))
    loaded[f"{prefix}.ffn"] = True


def _map_block_full(hf_state: dict, prefix: str, block, loaded: dict):
    """Copy full block (attention + FFN + norms) from a HF layer."""
    block.attn.wq.weight.copy_(_get_weight(hf_state, f"{prefix}.self_attn.q_proj.weight"))
    block.attn.wk.weight.copy_(_get_weight(hf_state, f"{prefix}.self_attn.k_proj.weight"))
    block.attn.wv.weight.copy_(_get_weight(hf_state, f"{prefix}.self_attn.v_proj.weight"))
    block.attn.wo.weight.copy_(_get_weight(hf_state, f"{prefix}.self_attn.o_proj.weight"))
    block.ffn.gate.weight.copy_(_get_weight(hf_state, f"{prefix}.mlp.gate_proj.weight"))
    block.ffn.up.weight.copy_(_get_weight(hf_state, f"{prefix}.mlp.up_proj.weight"))
    block.ffn.down.weight.copy_(_get_weight(hf_state, f"{prefix}.mlp.down_proj.weight"))
    block.attn_norm.weight.copy_(_get_weight(hf_state, f"{prefix}.input_layernorm.weight"))
    block.ffn_norm.weight.copy_(_get_weight(hf_state, f"{prefix}.post_attention_layernorm.weight"))
    loaded[f"{prefix}.full"] = True


def _detect_hf_config(hf_model) -> tuple[int, int, int, int, int, list[str] | None]:
    """Return (hidden_size, n_layers, n_heads, n_kv_heads, vocab_size, layer_types)."""
    cfg = hf_model.config
    # Qwen3.5 nests text config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    layer_types = getattr(cfg, "layer_types", None)
    return (
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        cfg.vocab_size,
        layer_types,
    )


def load_hf_weights(model, hf_model_id: str, dtype: Optional[torch.dtype] = None, device: str = "cpu"):
    """Load pretrained HF weights into compatible Prelude / Coda / Embed layers.

    Args:
        model: OpenMythos instance (must have ``attn_type="gqa"``).
        hf_model_id: HuggingFace model ID.
        dtype: torch dtype for the loaded checkpoint (default = model dtype).
        device: Where to stage the HF model before copying (default ``"cpu"``).

    Raises:
        ValueError: If ``model.cfg.attn_type != "gqa"``.
    """
    if model.cfg.attn_type != "gqa":
        raise ValueError("Hybrid loading requires attn_type='gqa'. Standard HF models do not implement MLA.")

    print(f"[Hybrid] Loading HF weights from: {hf_model_id}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=dtype or next(model.parameters()).dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    hf_state = hf_model.state_dict()

    # --- inspect HF architecture before deleting model ---
    hf_hidden, hf_layers, hf_heads, hf_kv_heads, hf_vocab, layer_types = _detect_hf_config(hf_model)
    del hf_model  # free memory

    print(f"[Hybrid] HF arch: layers={hf_layers}, dim={hf_hidden}, heads={hf_heads}/{hf_kv_heads}, vocab={hf_vocab}")

    # --- embedding ---
    embed_key = "model.embed_tokens.weight"
    if embed_key in hf_state:
        hf_vocab_real, hf_dim = hf_state[embed_key].shape
        my_vocab, my_dim = model.embed.weight.shape
        if hf_dim != my_dim:
            raise ValueError(f"Embed dim mismatch: HF={hf_dim}, OpenMythos={my_dim}")
        rows = min(hf_vocab_real, my_vocab)
        model.embed.weight.data[:rows].copy_(hf_state[embed_key][:rows])
        if hf_vocab_real != my_vocab:
            print(f"[Hybrid] WARNING vocab mismatch HF={hf_vocab_real} vs Mythos={my_vocab}; copied {rows} rows.")
    else:
        print("[Hybrid] embed_tokens.weight not found; skipped.")

    # --- Prelude: first N layers ---
    loaded_flags: dict = {}
    n_prelude = len(model.prelude)
    for i in range(n_prelude):
        prefix = f"model.layers.{i}"
        lt = layer_types[i] if layer_types else "full_attention"
        if lt == "full_attention":
            _map_block_full(hf_state, prefix, model.prelude[i], loaded_flags)
        else:
            print(f"[Hybrid] Prelude layer {i} is '{lt}'; loading FFN+norm only, attention kept random.")
            _map_block_ffn(hf_state, prefix, model.prelude[i], loaded_flags)

    # --- Coda: last M layers ---
    n_coda = len(model.coda)
    for j in range(n_coda):
        hf_idx = hf_layers - n_coda + j
        prefix = f"model.layers.{hf_idx}"
        lt = layer_types[hf_idx] if layer_types else "full_attention"
        if lt == "full_attention":
            _map_block_full(hf_state, prefix, model.coda[j], loaded_flags)
        else:
            print(f"[Hybrid] Coda layer {j} (HF idx {hf_idx}) is '{lt}'; loading FFN+norm only, attention kept random.")
            _map_block_ffn(hf_state, prefix, model.coda[j], loaded_flags)

    # --- final norm ---
    if "model.norm.weight" in hf_state:
        model.norm.weight.copy_(hf_state["model.norm.weight"])

    # --- lm_head ---
    if "lm_head.weight" in hf_state:
        hf_v, hf_d = hf_state["lm_head.weight"].shape
        my_v, my_d = model.head.weight.shape
        if hf_d == my_d and hf_v == my_v:
            model.head.weight.copy_(hf_state["lm_head.weight"])
        else:
            print(f"[Hybrid] lm_head shape mismatch ({hf_v},{hf_d}) vs ({my_v},{my_d}); skipped.")

    print(f"[Hybrid] Loaded {len(loaded_flags)} HF sub-layers into OpenMythos.")


def freeze_pretrained_layers(
    model,
    freeze_embed: bool = True,
    freeze_prelude: bool = True,
    freeze_coda: bool = True,
    freeze_norm: bool = True,
    freeze_head: bool = False,
):
    """Freeze layers carrying pretrained HF weights; leave RecurrentBlock trainable."""
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        should_freeze = False
        if freeze_embed and "embed" in name:
            should_freeze = True
        if freeze_prelude and "prelude" in name:
            should_freeze = True
        if freeze_coda and "coda" in name:
            should_freeze = True
        if freeze_norm and "norm" in name and "recurrent" not in name:
            should_freeze = True
        if freeze_head and "head" in name:
            should_freeze = True

        if should_freeze:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()

    total = trainable + frozen
    print(f"[Freeze] Frozen {frozen:,} ({frozen/total:.1%}) | Trainable {trainable:,} ({trainable/total:.1%})")
    rec_trainable = any(p.requires_grad for p in model.recurrent.parameters())
    print(f"[Freeze] RecurrentBlock trainable: {rec_trainable}")
