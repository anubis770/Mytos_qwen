#!/usr/bin/env python3
"""
Convert an OpenMythos PyTorch checkpoint (.pt) to HuggingFace-compatible
Safetensors format for easy sharing and fast loading.

Usage
-----
    python scripts/convert_to_safetensors.py \
        --checkpoint checkpoints/step_0001000.pt \
        --out_dir ./hf_upload \
        --repo_id your-username/open-mythos-hybrid

Dependencies
------------
    pip install safetensors huggingface-hub

Output
------
    hf_upload/
    ├── model.safetensors         # weights only
    ├── model.safetensors.index.json
    ├── config.json               # OpenMythos config
    ├── generation_config.json
    └── README.md                 # Model card

Notes
-----
* Optimizer state is dropped; only model weights are kept.
* This is a ONE-WAY export — you cannot resume training from Safetensors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def _open_mythos_state_dict_to_flat(state_dict: dict) -> dict[str, torch.Tensor]:
    """Convert nested OpenMythos state dict to flat Safetensors-compatible dict."""
    flat: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            flat[key] = value
        elif hasattr(value, "detach"):
            flat[key] = value.detach().cpu()
        else:
            print(f"[convert] Skipping non-tensor key: {key} ({type(value)})")
    return flat


def _mythos_config_to_json(cfg) -> dict:
    """Serialize MythosConfig dataclass to a plain dict for JSON export."""
    import dataclasses
    d = dataclasses.asdict(cfg)
    d["model_type"] = "open_mythos"
    d["architectures"] = ["OpenMythosForCausalLM"]
    return d


def convert_checkpoint(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    repo_id: str | None = None,
):
    """Load a PyTorch checkpoint, export weights to Safetensors, write config."""
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    step = int(ckpt.get("step", 0))
    cfg = ckpt["cfg"]
    vocab_size = int(ckpt.get("vocab_size", cfg.vocab_size))

    print(f"[convert] Step {step:,} | Config: dim={cfg.dim}, layers={cfg.prelude_layers}+{cfg.max_loop_iters}+{cfg.coda_layers}")

    # Export weights
    model_state = ckpt["model"]
    flat_state = _open_mythos_state_dict_to_flat(model_state)

    safetensors_path = out_dir / "model.safetensors"
    print(f"[convert] Writing Safetensors: {safetensors_path}")
    save_file(flat_state, str(safetensors_path), metadata={
        "step": str(step),
        "vocab_size": str(vocab_size),
        "format": "open_mythos_hybrid",
    })

    # Config JSON
    config = _mythos_config_to_json(cfg)
    config["vocab_size"] = vocab_size
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[convert] Written config: {config_path}")

    # Generation config
    gen_cfg = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 2048,
    }
    gen_cfg_path = out_dir / "generation_config.json"
    with open(gen_cfg_path, "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)

    # Model card README
    readme = f"""---
license: apache-2.0
base_model: sulpikar2/Qwen3.6-27B-hereticv3
library_name: transformers
tags:
  - open_mythos
  - recurrent_depth_transformer
  - hybrid_model
  - pytorch
---

# OpenMythos Hybrid (step {step:,})

This is a hybrid model combining **pre-trained weights** from
``sulpikar2/Qwen3.6-27B-hereticv3`` with a novel **Recurrent-Depth Transformer**
RecurrentBlock trained from scratch.

## Architecture

| Component | Source | Status |
|-----------|--------|--------|
| Prelude (first {cfg.prelude_layers} layers) | Qwen3.6-27B | Frozen / Pre-trained |
| RecurrentBlock (looped T={cfg.max_loop_iters} times) | **OpenMythos** | **Trained from scratch** |
| Coda (last {cfg.coda_layers} layers) | Qwen3.6-27B | Frozen / Pre-trained |

## Loading

```python
from safetensors.torch import load_file
import torch

state = load_file("model.safetensors")
# state is a dict[str, Tensor] mapping OpenMythos parameter names
```

## Training

Trained with AdamW on FineWeb-Edu using the OpenMythos training pipeline.

## Disclaimer

This is an experimental architecture. The Recurrent-Depth Transformer block
adds implicit latent reasoning through looped computation, but its benefit
at this scale is still under investigation.
"""
    readme_path = out_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"[convert] Written README: {readme_path}")

    # Index JSON (single shard for now)
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in flat_state.values())},
        "weight_map": {k: "model.safetensors" for k in flat_state.keys()},
    }
    index_path = out_dir / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    total_params = sum(p.numel() for p in flat_state.values())
    total_bytes = sum(p.numel() * p.element_size() for p in flat_state.values())
    print(f"[convert] Done. Total params: {total_params:,} | Size: {total_bytes / 1e9:.2f} GB")
    print(f"[convert] Output directory: {out_dir.resolve()}")

    if repo_id:
        print(f"\nTo upload to HuggingFace:")
        print(f"    huggingface-cli upload {repo_id} {out_dir} --repo-type model")


def main():
    parser = argparse.ArgumentParser(description="Convert OpenMythos .pt checkpoint to Safetensors")
    parser.add_argument("--checkpoint", required=True, help="Path to step_*.pt checkpoint")
    parser.add_argument("--out_dir", default="./hf_upload", help="Output directory")
    parser.add_argument("--repo_id", default=None, help="Optional HF repo ID for README")
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.out_dir, args.repo_id)


if __name__ == "__main__":
    main()
