# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenMythos is an open-source theoretical reconstruction of the Claude Mythos architecture — a Recurrent-Depth Transformer (RDT) that achieves deep reasoning through looped weight sharing rather than stacked layers. The entire codebase is a single Python module (`open_mythos/`).

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run a single test file
pytest test_main.py

# Run example
python example.py
```

## Architecture

The model flows through three stages:

```
Input → Prelude (standard layers) → [Recurrent Block × T] → Coda (standard layers) → Output
```

- **Prelude/Coda**: Standard transformer blocks with dense SwiGLU FFN
- **Recurrent Block**: One transformer block looped T times, with MoE FFN, LTI-stable injection, ACT halting, and per-iteration LoRA adaptation
- **Key insight**: The frozen encoded input `e` is injected at every loop iteration, preventing hidden state drift

## Code Structure

- `open_mythos/main.py` — All model code: `MythosConfig`, `OpenMythos`, and every sub-module (`RMSNorm`, RoPE, `GQAttention`, `MLAttention`, `MoEFFN`, `RecurrentBlock`, `LTIInjection`, `ACTHalting`, `LoRAAdapter`, `TransformerBlock`)
- `docs/open_mythos.md` — Full API reference; read this when working on the model
- `test_main.py` — Integration tests for forward pass, generation, and spectral radius validation

## Key Design Patterns

- **Two attention types**: MLA (default, ~10-20× smaller KV cache) or GQA (simpler, fewer KV heads)
- **LTI injection**: `A` is parameterized as `Diag(-exp(log_A))` then discretized via ZOH, guaranteeing `ρ(A) < 1` always
- **MoE only in Recurrent Block**: Prelude and Coda use dense SwiGLU; the recurrent block uses fine-grained routed + shared experts
- **Loop-index embedding**: Sinusoidal signal over recurrence depth, analogous to RoPE over sequence position
- **ACT halting**: Per-position early exit; cumulative sigmoid output determines when to stop looping

## Common Configurations

| Config | Use case |
|---|---|
| `MythosConfig()` | Default MLA (production-scale research) |
| `MythosConfig(attn_type="gqa", n_experts=8, ...)` | Minimal GQA for fast iteration |

See `docs/open_mythos.md` for the full configuration reference including all MLA/GQA fields.
