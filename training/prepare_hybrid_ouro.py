"""
Create hybrid OpenMythos checkpoint from Ouro-2.6B pretrained weights.
One-time setup before training.

Usage:
    python training/prepare_hybrid_ouro.py
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# HYBRID WEIGHT LOADER (Standalone)
# ============================================================================

class HybridOuroLoader:
    """Load Ouro pretrained → OpenMythos prelude+coda, random init RecurrentBlock"""
    
    @staticmethod
    def map_ouro_to_mythos(hf_model, mythos_model, freeze_pretrained=True):
        """Map Ouro weights to Prelude/Coda/Embed/Head of OpenMythos"""
        hf_state = hf_model.state_dict()
        mythos_state = mythos_model.state_dict()
        
        loaded_count = 0
        
        print("\n[Weight Transfer]")
        
        # 1. Embedding layer
        if "model.embed_tokens.weight" in hf_state:
            embed_key = "embed.weight"
            if embed_key in mythos_state:
                hf_shape = hf_state["model.embed_tokens.weight"].shape
                mythos_shape = mythos_state[embed_key].shape
                if hf_shape == mythos_shape:
                    mythos_state[embed_key].copy_(hf_state["model.embed_tokens.weight"])
                    loaded_count += 1
                    print(f"  ✓ Embedding: {mythos_shape}")
        
        # 2. Prelude layers (8 layers from first 8 Ouro layers)
        prelude_loaded = 0
        for layer_idx in range(min(8, 16)):  # Ouro has 16 layers
            hf_prefix = f"model.layers.{layer_idx}."
            mythos_prefix = f"prelude.layers.{layer_idx}."
            
            # Self-attention projections
            attn_keys = [
                ("self_attn.q_proj.weight", "attn.q_proj.weight"),
                ("self_attn.k_proj.weight", "attn.k_proj.weight"),
                ("self_attn.v_proj.weight", "attn.v_proj.weight"),
                ("self_attn.o_proj.weight", "attn.o_proj.weight"),
            ]
            
            for hf_key, mythos_key_suffix in attn_keys:
                hf_full_key = hf_prefix + hf_key
                mythos_full_key = mythos_prefix + mythos_key_suffix
                
                if hf_full_key in hf_state and mythos_full_key in mythos_state:
                    if hf_state[hf_full_key].shape == mythos_state[mythos_full_key].shape:
                        mythos_state[mythos_full_key].copy_(hf_state[hf_full_key])
                        prelude_loaded += 1
                        loaded_count += 1
            
            # LayerNorm weights
            norm_keys = [
                ("input_layernorm.weight", "norm1.weight"),
                ("post_attention_layernorm.weight", "norm2.weight"),
            ]
            
            for hf_key, mythos_key_suffix in norm_keys:
                hf_full_key = hf_prefix + hf_key
                mythos_full_key = mythos_prefix + mythos_key_suffix
                
                if hf_full_key in hf_state and mythos_full_key in mythos_state:
                    if hf_state[hf_full_key].shape == mythos_state[mythos_full_key].shape:
                        mythos_state[mythos_full_key].copy_(hf_state[hf_full_key])
                        prelude_loaded += 1
                        loaded_count += 1
        
        print(f"  ✓ Prelude layers (8 × 16 params): {prelude_loaded} weights")
        
        # 3. Coda layers (8 layers from Ouro layers 8-15)
        coda_loaded = 0
        for layer_idx in range(min(8, 16)):
            hf_src_idx = 8 + layer_idx  # Use Ouro layers 8-15 for Coda
            hf_prefix = f"model.layers.{hf_src_idx}."
            mythos_prefix = f"coda.layers.{layer_idx}."
            
            attn_keys = [
                ("self_attn.q_proj.weight", "attn.q_proj.weight"),
                ("self_attn.k_proj.weight", "attn.k_proj.weight"),
                ("self_attn.v_proj.weight", "attn.v_proj.weight"),
                ("self_attn.o_proj.weight", "attn.o_proj.weight"),
            ]
            
            for hf_key, mythos_key_suffix in attn_keys:
                hf_full_key = hf_prefix + hf_key
                mythos_full_key = mythos_prefix + mythos_key_suffix
                
                if hf_full_key in hf_state and mythos_full_key in mythos_state:
                    if hf_state[hf_full_key].shape == mythos_state[mythos_full_key].shape:
                        mythos_state[mythos_full_key].copy_(hf_state[hf_full_key])
                        coda_loaded += 1
                        loaded_count += 1
            
            norm_keys = [
                ("input_layernorm.weight", "norm1.weight"),
                ("post_attention_layernorm.weight", "norm2.weight"),
            ]
            
            for hf_key, mythos_key_suffix in norm_keys:
                hf_full_key = hf_prefix + hf_key
                mythos_full_key = mythos_prefix + mythos_key_suffix
                
                if hf_full_key in hf_state and mythos_full_key in mythos_state:
                    if hf_state[hf_full_key].shape == mythos_state[mythos_full_key].shape:
                        mythos_state[mythos_full_key].copy_(hf_state[hf_full_key])
                        coda_loaded += 1
                        loaded_count += 1
        
        print(f"  ✓ Coda layers (8 × 16 params): {coda_loaded} weights")
        
        # Load the combined state dict
        mythos_model.load_state_dict(mythos_state)
        
        # Freeze pretrained weights
        if freeze_pretrained:
            frozen_count = 0
            for name, param in mythos_model.named_parameters():
                if any(frozen in name for frozen in ["prelude", "coda", "embed", "head"]):
                    param.requires_grad = False
                    frozen_count += 1
        
        print(f"\n[Freeze Status]")
        print(f"  ✓ Total weights transferred: {loaded_count}")
        print(f"  ✓ Frozen (pretrained): {frozen_count} params")
        
        return mythos_model


# ============================================================================
# MAIN
# ============================================================================

def create_hybrid_checkpoint():
    """Create hybrid checkpoint from Ouro pretrained"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Import OpenMythos
    try:
        from open_mythos.main import OpenMythos, MythosConfig
    except ImportError:
        print("ERROR: open_mythos not installed. Run: pip install open-mythos")
        return
    
    # 1. Create OpenMythos config for 2.6B model
    print("\n[Step 1] Creating OpenMythos config...")
    cfg = MythosConfig(
        vocab_size=128256,           # Ouro tokenizer
        dim=2560,                    # Match Ouro hidden_size
        n_heads=16,                  # Match Ouro n_heads
        n_kv_heads=4,                # GQA (Ouro uses this)
        max_seq_len=8192,
        max_loop_iters=12,           # 12 recurrent loops
        prelude_layers=8,            # Encode input
        coda_layers=8,               # Decode output
        n_experts=32,                # 32 experts for domain diversity
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        lora_rank=32,
        attn_type="gqa",
        grad_ckpt=True,              # Gradient checkpointing for memory
        act_threshold=0.95,
    )
    print(f"  ✓ Config created")
    print(f"    - Hidden dim: {cfg.dim}")
    print(f"    - Heads: {cfg.n_heads}")
    print(f"    - Loop iterations: {cfg.max_loop_iters}")
    print(f"    - Experts: {cfg.n_experts} ({cfg.n_experts_per_tok} active per token)")
    
    # 2. Initialize OpenMythos
    print("\n[Step 2] Initializing OpenMythos model...")
    model = OpenMythos(cfg)
    print(f"  ✓ Model initialized")
    print(f"    - Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move to device
    if device == "cuda":
        model = model.cuda()
    
    # 3. Load Ouro pretrained
    print("\n[Step 3] Loading Ouro-2.6B-Thinking pretrained...")
    try:
        ouro = AutoModelForCausalLM.from_pretrained(
            "Bytedance/Ouro-2.6B-Thinking",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        print(f"  ✓ Ouro model loaded")
    except Exception as e:
        print(f"  ERROR loading Ouro: {e}")
        print(f"  Make sure you have internet and HF_TOKEN set if needed")
        return
    
    # 4. Transfer weights
    print("\n[Step 4] Transferring weights from Ouro to OpenMythos...")
    model = HybridOuroLoader.map_ouro_to_mythos(ouro, model, freeze_pretrained=True)
    print(f"  ✓ Weight transfer complete")
    
    # Delete Ouro to free memory
    del ouro
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 5. Create output directory
    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    # 6. Save hybrid checkpoint
    print("\n[Step 5] Saving hybrid checkpoint...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg.__dict__,
        "model_class": "OpenMythos",
        "base_model": "Bytedance/Ouro-2.6B-Thinking",
    }
    
    checkpoint_path = output_dir / "hybrid_ouro_mythos_3b.pt"
    torch.save(checkpoint, str(checkpoint_path))
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    print(f"    - Size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")
    
    # 7. Save config separately
    config_path = output_dir / "hybrid_ouro_mythos_3b_config.json"
    with open(config_path, "w") as f:
        # Convert config to dict if needed
        config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else cfg.to_dict()
        json.dump(config_dict, f, indent=2)
    print(f"  ✓ Config saved: {config_path}")
    
    # 8. Print summary
    print("\n" + "="*70)
    print("HYBRID CHECKPOINT CREATED SUCCESSFULLY")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Summary:")
    print(f"  Total parameters:     {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    print(f"    → RecurrentBlock, routers, MoE")
    print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/1e9:.2f}B)")
    print(f"    → Prelude, Coda, Embedding, Head")
    
    print(f"\nNext step:")
    print(f"  python training/train_from_hybrid_ouro.py")
    print("="*70)


if __name__ == "__main__":
    create_hybrid_checkpoint()
