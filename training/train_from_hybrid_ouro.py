"""
Train OpenMythos from hybrid checkpoint on FineWeb-Edu dataset.

Assumes prepare_hybrid_ouro.py has been run first.

Usage:
    python training/train_from_hybrid_ouro.py
    
Resume from checkpoint:
    python training/train_from_hybrid_ouro.py --resume checkpoint-step-010000.pt
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import sys

# ============================================================================
# CONFIG
# ============================================================================

BATCH_SIZE = 4  # Adjust based on GPU memory
GRAD_ACCUM_STEPS = 2
LR = 3e-4
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 2000
NUM_TRAINING_STEPS = 50000  # ~30B tokens
EVAL_STEPS = 500
SAVE_STEPS = 1000
N_LOOPS = 12  # Recurrent loops during training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints")

# ============================================================================
# DATASET
# ============================================================================

class FineWebEduDataset:
    """Stream FineWeb-Edu dataset"""
    
    def __init__(self, tokenizer, max_seq_len=8192):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
    
    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            tokens = self.tokenizer(sample["text"], truncation=False)["input_ids"]
            buffer.extend(tokens)
            
            while len(buffer) >= self.max_seq_len:
                chunk = buffer[:self.max_seq_len]
                buffer = buffer[self.max_seq_len:]
                
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


# ============================================================================
# TRAINING
# ============================================================================

def train(resume_from=None):
    """Main training loop"""
    
    from open_mythos.main import OpenMythos, MythosConfig
    
    print("="*70)
    print("TRAINING OPENMYTHOS FROM HYBRID CHECKPOINT")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Recurrent loops: {N_LOOPS}")
    
    # 1. Load hybrid checkpoint
    print("\n[Step 1] Loading hybrid checkpoint...")
    checkpoint_path = CHECKPOINT_DIR / "hybrid_ouro_mythos_3b.pt"
    config_path = CHECKPOINT_DIR / "hybrid_ouro_mythos_3b_config.json"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"Run: python training/prepare_hybrid_ouro.py")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    with open(config_path) as f:
        config_dict = json.load(f)
    
    cfg = MythosConfig(**config_dict)
    model = OpenMythos(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ Loaded checkpoint: {checkpoint_path}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total params: {total_params:,}")
    print(f"  ✓ Trainable params: {trainable_params:,}")
    
    # 2. Setup tokenizer & dataset
    print("\n[Step 2] Setting up tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("Bytedance/Ouro-2.6B-Thinking")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")
    
    dataset = FineWebEduDataset(tokenizer, max_seq_len=cfg.max_seq_len)
    print(f"  ✓ Dataset initialized")
    
    # 3. Setup optimizer & scheduler
    print("\n[Step 3] Setting up optimizer...")
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params_list,
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_TRAINING_STEPS,
        eta_min=1e-5,
    )
    print(f"  ✓ Optimizer: AdamW (lr={LR})")
    print(f"  ✓ Scheduler: CosineAnnealingLR (T_max={NUM_TRAINING_STEPS})")
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        resume_path = CHECKPOINT_DIR / resume_from
        if resume_path.exists():
            print(f"\n[Step 2b] Resuming from checkpoint...")
            resume_ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            start_step = resume_ckpt["step"]
            print(f"  ✓ Resumed from step {start_step}")
        else:
            print(f"  WARNING: Resume checkpoint not found: {resume_path}")
    
    # 4. Training loop
    print("\n[Step 4] Starting training...")
    print("="*70)
    
    model.train()
    total_loss = 0.0
    step = start_step
    
    pbar = tqdm(dataset, desc="Training", initial=start_step)
    for batch in pbar:
        # Prepare batch
        input_ids = batch["input_ids"].unsqueeze(0).to(DEVICE)  # (1, T)
        labels = batch["labels"].unsqueeze(0).to(DEVICE)        # (1, T)
        
        # Forward pass
        logits = model(input_ids, n_loops=N_LOOPS)  # (1, T, vocab)
        
        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="mean",
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        # Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        step += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "step": step,
        })
        
        # Logging every 100 steps
        if step % 100 == 0:
            avg_loss = total_loss / 100
            print(f"\nStep {step:6d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            total_loss = 0.0
        
        # Save checkpoint every 1000 steps
        if step % SAVE_STEPS == 0:
            checkpoint_save_path = CHECKPOINT_DIR / f"checkpoint-step-{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config_dict,
            }, checkpoint_save_path)
            print(f"\n✓ Checkpoint saved: {checkpoint_save_path}")
        
        # Stop at max training steps
        if step >= NUM_TRAINING_STEPS + start_step:
            print(f"\nReached max training steps ({NUM_TRAINING_STEPS})")
            break
    
    # 5. Save final model
    print("\n[Step 5] Saving final model...")
    final_path = CHECKPOINT_DIR / "mythos-3b-final-trained.pt"
    torch.save(model.state_dict(), final_path)
    print(f"  ✓ Final model saved: {final_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}/")
    print(f"Final model: {final_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenMythos from hybrid checkpoint")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (e.g., 'checkpoint-step-010000.pt')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Learning rate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_TRAINING_STEPS,
        help="Number of training steps"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=N_LOOPS,
        help="Number of recurrent loops"
    )
    
    args = parser.parse_args()
    
    # Update globals from args
    BATCH_SIZE = args.batch_size
    LR = args.lr
    NUM_TRAINING_STEPS = args.steps
    N_LOOPS = args.loops
    
    try:
        train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
