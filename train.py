# ncn_project/train.py

"""
Training script for NCN-Modulated Language Model.

Features:
- BinaryDataset (Memmap) for instant data loading with correct stride.
- Custom CUDA Kernels (via architecture imports).
- PyTorch 2.x / 1.x compatibility.
- Gradient Checkpointing support.
- FULL DETAILED LOGGING (Loss, Reg, PPL, LR, Scale, Tok/s, Progress).
- FULL Resume capability (Model, Optimizer, Scheduler, Scaler).
- CRITICAL FIX: Architecture arguments sync with checkpoint before Dataset creation.
- CRITICAL FIX: Data Loader Fast-Forwarding on Resume.
- CRITICAL FIX: Explicit Checkpoint Logging.
- CRITICAL FIX: Correct TQDM Progress Bar (No double counting).
- CRITICAL FIX: LR Override on Resume (Forces new LR even when loading optimizer state).

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Original Author: Michael Morgan
Date: 2025-11-25
Github: https://github.com/Mmorgan-ML
Email: mmorgankorea@gmail.com
"""

import os
import sys
import argparse
import time
import math
import logging
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.serialization
import torch.backends.cudnn as cudnn 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- AMP Import Handling ---
try:
    from torch.amp import autocast, GradScaler
    TORCH_2_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    TORCH_2_AMP = False

from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# --- Import custom modules ---
try:
    from ncn_architecture import NCNConfig, ModulatedLLM
    from tokenizer import Tokenizer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ncn_architecture import NCNConfig, ModulatedLLM
    from tokenizer import Tokenizer
    print("Added project directory to sys.path")

torch.serialization.add_safe_globals([NCNConfig])

# --- Constants & Logging ---
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training.log"
DEFAULT_TOKENIZER_PATH = "gpt2_tokenizer_files"
DEFAULT_DATA_FILE = "train.bin"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# --- WandB Setup ---
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("wandb not installed. WandB logging disabled.")


# --- Helper Functions ---
def format_param_count(count):
    if count > 1e9: return f"{count / 1e9:.2f}B"
    if count > 1e6: return f"{count / 1e6:.2f}M"
    if count > 1e3: return f"{count / 1e3:.2f}K"
    return str(count)

def set_seed(seed: int):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_ddp():
    backend = 'nccl'; local_rank = int(os.environ.get("LOCAL_RANK", -1)); world_size = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", -1))
    if world_size > 1:
        if local_rank == -1 or rank == -1: raise RuntimeError("DDP environment variables not set.")
        logger.info(f"Initializing DDP: WS={world_size}, R={rank}, LR={local_rank}"); dist.init_process_group(backend=backend); torch.cuda.set_device(local_rank); logger.info(f"DDP initialized: Rank {rank}/{world_size} on cuda:{local_rank}.")
        is_ddp = True; is_master = (rank == 0)
    else: logger.info("DDP not enabled."); is_ddp = False; is_master = True; local_rank = 0; rank = 0
    return is_ddp, rank, world_size, local_rank, is_master

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group(); logger.info("DDP process group destroyed.")

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    if not os.path.exists(tokenizer_path): raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' not found.")
    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_path); logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
            logger.warning(f"Tokenizer invalid PAD ID. Using standard GPT-2 EOS (50256) as PAD.")
            tokenizer.pad_token_id = 50256
        return tokenizer
    except Exception as e: logger.error(f"Failed to load tokenizer: {e}", exc_info=True); raise

# --- High Performance Binary Dataset (FIXED STRIDE) ---
class BinaryDataset(Dataset):
    def __init__(self, bin_file: str, block_size: int, split: str = "train", val_ratio: float = 0.05):
        self.block_size = block_size
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file {bin_file} not found. Run prepare_data.py first!")
        
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        total_tokens = len(self.data)
        split_idx = int(total_tokens * (1.0 - val_ratio))
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = split_idx
        elif split == "val":
            self.start_idx = split_idx
            self.end_idx = total_tokens
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
        available_tokens = self.end_idx - self.start_idx
        # Correct stride calculation: non-overlapping chunks
        self.num_sequences = (available_tokens - 1) // self.block_size

        print(f"Dataset ({split}): {self.num_sequences} sequences (Non-overlapping)")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Stride = block_size
        start_offset = self.start_idx + (idx * self.block_size)
        
        chunk = torch.from_numpy(self.data[start_offset : start_offset + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- Checkpoint Functions ---
def save_checkpoint(state: Dict[str, Any], directory: str, filename: str = "checkpoint.pt", is_master: bool = True):
    if not is_master: return
    os.makedirs(directory, exist_ok=True); filepath = os.path.join(directory, filename); 
    logger.info(f"Saving checkpoint to {filepath}...");
    try: 
        torch.save(state, filepath); 
        # UPDATED: Explicitly name the file in success log
        logger.info(f"Checkpoint successfully saved: {filename}")
    except Exception as e: logger.error(f"ERROR saving checkpoint: {e}", exc_info=True)

def load_model_state_dict(filepath: str, model: Union[ModulatedLLM, DDP], device: torch.device) -> bool:
    if not os.path.exists(filepath): logger.error(f"Model checkpoint not found: {filepath}"); return False
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            target_model = model.module if isinstance(model, DDP) else model; load_result = target_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("Model state_dict loaded (strict=False)."); return True
        return False
    except Exception as e: logger.error(f"Error loading model state_dict: {e}", exc_info=True); return False

def load_resume_metadata(filepath: str, device: torch.device) -> Tuple[Optional[NCNConfig], int, int, float]:
    if not os.path.exists(filepath): return None, 0, 0, float('inf')
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        cfg_data = checkpoint.get('config')
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_val = checkpoint.get('best_val_loss', float('inf'))
        return cfg_data, start_epoch, global_step, best_val
    except Exception as e: logger.error(f"Error loading resume state: {e}"); return None, 0, 0, float('inf')

def load_training_state(filepath: str, optimizer: optim.Optimizer, scheduler: Any, scaler: GradScaler, device: torch.device, new_lr: Optional[float] = None):
    if not os.path.exists(filepath): return
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        if 'optimizer_state_dict' in checkpoint and optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state restored.")
            
        if 'scheduler_state_dict' in checkpoint and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state restored.")
            
        if 'scaler_state_dict' in checkpoint and scaler:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("GradScaler state restored.")
        
        # --- LR OVERRIDE LOGIC ---
        if new_lr is not None:
            logger.info(f"!!! Overriding Optimizer/Scheduler LR with new value: {new_lr} !!!")
            # 1. Update Optimizer Param Groups
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
                # Update initial_lr if present (common in schedulers)
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = new_lr
            
            # 2. Update Scheduler Base LRs (Critical for Correct Decay)
            if scheduler and hasattr(scheduler, 'base_lrs'):
                for i in range(len(scheduler.base_lrs)):
                    scheduler.base_lrs[i] = new_lr
            
    except Exception as e:
        logger.warning(f"Failed to load training state: {e}")

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train NCN-Modulated Language Model")
    parser.add_argument('--data_file', type=str, default=DEFAULT_DATA_FILE, help="Path to train.bin.")
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH, help="Path to tokenizer.")
    parser.add_argument('--val_ratio', type=float, default=0.05, help="Ratio of data to use for validation.")
    parser.add_argument('--validation_steps', type=int, default=200, help="Number of batches to run for validation.")
    # Model Config
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence length.")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension.")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers.")
    parser.add_argument("--n_head", type=int, default=12, help="Number of heads.")
    parser.add_argument("--dim_feedforward", type=int, default=None, help="FFN hidden dim.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5, help="LayerNorm epsilon.")
    parser.add_argument("--tie_weights", action=argparse.BooleanOptionalAction, default=True, help="Tie embedding/output weights.")
    # NCN Config
    parser.add_argument("--ncn_input_dim", type=int, default=None, help="NCN input dim (default: d_model).")
    parser.add_argument("--ncn_hidden_dim", type=int, default=128, help="NCN hidden dimension.")
    parser.add_argument("--ncn_heads", type=int, default=4, help="Number of heads for NCN attention pooling.")
    parser.add_argument("--mod_signal_names", nargs='+', default=["gain", "precision", "ffn_gate"], help="NCN modulation signal names.")
    parser.add_argument("--ncn_act_fn", type=str, default="relu", help="NCN activation.")
    # Training Params
    parser.add_argument('--num_epochs', type=int, required=True, help="Total training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size per device.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Accumulate gradients.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument('--warmup_steps', type=int, default=100, help="LR warmup steps.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    # Optimization & Memory
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing to save VRAM.")
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./ncn_checkpoints', help="Checkpoint save directory.")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Resume from checkpoint path.")
    parser.add_argument('--checkpoint_frequency', type=int, default=5000, help="Save ckpt every N steps.")
    # Logging
    parser.add_argument('--log_frequency', type=int, default=100, help="Log every N steps.")
    parser.add_argument("--use_wandb", action='store_true', help="Enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, default="ncn-llm-train", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")
    # Hardware
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help="Device.")
    parser.add_argument("--num_workers", type=str, default="0", help="Dataloader workers.")
    parser.add_argument("--use_amp", action='store_true', help="Enable Automatic Mixed Precision (AMP).")
    args = parser.parse_args()
    if args.d_model:
        if args.dim_feedforward is None: args.dim_feedforward = 4 * args.d_model
        if args.ncn_input_dim is None: args.ncn_input_dim = args.d_model
    return args

# --- Collate Function ---
def collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return torch.stack(xs), torch.stack(ys)

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model: Union[ModulatedLLM, DDP], loader: DataLoader, device: torch.device, args: argparse.Namespace) -> float:
    if loader is None: return float('inf')
    model.eval(); total_loss = 0.0; num_samples = 0
    criterion_eval = nn.CrossEntropyLoss(reduction='sum')
    if args.is_master: logger.info("Starting validation...")

    limit_steps = args.validation_steps if args.validation_steps > 0 else len(loader)
    
    for i, (input_ids, target_ids) in enumerate(loader):
        if i >= limit_steps: break
        
        input_ids, target_ids = input_ids.to(device, non_blocking=True), target_ids.to(device, non_blocking=True)
        batch_size, seq_len = input_ids.size()
        
        if TORCH_2_AMP and args.use_amp:
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=True):
                logits, _, _ = model(input_ids=input_ids)
                loss = criterion_eval(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        elif args.use_amp: 
            with torch.cuda.amp.autocast(enabled=True):
                logits, _, _ = model(input_ids=input_ids)
                loss = criterion_eval(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        else:
             logits, _, _ = model(input_ids=input_ids)
             loss = criterion_eval(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
        total_loss += loss.item()
        num_samples += batch_size * seq_len

    if args.is_ddp:
        stats = torch.tensor([total_loss, num_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, num_samples = stats[0].item(), stats[1].item()

    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    perplexity = math.exp(avg_loss) if 0 < avg_loss < 700 else float('inf')
    
    if args.is_master: logger.info(f"Validation Result: Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    model.train(); return avg_loss

# --- Main ---
def main():
    args = parse_arguments()
    is_ddp, rank, world_size, local_rank, is_master = setup_ddp()
    args.is_ddp, args.rank, args.world_size, args.local_rank, args.is_master = is_ddp, rank, world_size, local_rank, is_master
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Enable cuDNN benchmark
    if device.type == "cuda":
        cudnn.benchmark = True
        if is_master: logger.info("Enabled torch.backends.cudnn.benchmark for optimization.")

    if args.num_workers.isdigit(): args.num_workers = int(args.num_workers)
    else: args.num_workers = 0

    set_seed(args.seed + rank)
    
    if is_master: 
        logger.info("Starting NCN Training (Binary Mode)")
        logger.info(f"Device: {device}, AMP: {args.use_amp}, Gradient Checkpointing: {args.gradient_checkpointing}")
        logger.info("="*40)
        logger.info("Command Line Arguments:")
        logger.info(json.dumps(vars(args), indent=2, default=str))
        logger.info("="*40)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if is_master and WANDB_AVAILABLE and args.use_wandb:
        run_name = args.wandb_run_name or f"ncn_bin_{time.strftime('%Y%m%d_%H%M%S')}";
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    try: tokenizer = load_tokenizer(args.tokenizer_path)
    except Exception: return
    args.vocab_size = tokenizer.vocab_size

    # --- 1. PRE-LOAD METADATA ---
    model_config = None
    start_epoch = 0; global_step = 0; best_val_loss = float('inf')
    ckpt_path = None

    if args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint
        if args.resume_checkpoint == 'latest': ckpt_path = f"{args.checkpoint_dir}/checkpoint_latest.pt"
        if args.resume_checkpoint == 'best': ckpt_path = f"{args.checkpoint_dir}/checkpoint_best.pt"
        
        # Load metadata BEFORE dataset creation
        loaded_config, start_epoch, global_step, best_val_loss = load_resume_metadata(ckpt_path, device)
        
        if loaded_config:
            model_config = loaded_config
            if is_master: logger.info(f"Resumed metadata from {ckpt_path}. Epoch: {start_epoch+1}, Step: {global_step}")
            
            # CRITICAL SYNC: Overwrite CLI args with Checkpoint Config to prevent data loader crashes
            if is_master:
                logger.info("Overwriting CLI args with Checkpoint Config to ensure compatibility:")
                logger.info(f"  block_size: {args.block_size} -> {model_config.max_position_embeddings}")
                logger.info(f"  d_model:    {args.d_model} -> {model_config.d_model}")
            
            args.block_size = model_config.max_position_embeddings
            args.d_model = model_config.d_model
            args.n_layer = model_config.num_layers
            args.n_head = model_config.nhead
            # Sync vocab size as well
            args.vocab_size = model_config.vocab_size

    # --- 2. INIT DATASET (Now using the correct synced args.block_size) ---
    try:
        if is_master: logger.info(f"Mapping binary data: {args.data_file} ...")
        train_dataset = BinaryDataset(args.data_file, args.block_size, split="train", val_ratio=args.val_ratio)
        val_dataset = BinaryDataset(args.data_file, args.block_size, split="val", val_ratio=args.val_ratio)
    except FileNotFoundError as e:
        if is_master: logger.error(str(e))
        return

    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler, 
        num_workers=args.num_workers,
        pin_memory=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, 
        collate_fn=collate_fn
    )

    # --- 3. INIT MODEL ---
    if model_config is None:
        if is_master: logger.info("Initializing new NCN model...")
        model_config = NCNConfig(
            vocab_size=args.vocab_size, d_model=args.d_model, nhead=args.n_head, num_layers=args.n_layer, 
            dim_feedforward=args.dim_feedforward, dropout=args.dropout, max_position_embeddings=args.block_size, 
            ncn_input_dim=args.ncn_input_dim, ncn_hidden_dim=args.ncn_hidden_dim, ncn_heads=args.ncn_heads, 
            num_mod_signals=len(args.mod_signal_names), modulation_signal_names=args.mod_signal_names, 
            ncn_activation_fn=args.ncn_act_fn, tie_weights=args.tie_weights,
            gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        # Allow CLI overriding of gradient_checkpointing even on resume
        if hasattr(model_config, 'gradient_checkpointing'):
             model_config.gradient_checkpointing = args.gradient_checkpointing
             if is_master: logger.info(f"Updated gradient_checkpointing to {args.gradient_checkpointing}")

    if is_master:
        logger.info("Model Configuration Details:")
        logger.info(f"  Gradient Checkpointing: {model_config.gradient_checkpointing}")
        logger.info(f"  Vocabulary Size: {model_config.vocab_size}")
        logger.info(f"  Context Length: {model_config.max_position_embeddings}")
        logger.info(f"  Embedding Dim: {model_config.d_model}")
        logger.info(f"  Layers: {model_config.num_layers}")
        logger.info("-" * 40)

    model = ModulatedLLM(model_config).to(device)
    
    # --- LOAD WEIGHTS ---
    if args.resume_checkpoint and model_config and ckpt_path:
        load_model_state_dict(ckpt_path, model, device)

    if is_ddp: model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- INIT OPTIMIZER & SCALER ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if TORCH_2_AMP and args.use_amp:
        scaler = GradScaler('cuda', enabled=True)
    elif args.use_amp:
        scaler = GradScaler(enabled=True)
    else:
        scaler = GradScaler(enabled=False)

    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # --- LOAD TRAINING STATE (Optim/Sched/Scaler) ---
    if args.resume_checkpoint and model_config and ckpt_path:
        # Pass args.lr as the new override value
        load_training_state(ckpt_path, optimizer, scheduler, scaler, device, new_lr=args.lr)

    if is_master: 
        logger.info(f"Training started. Total Sequences: {len(train_dataset)}")
        logger.info(f"Steps per epoch: {len(train_loader)}")

    # Initialize performance tracking vars
    tokens_processed_since_log = 0
    last_log_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        if is_ddp: train_sampler.set_epoch(epoch)
        model.train()
        
        # --- FAST FORWARD LOGIC ---
        # Calculate how many micro-batches we have already done in this epoch
        resume_index = 0
        if args.resume_checkpoint and epoch == start_epoch:
            # We track global_step (Optimization steps)
            # Micro-batches done = global_step * accumulation_steps
            total_micro_batches_done = global_step * args.gradient_accumulation_steps
            # Micro-batches done IN THIS EPOCH
            resume_index = total_micro_batches_done % len(train_loader)
            if is_master and resume_index > 0:
                logger.info(f"Resuming mid-epoch. Fast-forwarding {resume_index} micro-batches...")

        # Initialize TQDM without 'initial' to show correct progress bar
        epoch_iter = tqdm(train_loader, desc=f"Ep {epoch+1}", disable=not is_master)
        
        for batch_idx, (x, y) in enumerate(epoch_iter):
            # Skip batches until we reach where we left off
            if batch_idx < resume_index:
                continue

            x, y = x.to(device), y.to(device)
            
            # Forward
            if TORCH_2_AMP and args.use_amp:
                device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    logits, _, reg_loss = model(input_ids=x)
                    task_loss = nn.CrossEntropyLoss()(logits.view(-1, args.vocab_size), y.view(-1))
                    loss = (task_loss + reg_loss) / args.gradient_accumulation_steps
            elif args.use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits, _, reg_loss = model(input_ids=x)
                    task_loss = nn.CrossEntropyLoss()(logits.view(-1, args.vocab_size), y.view(-1))
                    loss = (task_loss + reg_loss) / args.gradient_accumulation_steps
            else:
                 logits, _, reg_loss = model(input_ids=x)
                 task_loss = nn.CrossEntropyLoss()(logits.view(-1, args.vocab_size), y.view(-1))
                 loss = (task_loss + reg_loss) / args.gradient_accumulation_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            # Track tokens
            tokens_processed_since_log += x.numel()

            # Optimizer Step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # --- DETAILED LOGGING ---
                if global_step % args.log_frequency == 0 and is_master:
                    current_time = time.time()
                    elapsed_time = current_time - last_log_time
                    
                    # Calculate throughput
                    tok_sec = tokens_processed_since_log / elapsed_time if elapsed_time > 0 else 0.0
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    loss_val = task_loss.item()
                    ppl = math.exp(loss_val) if loss_val < 20 else float('inf')
                    scale_val = scaler.get_scale()
                    
                    # Percentage Progress
                    pct = (batch_idx + 1) / len(train_loader) * 100
                    
                    # Full formatted string
                    log_str = (f"Step: {global_step} | Ep: {epoch+1} [{batch_idx+1}/{len(train_loader)} ({pct:.1f}%)] | "
                               f"Loss: {loss_val:.4f} | Reg: {reg_loss.item():.4f} | PPL: {ppl:.2f} | "
                               f"LR: {current_lr:.2e} | Scale: {scale_val:.1f} | Tok/s: {tok_sec:.0f}")
                    
                    logger.info(log_str)
                    epoch_iter.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")
                    
                    if WANDB_AVAILABLE and args.use_wandb:
                        wandb.log({
                            "train/loss": loss_val, 
                            "train/reg_loss": reg_loss.item(), 
                            "train/perplexity": ppl, 
                            "lr": current_lr,
                            "train/tokens_per_sec": tok_sec,
                            "train/amp_scale": scale_val
                        }, step=global_step)
                        
                    # Reset counters
                    tokens_processed_since_log = 0
                    last_log_time = current_time

                # Checkpointing
                if args.checkpoint_frequency > 0 and global_step % args.checkpoint_frequency == 0:
                     save_checkpoint({
                         'epoch': epoch, 
                         'global_step': global_step, 
                         'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'scaler_state_dict': scaler.state_dict(),
                         'config': model_config
                     }, args.checkpoint_dir, f"checkpoint_step_{global_step}.pt", is_master)

        val_loss = evaluate(model, val_loader, device, args)
        
        if is_master:
            is_best = val_loss < best_val_loss
            if is_best: best_val_loss = val_loss
            
            save_checkpoint({
                'epoch': epoch + 1, 
                'global_step': global_step, 
                'best_val_loss': best_val_loss,
                'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': model_config
            }, args.checkpoint_dir, "checkpoint_latest.pt", is_master)
            
            if is_best:
                save_checkpoint({
                'epoch': epoch + 1, 
                'global_step': global_step, 
                'best_val_loss': best_val_loss,
                'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': model_config
            }, args.checkpoint_dir, "checkpoint_best.pt", is_master)
        
        # Reset resume index for subsequent epochs
        resume_index = 0

    if is_master: logger.info("Training Complete.")
    cleanup_ddp()

if __name__ == "__main__":
    main()