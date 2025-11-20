# ncn_project/train.py

"""
Training script for NCN-Modulated Language Model.

Features IterableDataset for efficient data handling, detailed checkpointing/resumption,
DDP support for multi-GPU training, validation loop with pre-tokenization caching,
and optional WandB logging, tailored for the NCN architecture.

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Original Author: Michael Morgan
Date: 2025-04-30
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
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.serialization
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler # Keep GradScaler from cuda.amp
# Updated Autocast import for PyTorch 2.x compatibility
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

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

# --- Allowlist NCNConfig for torch.load ---
torch.serialization.add_safe_globals([NCNConfig])

# --- Constants & Logging ---
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training.log"
DEFAULT_TOKENIZER_PATH = "gpt2_tokenizer_files"
DEFAULT_DATA_DIR = "training_data"
TOKENIZED_VAL_CACHE = "tokenized_val_data.pt"

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
    logger.warning("wandb not installed. WandB logging disabled. Run `pip install wandb` to enable.")


# --- Helper Functions ---
def format_param_count(count):
    """Formats parameter count into K, M, B."""
    if count > 1e9: return f"{count / 1e9:.2f}B"
    if count > 1e6: return f"{count / 1e6:.2f}M"
    if count > 1e3: return f"{count / 1e3:.2f}K"
    return str(count)

def find_text_files(root_dir: str) -> List[str]:
    """Recursively finds all .txt files in a directory."""
    txt_files = []
    if not os.path.isdir(root_dir): logger.error(f"Data directory '{root_dir}' not found."); return []
    logger.info(f"Scanning for .txt files in '{root_dir}'...");
    for root, _, files in os.walk(root_dir): [txt_files.append(os.path.join(root, file)) for file in files if file.lower().endswith(".txt")]
    logger.info(f"Found {len(txt_files)} .txt files.")
    return txt_files

def count_total_tokens_multi_file(file_paths: List[str], tokenizer: Tokenizer, sequence_length: int) -> Tuple[Optional[int], Optional[int]]:
    """Counts total tokens by tokenizing multiple files line by line."""
    if not file_paths: logger.error("No text files for token counting."); return None, None
    logger.info(f"Pre-calculating total tokens across {len(file_paths)} files...");
    total_tokens = 0; total_lines = 0; skipped_lines = 0
    for file_path in tqdm(file_paths, desc="Counting Tokens", unit="file"):
        file_tokens = 0; file_lines = 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    file_lines += 1
                    if not line.strip(): continue
                    try:
                        encoded = tokenizer.encode(line, add_special_tokens=False)
                        ids = encoded['input_ids'] if isinstance(encoded, dict) else encoded
                        if isinstance(ids, list): file_tokens += len(ids)
                        else: skipped_lines += 1
                    except Exception: skipped_lines += 1
        except FileNotFoundError: logger.warning(f"File not found during token count: {file_path}. Skipping."); continue
        except Exception as e: logger.warning(f"Error reading file {file_path} during token count: {e}. Skipping."); continue
        total_tokens += file_tokens; total_lines += file_lines
    logger.info(f"Token counting complete. Processed {total_lines} lines.");
    if skipped_lines > 0: logger.warning(f"Skipped {skipped_lines} lines during counting.")
    total_seq_approx = total_tokens // sequence_length if sequence_length > 0 else 0
    logger.info(f"Total tokens: {format_param_count(total_tokens)} (raw: {total_tokens}), Approx sequences: {format_param_count(total_seq_approx)}")
    return total_tokens, total_seq_approx

def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_ddp():
    """Initializes Torch Distributed."""
    backend = 'nccl'; local_rank = int(os.environ.get("LOCAL_RANK", -1)); world_size = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", -1))
    if world_size > 1:
        if local_rank == -1 or rank == -1: raise RuntimeError("DDP environment variables not set.")
        logger.info(f"Initializing DDP: WS={world_size}, R={rank}, LR={local_rank}"); dist.init_process_group(backend=backend); torch.cuda.set_device(local_rank); logger.info(f"DDP initialized: Rank {rank}/{world_size} on cuda:{local_rank}.")
        is_ddp = True; is_master = (rank == 0)
    else: logger.info("DDP not enabled."); is_ddp = False; is_master = True; local_rank = 0; rank = 0
    return is_ddp, rank, world_size, local_rank, is_master

def cleanup_ddp():
    """Cleans up Torch Distributed."""
    if dist.is_initialized(): dist.destroy_process_group(); logger.info("DDP process group destroyed.")

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Loads the custom tokenizer, handling PAD token."""
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    if not os.path.exists(tokenizer_path): raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' not found.")
    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_path); logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
            logger.warning(f"Tokenizer invalid PAD ID ({tokenizer.pad_token_id}). Using EOS."); eos_token = getattr(tokenizer, 'eos_token', None) or (tokenizer.model.id_to_token(50256) if hasattr(tokenizer.model, 'id_to_token') else None) or "<|endoftext|>"
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is not None and eos_token_id != tokenizer.unk_token_id: logger.warning(f"Using EOS '{eos_token}' ({eos_token_id}) as PAD."); tokenizer.pad_token_id = eos_token_id;
            else: logger.error(f"Could not find valid EOS ('{eos_token}') as PAD."); raise ValueError("Need PAD/EOS token.")
        else: logger.info(f"Using PAD token ID: {tokenizer.pad_token_id} ('{tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)}')")
        return tokenizer
    except Exception as e: logger.error(f"Failed to load tokenizer: {e}", exc_info=True); raise

# --- Datasets ---
class MultiFileTextDataset(IterableDataset):
    """Streams and tokenizes lines from multiple text files for training. Supports num_workers > 0."""
    def __init__(self, file_paths: List[str], tokenizer: Tokenizer, sequence_length: int, skip_sequences: int = 0):
        self.file_paths = file_paths; self.tokenizer = tokenizer; self.sequence_length = sequence_length; self.skip_sequences = max(0, skip_sequences)
        if self.skip_sequences > 0: logger.info(f"Dataset Iterator: Skipping approx {self.skip_sequences} sequences (distributed).")
    
    def __iter__(self):
        # Worker logic for sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # Single process
            files_to_read = self.file_paths
            skip_target = self.skip_sequences
        else: # Multi-process: Split files among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Calculate chunk size per worker
            per_worker = int(math.ceil(len(self.file_paths) / float(num_workers)))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            files_to_read = self.file_paths[iter_start:iter_end]
            # Distribute skip sequences among workers roughly evenly
            skip_target = self.skip_sequences // num_workers
            
        token_buffer = []; sequences_yielded_total = 0; sequences_skipped_count = 0
        
        for file_path in files_to_read:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
                    for line_num, line in enumerate(file_handle):
                        if not line.strip(): continue
                        try:
                            encoded = self.tokenizer.encode(line, add_special_tokens=False); ids = encoded.get('input_ids') if isinstance(encoded, dict) else encoded
                            if not ids or not isinstance(ids, list): continue
                            token_buffer.extend(ids)
                            while len(token_buffer) >= self.sequence_length + 1:
                                chunk = token_buffer[:self.sequence_length + 1]; token_buffer = token_buffer[self.sequence_length:]
                                if sequences_skipped_count < skip_target:
                                    sequences_skipped_count += 1
                                    # Only print progress from one worker to avoid log spam
                                    if worker_info is None or worker_info.id == 0:
                                        if sequences_skipped_count % 5000 == 0: print(f"  ... skipped {sequences_skipped_count}/{skip_target} (worker local)", end='\r', file=sys.stderr)
                                    continue
                                x = torch.tensor(chunk[:-1], dtype=torch.long); y = torch.tensor(chunk[1:], dtype=torch.long)
                                yield x, y; sequences_yielded_total += 1
                        except Exception: continue
            except FileNotFoundError: logger.warning(f"File not found: {file_path}. Skipping."); continue
            except Exception as e: logger.warning(f"Error reading {file_path}: {e}. Skipping."); continue
            
        if skip_target > 0 and (worker_info is None or worker_info.id == 0): 
             print(' ' * 80, end='\r', file=sys.stderr); 

class ValidationTextDataset(Dataset):
    """Loads and uses pre-tokenized validation data."""
    def __init__(self, token_ids: List[int], block_size: int): # Takes token_ids directly
        self.block_size = block_size; self.token_ids = token_ids; self.num_sequences = max(0, len(self.token_ids) - self.block_size)
        if len(self.token_ids) < block_size + 1: logger.warning(f"Validation data has only {len(self.token_ids)} tokens < block_size+1.")
    def __len__(self): return self.num_sequences
    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.block_size + 1];
        if len(chunk) < self.block_size + 1: raise IndexError(f"Index {idx} invalid for block_size {self.block_size}")
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

# --- Checkpoint Functions ---
def save_checkpoint(state: Dict[str, Any], directory: str, filename: str = "checkpoint.pt", is_master: bool = True):
    """Saves checkpoint state to file."""
    if not is_master: return
    required = ['epoch', 'global_step', 'model_state_dict', 'optimizer_state_dict', 'config']
    if not all(k in state for k in required): logger.warning(f"Ckpt missing keys: {[k for k in required if k not in state]}")
    os.makedirs(directory, exist_ok=True); filepath = os.path.join(directory, filename); logger.info(f"Saving checkpoint to {filepath}...");
    try: torch.save(state, filepath); logger.info("Checkpoint saved.")
    except Exception as e: logger.error(f"ERROR saving checkpoint: {e}", exc_info=True)

def load_model_state_dict(filepath: str, model: Union[ModulatedLLM, DDP], device: torch.device) -> bool:
    """Loads model state dict from checkpoint."""
    if not os.path.exists(filepath): logger.error(f"Model checkpoint not found: {filepath}"); return False
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            target_model = model.module if isinstance(model, DDP) else model; load_result = target_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if load_result.missing_keys: logger.warning(f"Missing keys loading model state: {load_result.missing_keys}")
            if load_result.unexpected_keys: logger.warning(f"Unexpected keys loading model state: {load_result.unexpected_keys}")
            logger.info("Model state_dict loaded (strict=False)."); return True
        else: logger.error("'model_state_dict' not found in checkpoint."); return False
    except Exception as e: logger.error(f"Error loading model state_dict: {e}", exc_info=True); return False

def load_optimizer_and_scaler_state_dict(filepath: str, optimizer: optim.Optimizer, scaler: Optional[GradScaler], device: torch.device, new_lr: Optional[float] = None) -> Tuple[bool, bool]:
    """Loads optimizer and scaler state dicts."""
    opt_loaded, scaler_loaded = False, False
    if not os.path.exists(filepath): logger.warning(f"Opt/Scaler checkpoint not found: {filepath}."); return False, False
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict']); logger.info("Optimizer state loaded."); opt_loaded = True
            except ValueError as e: logger.warning(f"Could not load optimizer state (param mismatch?): {e}."); opt_loaded = False
            if opt_loaded and new_lr is not None: logger.info(f"Setting loaded optimizer LR to: {new_lr}"); [pg.update(lr=new_lr) for pg in optimizer.param_groups]
            logger.info(f"Optimizer LR after load: {optimizer.param_groups[0]['lr']:.2e}")
        else: logger.warning("Optimizer state not found.")
        if scaler and 'scaler_state_dict' in checkpoint:
            try: scaler.load_state_dict(checkpoint['scaler_state_dict']); logger.info("GradScaler state loaded."); scaler_loaded = True
            except Exception as se: logger.warning(f"Could not load GradScaler state: {se}"); scaler_loaded = False
        elif scaler: logger.warning("GradScaler state not found.")
        return opt_loaded, scaler_loaded
    except Exception as e: logger.error(f"Error loading opt/scaler state: {e}", exc_info=True); return False, False

def load_resume_state(filepath: str, device: torch.device) -> Tuple[Optional[NCNConfig], int, int, int, int, float]:
    """Loads training state variables and NCNConfig from checkpoint."""
    if not os.path.exists(filepath): logger.error(f"Resume checkpoint not found: {filepath}"); return None, 0, 0, 0, 0, float('inf')
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        cfg_data = checkpoint.get('config'); model_config = cfg_data if isinstance(cfg_data, NCNConfig) else None
        if model_config is None: logger.error(f"Loaded 'config' not NCNConfig instance ({type(cfg_data)})."); return None, 0, 0, 0, 0, float('inf')
        start_epoch = checkpoint.get('epoch', 0); global_step = checkpoint.get('global_step', 0); processed_tokens = checkpoint.get('processed_tokens', 0); seq_skip = checkpoint.get('sequences_processed_this_epoch', 0); best_val = checkpoint.get('best_val_loss', float('inf'))
        if not all(isinstance(v, int) for v in [start_epoch, global_step, processed_tokens, seq_skip]): logger.warning("Loaded non-integer state vars. Resetting."); return model_config, 0, 0, 0, 0, float('inf')
        logger.info(f"Loaded resume state: Epoch {start_epoch+1}, Step {global_step}, Tokens {processed_tokens}, Seq Skip {seq_skip}")
        if best_val != float('inf'): logger.info(f"Loaded best_val_loss: {best_val:.4f}")
        return model_config, start_epoch, global_step, processed_tokens, seq_skip, best_val
    except Exception as e: logger.error(f"Error loading resume state: {e}", exc_info=True); return None, 0, 0, 0, 0, float('inf')

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train NCN-Modulated Language Model")
    # Data/Tokenizer
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help="ROOT training data directory.")
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH, help="Path to tokenizer.")
    parser.add_argument('--train_val_split', type=float, default=0.98, help="Train/Val split ratio. Consider 0.95 for larger val set.")
    parser.add_argument('--val_token_cache', type=str, default=TOKENIZED_VAL_CACHE, help="Path to save/load tokenized validation data cache.")
    parser.add_argument('--validation_steps', type=int, default=500, help="Number of batches to run for validation. 0 to run on full dataset.")
    # Model Config
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence length.")
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocab size (from tokenizer).")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension.")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers.")
    parser.add_argument("--n_head", type=int, default=12, help="Number of heads.")
    parser.add_argument("--dim_feedforward", type=int, default=None, help="FFN hidden dim (default: 4*d_model).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5, help="LayerNorm epsilon.")
    parser.add_argument("--tie_weights", action=argparse.BooleanOptionalAction, default=True, help="Tie embedding/output weights.")
    # NCN Config
    parser.add_argument("--ncn_input_dim", type=int, default=None, help="NCN input dim (default: d_model).")
    parser.add_argument("--ncn_hidden_dim", type=int, default=128, help="NCN hidden dimension.")
    parser.add_argument("--ncn_heads", type=int, default=4, help="Number of heads for NCN attention pooling.")
    parser.add_argument("--mod_signal_names", nargs='+', default=["gain", "precision", "ffn_gate"], help="NCN modulation signal names.")
    parser.add_argument("--ncn_act_fn", type=str, default="relu", help="NCN activation (relu, gelu, tanh, sigmoid).")
    # Training Params
    parser.add_argument('--num_epochs', type=int, required=True, help="Total training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size per device.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument('--adam_betas', type=lambda s: tuple(map(float, s.split(','))), default=(0.9, 0.95), help="AdamW betas (e.g., 0.9,0.95).")
    parser.add_argument('--warmup_steps', type=int, default=100, help="LR warmup steps.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping (0 to disable).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./ncn_checkpoints', help="Checkpoint save directory.")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Resume from checkpoint path ('latest'/'best').")
    parser.add_argument('--checkpoint_frequency', type=int, default=5000, help="Save ckpt every N steps (0=disable).")
    parser.add_argument('--save_best_only', action='store_true', help="Only keep best ckpt by val loss.")
    # Logging/Monitoring
    parser.add_argument('--log_frequency', type=int, default=100, help="Log every N steps.")
    parser.add_argument('--total_tokens', type=int, default=None, help="Optional: Pre-calculated total tokens.")
    parser.add_argument("--use_wandb", action='store_true', help="Enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, default="ncn-llm-train", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")
    # Hardware/Performance
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help="Device ('auto').")
    parser.add_argument("--num_workers", type=str, default="0", help="Dataloader workers. '0' (default), int, or 'auto'.")
    parser.add_argument("--use_amp", action='store_true', help="Enable Automatic Mixed Precision (AMP).")
    args = parser.parse_args()
    if args.d_model:
        if args.dim_feedforward is None: args.dim_feedforward = 4 * args.d_model
        if args.ncn_input_dim is None: args.ncn_input_dim = args.d_model
        if args.n_head and args.d_model % args.n_head != 0: parser.error(f"d_model ({args.d_model}) must be divisible by n_head ({args.n_head})")
    return args

# --- Collate Function ---
def collate_fn(batch):
    """Collates batches, handling potential empty items."""
    valid_items = [item for item in batch if item is not None and item[0].numel() > 0 and item[1].numel() > 0]
    if not valid_items: return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long)
    try:
        xs = [item[0] for item in valid_items]; ys = [item[1] for item in valid_items]
        seq_len_x = xs[0].shape[0]; seq_len_y = ys[0].shape[0]
        batch_x = torch.stack(xs); batch_y = torch.stack(ys)
        return batch_x, batch_y
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}.", exc_info=True)
        try: seq_len_x = batch[0][0].shape[0]; seq_len_y = batch[0][1].shape[0]
        except: seq_len_x = 0; seq_len_y = 0
        return torch.empty((0, seq_len_x), dtype=torch.long), torch.empty((0, seq_len_y), dtype=torch.long)

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model: Union[ModulatedLLM, DDP], loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Optional[float]:
    """Evaluates the model on the validation set."""
    if loader is None or len(loader) == 0:
        if args.is_master: logger.warning("Validation loader empty, skipping eval.")
        return None
    model.eval(); total_loss = 0.0; num_samples = 0
    criterion_eval = nn.CrossEntropyLoss(reduction='sum')
    if args.is_master: logger.info("Starting validation...")

    num_loader_batches = len(loader)
    progress_bar_total = num_loader_batches
    if args.validation_steps > 0:
        progress_bar_total = min(args.validation_steps, num_loader_batches)
    val_iterator = tqdm(loader, desc="Validation", unit="batch", total=progress_bar_total, disable=(not args.is_master), leave=False)

    for batch_idx, (input_ids, target_ids) in enumerate(val_iterator):
        if args.validation_steps > 0 and batch_idx >= args.validation_steps:
            break

        if input_ids.numel() == 0: continue
        input_ids, target_ids = input_ids.to(device, non_blocking=True), target_ids.to(device, non_blocking=True)
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        with torch.amp.autocast(device_type=device.type if args.use_amp else 'cpu', enabled=args.use_amp):
            try:
                # Unpack returned tuple: (logits, past_key_values, reg_loss)
                logits, _, _ = model(input_ids=input_ids, attention_mask=None)
                vocab_size_eval = logits.size(-1)
                loss = criterion_eval(logits.view(-1, vocab_size_eval), target_ids.view(-1))
            except Exception as e:
                 if args.is_master: logger.error(f"Error during validation batch {batch_idx}: {e}", exc_info=True)
                 continue
        if not torch.isnan(loss) and not torch.isinf(loss): total_loss += loss.item(); num_samples += batch_size * seq_len
        else: logger.warning(f"NaN/Inf loss in validation batch {batch_idx}.") if args.is_master else None

    if args.is_ddp:
        loss_t = torch.tensor([total_loss], device=device); samples_t = torch.tensor([num_samples], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM); dist.all_reduce(samples_t, op=dist.ReduceOp.SUM)
        total_loss_global, total_samples_global = loss_t.item(), samples_t.item()
    else: total_loss_global, total_samples_global = total_loss, num_samples
    final_avg_loss = total_loss_global / total_samples_global if total_samples_global > 0 else float('inf')
    perplexity = math.exp(final_avg_loss) if 0 < final_avg_loss < float('inf') else float('inf')
    if args.is_master: logger.info(f"Validation Complete: Avg Loss: {final_avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    model.train(); return final_avg_loss

# --- Main Training Function ---
def main():
    args = parse_arguments()
    is_ddp, rank, world_size, local_rank, is_master = setup_ddp()
    args.is_ddp, args.rank, args.world_size, args.local_rank, args.is_master = is_ddp, rank, world_size, local_rank, is_master
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"); amp_enabled = args.use_amp and device.type == 'cuda'
    
    # --- Auto-detect num_workers ---
    if args.num_workers.lower() == 'auto':
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            if cpu_count >= 4:
                args.num_workers = 2
            elif cpu_count > 1:
                args.num_workers = 1
            else:
                args.num_workers = 0
            if is_master: logger.info(f"Auto-detected num_workers: {args.num_workers} (based on {cpu_count} cores)")
        except Exception as e:
            logger.warning(f"Auto-detection failed ({e}). Defaulting num_workers to 0.")
            args.num_workers = 0
    else:
        try:
            args.num_workers = int(args.num_workers)
        except ValueError:
            logger.error(f"Invalid num_workers value: {args.num_workers}. Using 0.")
            args.num_workers = 0
    # -------------------------------

    set_seed(args.seed + rank)
    if is_master: logger.info("Starting NCN Training Script"); logger.info(f"Args: {vars(args)}"); logger.info(f"Device: {device}, DDP: {is_ddp}, AMP: {amp_enabled}"); os.makedirs(args.checkpoint_dir, exist_ok=True)
    if is_master and WANDB_AVAILABLE and args.use_wandb:
        run_name = args.wandb_run_name or f"ncn_{time.strftime('%Y%m%d_%H%M%S')}";
        try: wandb.init(project=args.wandb_project, name=run_name, config=vars(args)); logger.info("WandB initialized.")
        except Exception as e: logger.error(f"WandB init failed: {e}. Disabling."); args.use_wandb = False

    try: tokenizer = load_tokenizer(args.tokenizer_path)
    except Exception: logger.error("Tokenizer loading failed. Exiting.", exc_info=True); cleanup_ddp(); return
    args.vocab_size = tokenizer.vocab_size

    all_text_files = find_text_files(args.data_dir)
    if not all_text_files: logger.error(f"No .txt files in {args.data_dir}. Exiting."); cleanup_ddp(); return
    random.Random(args.seed).shuffle(all_text_files); split_idx = int(args.train_val_split * len(all_text_files)); train_files = all_text_files[:split_idx]; val_files = all_text_files[split_idx:]
    if is_master: logger.info(f"Data split: {len(train_files)} train, {len(val_files)} val files.")

    total_tokens_in_dataset = None;
    if args.total_tokens and args.total_tokens > 0: total_tokens_in_dataset = args.total_tokens; logger.info(f"Using pre-calculated tokens: {format_param_count(total_tokens_in_dataset)}") if is_master else None
    elif args.block_size > 0:
        if is_master: logger.info("Calculating total train tokens..."); total_tokens_in_dataset, _ = count_total_tokens_multi_file(train_files, tokenizer, args.block_size); logger.info(f"Master token count done.")
        if is_ddp: token_count_tensor = torch.tensor([total_tokens_in_dataset if is_master and total_tokens_in_dataset is not None else 0], dtype=torch.long, device=device); dist.broadcast(token_count_tensor, src=0); total_tokens_in_dataset = token_count_tensor.item() if rank != 0 else total_tokens_in_dataset; dist.barrier()
        if (total_tokens_in_dataset is None or total_tokens_in_dataset == 0) and is_master: logger.warning("Token count resulted in 0. Progress estimate may be inaccurate.")
        elif is_master and not args.total_tokens and total_tokens_in_dataset is not None and total_tokens_in_dataset > 0:
            logger.info(f"TOKEN COUNT COMPLETED. For future runs to skip this, use: --total_tokens {total_tokens_in_dataset}")

    model: Optional[Union[ModulatedLLM, DDP]] = None; optimizer: Optional[optim.AdamW] = None; scaler = GradScaler(enabled=amp_enabled); model_config: Optional[NCNConfig] = None; start_epoch = 0; sequences_to_skip_on_resume = 0; global_step = 0; processed_tokens_offset = 0; best_val_loss = float('inf')

    if args.resume_checkpoint:
        if is_master: logger.info(f"Attempting resume from: {args.resume_checkpoint}")
        ckpt_path_str = args.resume_checkpoint
        if is_master:
            if args.resume_checkpoint.lower() == 'latest': latest_ckpt = Path(args.checkpoint_dir) / "checkpoint_latest.pt"; ckpt_path_str = str(latest_ckpt) if latest_ckpt.exists() else None
            elif args.resume_checkpoint.lower() == 'best': best_ckpt = Path(args.checkpoint_dir) / "checkpoint_best.pt"; ckpt_path_str = str(best_ckpt) if best_ckpt.exists() else None
            if not ckpt_path_str or not os.path.exists(ckpt_path_str): logger.warning(f"Checkpoint path '{ckpt_path_str}' invalid. Starting fresh."); ckpt_path_str = "NOT_FOUND"
        if is_ddp: path_list = [ckpt_path_str if is_master else None]; dist.broadcast_object_list(path_list, src=0); ckpt_path_str = path_list[0]; dist.barrier()

        if ckpt_path_str == "NOT_FOUND" or ckpt_path_str is None: args.resume_checkpoint = None; logger.warning("Resume failed: Checkpoint not found.") if is_master else None
        else:
            loaded_config, start_epoch, global_step, processed_tokens_offset, sequences_to_skip_on_resume, best_val_loss = load_resume_state(ckpt_path_str, device)
            if loaded_config is None: logger.error("Config load failed. Cannot resume."); cleanup_ddp(); return
            model_config = loaded_config; args.block_size = model_config.max_position_embeddings; model_config.vocab_size = args.vocab_size
            
            if is_master:
                logger.info("\n" + "="*60)
                logger.info(f"  SUCCESSFULLY RESUMING TRAINING FROM CHECKPOINT")
                logger.info(f"  Checkpoint Path: {ckpt_path_str}")
                logger.info(f"  Resuming at Epoch: {start_epoch + 1}, Global Step: {global_step}")
                logger.info("="*60 + "\n")
            
            if model_config.d_model % model_config.nhead != 0: logger.error(f"Loaded config incompatible d_model/nhead."); cleanup_ddp(); return
            try: model = ModulatedLLM(model_config).to(device)
            except Exception as e: logger.error(f"Model init from loaded config failed: {e}", exc_info=True); cleanup_ddp(); return
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.adam_betas)
            load_model_state_dict(ckpt_path_str, model, device); load_optimizer_and_scaler_state_dict(ckpt_path_str, optimizer, scaler, device, new_lr=args.lr)
            if is_master: logger.info("Resumed model/optimizer setup complete.")

    if model is None:
        if is_master: logger.info("Initializing new model...")
        if not args.d_model or not args.n_layer or not args.n_head: logger.error("Missing required model dimensions. Exiting."); cleanup_ddp(); return
        model_config = NCNConfig(
            vocab_size=args.vocab_size, d_model=args.d_model, nhead=args.n_head, num_layers=args.n_layer, 
            dim_feedforward=args.dim_feedforward, dropout=args.dropout, max_position_embeddings=args.block_size, 
            layer_norm_eps=args.layer_norm_eps, ncn_input_dim=args.ncn_input_dim, ncn_hidden_dim=args.ncn_hidden_dim,
            ncn_heads=args.ncn_heads, # Added NCN heads
            num_mod_signals=len(args.mod_signal_names), modulation_signal_names=args.mod_signal_names, 
            ncn_activation_fn=args.ncn_act_fn, tie_weights=args.tie_weights
        )
        try: model = ModulatedLLM(model_config).to(device); optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.adam_betas)
        except Exception as e: logger.error(f"Error creating new NCN model: {e}", exc_info=True); cleanup_ddp(); return
        if is_master: logger.info("Initialized new model and optimizer.")

    if is_ddp: model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False); logger.info("Model wrapped with DDP.") if is_master else None

    if is_master:
        logger.info("\n--- Final Model Configuration ---")
        config_to_print = model.module.config if is_ddp else model.config
        if config_to_print:
            args.block_size = config_to_print.max_position_embeddings
            logger.info(f"Using final block_size: {args.block_size}")
            for key, value in vars(config_to_print).items(): logger.info(f"  {key}: {value}")
        else: logger.error("Model configuration not accessible."); cleanup_ddp(); return
        total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total Parameters: {format_param_count(total_params)}"); logger.info(f"  Trainable Parameters: {format_param_count(trainable_params)}")
        logger.info("---------------------------\n")

    effective_batch_size = args.batch_size * world_size; steps_per_epoch_approx = 0
    if total_tokens_in_dataset and total_tokens_in_dataset > 0 and args.block_size > 0: total_sequences_approx = total_tokens_in_dataset // args.block_size
    else: total_sequences_approx = 0
    if total_sequences_approx and effective_batch_size > 0: steps_per_epoch_approx = max(1, math.ceil(total_sequences_approx / effective_batch_size))
    if is_master: logger.info(f"Final effective BS: {effective_batch_size}, Estimated steps/epoch: ~{steps_per_epoch_approx or 'N/A'}")

    remaining_epochs = args.num_epochs - start_epoch; total_training_steps = steps_per_epoch_approx * remaining_epochs if steps_per_epoch_approx > 0 else 1_000_000
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max(1, total_training_steps))
    if args.resume_checkpoint and ckpt_path_str != "NOT_FOUND" and ckpt_path_str is not None:
        try:
            checkpoint_sched = torch.load(ckpt_path_str, map_location=device, weights_only=False)
            if 'scheduler_state_dict' in checkpoint_sched: scheduler.load_state_dict(checkpoint_sched['scheduler_state_dict']); logger.info(f"Scheduler state loaded (last_epoch={scheduler.last_epoch}).") if is_master else None
            else: logger.warning("Scheduler state not found. Advancing manually."); [scheduler.step() for _ in range(global_step)] if is_master and global_step > 0 else None
        except Exception as e: logger.warning(f"Could not load/advance scheduler state: {e}. Starting fresh."); [scheduler.step() for _ in range(global_step)] if is_master and global_step > 0 else None

    val_loader = None
    if val_files:
        val_token_cache_path = Path(args.val_token_cache)
        val_tokens = []
        if is_master:
            if val_token_cache_path.exists():
                try: logger.info(f"Loading tokenized validation data from cache: {val_token_cache_path}"); val_tokens = torch.load(val_token_cache_path); logger.info(f"Loaded {len(val_tokens)} validation tokens from cache.")
                except Exception as e: logger.warning(f"Failed to load validation cache ({e}). Retokenizing.") ; val_tokens = []
            if not val_tokens:
                logger.info("Tokenizing validation data...")
                for file_path in tqdm(val_files, desc="Tokenizing Validation", unit="file"):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text = f.read()
                        if text: encoded = tokenizer.encode(text, add_special_tokens=False); ids = encoded['input_ids'] if isinstance(encoded, dict) else encoded
                        if ids and isinstance(ids, list): val_tokens.extend(ids); val_tokens.append(tokenizer.pad_token_id)
                    except Exception as e: logger.warning(f"Error tokenizing val file {file_path}: {e}")
                try: logger.info(f"Saving tokenized validation data ({len(val_tokens)} tokens) to cache: {val_token_cache_path}"); torch.save(val_tokens, val_token_cache_path)
                except Exception as e: logger.warning(f"Failed to save validation cache: {e}")
        if is_ddp: objects = [val_tokens if is_master else None]; dist.broadcast_object_list(objects, src=0); val_tokens = objects[0] if not is_master else val_tokens; dist.barrier()
        if val_tokens:
             if is_master: logger.info(f"Total validation tokens prepared: {len(val_tokens)}")
             val_dataset = ValidationTextDataset(val_tokens, args.block_size); val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
             val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False); logger.info("Validation DataLoader created.") if is_master else None
        elif is_master: logger.warning("No validation tokens loaded/tokenized. Skipping validation.")

    criterion = nn.CrossEntropyLoss()
    if is_master: logger.info(f"--- Starting Training: Epochs {start_epoch+1} to {args.num_epochs} ---")
    tokens_processed_session = 0; cumulative_loss_since_log = 0.0; steps_since_log = 0; script_start_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_iter_start_time = time.time(); logger.info(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---") if is_master else None; model.train()
        current_epoch_files = train_files[:]; random.shuffle(current_epoch_files)
        skip_sequences = sequences_to_skip_on_resume if epoch == start_epoch and args.resume_checkpoint else 0
        if skip_sequences > 0 and is_master: logger.info(f"Resuming Epoch {epoch+1}. Skipping {skip_sequences} sequences.")
        train_dataset = MultiFileTextDataset(current_epoch_files, tokenizer, args.block_size, skip_sequences)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
        initial_display = (global_step % steps_per_epoch_approx) if epoch == start_epoch and args.resume_checkpoint and steps_per_epoch_approx > 0 else 0
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch", total=steps_per_epoch_approx or None, initial=initial_display, disable=(not is_master), leave=True)
        sequences_processed_epoch_run = 0

        for batch_idx, batch_data in enumerate(epoch_iterator):
            batch_start_time = time.time(); batch_x, batch_y = batch_data
            if batch_x.numel() == 0: continue
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            try:
                # Updated autocast call for PyTorch 2.x
                with torch.amp.autocast(device_type=device.type if args.use_amp else 'cpu', enabled=args.use_amp):
                    # Unpack tuple return: logits, past_kv, reg_loss
                    logits, _, reg_loss = model(input_ids=batch_x)
                    task_loss = criterion(logits.view(-1, args.vocab_size), batch_y.view(-1))
                    # Add homeostatic regularization
                    loss = task_loss + reg_loss
                
                if torch.isnan(loss) or torch.isinf(loss): logger.warning(f"Invalid loss ({loss.item():.4f}) @ Step {global_step+1}.") if is_master else None; continue
                
                # Scaler Logic to fix LR Scheduler warning
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Capture scale before step
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                # Only step scheduler if optimizer step wasn't skipped (scale didn't decrease)
                if scale_after >= scale_before:
                    scheduler.step()
                    
            except Exception as e: logger.error(f"ERROR @ Step {global_step+1}: {e}", exc_info=True) if is_master else None; optimizer.zero_grad(set_to_none=True); continue

            current_loss = loss.item(); cumulative_loss_since_log += current_loss; current_batch_tokens = batch_x.numel(); tokens_processed_session += current_batch_tokens * world_size; steps_since_log += 1; sequences_processed_epoch_run += batch_x.size(0) * world_size; global_step += 1

            if is_master and args.log_frequency > 0 and global_step % args.log_frequency == 0 and steps_since_log > 0:
                avg_loss = cumulative_loss_since_log / steps_since_log; perplexity = math.exp(min(avg_loss, 700)) if avg_loss > 0 else float('inf'); current_lr = optimizer.param_groups[0]['lr']; batch_time = time.time() - batch_start_time; tokens_per_sec = (current_batch_tokens * world_size) / batch_time if batch_time > 0 else 0
                epoch_batch_num = (global_step - 1) % steps_per_epoch_approx + 1 if steps_per_epoch_approx else global_step; epoch_perc = min(100.0, (epoch_batch_num / steps_per_epoch_approx) * 100) if steps_per_epoch_approx else 0; epoch_prog = f"{epoch_batch_num}/{steps_per_epoch_approx} ({epoch_perc:.1f}%)" if steps_per_epoch_approx else f"Step {global_step}"
                amp_scale_str = f"Scale: {scaler.get_scale():.1f} | " if amp_enabled else ""; 
                # Added Reg loss to log
                log_str = f"Step: {global_step} | Ep: {epoch+1} [{epoch_prog}] | Loss: {avg_loss:.4f} | Reg: {reg_loss.item():.4f} | PPL: {perplexity:.2f} | LR: {current_lr:.2e} | {amp_scale_str}Tok/s: {tokens_per_sec:.0f}"; logger.info(log_str); epoch_iterator.set_description(f"Epoch {epoch+1}/{args.num_epochs} (Loss: {avg_loss:.4f})")
                if WANDB_AVAILABLE and args.use_wandb: epoch_frac = epoch + (epoch_batch_num / steps_per_epoch_approx if steps_per_epoch_approx else 0); wandb.log({"train/loss": avg_loss, "train/reg_loss": reg_loss.item(), "train/perplexity": perplexity, "train/learning_rate": current_lr, "train/tokens_per_sec_effective": tokens_per_sec, "train/amp_scale": scaler.get_scale() if amp_enabled else 0.0, "global_step": global_step, "epoch_frac": epoch_frac}, commit=True)
                cumulative_loss_since_log = 0.0; steps_since_log = 0

            if is_master and args.checkpoint_frequency > 0 and global_step % args.checkpoint_frequency == 0:
                 effective_seq = skip_sequences + sequences_processed_epoch_run; ckpt_state = { 'epoch': epoch, 'sequences_processed_this_epoch': effective_seq, 'global_step': global_step, 'processed_tokens': processed_tokens_offset + tokens_processed_session, 'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict() if amp_enabled else None, 'config': model_config, 'best_val_loss': best_val_loss, 'device_type': device.type }; save_checkpoint(ckpt_state, args.checkpoint_dir, f"checkpoint_step_{global_step}.pt", is_master)

        epoch_iterator.close(); epoch_elapsed = time.time() - epoch_iter_start_time; avg_loss_final = cumulative_loss_since_log / steps_since_log if steps_since_log > 0 else float('nan'); ppl_final = math.exp(min(avg_loss_final, 700)) if not math.isnan(avg_loss_final) and avg_loss_final > 0 else float('inf')
        if is_master: logger.info(f"Epoch {epoch+1} finished data pass. Time: {time.strftime('%H:%M:%S', time.gmtime(epoch_elapsed))}. Final interval Loss: {avg_loss_final:.4f}, PPL: {ppl_final:.2f}")

        val_loss = None
        if val_loader:
            if is_ddp: dist.barrier()
            val_loss = evaluate(model, val_loader, device, args)
            if is_ddp: dist.barrier()

        if is_master:
            is_best = False;
            if val_loss is not None and val_loss < best_val_loss: best_val_loss = val_loss; is_best = True; logger.info(f"** New best val loss: {best_val_loss:.4f} @ Epoch {epoch+1} **")
            logger.info(f"Saving end-of-epoch {epoch+1} checkpoint...")
            ckpt_state = { 'epoch': epoch + 1, 'sequences_processed_this_epoch': 0, 'global_step': global_step, 'processed_tokens': processed_tokens_offset + tokens_processed_session, 'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict() if amp_enabled else None, 'config': model_config, 'val_loss': val_loss, 'best_val_loss': best_val_loss, 'device_type': device.type }
            save_checkpoint(ckpt_state, args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt", is_master); save_checkpoint(ckpt_state, args.checkpoint_dir, f"checkpoint_latest.pt", is_master)
            if is_best: save_checkpoint(ckpt_state, args.checkpoint_dir, f"checkpoint_best.pt", is_master); 

        if is_master and WANDB_AVAILABLE and args.use_wandb:
             epoch_metrics = {"epoch": epoch + 1, "epoch/duration_seconds": epoch_elapsed};
             if val_loss is not None: epoch_metrics["val/epoch_loss"] = val_loss; epoch_metrics["val/epoch_perplexity"] = math.exp(val_loss) if 0 < val_loss < float('inf') else float('inf')
             wandb.log(epoch_metrics, step=global_step, commit=True)

        sequences_to_skip_on_resume = 0; args.resume_checkpoint = None # Reset for next epoch loop

    total_time = time.time() - script_start_time
    if is_master:
        logger.info(f"\n--- Training Finished After Reaching Epoch {args.num_epochs} ---"); logger.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}."); logger.info(f"Final Step: {global_step}"); logger.info(f"Total Tokens: {format_param_count(processed_tokens_offset + tokens_processed_session)}")
        logger.info("Saving final model state..."); final_state = { 'epoch': args.num_epochs, 'sequences_processed_this_epoch': 0,'global_step': global_step, 'processed_tokens': processed_tokens_offset + tokens_processed_session, 'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict() if amp_enabled else None, 'config': model_config, 'val_loss': val_loss, 'best_val_loss': best_val_loss, 'device_type': device.type }; save_checkpoint(final_state, args.checkpoint_dir, "checkpoint_final.pt", is_master); logger.info("Final model saved.")
        if WANDB_AVAILABLE and args.use_wandb: wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    main()