# ncn_project/prepare_data.py

"""
Data preparation script to create train.bin file from a directory of .txt files in training_data.

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Original Author: Michael Morgan
Date: 2025-11-24
Github: https://github.com/Mmorgan-ML
Email: mmorgankorea@gmail.com
Twitter: @Mmorgan_ML
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("training_data")     # Folder containing .txt files
OUTPUT_FILE = "train.bin"            # Output binary file
TOKENIZER_PATH = "gpt2_tokenizer_files" # Path to your tokenizer json/model files

def prepare():
    # 1. Check directories
    if not DATA_DIR.exists():
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    # 2. Load Tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = None
    try:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        print(">> SUCCESS: Using HuggingFace Fast Tokenizer (Rust-backed).")
    except Exception as e:
        print(f">> WARNING: Could not load Fast Tokenizer ({e}). Falling back to local.")
        try:
            from tokenizer import Tokenizer
            tokenizer = Tokenizer.from_pretrained(TOKENIZER_PATH)
        except ImportError:
            print("Error: Could not import 'Tokenizer' from tokenizer.py.")
            sys.exit(1)

    # Check Vocab Size for Data Type
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    if vocab_size < 65535:
        dtype = np.uint16
        print(f"Vocab size is {vocab_size}. Using uint16 (efficient).")
    else:
        dtype = np.int32
        print(f"Vocab size is {vocab_size}. Using int32.")

    # 3. Get EOS token
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_id is None:
        eos_id = 50256 
    print(f"Using EOS Token ID: {eos_id}")

    # 4. Gather Files
    files = sorted(list(DATA_DIR.rglob("*.txt")))
    if not files:
        print(f"No .txt files found in {DATA_DIR}.")
        return

    total_size = sum(os.path.getsize(f) for f in files)
    print(f"Found {len(files)} files. Total size: {total_size / 1024 / 1024:.2f} MB")

    # 5. Processing Loop (STREAMING)
    print(f"Writing to {OUTPUT_FILE}...")
    token_count = 0
    buffer = []
    BUFFER_FLUSH_SIZE = 500_000 # Flush to disk every 500k tokens
    
    # We aggregate text into small chunks to speed up tokenization calls
    # without blowing up RAM.
    TEXT_CHUNK_SIZE = 1024 * 1024 * 5 # Process 5MB of text at a time
    
    with open(OUTPUT_FILE, "wb") as f_out:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Tokenizing") as pbar:
            for file_path in files:
                try:
                    pbar.set_description(f"Proc {file_path.name[:15]}")
                    
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f_in:
                        text_accumulator = []
                        current_acc_size = 0
                        
                        for line in f_in:
                            line_len = len(line.encode('utf-8'))
                            text_accumulator.append(line)
                            current_acc_size += line_len
                            
                            # If accumulator is full, tokenize it
                            if current_acc_size >= TEXT_CHUNK_SIZE:
                                text_block = "".join(text_accumulator)
                                ids = tokenizer.encode(text_block, add_special_tokens=False)
                                buffer.extend(ids)
                                token_count += len(ids)
                                pbar.update(current_acc_size)
                                
                                # Reset accumulator
                                text_accumulator = []
                                current_acc_size = 0
                                
                                # Flush buffer to disk if full
                                if len(buffer) >= BUFFER_FLUSH_SIZE:
                                    f_out.write(np.array(buffer, dtype=dtype).tobytes())
                                    buffer = []

                        # Process remaining lines in this file
                        if text_accumulator:
                            text_block = "".join(text_accumulator)
                            ids = tokenizer.encode(text_block, add_special_tokens=False)
                            buffer.extend(ids)
                            token_count += len(ids)
                            pbar.update(current_acc_size)
                    
                    # Add EOS at end of file
                    buffer.append(eos_id)
                    token_count += 1
                    
                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
        
        # Final flush
        if buffer:
            f_out.write(np.array(buffer, dtype=dtype).tobytes())

    print(f"\nSuccess! Saved {OUTPUT_FILE}")
    print(f"Total Tokens: {token_count}")

if __name__ == "__main__":
    prepare()