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

    # 2. Load Tokenizer (Attempt Rust Acceleration)
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = None
    use_fast = False

    try:
        from transformers import PreTrainedTokenizerFast
        # Try loading as a Fast tokenizer (Rust backend)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        use_fast = True
        print(">> SUCCESS: Using HuggingFace Fast Tokenizer (Rust-backed). Expect high speeds.")
    except Exception as e:
        print(f">> WARNING: Could not load Fast Tokenizer ({e}).")
        print(">> Falling back to local tokenizer.py (Python-backed). Expect SLOW speeds.")
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
        print(f"Defaulting to GPT-2 standard EOS ID: {eos_id}")
    else:
        print(f"Using EOS Token ID: {eos_id}")

    # 4. Gather Files Recursively
    print(f"Scanning '{DATA_DIR}' and all subdirectories for .txt files...")
    files = sorted(list(DATA_DIR.rglob("*.txt")))
    
    if not files:
        print(f"No .txt files found in {DATA_DIR} or its subdirectories.")
        return

    total_size = sum(os.path.getsize(f) for f in files)
    print(f"Found {len(files)} files. Total raw text size: {total_size / 1024 / 1024:.2f} MB")

    token_count = 0
    
    # 5. Processing Loop
    print(f"Writing to {OUTPUT_FILE}...")
    
    # We create a buffer to avoid writing to disk for every single file (speeds up HDD IO)
    buffer = []
    BUFFER_SIZE = 100_000 # Flush every 100k tokens
    
    with open(OUTPUT_FILE, "wb") as f_out:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Tokenizing") as pbar:
            for file_path in files:
                try:
                    pbar.set_description(f"Processing {file_path.name[:20]}")
                    
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f_in:
                        text = f_in.read()
                    
                    if not text.strip():
                        pbar.update(len(text.encode('utf-8')))
                        continue

                    # Encode
                    if use_fast:
                        # transformers library returns a list directly usually
                        encoded = tokenizer.encode(text, add_special_tokens=False)
                        ids = encoded
                    else:
                        # Local tokenizer
                        encoded = tokenizer.encode(text, add_special_tokens=False)
                        ids = encoded['input_ids'] if isinstance(encoded, dict) else encoded

                    # Append EOS
                    ids.append(eos_id)
                    
                    # Add to buffer
                    buffer.extend(ids)
                    token_count += len(ids)
                    
                    # Flush buffer if full
                    if len(buffer) >= BUFFER_SIZE:
                        arr = np.array(buffer, dtype=dtype)
                        f_out.write(arr.tobytes())
                        buffer = [] # Clear buffer

                    # Update Progress Bar
                    pbar.update(len(text.encode('utf-8')))

                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
        
        # Flush remaining buffer
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            f_out.write(arr.tobytes())

    print(f"\nSuccess! Saved {OUTPUT_FILE}")
    print(f"Total Tokens: {token_count}")
    print(f"File Size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    prepare()