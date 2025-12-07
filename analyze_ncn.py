# ncn_project/analyze_ncn.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from ncn_architecture import ModulatedLLM, NCNConfig
from tokenizer import Tokenizer

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints_run1/checkpoint_best.pt"
TOKENIZER_PATH = "gpt2_tokenizer_files"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    config = checkpoint['config']
    # Ensure config matches inference mode
    config.gradient_checkpointing = False 
    
    model = ModulatedLLM(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # HOT-FIX: Restore the HardTanh activation used during training
    import torch.nn as nn
    if hasattr(model.ncn, 'activation'):
        model.ncn.activation = nn.Hardtanh(min_val=0.0, max_val=3.0)
        
    print("Model loaded successfully.")
    return model

def get_input_ids(tokenizer, text):
    """Helper to handle tokenizer output variance (dict vs list)"""
    encoded = tokenizer.encode(text)
    if isinstance(encoded, dict):
        return encoded['input_ids']
    return encoded

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    ids = get_input_ids(tokenizer, prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    # Simple generation loop
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _, _ = model(input_ids=input_tensor)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            print(".", end="", flush=True)
            
            if next_token.item() == tokenizer.pad_token_id: # EOS check
                break
                
    print(" Done.")
    # Decode only the generated part
    full_sequence = input_tensor[0].tolist()
    decoded = tokenizer.decode(full_sequence)
    print(f"\n--- OUTPUT ---\n{decoded}\n--------------\n")

def run_brain_scan(model, tokenizer, text):
    """
    Feeds text to the model and records NCN values (Gain/Precision) for every token.
    """
    print(f"Running Brain Scan on: '{text}'")
    ids = get_input_ids(tokenizer, text)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    # We need to capture the NCN output. 
    captured_signals = []

    def hook_fn(module, input, output):
        # output is a dict: {'gain': ..., 'precision': ..., 'ffn_gate': ...}
        snapshot = {}
        for k, v in output.items():
            # Detach, move to CPU, remove batch dim
            snapshot[k] = v.detach().cpu().squeeze(0).numpy() 
        captured_signals.append(snapshot)

    # Register hook on the NCN module
    handle = model.ncn.register_forward_hook(hook_fn)
    
    # Run Forward Pass
    with torch.no_grad():
        model(input_ids=input_tensor)
    
    handle.remove() # Clean up
    
    # Process Data
    if not captured_signals:
        print("Error: No signals captured. Check model architecture names.")
        return

    data = captured_signals[0] # Take the first forward pass
    tokens = [tokenizer.decode([i]) for i in ids]
    
    # Setup Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. Gain Heatmap
    # Shape: (Seq, Layers, 1) -> Squeeze -> (Seq, Layers) -> Transpose -> (Layers, Seq)
    gain_map = data['gain'].squeeze().T
    sns.heatmap(gain_map, ax=axes[0], cmap="viridis", cbar=True, yticklabels=True)
    axes[0].set_title("Layer Gain (Signal Strength)")
    axes[0].set_ylabel("Layer Depth")
    
    # 2. Precision Heatmap
    prec_map = data['precision'].squeeze().T
    sns.heatmap(prec_map, ax=axes[1], cmap="magma", cbar=True, yticklabels=True)
    axes[1].set_title("Attention Precision (Entropy Control)")
    axes[1].set_ylabel("Layer Depth")

    # 3. Gate Heatmap
    gate_map = data['ffn_gate'].squeeze().T
    sns.heatmap(gate_map, ax=axes[2], cmap="coolwarm", cbar=True, yticklabels=True)
    axes[2].set_title("FFN Gating (Metabolic Activation)")
    axes[2].set_ylabel("Layer Depth")
    
    # X-Axis Formatting
    axes[2].set_xticks(np.arange(len(tokens)) + 0.5)
    axes[2].set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    output_file = "brain_scan_trace.png"
    plt.savefig(output_file, dpi=300)
    print(f"Saved '{output_file}'. Check your folder!")
    plt.show()

if __name__ == "__main__":
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_PATH)
    # Patch pad_token if missing
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 50256
        
    model = load_model()
    
    # 1. Test Generation
    print("\n=== GENERATION TEST ===")
    generate_text(model, tokenizer, "Once upon a time, there was a tiny dragon who loved to")
    
    # 2. Run the Scientific Trace
    print("\n=== MECHANISTIC ANALYSIS ===")
    test_sentence = "Timmy wanted an apple. He looked in the box, but it was empty."
    run_brain_scan(model, tokenizer, test_sentence)