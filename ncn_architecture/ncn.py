# ncn_architecture/ncn.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.25
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import NCNConfig

# Try to import custom CUDA kernel
try:
    from .cuda_kernels import ncn_actuator_cuda
except ImportError:
    ncn_actuator_cuda = None

# Activation functions specifically for the NCN internal layers
_NCN_ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
}

class NeuromodulatoryControlNetwork(nn.Module):
    """
    Neuromodulatory Control Network (NCN).

    Updated to use Attention Pooling (Salience) and Layer-Wise Vectorized Outputs.
    Handles both 2D (generation) and 3D (training) inputs for Phasic Modulation.
    """
    def __init__(self, config: NCNConfig):
        """
        Initializes the NeuromodulatoryControlNetwork.

        Args:
            config (NCNConfig): Configuration object containing NCN parameters.
        """
        super().__init__()
        self.config = config
        self.input_dim = config.ncn_input_dim
        self.hidden_dim = config.ncn_hidden_dim
        self.num_mod_signals = config.num_mod_signals
        self.num_layers = config.num_layers
        self.signal_names = config.modulation_signal_names

        # --- 1. Salience Pooling (Attention Pooling) ---
        # Learnable Query Vector: "What patterns trigger modulation?"
        # Shape: (1, 1, input_dim)
        self.control_query = nn.Parameter(torch.randn(1, 1, self.input_dim))
        
        # Attention mechanism to pool the context
        self.pooling_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=config.ncn_heads,
            batch_first=True
        )

        # --- 2. NCN Core Architecture ---
        # Layer 1: Projection from pooled context + optional hidden state
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Layer 2: Output projection to (Num_Signals * Num_Layers)
        # This enables layer-specific control vectors.
        self.layer2 = nn.Linear(self.hidden_dim, config.ncn_output_dim)

        # --- Initialization Stability Protocol ---
        # Initialize output biases to ensure neutral identity state at t=0
        self._init_output_biases()

        # Get NCN internal activation function
        activation_name = config.ncn_activation_fn.lower()
        self.activation = _NCN_ACTIVATION_FUNCTIONS.get(activation_name)
        if self.activation is None:
             raise ValueError(f"Unsupported ncn_activation_fn: {config.ncn_activation_fn}")

        # --- Define Signal Transformations ---
        self.signal_transforms = {}
        for name in self.signal_names:
            if name == "gain":
                # Range: [0.5, 1.5] centered at 1.0
                self.signal_transforms[name] = lambda x: torch.sigmoid(x) + 0.5
            elif name == "precision": # Replaces attention_temp
                # Range: (0, +inf). 1.0 is neutral.
                # Softplus(x) + 0.01 guarantees positivity.
                self.signal_transforms[name] = lambda x: F.softplus(x) + 0.01
            elif name == "ffn_gate":
                # Range: [0, 1]
                self.signal_transforms[name] = torch.sigmoid
            else:
                 raise ValueError(f"Unknown modulation signal name '{name}' specified in config.")

    def _init_output_biases(self):
        """
        Initializes the biases of the final output layer to ensure the NCN starts
        in a neutral identity state (Mitigates Entropy Shock & Metabolic Throttling).
        """
        with torch.no_grad():
            # Reset all biases to 0.0 first (Standard)
            self.layer2.bias.zero_()
            
            # Create a view that shares memory with the actual bias tensor.
            # Shape matches the reshaping logic in forward(): (Batch, Seq, NumLayers, NumSignals)
            # The linear layer outputs flattened (NumLayers * NumSignals).
            bias_view = self.layer2.bias.view(self.num_layers, self.num_mod_signals)

            for i, name in enumerate(self.signal_names):
                if name == "precision":
                    # Target: Beta = 1.0 (Standard Attention)
                    # Beta = Softplus(b) + 0.01
                    # 1.0 = ln(1 + e^b) + 0.01 -> b ~= 0.525
                    bias_view[:, i].fill_(0.525)
                elif name == "ffn_gate":
                    # Target: Gamma ~= 0.95 (Mostly Open)
                    # Gamma = Sigmoid(b)
                    # 0.95 = 1 / (1 + e^-b) -> b ~= 2.94 -> Round to 3.0
                    bias_view[:, i].fill_(3.0)
                # "gain" naturally targets 1.0 at b=0.0 (Sigmoid(0) + 0.5 = 1.0), so no update needed.

    def forward(
        self, 
        control_input: torch.Tensor, 
        current_hidden_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes layer-wise modulation signals.

        Args:
            control_input (torch.Tensor): The context embeddings (Batch, Seq, Dim).
            current_hidden_state (Optional[torch.Tensor]): The hidden state of the current token(s).
                Can be (Batch, Dim) [Generation] or (Batch, Seq, Dim) [Training].

        Returns:
            Dict[str, torch.Tensor]: Keys are signal names. Values are tensors of shape
                                     (Batch, Seq, Num_Layers, 1) if training,
                                     or (Batch, Num_Layers, 1) if generation (Seq=1 implied).
        """
        batch_size = control_input.size(0)

        # --- Step 1: Salience Pooling (Global Context) ---
        # Expand query for the batch: (Batch, 1, Dim)
        query = self.control_query.expand(batch_size, -1, -1)
        
        # Attend to the control_input (Key/Value) using the learnable Query
        # pooled_context shape: (Batch, 1, Dim)
        pooled_context, _ = self.pooling_attn(query, control_input, control_input)
        
        combined_input = pooled_context # Keep as (Batch, 1, Dim) for easier broadcasting

        # --- Step 2: Phasic Integration (Local Context) ---
        seq_len = 1
        if current_hidden_state is not None:
            if current_hidden_state.dim() == 3:
                # Training Mode: Input is (Batch, Seq, Dim)
                seq_len = current_hidden_state.size(1)
                # Add Global (B, 1, D) + Local (B, S, D) -> Broadcasts correctly to (B, S, D)
                combined_input = combined_input + current_hidden_state
            else:
                # Generation Mode: Input is (Batch, Dim) or (Batch, 1, Dim)
                if current_hidden_state.dim() == 2:
                    current_hidden_state = current_hidden_state.unsqueeze(1) # (B, 1, D)
                combined_input = combined_input + current_hidden_state
                seq_len = combined_input.size(1)
        else:
            # If no phasic input, assume Seq=1 or broadcast later. 
            # Currently we assume pooled context is (B, 1, D)
            pass

        # --- Step 3: NCN Core Computation ---
        # combined_input is (Batch, Seq, Dim)
        hidden = self.activation(self.layer1(combined_input))
        
        # Output Shape: (Batch, Seq, Num_Signals * Num_Layers)
        mod_signals_flat = self.layer2(hidden) 

        # --- Step 4: Reshape and Transform ---
        # Reshape to (Batch, Seq, Num_Layers, Num_Signals)
        mod_signals_reshaped = mod_signals_flat.view(batch_size, seq_len, self.num_layers, self.num_mod_signals)

        mod_signals_processed = {}
        
        # High-Performance CUDA Path
        # We only use the kernel if:
        # 1. Kernel is loaded
        # 2. Data is on GPU
        # 3. We have exactly 3 signals (Kernel hardcoded for Gain/Prec/Gate triplet)
        if ncn_actuator_cuda is not None and mod_signals_flat.is_cuda and self.num_mod_signals == 3:
            # The kernel processes the triplet in a Structure-of-Arrays coalesced manner.
            # Expected Input: (..., 3) -> [raw_gain, raw_prec, raw_gate]
            # Returns: (..., ) gain, precision, gate
            
            g, p, f = ncn_actuator_cuda(mod_signals_reshaped)
            
            # The rest of the architecture expects (..., 1) for broadcasting
            mod_signals_processed["gain"] = g.unsqueeze(-1)
            mod_signals_processed["precision"] = p.unsqueeze(-1)
            mod_signals_processed["ffn_gate"] = f.unsqueeze(-1)
            
        else:
            # Fallback (CPU or custom signal config)
            for i, name in enumerate(self.signal_names):
                # Slice specific signal across all layers: (Batch, Seq, Num_Layers, 1)
                raw_signal_slice = mod_signals_reshaped[:, :, :, i].unsqueeze(-1)
                
                transform_func = self.signal_transforms.get(name)
                if transform_func:
                    processed_signal = transform_func(raw_signal_slice)
                    mod_signals_processed[name] = processed_signal

        return mod_signals_processed