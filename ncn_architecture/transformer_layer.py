# ncn_architecture/transformer_layer.py

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
from typing import Optional, Dict, Tuple

# Import components from within the package
from .config import NCNConfig
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward

class ModulatedTransformerLayer(nn.Module):
    """
    A single Transformer layer incorporating Neuromodulatory Control Network (NCN) signals.

    Implements Multi-Head Self-Attention and Feed-Forward blocks with Layer Normalization,
    allowing modulation signals to dynamically alter attention precision, FFN gating,
    and overall layer gains. Uses pre-LayerNorm structure.
    """
    def __init__(self, config: NCNConfig):
        """
        Initializes the ModulatedTransformerLayer.

        Args:
            config (NCNConfig): Configuration object with model hyperparameters.
        """
        super().__init__()
        self.config = config

        # Self-Attention mechanism (Updated for Flash Attention & Caching)
        self.self_attn = MultiHeadAttention(config, batch_first=True)

        # Feed-Forward network
        self.feed_forward = PositionwiseFeedForward(config)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

        # Store expected signal names for clarity (optional)
        self.expected_signals = config.modulation_signal_names

    def forward(
        self,
        src: torch.Tensor,
        mod_signals: Optional[Dict[str, torch.Tensor]] = None,
        src_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the Modulated Transformer Layer.

        Args:
            src (torch.Tensor): Input sequence tensor (batch, seq_len, d_model).
            mod_signals (Optional[Dict[str, torch.Tensor]]): Dictionary containing modulation
                signals specific to this layer. Keys match config.modulation_signal_names.
                Values are tensors (batch, 1). Defaults to None.
            src_mask (Optional[torch.Tensor]): Mask for the self-attention layer.
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached keys/values for this layer.
            use_cache (bool): Whether to return the updated cache.
            is_causal (bool): Whether this is a causal (autoregressive) step.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 
            - Output tensor (batch, seq_len, d_model).
            - Updated key/value cache if use_cache is True.
        """
        # --- Retrieve Modulation Signals ---
        mod_signals = mod_signals or {} 

        # Extract signals. Note "precision" replaces "attention_temp".
        # Inputs are expected to be (Batch, 1) from the NCN layer slicing.
        mod_gain = mod_signals.get("gain") 
        attn_precision = mod_signals.get("precision")
        ffn_gate = mod_signals.get("ffn_gate")

        # Ensure Gain broadcasts: (Batch, 1) -> (Batch, 1, 1)
        if mod_gain is not None and mod_gain.dim() == 2:
             mod_gain = mod_gain.unsqueeze(-1)

        # --- Pre-Normalization Structure ---

        # 1. Self-Attention Block (Norm -> Attention -> Dropout -> Residual)
        residual = src
        x = self.norm1(src)

        # Pass signals and cache to updated MHA
        attn_output, present_key_value = self.self_attn(
            hidden_states=x,
            attention_precision=attn_precision,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attn_mask=src_mask,
            is_causal=is_causal
        )

        # Apply Gain modulation to attention output
        if mod_gain is not None:
            attn_output = attn_output * mod_gain

        # Add residual connection
        src = residual + self.dropout(attn_output)

        # 2. Feed-Forward Block (Norm -> FFN -> Dropout -> Residual)
        residual = src
        x = self.norm2(src)

        # Pass FFN gating modulation signal to FFN
        ff_output = self.feed_forward(x, ffn_gate=ffn_gate)

        # Apply Gain modulation to FFN output
        if mod_gain is not None:
            ff_output = ff_output * mod_gain

        # Add residual connection
        src = residual + self.dropout(ff_output)

        return src, present_key_value