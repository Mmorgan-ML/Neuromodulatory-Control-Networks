# ncn_architecture/transformer_layer.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.25
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
 Twitter: @Mmorgan_ML
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

# Import components from within the package
try:
    from .config import NCNConfig
    from .attention import MultiHeadAttention
    from .feedforward import PositionwiseFeedForward
    # Attempt to import custom kernel wrappers
    from .cuda_kernels import fused_modulated_add, rms_norm_cuda
except ImportError:
    # Fallback/Direct run
    from config import NCNConfig
    from attention import MultiHeadAttention
    from feedforward import PositionwiseFeedForward
    
    # If running directly or compilation failed, define simple fallbacks
    def fused_modulated_add(x, residual, gain):
        return x * gain + residual

    def rms_norm_cuda(x, weight, eps):
        # Fallback: Explicit float cast for precision, then cast back
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).type_as(x) * weight

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Faster than LayerNorm (no mean subtraction) and standard in modern LLMs (Llama).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Optimized Fused Kernel (Automatic FP32 accumulation -> FP16 output)
        return rms_norm_cuda(x, self.weight, self.eps)

class ModulatedTransformerLayer(nn.Module):
    """
    A single Transformer layer incorporating Neuromodulatory Control Network (NCN) signals.
    Now optimized with Custom CUDA Kernels and RMSNorm.
    """
    def __init__(self, config: NCNConfig):
        super().__init__()
        self.config = config

        # Self-Attention mechanism
        self.self_attn = MultiHeadAttention(config, batch_first=True)

        # Feed-Forward network
        self.feed_forward = PositionwiseFeedForward(config)

        # Normalization: Replaced LayerNorm with RMSNorm for speed
        self.norm1 = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

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
        Forward pass optimized with Fused Kernels.
        """
        mod_signals = mod_signals or {} 

        # Extract signals (Batch, 1) -> unsqueeze to (Batch, 1, 1)
        mod_gain = mod_signals.get("gain") 
        attn_precision = mod_signals.get("precision")
        ffn_gate = mod_signals.get("ffn_gate")

        if mod_gain is not None and mod_gain.dim() == 2:
             mod_gain = mod_gain.unsqueeze(-1)

        # 1. Self-Attention Block
        residual = src
        x = self.norm1(src)

        attn_output, present_key_value = self.self_attn(
            hidden_states=x,
            attention_precision=attn_precision,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attn_mask=src_mask,
            is_causal=is_causal
        )
        
        # Apply Dropout to attention output (standard Transformer logic)
        attn_output = self.dropout(attn_output)

        # Fused Modulation + Residual
        # Math: src = attn_output * gain + residual
        if mod_gain is not None:
            src = fused_modulated_add(attn_output, residual, mod_gain)
        else:
            src = residual + attn_output

        # 2. Feed-Forward Block
        residual = src
        x = self.norm2(src)

        ff_output = self.feed_forward(x, ffn_gate=ffn_gate)
        ff_output = self.dropout(ff_output)

        # Fused Modulation + Residual
        # Math: src = ff_output * gain + residual
        if mod_gain is not None:
            src = fused_modulated_add(ff_output, residual, mod_gain)
        else:
            src = residual + ff_output

        return src, present_key_value