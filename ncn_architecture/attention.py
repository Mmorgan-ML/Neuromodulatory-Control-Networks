# ncn_architecture/attention.py

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
from typing import Optional, Tuple

# Assuming config is imported correctly if run as part of the package
try:
    from .config import NCNConfig
    # Import Custom CUDA Kernels
    from .cuda_kernels import kv_cache_update_cuda
except ImportError:
    # Allow running script directly for testing/inspection
    from config import NCNConfig
    # Fallback definition if cuda_kernels not found
    def kv_cache_update_cuda(cache, new_tok, positions):
        pass


class MultiHeadAttention(nn.Module):
    """
    Flash Attention compatible Multi-Head Attention mechanism with NCN modulation
    and Key-Value (KV) Caching support.
    """
    def __init__(self, config: NCNConfig, batch_first: bool = True):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            config (NCNConfig): Configuration object with model hyperparameters.
            batch_first (bool): If True, assumes input tensors have shape (batch, seq, dim).
                                Defaults to True.
        """
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})")
        
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.batch_first = batch_first

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_precision: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the multi-head attention using Flash Attention.

        Args:
            hidden_states (torch.Tensor): Input tensor (batch, seq_len, d_model).
            attention_precision (Optional[torch.Tensor]): Modulation signal for attention precision.
                 Acts as a multiplicative factor on Q (inverse temperature).
                 Expected shape from NCN: (batch, 1, 1) or broadcastable.
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached keys and values. 
                 Can be a growing tuple or a pre-allocated buffer.
            use_cache (bool): Whether to return the current keys and values.
            attn_mask (Optional[torch.Tensor]): Attention mask.
            is_causal (bool): Whether to apply a causal mask (auto-handled by SDPA if True).
            position_ids (Optional[torch.Tensor]): Positions of the current tokens. 
                 Required for Zero-Copy Kernel updates.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            - The attention output tensor (batch, seq_len, d_model).
            - The updated (key, value) cache if use_cache=True, else None.
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 1. Linear Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape for SDPA: (Batch, Head, Seq, Head_Dim)
        # Transpose to (Batch, Head, Seq, Head_Dim)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # 3. KV Caching Logic
        current_key_value = None
        
        if use_cache:
            # OPTIMIZED ZERO-COPY PATH
            # Conditions: We have a cache buffer, position_ids, and we are on CUDA (implicit in kernel check)
            if past_key_value is not None and position_ids is not None:
                past_k, past_v = past_key_value
                
                # Check if this is likely a pre-allocated buffer (Seq dim of cache > Seq dim of update)
                # And ensure inputs are contiguous for the kernel
                if past_k.size(2) > k.size(2): # past_k is (B, H, MaxSeq, D) usually? 
                    # Note: Previous PyTorch cat logic implies (Batch, Head, Seq, Dim).
                    # If using pre-allocated buffer, usually it is (Batch, MaxSeq, Head, Dim) or (Batch, Head, MaxSeq, Dim).
                    # The kernel expects (Batch, MaxSeq, Head, Dim) layout usually? 
                    # Let's check kernel definition in cuda_kernels.py:
                    # "kv_update_kernel... stride_b, stride_s, stride_h, stride_d"
                    # It handles any stride. We just need to ensure we pass the right tensors.
                    
                    # Update In-Place via Kernel
                    # We expect position_ids to align with 'k' (Batch, Seq). 
                    # If processing multiple tokens (Seq > 1), position_ids should be (Batch, Seq).
                    # The kernel implementation provided earlier takes 'positions' as (Batch,) implying 1 token per batch.
                    # If seq_len > 1 (Prompt processing), we usually don't use the specific 'update at pos' kernel 
                    # unless we loop or have a specific prompt kernel.
                    # For now, we use the kernel if seq_len == 1 (Generation Step).
                    
                    if seq_len == 1 and k.is_cuda:
                        # Invoke CUDA Kernel
                        kv_cache_update_cuda(past_k, k, position_ids[:, 0].contiguous())
                        kv_cache_update_cuda(past_v, v, position_ids[:, 0].contiguous())
                        
                        # Return the buffers as the "current" key value
                        current_key_value = (past_k, past_v)
                        
                        # For Attention computation, we must use the valid part of the cache.
                        # We calculate the max position to determine the slice.
                        # Assuming left-padding or standard generation, max_pos tells us how much history exists.
                        max_pos = position_ids.max().item()
                        
                        # Slice for SDPA: (Batch, Head, Valid_Seq, Dim)
                        # Assumes Cache Layout: (Batch, Head, MaxSeq, Dim) based on line 87 reshape
                        k = past_k[:, :, :max_pos+1, :]
                        v = past_v[:, :, :max_pos+1, :]
                        
                    else:
                        # Fallback for Prompt Processing (Seq > 1) or CPU
                        # Append new tokens to history
                        k = torch.cat((past_k, k), dim=2)
                        v = torch.cat((past_v, v), dim=2)
                        current_key_value = (k, v)
                else:
                     # past_key_value is likely just the previous step's tensor, not a full buffer
                     k = torch.cat((past_k, k), dim=2)
                     v = torch.cat((past_v, v), dim=2)
                     current_key_value = (k, v)
            
            elif past_key_value is not None:
                # Standard fallback (No position_ids provided)
                past_k, past_v = past_key_value
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
                current_key_value = (k, v)
            else:
                # No past data, initialize cache
                current_key_value = (k, v)

        # 4. NCN Modulation: Precision scaling
        # Instead of dividing logits by temp, we multiply Q by sqrt(precision) or similar.
        # Standard Attention: Softmax(QK^T / sqrt(d)) V
        # Modulated: Softmax((QK^T * precision) / sqrt(d)) V
        # This is equivalent to scaling Q by precision before the dot product.
        if attention_precision is not None:
            # attention_precision shape expected: (Batch, 1, 1) or (Batch, 1)
            # We need to broadcast to (Batch, 1, 1, 1) to multiply q which is (Batch, Head, Seq, Dim)
            if attention_precision.dim() == 1:
                beta = attention_precision.view(batch_size, 1, 1, 1)
            elif attention_precision.dim() == 2:
                beta = attention_precision.view(batch_size, 1, 1, 1) # Assuming (Batch, 1)
            elif attention_precision.dim() == 3:
                 beta = attention_precision.unsqueeze(1) # (Batch, 1, 1, 1)
            else:
                 # Fallback or error, simplified for now
                 beta = attention_precision.view(batch_size, 1, 1, 1)
            
            # --- CRITICAL STABILITY FIX ---
            # Clamp precision to prevent logits from exceeding FP16 range.
            beta = beta.clamp(max=4.0)

            # Apply modulation
            q = q * beta

        # 5. Flash Attention (Scaled Dot Product Attention)
        # is_causal=True handles the triangular mask efficiently
        # If attn_mask is provided (e.g. padding mask), pass it. 
        # Note: SDPA merges attn_mask and is_causal logic if needed, but explicit mask overrides causal usually.
        dropout_p = 0.0 if not self.training else 0.1 # Hardcoded dropout fallback or use config
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal if attn_mask is None else False
        )

        # 6. Combine heads and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, current_key_value