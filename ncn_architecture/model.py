# ncn_architecture/model.py

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
import math
from typing import Optional, Tuple, List

# Import components from within the package
try: # Try relative import first
    from .config import NCNConfig
    from .ncn import NeuromodulatoryControlNetwork
    # Import RMSNorm from transformer_layer to ensure consistency
    from .transformer_layer import ModulatedTransformerLayer, RMSNorm
except ImportError: # Fallback for running script directly
    from config import NCNConfig
    from ncn import NeuromodulatoryControlNetwork
    from transformer_layer import ModulatedTransformerLayer, RMSNorm


class ModulatedLLM(nn.Module):
    """
    The main Language Model incorporating Neuromodulatory Control Networks (NCNs).
    """
    def __init__(self, config: NCNConfig):
        """
        Initializes the ModulatedLLM.
        """
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # --- Neuromodulatory Control Network ---
        self.ncn = NeuromodulatoryControlNetwork(config)

        # --- Transformer Layers ---
        self.transformer_layers = nn.ModuleList(
            [ModulatedTransformerLayer(config) for _ in range(config.num_layers)]
        )

        # --- Final Normalization and Output Head ---
        # Updated to RMSNorm
        self.final_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Apply custom weight initialization
        self.apply(self._init_weights)

        # Tie the weights of the token embeddings and the output projection layer
        if self.config.tie_weights:
            self.token_embeddings.weight = self.output_head.weight
            print("Weight tying enabled between token embeddings and output head.")

        # Report number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model initialized. Trainable parameters: {trainable_params/1e6:.2f} M")


    def _init_weights(self, module):
        """Applies custom weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                 with torch.no_grad():
                     module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            # Handle both LayerNorm and RMSNorm (RMSNorm has no bias)
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        """
        Forward pass of the ModulatedLLM.
        Supports Gradient Checkpointing.
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # 1. Calculate Embeddings
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2) 
            
        tok_emb = self.token_embeddings(input_ids)
        
        # Create position IDs based on past length
        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # Broadcast
        pos_emb = self.position_embeddings(position_ids)

        x = self.dropout(tok_emb + pos_emb)

        # 2. Compute Neuromodulatory Signals
        mod_signals = self.ncn(control_input=x, current_hidden_state=x)

        # 3. Compute Homeostatic Regularization Loss
        reg_loss = torch.tensor(0.0, device=device)
        for signal_name, signal_tensor in mod_signals.items():
            loss_component = torch.mean((signal_tensor - 1.0) ** 2)
            reg_loss += loss_component
        
        reg_loss = reg_loss * self.config.homeostatic_lambda

        # 4. Pass through Modulated Transformer Layers
        hidden_states = x
        new_key_values = [] if use_cache else None
        
        is_causal = True if past_key_values is None else False
        
        for i, layer in enumerate(self.transformer_layers):
            layer_signals = {
                k: v[:, :, i, :] for k, v in mod_signals.items()
            }

            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Checkpointing logic handled via config flag in train.py (via model config)
            if self.config.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                hidden_states, layer_present = checkpoint(
                    layer,
                    hidden_states,
                    layer_signals,
                    attention_mask,
                    layer_past,
                    False, 
                    is_causal,
                    use_reentrant=False 
                )
            else:
                hidden_states, layer_present = layer(
                    src=hidden_states,
                    mod_signals=layer_signals,
                    src_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    is_causal=is_causal
                )
            
            if use_cache:
                new_key_values.append(layer_present)

        x = hidden_states

        # 6. Final Normalization and Output Head
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits, new_key_values, reg_loss