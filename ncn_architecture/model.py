# ncn_architecture/model.py

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
import math
from typing import Optional, Tuple, List

# Import components from within the package
try: # Try relative import first
    from .config import NCNConfig
    from .ncn import NeuromodulatoryControlNetwork
    from .transformer_layer import ModulatedTransformerLayer
except ImportError: # Fallback for running script directly
    from config import NCNConfig
    from ncn import NeuromodulatoryControlNetwork
    from transformer_layer import ModulatedTransformerLayer


class ModulatedLLM(nn.Module):
    """
    The main Language Model incorporating Neuromodulatory Control Networks (NCNs).

    This model uses a standard Transformer architecture (GPT-style) where each layer's
    behavior can be dynamically modulated by signals computed by a parallel NCN.
    
    Updated for:
    1. Layer-Wise Modulation vectors.
    2. KV Caching (Autoregressive generation).
    3. Homeostatic Regularization.
    """
    def __init__(self, config: NCNConfig):
        """
        Initializes the ModulatedLLM.

        Args:
            config (NCNConfig): Configuration object with model and NCN hyperparameters.
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

        # --- Final Layer Normalization and Output Head ---
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        """
        Forward pass of the ModulatedLLM.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length).
            attention_mask: Mask to avoid attending to padding tokens (batch, seq).
            past_key_values: List of tuples containing cached keys and values for each layer.
            use_cache: Boolean, whether to return KV cache.

        Returns:
            Tuple containing:
            - logits: (batch, seq, vocab_size)
            - new_key_values: List of updated caches
            - reg_loss: Scalar tensor representing homeostatic regularization loss
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # 1. Calculate Embeddings
        # If using cache (decoding), we typically only pass the last token, so position needs offset
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2) # (B, Head, Seq, Dim) -> size(2) is seq
            
        tok_emb = self.token_embeddings(input_ids)
        
        # Create position IDs based on past length
        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # Broadcast
        pos_emb = self.position_embeddings(position_ids)

        x = self.dropout(tok_emb + pos_emb)

        # 2. Compute Neuromodulatory Signals
        # NCN now uses internal Attention Pooling and handles sequence broadcasting internally.
        # Returns: Dict {signal_name: (Batch, Seq, NumLayers, 1)}
        mod_signals = self.ncn(control_input=x, current_hidden_state=x)

        # 3. Compute Homeostatic Regularization Loss
        # Penalty = lambda * sum((signal - 1.0)^2)
        reg_loss = torch.tensor(0.0, device=device)
        for signal_name, signal_tensor in mod_signals.items():
            # Mean squared error from neutral state (1.0)
            loss_component = torch.mean((signal_tensor - 1.0) ** 2)
            reg_loss += loss_component
        
        reg_loss = reg_loss * self.config.homeostatic_lambda

        # 4. Pass through Modulated Transformer Layers
        hidden_states = x
        new_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.transformer_layers):
            # Extract signals specific to this layer.
            # mod_signals[k] is (Batch, Seq, NumLayers, 1) -> We select layer 'i' from dim 2.
            # Result: (Batch, Seq, 1).
            # The transformer components (MHA, FFN) are broadcasting-compatible with (B, S, 1).
            layer_signals = {
                k: v[:, :, i, :] for k, v in mod_signals.items()
            }

            # Get past cache for this layer
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, layer_present = layer(
                src=hidden_states,
                mod_signals=layer_signals,
                src_mask=attention_mask, # Padding mask
                past_key_value=layer_past,
                use_cache=use_cache,
                is_causal=True if past_key_values is None else False # Use causal mask if generating from scratch
            )
            
            if use_cache:
                new_key_values.append(layer_present)

        x = hidden_states

        # 6. Final Normalization and Output Head
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits, new_key_values, reg_loss