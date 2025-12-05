# ncn_architecture/feedforward.py

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
import torch.nn.functional as F
from typing import Optional

# Assuming config is imported correctly if run as part of the package
try:
    from .config import NCNConfig
except ImportError:
    # Allow running script directly for testing/inspection
    from config import NCNConfig


_ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
}

class PositionwiseFeedForward(nn.Module):
    """
    Standard Positionwise Feed-Forward Network (FFN) for Transformers,
    with optional gating/scaling modulation applied to the output.
    """
    def __init__(self, config: NCNConfig):
        """
        Initializes the PositionwiseFeedForward layer.

        Args:
            config (NCNConfig): Configuration object with model hyperparameters.
        """
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)

        # Using GELU as the activation, which is standard for GPT-style models.
        # This is now sourced from the dictionary for better code organization.
        try:
            self.activation = _ACTIVATION_FUNCTIONS["gelu"]
        except KeyError:
            raise RuntimeError("Default 'gelu' activation not found in _ACTIVATION_FUNCTIONS.")


    def forward(
        self,
        x: torch.Tensor,
        ffn_gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the FFN.

        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, d_model).
            ffn_gate (Optional[torch.Tensor]): Modulation signal to scale the output.
                 Expected shape from NCN: (batch, 1) or (batch, 1, 1). Defaults to 1.0 if None.

        Returns:
            torch.Tensor: Output tensor (batch, seq_len, d_model).
        """
        # Apply first linear layer, activation, and dropout
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply second linear layer
        x = self.linear2(x)

        # Apply Modulation: Output Gating/Scaling
        if ffn_gate is not None:
            # Ensure broadcasting: (Batch, 1, 1) for multiplication with (Batch, Seq, Dim)
            # The new model.py will likely pass (Batch, 1) slices from the layer-wise vector.
            if ffn_gate.dim() == 1:
                ffn_gate = ffn_gate.view(-1, 1, 1)
            elif ffn_gate.dim() == 2:
                ffn_gate = ffn_gate.unsqueeze(1) # (B, 1) -> (B, 1, 1)
            
            # Apply gating
            x = x * ffn_gate

        return x