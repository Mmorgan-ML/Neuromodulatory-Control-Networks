# ncn_architecture/config.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.25
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class NCNConfig:
    """
    Configuration class for the NCN-Modulated Language Model.

    This dataclass centralizes all hyperparameters for both the base Transformer model
    and the Neuromodulatory Control Network (NCN). It includes validation logic
    in its __post_init__ method to ensure configuration consistency.
    """
    # Standard Transformer parameters (with GPT-2 defaults)
    vocab_size: int = 50257
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: Optional[int] = None # Set in __post_init__
    dropout: float = 0.1
    max_position_embeddings: int = 2048 # Updated to 2048 for modern contexts
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_weights: bool = True # Controls tying of input/output token embedding weights

    # NCN Specific parameters
    ncn_input_dim: Optional[int] = None # Set in __post_init__
    ncn_hidden_dim: int = 128
    ncn_heads: int = 4 # Added for NCN Attention Pooling
    num_mod_signals: int = 3
    # Updated default signal names to use "precision" instead of "attention_temp"
    modulation_signal_names: List[str] = field(default_factory=lambda: ["gain", "precision", "ffn_gate"])
    ncn_activation_fn: str = "relu"
    
    # Regularization
    homeostatic_lambda: float = 0.01 # Strength of penalty for deviation from neutral state

    # Internal attribute for layer-wise output dimension
    ncn_output_dim: Optional[int] = None

    def __post_init__(self):
        """Post-initialization checks and default value setting."""
        if self.dim_feedforward is None:
            self.dim_feedforward = 4 * self.d_model
        if self.ncn_input_dim is None:
            self.ncn_input_dim = self.d_model
        if len(self.modulation_signal_names) != self.num_mod_signals:
            raise ValueError(f"Length of modulation_signal_names ({len(self.modulation_signal_names)}) must match num_mod_signals ({self.num_mod_signals}).")
        
        allowed_activations = ["relu", "gelu", "tanh", "sigmoid"]
        if self.ncn_activation_fn.lower() not in allowed_activations:
             raise ValueError(f"ncn_activation_fn must be one of {allowed_activations}")

        # Calculate total output dimension for NCN: (Num_Signals * Num_Layers)
        # This enables layer-wise modulation vectors.
        self.ncn_output_dim = self.num_mod_signals * self.num_layers