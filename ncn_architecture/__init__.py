# ncn_architecture/__init__.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.25
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
"""

"""
Neuromodulatory Control Network (NCN) Architecture Package.

This package contains the PyTorch modules for implementing an LLM
augmented with Neuromodulatory Control Networks, based on the
proposal by Michael Morgan (2025).
"""

# Expose key classes at the package level for easier import
from .config import NCNConfig
from .model import ModulatedLLM
from .ncn import NeuromodulatoryControlNetwork
from .transformer_layer import ModulatedTransformerLayer
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


# Define the public API of the package
__all__ = [
    "NCNConfig",
    "ModulatedLLM",
    "NeuromodulatoryControlNetwork",
    "ModulatedTransformerLayer",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
]