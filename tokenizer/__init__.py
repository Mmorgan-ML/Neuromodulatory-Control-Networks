# tokenizer/__init__.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.12
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
 Twitter: @Mmorgan_ML
"""

"""
Reusable Tokenizer Package.

Provides the Tokenizer class and its underlying components for text
tokenization, including normalization, pre-tokenization, sub-word modeling (BPE),
and post-processing.
"""

# Import the main tokenizer class to make it available directly
# e.g., from echonet_tokenizer import Tokenizer
from .tokenizer import Tokenizer

# Define the public API of the package
__all__ = [
    "Tokenizer", 
    # You could optionally expose other core components here if desired, e.g.:
    # "BPEModel",
    # "TemplateProcessor",
    # "WhitespaceSplit",
    # "ByteLevel",
    # "Lowercase",
]

# No package-level initialization code needed for this basic setup.