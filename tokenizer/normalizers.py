# tokenizer/normalizers.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.12
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
"""

import unicodedata
from typing import List, Type

class Normalizer:
    """Base class for Normalizers (Optional Interface)."""
    def normalize_str(self, text: str) -> str:
        raise NotImplementedError

class Lowercase(Normalizer):
    """Converts the input string to lowercase."""
    def normalize_str(self, text: str) -> str:
        """
        Converts the input string to lowercase.

        Args:
            text (str): The input string.

        Returns:
            str: The lowercased string.
        """
        return text.lower()

class NFC(Normalizer):
    """Applies Unicode NFC normalization to the input string."""
    def normalize_str(self, text: str) -> str:
        """
        Applies Unicode NFC normalization.

        Args:
            text (str): The input string.

        Returns:
            str: The NFC normalized string.
        """
        return unicodedata.normalize('NFC', text)

class NFKC(Normalizer):
    """Applies Unicode NFKC normalization to the input string."""
    def normalize_str(self, text: str) -> str:
        """
        Applies Unicode NFKC normalization.

        Args:
            text (str): The input string.

        Returns:
            str: The NFKC normalized string.
        """
        return unicodedata.normalize('NFKC', text)

class NFD(Normalizer):
    """Applies Unicode NFD normalization to the input string."""
    def normalize_str(self, text: str) -> str:
        """
        Applies Unicode NFD normalization.

        Args:
            text (str): The input string.

        Returns:
            str: The NFD normalized string.
        """
        return unicodedata.normalize('NFD', text)

class NFKD(Normalizer):
    """Applies Unicode NFKD normalization to the input string."""
    def normalize_str(self, text: str) -> str:
        """
        Applies Unicode NFKD normalization.

        Args:
            text (str): The input string.

        Returns:
            str: The NFKD normalized string.
        """
        return unicodedata.normalize('NFKD', text)


class StripAccents(Normalizer):
    """Removes accents/diacritics from characters in the input string."""
    def normalize_str(self, text: str) -> str:
        """
        Strips accents from the input string.
        This typically involves NFD normalization followed by filtering non-spacing marks.

        Args:
            text (str): The input string.

        Returns:
            str: The string with accents removed.
        """
        # Normalize to NFD (decompose characters and accents)
        normalized_text = unicodedata.normalize('NFD', text)
        # Filter out non-spacing marks (accents)
        output = "".join(c for c in normalized_text if unicodedata.category(c) != 'Mn')
        # Re-normalize to NFC to potentially recompose characters if needed
        # (though after stripping, recomposition might be less relevant,
        # keeping it for consistency with some libraries)
        return unicodedata.normalize('NFC', output)

class Sequence(Normalizer):
    """
    Applies a sequence of normalizers in the order they are given.

    Args:
        normalizers (List[Normalizer]): A list of normalizer objects to apply.
    """
    def __init__(self, normalizers: List[Normalizer]):
        if not isinstance(normalizers, list) or not all(isinstance(n, Normalizer) for n in normalizers):
             raise TypeError("Expected a list of Normalizer instances.")
        self.normalizers = normalizers

    def normalize_str(self, text: str) -> str:
        """
        Applies each normalizer in the sequence to the text.

        Args:
            text (str): The input string.

        Returns:
            str: The normalized string after applying all normalizers.
        """
        for normalizer in self.normalizers:
            text = normalizer.normalize_str(text)
        return text

# Example of how you might combine normalizers (used later in the main Tokenizer class)
# default_normalizer = Sequence([NFC(), Lowercase(), StripAccents()])