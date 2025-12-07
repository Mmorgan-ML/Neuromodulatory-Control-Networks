# tokenizer/pre_tokenizers.py

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

import regex as re
from typing import List

class PreTokenizer:
    """Base class for PreTokenizers (Optional Interface)."""
    def pre_tokenize_str(self, text: str) -> List[str]:
        """
        Splits the input string into preliminary tokens (words/units).

        Args:
            text (str): The input string (usually already normalized).

        Returns:
            List[str]: A list of string splits.
        """
        raise NotImplementedError

class WhitespaceSplit(PreTokenizer):
    """Splits the input string based on whitespace characters."""
    def pre_tokenize_str(self, text: str) -> List[str]:
        """
        Splits the text by whitespace.

        Args:
            text (str): The input string.

        Returns:
            List[str]: A list of strings split by whitespace.
        """
        return text.split()

class WhitespacePunctuationSplit(PreTokenizer):
    """
    Splits the input string based on whitespace and punctuation.
    Punctuation marks are treated as separate tokens.
    """
    def __init__(self):
        # Regex to split by whitespace or keep common punctuation as separate tokens
        # This can be customized significantly based on desired punctuation handling
        # Example: Keep basic punctuation separate, split by whitespace
        # Consider edge cases like contractions ('s, 't, 're, etc.)
        # A common approach is inspired by GPT-2/BERT basic tokenization patterns
        # This regex tries to capture words, numbers, and individual punctuation/symbols
        # It's a simplified example; robust tokenizers use more complex regex or rules
        self._regex_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


    def pre_tokenize_str(self, text: str) -> List[str]:
        """
        Splits the text by whitespace and punctuation using regex.

        Args:
            text (str): The input string.

        Returns:
            List[str]: A list of strings (words and punctuation).
        """
        # Use re.findall to get all matching patterns, which handles spaces correctly
        # Filter out empty strings that might result from the regex
        return [token for token in self._regex_pattern.findall(text) if token and not token.isspace()]


class ByteLevel(PreTokenizer):
    """
    Treats each byte of the input string as a token.
    Maps bytes to UTF-8 characters for representation.

    Args:
        add_prefix_space (bool): Whether to add a space prefix to the first word
                                 if the input string doesn't start with one.
                                 Helps distinguish start-of-word tokens. Defaults to True.
        use_regex (bool): Whether to use regex for splitting words before byte conversion.
                          If False, uses simple whitespace split. Defaults to True (recommended).
    """
    def __init__(self, add_prefix_space: bool = True, use_regex: bool = True):
        self.add_prefix_space = add_prefix_space
        # Using a simplified regex for splitting if use_regex is True
        self._regex_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.use_regex = use_regex
        # Create the byte-to-char mapping
        self._byte_to_unicode = self._create_byte_map()

    def _create_byte_map(self) -> dict[int, str]:
        """Creates a mapping from bytes (0-255) to printable unicode characters."""
        map_dict = {}
        # Map the first 256 unicode characters, shifting printable ones
        # Goal: ensure every byte 0-255 has a unique character representation
        n = 0
        for i in range(256):
            if i < 33 or i > 126: # Non-printable ASCII range (excluding space)
                map_dict[i] = chr(256 + n) # Shift to higher unicode plane
                n += 1
            else:
                map_dict[i] = chr(i) # Use the character itself
        return map_dict

    def _map_bytes_to_unicode(self, byte_list: bytes) -> str:
        """Maps a list of bytes to a string using the byte map."""
        return "".join(self._byte_to_unicode[b] for b in byte_list)

    def pre_tokenize_str(self, text: str) -> List[str]:
        """
        Splits text into words/units, then maps bytes of each unit to unicode chars.

        Args:
            text (str): The input string.

        Returns:
            List[str]: A list of strings, where each string represents an original
                       word/unit mapped byte-by-byte to unicode characters.
        """
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text # Add prefix space if needed

        # Initial split into words/units
        if self.use_regex:
             splits = [token for token in self._regex_pattern.findall(text) if token and not token.isspace()]
        else:
             splits = text.split() # Simple whitespace split

        # Map bytes of each split to the custom unicode characters
        byte_level_splits = []
        for split in splits:
            byte_representation = split.encode('utf-8')
            mapped_string = self._map_bytes_to_unicode(byte_representation)
            byte_level_splits.append(mapped_string)

        return byte_level_splits