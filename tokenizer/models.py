# tokenizer/models.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.12
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
"""

from typing import List, Dict, Tuple, Optional

class Model:
    """Base class for sub-word tokenization models (Optional Interface)."""
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a single pre-tokenized word/string into sub-words.

        Args:
            text (str): A single word or string unit produced by the PreTokenizer.

        Returns:
            List[str]: A list of sub-word tokens.
        """
        raise NotImplementedError

    def token_to_id(self, token: str) -> Optional[int]:
        """Converts a sub-word token string to its ID."""
        raise NotImplementedError

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Converts a token ID back to its sub-word string."""
        raise NotImplementedError


class BPEModel(Model):
    """
    Implements Byte Pair Encoding (BPE) sub-word tokenization.

    Requires a vocabulary mapping tokens to IDs and a set of merge rules
    defining merge priorities.

    Args:
        vocab (Dict[str, int]): Dictionary mapping sub-word tokens to integer IDs.
        merges (Dict[Tuple[str, str], int]): Dictionary mapping merge pairs
            (tuples of strings) to their merge rank/priority (lower means higher priority).
        unk_token (str, optional): The string representation for unknown tokens.
             Defaults to "[UNK]". If None, unknown tokens might cause errors.
        end_of_word_suffix (str, optional): Suffix added to the end of characters
            if the original word is not preceded by whitespace (or if using ByteLevel).
            Helps distinguish sub-words within a word vs. whole words.
            Defaults to "</w>". Set to None or "" if not needed.
    """
    def __init__(self,
                 vocab: Dict[str, int],
                 merges: Dict[Tuple[str, str], int],
                 unk_token: Optional[str] = "[UNK]",
                 end_of_word_suffix: Optional[str] = "</w>",
                 **kwargs): # Accept additional kwargs (like cache from original HF version)
        self.vocab = vocab
        self.merges = merges
        self.unk_token = unk_token
        self.unk_token_id = self.vocab.get(unk_token) if unk_token else None
        self.end_of_word_suffix = end_of_word_suffix
        # Create reverse mapping for decoding
        self._id_to_token_map = {v: k for k, v in vocab.items()}
        # Cache for tokenization results
        self.cache = {}
        # Store any extra kwargs if needed by subclasses or specific implementations
        self.kwargs = kwargs


    def token_to_id(self, token: str) -> Optional[int]:
        """Converts a sub-word token string to its ID."""
        token_id = self.vocab.get(token, self.unk_token_id)
        # No need to raise error here, just return None or unk_token_id
        return token_id

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Converts a token ID back to its sub-word string."""
        return self._id_to_token_map.get(token_id)

    def _get_pairs(self, word_units: List[str]) -> set[Tuple[str, str]]:
        """
        Generates all adjacent pairs from a list of units.

        Args:
            word_units (List[str]): List of current sub-word units for a word.

        Returns:
            set[Tuple[str, str]]: A set of adjacent pairs.
        """
        pairs = set()
        if len(word_units) < 2:
            return pairs
        prev_char = word_units[0]
        for i in range(1, len(word_units)):
            pairs.add((prev_char, word_units[i]))
            prev_char = word_units[i]
        return pairs

    def tokenize(self, word: str) -> List[str]:
        """
        Applies BPE tokenization to a single pre-tokenized word.

        Args:
            word (str): The word to tokenize (output from PreTokenizer).

        Returns:
            List[str]: A list of BPE sub-word tokens.
        """
        if word in self.cache:
            return self.cache[word]

        if not word: # Handle empty string input
            return []

        word_units = list(word)

        if self.end_of_word_suffix and len(word_units) > 0:
             # Ensure suffix isn't added if word is only the suffix itself
            if "".join(word_units) != self.end_of_word_suffix:
                 word_units[-1] = word_units[-1] + self.end_of_word_suffix

        # If after adding suffix, it's still less than 2, return as is
        if len(word_units) < 2:
             self.cache[word] = word_units
             return word_units

        while True:
            pairs = self._get_pairs(word_units)

            # *** FIX: Check if pairs set is empty before calling min ***
            if not pairs:
                break

            best_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))

            if self.merges.get(best_pair, float('inf')) == float('inf'):
                break # No more merges found

            new_word_units = []
            i = 0
            while i < len(word_units):
                try:
                    # Find the first index of the first part of the best pair
                    j = word_units.index(best_pair[0], i)
                    # Add preceding units
                    new_word_units.extend(word_units[i:j])
                    # Check if the next unit forms the best pair
                    if j < len(word_units) - 1 and word_units[j+1] == best_pair[1]:
                        # Perform merge
                        new_word_units.append(best_pair[0] + best_pair[1])
                        i = j + 2 # Move index past the merged pair
                    else:
                        # If not the best pair, add the current unit and advance index
                        new_word_units.append(word_units[j])
                        i = j + 1
                except ValueError:
                    # First part of best_pair not found in remaining sequence
                    new_word_units.extend(word_units[i:])
                    break # Exit inner loop

            word_units = new_word_units

        self.cache[word] = word_units
        return word_units

    @staticmethod
    def from_files(vocab_path: str, merges_path: str, **kwargs) -> "BPEModel":
        """Loads the BPE model vocabulary and merges from files."""
        import json
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        except Exception as e:
            raise IOError(f"Could not read vocab file at {vocab_path}: {e}")

        merges = {}
        try:
            with open(merges_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                start_rank = 0
                if not first_line.startswith("#version"):
                    parts = first_line.split()
                    if len(parts) == 2:
                        merges[(parts[0], parts[1])] = start_rank
                        start_rank += 1
                    # else: handle potential malformed first line if needed

                for i, line in enumerate(f, start=start_rank):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges[(parts[0], parts[1])] = i
                    # else: handle potential empty/malformed lines

        except Exception as e:
            raise IOError(f"Could not read merges file at {merges_path}: {e}")

        return BPEModel(vocab=vocab, merges=merges, **kwargs)