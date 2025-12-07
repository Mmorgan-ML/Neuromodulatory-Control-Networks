# tokenizer/processors.py

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

from typing import List, Dict, Optional, Tuple, Literal

class PostProcessor:
    """Base class for PostProcessors (Optional Interface)."""
    def process(self,
                token_ids_a: List[int],
                token_ids_b: Optional[List[int]] = None,
                max_length: Optional[int] = None,
                truncation: bool = False,
                padding: bool = False,
                add_special_tokens: bool = True,
                **kwargs) -> Dict[str, List[int]]:
        """
        Processes encoded token IDs (potentially adding special tokens, truncating, padding).

        Args:
            token_ids_a (List[int]): List of token IDs for the first sequence.
            token_ids_b (Optional[List[int]], optional): List of token IDs for the
                second sequence (for paired inputs). Defaults to None.
            max_length (Optional[int], optional): Maximum sequence length. Required if
                truncation or padding is True. Defaults to None.
            truncation (bool, optional): Whether to truncate sequences to max_length.
                Defaults to False.
            padding (bool, optional): Whether to pad sequences to max_length.
                 Defaults to False.
            add_special_tokens (bool, optional): Whether to add special tokens based
                 on the processor's template. Defaults to True.

        Returns:
            Dict[str, List[int]]: A dictionary containing processed "input_ids",
                                  "token_type_ids", and "attention_mask".
        """
        raise NotImplementedError


class TemplateProcessor(PostProcessor):
    """
    Processes token IDs based on template strings for single and paired sequences,
    handling special tokens, truncation, and padding.

    Args:
        template_single (str): Template for single sequences (e.g., "[CLS] $A [SEP]").
                                Use $A as the placeholder for sequence A.
        template_pair (str): Template for sequence pairs (e.g., "[CLS] $A [SEP] $B [SEP]").
                             Use $A and $B as placeholders.
        special_tokens (Dict[str, Dict]): Dictionary mapping special token strings
            (like "[CLS]", "[SEP]") to their properties, MUST include 'id' (token ID)
            and 'type_id' (token type/segment ID).
            Example: {"[CLS]": {"id": 101, "type_id": 0}, "[SEP]": {"id": 102, "type_id": 0}}
        pad_token (str): The string representation of the padding token (e.g., "[PAD]").
        pad_token_id (int): The ID of the padding token.
        pad_token_type_id (int, optional): The token type ID for padding tokens. Defaults to 0.

    """
    def __init__(self,
                 template_single: str,
                 template_pair: str,
                 special_tokens: Dict[str, Dict],
                 pad_token: str,
                 pad_token_id: int,
                 pad_token_type_id: int = 0):

        self.template_single = template_single.split()
        self.template_pair = template_pair.split()
        self.special_tokens = special_tokens
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.pad_token_type_id = pad_token_type_id

        # Validate special tokens structure
        for token, data in special_tokens.items():
            if 'id' not in data or 'type_id' not in data:
                raise ValueError(f"special_tokens entry for '{token}' must contain 'id' and 'type_id'.")


    def _build_sequence(self,
                        template: List[str],
                        ids_a: List[int],
                        ids_b: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        """Builds input_ids and token_type_ids based on the template."""
        input_ids = []
        token_type_ids = []
        type_id_counter = 0 # Start with type 0

        for part in template:
            if part == "$A":
                input_ids.extend(ids_a)
                token_type_ids.extend([type_id_counter] * len(ids_a))
            elif part == "$B":
                if ids_b is None:
                    raise ValueError("Template contains $B but second sequence (ids_b) not provided.")
                # Increment type_id for the second sequence if template indicates segment change
                # Heuristic: Change type ID after the *first* occurrence of a special token
                # associated with type_id 0, assuming it's a separator like [SEP].
                # More robust tokenizers often explicitly define type IDs for pairs.
                # Simple approach: if B exists, use type 1.
                type_id_counter = 1 # Assume B is always type 1 if present
                input_ids.extend(ids_b)
                token_type_ids.extend([type_id_counter] * len(ids_b))
            elif part in self.special_tokens:
                token_info = self.special_tokens[part]
                input_ids.append(token_info['id'])
                # Use specified type_id, but ensure it respects current segment if needed
                # For simplicity here, just use the defined type_id.
                token_type_ids.append(token_info['type_id'])
                # Optional: could advance type_id_counter here based on token rules
            else:
                raise ValueError(f"Unknown part '{part}' in template.")

        return input_ids, token_type_ids

    def _truncate_sequences(self,
                             ids_a: List[int],
                             ids_b: Optional[List[int]],
                             num_special_tokens: int,
                             max_length: int,
                             strategy: Literal['longest_first', 'only_first', 'only_second'] = 'longest_first',
                             stride: int = 0) -> Tuple[List[int], Optional[List[int]]]:
        """Truncates sequences according to max_length and strategy."""
        # This is a simplified truncation logic. Real tokenizers handle stride etc.
        total_len = len(ids_a) + (len(ids_b) if ids_b else 0)
        num_to_remove = total_len - (max_length - num_special_tokens)

        if num_to_remove <= 0:
            return ids_a, ids_b

        if ids_b is None or strategy == 'only_first':
            ids_a = ids_a[:-num_to_remove] # Truncate from end
        elif strategy == 'only_second':
            ids_b = ids_b[:-num_to_remove] # Truncate from end
        elif strategy == 'longest_first':
             # Truncate from the longest sequence first
            for _ in range(num_to_remove):
                if len(ids_a) > len(ids_b):
                    ids_a.pop()
                else:
                    ids_b.pop()
        else:
             raise ValueError(f"Invalid truncation strategy: {strategy}")

        return ids_a, ids_b

    def process(self,
                token_ids_a: List[int],
                token_ids_b: Optional[List[int]] = None,
                max_length: Optional[int] = None,
                truncation: bool = False,
                padding: bool = False,
                add_special_tokens: bool = True,
                padding_side: Literal['right', 'left'] = 'right',
                truncation_strategy: Literal['longest_first', 'only_first', 'only_second'] = 'longest_first',
                **kwargs) -> Dict[str, List[int]]:
        """
        Processes encoded token IDs.

        Args:
             (See PostProcessor base class and __init__ for others)
             padding_side (Literal['right', 'left'], optional): Side to pad on. Defaults to 'right'.
             truncation_strategy (Literal[...], optional): Truncation strategy for pairs.
                  Defaults to 'longest_first'.

        Returns:
            Dict[str, List[int]]: Processed encodings.
        """
        if padding or truncation:
            if max_length is None:
                raise ValueError("max_length must be specified if truncation or padding is True.")

        ids_a = token_ids_a[:] # Work with copies
        ids_b = token_ids_b[:] if token_ids_b is not None else None

        # Select template based on whether it's a pair
        template = self.template_pair if ids_b is not None else self.template_single

        # 1. Handle Special Tokens Addition
        if add_special_tokens:
             num_special = sum(1 for part in template if part in self.special_tokens)

             # 2. Truncation (if enabled) - applied *before* adding special tokens
             if truncation:
                 ids_a, ids_b = self._truncate_sequences(
                     ids_a, ids_b, num_special, max_length, truncation_strategy
                 )

             # Build sequence with special tokens
             input_ids, token_type_ids = self._build_sequence(template, ids_a, ids_b)

        else: # No special tokens
             if ids_b is not None:
                 input_ids = ids_a + ids_b
                 token_type_ids = ([0] * len(ids_a)) + ([1] * len(ids_b)) # Basic type IDs
             else:
                 input_ids = ids_a
                 token_type_ids = [0] * len(ids_a)

             # Truncate combined sequence if needed (without accounting for special tokens)
             if truncation and len(input_ids) > max_length:
                 input_ids = input_ids[:max_length]
                 token_type_ids = token_type_ids[:max_length]


        # 3. Padding (if enabled)
        attention_mask = [1] * len(input_ids)
        if padding and len(input_ids) < max_length:
            num_pad = max_length - len(input_ids)
            pad_ids = [self.pad_token_id] * num_pad
            pad_type_ids = [self.pad_token_type_id] * num_pad
            pad_attn_mask = [0] * num_pad # Mask padding tokens

            if padding_side == 'right':
                input_ids.extend(pad_ids)
                token_type_ids.extend(pad_type_ids)
                attention_mask.extend(pad_attn_mask)
            elif padding_side == 'left':
                input_ids = pad_ids + input_ids
                token_type_ids = pad_type_ids + token_type_ids
                attention_mask = pad_attn_mask + attention_mask
            else:
                raise ValueError(f"Invalid padding_side: {padding_side}")

        # 4. Final Check (Optional): If padding was not enabled but truncation was,
        # ensure length doesn't exceed max_length (should be handled by truncation, but as safeguard)
        if not padding and truncation and len(input_ids) > max_length:
             input_ids = input_ids[:max_length]
             token_type_ids = token_type_ids[:max_length]
             attention_mask = attention_mask[:max_length]


        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }