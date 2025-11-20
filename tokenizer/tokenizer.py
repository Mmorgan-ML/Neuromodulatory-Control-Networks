# tokenizer/tokenizer.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.04.12
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
"""

import json
import os
# Added necessary imports here:
from typing import List, Optional, Dict, Union, Tuple, Literal
import torch # For return_tensors='pt'
import numpy as np # For return_tensors='np'
import re # For regular expression operations in decode cleanup

# Import components using relative paths
from .normalizers import Normalizer, Sequence as NormalizerSequence # Alias Sequence
from .pre_tokenizers import PreTokenizer
from .pre_tokenizers import ByteLevel # Import ByteLevel for isinstance check in decode
from .models import Model, BPEModel # Assuming BPE for now
from .processors import PostProcessor, TemplateProcessor

class Tokenizer:
    """
    Orchestrates the tokenization pipeline using configured components.

    Combines normalization, pre-tokenization, sub-word modeling (e.g., BPE),
    and post-processing steps.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping tokens to IDs.
        merges (Optional[Dict[Tuple[str, str], int]], optional): Merge rules for BPE.
            Required if model is BPEModel. Defaults to None.
        unk_token (str, optional): Unknown token string. Defaults to "[UNK]".
        model_type (str, optional): Type of the core sub-word model.
            Defaults to "BPE".
        normalizer (Optional[Normalizer], optional): Normalizer instance.
            Defaults to None (no normalization).
        pre_tokenizer (Optional[PreTokenizer], optional): Pre-tokenizer instance.
            Defaults to None (implies model handles raw normalized string).
        post_processor (Optional[PostProcessor], optional): Post-processor instance.
             Defaults to None (raw token IDs are returned).
        **kwargs: Additional kwargs passed to the underlying model (e.g., end_of_word_suffix for BPE).
    """
    def __init__(
        self,
        vocab: Dict[str, int],
        merges: Optional[Dict[Tuple[str, str], int]] = None,
        unk_token: str = "[UNK]",
        model_type: str = "BPE",
        normalizer: Optional[Normalizer] = None,
        pre_tokenizer: Optional[PreTokenizer] = None,
        post_processor: Optional[PostProcessor] = None,
        **kwargs
    ):
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.post_processor = post_processor
        self.unk_token = unk_token
        self._vocab_str_to_int = vocab
        self._vocab_int_to_str = {i: s for s, i in vocab.items()}
        self.model_type = model_type

        # Instantiate the core model based on type
        if model_type.upper() == "BPE":
            if merges is None:
                raise ValueError("Merges must be provided for BPEModel.")
            # Pass kwargs like end_of_word_suffix to the BPEModel
            self.model = BPEModel(vocab, merges, unk_token=unk_token, **kwargs)
        # Add elif blocks here for other model types like WordPiece, Unigram, etc.
        # elif model_type.upper() == "WORDPIECE":
        #     self.model = WordPieceModel(vocab, unk_token=unk_token, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Store common token IDs for convenience
        self.unk_token_id = self.model.token_to_id(unk_token) if unk_token else None

        if post_processor and hasattr(post_processor, 'pad_token_id'):
            self.pad_token_id = post_processor.pad_token_id
        else:
             # Try to get pad_token_id from vocab if post_processor doesn't define it
             # Assume a default pad token string if none specified in config
             pad_token_str = "[PAD]"
             if post_processor and hasattr(post_processor, 'pad_token'):
                 pad_token_str = post_processor.pad_token
             self.pad_token_id = self.model.token_to_id(pad_token_str)
             if self.pad_token_id is None:
                 # Handle case where default PAD token isn't even in the vocab
                 # print(f"Warning: Default PAD token '{pad_token_str}' not found in vocab.")
                 self.pad_token_id = -1 # Or raise an error?

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self._vocab_str_to_int)

    def _normalize(self, text: str) -> str:
        """Applies normalization."""
        return self.normalizer.normalize_str(text) if self.normalizer else text

    def _pre_tokenize(self, normalized_text: str) -> List[str]:
        """Applies pre-tokenization."""
        return self.pre_tokenizer.pre_tokenize_str(normalized_text) if self.pre_tokenizer else [normalized_text]

    def _tokenize_sequence(self, sequence: str) -> List[str]:
        """Tokenizes a single normalized sequence."""
        normalized = self._normalize(sequence)
        pre_tokenized = self._pre_tokenize(normalized)
        subword_tokens = []
        for word in pre_tokenized:
            subword_tokens.extend(self.model.tokenize(word))
        return subword_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token or list of tokens to their corresponding IDs."""
        if isinstance(tokens, str):
            token_id = self.model.token_to_id(tokens)
            if token_id is None: # Should only happen if unk_token is None and token is OOV
                 raise ValueError(f"Token '{tokens}' not found in vocab and no unk_token defined.")
            return token_id
        ids = []
        for token in tokens:
             token_id = self.model.token_to_id(token)
             if token_id is None:
                 raise ValueError(f"Token '{token}' not found in vocab and no unk_token defined.")
             ids.append(token_id)
        return ids

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """Converts an ID or list of IDs to their token strings."""
        if isinstance(ids, int):
            token = self.model.id_to_token(ids)
            # Check if it's a special token *before* potentially replacing with unk_token
            is_special = self._is_special_token(token)
            if skip_special_tokens and is_special:
                return ""
            return token if token is not None else self.unk_token # Use unk_token if ID is invalid

        tokens = [self.model.id_to_token(id_) for id_ in ids]
        processed_tokens = []
        for token in tokens:
            is_special = self._is_special_token(token)
            if skip_special_tokens and is_special:
                continue
            processed_tokens.append(token if token is not None else self.unk_token)
        return processed_tokens

    def _is_special_token(self, token: Optional[str]) -> bool:
        """Checks if a token is considered a special token by the post-processor."""
        if token is None: return False
        # Check against known special tokens defined in the post_processor
        if self.post_processor and hasattr(self.post_processor, 'special_tokens'):
            # Check if the token string itself is a key in the special_tokens dict
            if token in self.post_processor.special_tokens:
                return True
        # Check against known UNK token
        if token == self.unk_token:
            return True
        # Check against known PAD token (get string representation)
        pad_token_str = self.model.id_to_token(self.pad_token_id) if self.pad_token_id is not None and self.pad_token_id != -1 else None
        if token == pad_token_str:
            return True
        # Add other common patterns if needed, e.g., specific control codes
        return False


    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, Literal['longest', 'max_length']] = False, # Allow string options
        truncation: Union[bool, str, Literal['longest_first', 'only_first', 'only_second']] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Literal["pt", "np"]] = None, # Use Literal here
        **kwargs # Passed to post_processor
    ) -> Dict:
        """
        Encodes a single text or pair of texts into model inputs.

        Args:
            text (str): The first sequence to encode.
            text_pair (Optional[str], optional): The second sequence to encode (for pairs).
                 Defaults to None.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.
            padding (Union[bool, str, Literal['longest', 'max_length']], optional): Whether/how to pad.
                 Defaults to False.
            truncation (Union[bool, str, Literal[...]], optional): Whether/how to truncate.
                 Defaults to False.
            max_length (Optional[int], optional): Max length for padding/truncation. Defaults to None.
            return_tensors (Optional[Literal["pt", "np"]], optional): If set, returns tensors
                 of the specified type (PyTorch or NumPy). Defaults to None (returns lists).
            **kwargs: Additional arguments for the post-processor's process method
                      (e.g., padding_side, truncation_strategy).

        Returns:
            Dict: Dictionary containing input_ids, token_type_ids, attention_mask.
                  Values are lists of ints by default, or tensors if return_tensors is set.
        """
        # 1. Tokenize text into sub-word strings
        tokens_a = self._tokenize_sequence(text)

        tokens_b = None
        if text_pair is not None:
            tokens_b = self._tokenize_sequence(text_pair)

        # 2. Convert sub-word strings to IDs
        ids_a = self.convert_tokens_to_ids(tokens_a)
        ids_b = self.convert_tokens_to_ids(tokens_b) if tokens_b is not None else None

        # 3. Post-process
        if self.post_processor:
            # Determine padding/truncation based on input args
            do_padding = bool(padding) or padding == "max_length" or padding == "longest"
            do_truncation = bool(truncation) or isinstance(truncation, str) # Check if strategy is specified
            trunc_strat = truncation if isinstance(truncation, str) and truncation in ['longest_first', 'only_first', 'only_second'] else "longest_first" # Default strategy


            output = self.post_processor.process(
                token_ids_a=ids_a,
                token_ids_b=ids_b,
                max_length=max_length,
                truncation=do_truncation,
                padding=do_padding,
                add_special_tokens=add_special_tokens,
                truncation_strategy=trunc_strat,
                **kwargs # Pass other args like padding_side
            )
        else: # Raw IDs if no processor
             # Basic handling if no post-processor exists
             if ids_b is not None:
                 input_ids = ids_a + ids_b
                 # Assign basic type IDs (0 for first seq, 1 for second)
                 token_type_ids = ([0] * len(ids_a)) + ([1] * len(ids_b))
             else:
                 input_ids = ids_a
                 token_type_ids = [0] * len(ids_a)
             attention_mask = [1] * len(input_ids) # No padding assumed

             # Manual truncation if requested (simple head truncation)
             if truncation and max_length is not None and len(input_ids) > max_length:
                 input_ids = input_ids[:max_length]
                 token_type_ids = token_type_ids[:max_length]
                 attention_mask = attention_mask[:max_length]

             # Manual padding if requested (simple right padding)
             if padding and max_length is not None and len(input_ids) < max_length:
                  pad_len = max_length - len(input_ids)
                  if self.pad_token_id is None or self.pad_token_id == -1:
                       raise ValueError("Padding requested but pad_token_id is not validly set.")
                  input_ids.extend([self.pad_token_id] * pad_len)
                  # Assume pad token type id is 0 if not specified otherwise
                  pad_type = self.post_processor.pad_token_type_id if self.post_processor and hasattr(self.post_processor, 'pad_token_type_id') else 0
                  token_type_ids.extend([pad_type] * pad_len)
                  attention_mask.extend([0] * pad_len)

             output = {
                 "input_ids": input_ids,
                 "token_type_ids": token_type_ids,
                 "attention_mask": attention_mask
             }


        # Convert to tensors if requested
        if return_tensors == "pt":
            # import torch should be at the top now
            for key in output:
                # Pad lists within the batch to the same length before converting to tensor
                # This typically happens in a data collator, but basic tensor conversion shown here
                output[key] = torch.tensor(output[key]) # Simple conversion assumes pre-padded/single item
        elif return_tensors == "np":
            # import numpy as np should be at the top now
            for key in output:
                output[key] = np.array(output[key])

        return output

    # Make the tokenizer callable
    __call__ = encode

    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor, np.ndarray], # Use np alias
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True
        ) -> str:
        """
        Decodes a sequence of token IDs back into a string.

        Args:
            token_ids (Union[List[int], torch.Tensor, np.ndarray]): List/tensor of token IDs.
            skip_special_tokens (bool, optional): Remove special tokens. Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Clean BPE/WP artifacts. Defaults to True.

        Returns:
            str: The decoded string.
        """
        # Convert tensor to list if necessary
        if hasattr(token_ids, 'tolist'): # Covers torch and numpy
             ids_list = token_ids.tolist()
        else:
             ids_list = list(token_ids) # Assume it's already list-like

        # Convert IDs to tokens, handling potential skipping internally
        tokens = self.convert_ids_to_tokens(ids_list, skip_special_tokens=skip_special_tokens)

        # --- Decoding Cleanup ---
        if not tokens:
            return ""

        # 1. Handle ByteLevel decoding FIRST if applicable
        # More robust check: Requires knowledge of how ByteLevel was configured/used
        # This heuristic might incorrectly identify non-byte-level tokens
        # is_byte_level = False # Disable simple heuristic for now
        # A better approach is to store byte_level info in tokenizer_config.json
        if isinstance(self.pre_tokenizer, ByteLevel): # Check type of pre_tokenizer
             try:
                 byte_values = []
                 inv_byte_map = {v: k for k, v in self.pre_tokenizer._byte_to_unicode.items()}
                 for token in tokens:
                      # Each token should be a single character in the byte map
                      if len(token) == 1 and token in inv_byte_map:
                          byte_values.append(inv_byte_map[token])
                      # else: What to do with multi-char tokens or tokens not in map?
                      # This suggests the input wasn't purely byte-level or included specials
                      # We might need to handle this more gracefully or assume pure byte tokens here

                 if byte_values: # Proceed only if we successfully mapped some bytes
                      decoded_bytes = bytes(byte_values)
                      text = decoded_bytes.decode('utf-8', errors='replace')
                      # Further cleanup might be needed depending on pre-tokenizer's space handling
                      return text.strip() if clean_up_tokenization_spaces else text

             except Exception as e:
                 # print(f"Warning: Byte-level decoding failed: {e}. Falling back.")
                 pass # Fallback to general decoding

        # 2. General BPE/WordPiece/etc. cleanup
        text = "".join(tokens)
        if clean_up_tokenization_spaces:
            # Generic cleanup - this might need customization per tokenizer type
            # Example for BPE with suffix:
            if isinstance(self.model, BPEModel) and self.model.end_of_word_suffix:
                 text = text.replace(self.model.end_of_word_suffix, ' ')

            # General space cleanup
            text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with one

        return text


    # --- Saving and Loading ---
    # save_pretrained and from_pretrained remain largely the same as before
    # Need to ensure component class names in config match actual class names
    # And handle loading/saving of component-specific args (like ByteLevel options)

    def save_pretrained(self, save_directory: str):
        """Saves tokenizer files (vocab, merges, config) to a directory."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # 1. Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # Use underlying vocab map
            json.dump(self.model.vocab if hasattr(self.model, 'vocab') else self._vocab_str_to_int,
                      f, ensure_ascii=False, indent=2)

        # 2. Save merges (if BPE)
        if isinstance(self.model, BPEModel):
            merges_file = os.path.join(save_directory, "merges.txt")
            sorted_merges = sorted(self.model.merges.items(), key=lambda item: item[1])
            with open(merges_file, 'w', encoding='utf-8') as f:
                f.write("#version: 0.2\n")
                for (tok1, tok2), rank in sorted_merges:
                    f.write(f"{tok1} {tok2}\n")

        # 3. Save tokenizer configuration
        config = self._get_config()
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _get_config(self) -> Dict:
         """Helper to generate the config dictionary for saving."""
         # Get pad token string representation
         pad_token_str = self.model.id_to_token(self.pad_token_id) if self.pad_token_id is not None and self.pad_token_id != -1 else "[PAD]"

         config = {
             "model_type": self.model_type,
             "unk_token": self.unk_token,
             "pad_token": pad_token_str,
             "normalizer": self.normalizer.__class__.__name__ if self.normalizer else None,
             "pre_tokenizer": self.pre_tokenizer.__class__.__name__ if self.pre_tokenizer else None,
             "post_processor": self.post_processor.__class__.__name__ if self.post_processor else None,
             "model_kwargs": {},
             "pre_tokenizer_kwargs": {},
             "post_processor_kwargs": {}
         }
         # Store model specific args (like end_of_word_suffix for BPE)
         if isinstance(self.model, BPEModel):
             config["model_kwargs"]["end_of_word_suffix"] = self.model.end_of_word_suffix
         # Store pre_tokenizer specific args (like add_prefix_space for ByteLevel)
         if isinstance(self.pre_tokenizer, ByteLevel):
             config["pre_tokenizer_kwargs"]["add_prefix_space"] = self.pre_tokenizer.add_prefix_space
             config["pre_tokenizer_kwargs"]["use_regex"] = self.pre_tokenizer.use_regex # Added this arg
         # Store post_processor specific args (like templates for TemplateProcessor)
         if isinstance(self.post_processor, TemplateProcessor):
             config["post_processor_kwargs"]["template_single"] = " ".join(self.post_processor.template_single)
             config["post_processor_kwargs"]["template_pair"] = " ".join(self.post_processor.template_pair)
             config["post_processor_kwargs"]["special_tokens"] = self.post_processor.special_tokens
             # pad_token/id handled above

         return config


    @classmethod
    def from_pretrained(cls, load_directory: str) -> "Tokenizer": # Updated return type hint
        """Loads tokenizer files from a directory."""
        if not os.path.isdir(load_directory):
            raise EnvironmentError(f"Directory not found: {load_directory}")

        # 1. Load configuration
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        if not os.path.exists(config_file):
             # If config doesn't exist, try loading from HF style config? Risky.
             # For now, require the config saved by save_pretrained.
             raise EnvironmentError(f"tokenizer_config.json not found in {load_directory}. Use save_pretrained first or load programmatically.")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 2. Load vocabulary
        vocab_file = os.path.join(load_directory, "vocab.json")
        if not os.path.exists(vocab_file):
             raise EnvironmentError(f"vocab.json not found in {load_directory}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # 3. Load merges (if BPE)
        merges = None
        model_type = config.get("model_type", "BPE") # Default to BPE if not specified
        if model_type.upper() == "BPE":
            merges_file = os.path.join(load_directory, "merges.txt")
            if not os.path.exists(merges_file):
                 raise EnvironmentError(f"merges.txt not found in {load_directory} (required for BPE)")
            merges = {}
            try:
                 with open(merges_file, 'r', encoding='utf-8') as f:
                     first_line = f.readline().strip()
                     if not first_line.startswith("#version"):
                         parts = first_line.split()
                         if len(parts) == 2: merges[(parts[0], parts[1])] = 0
                     for i, line in enumerate(f, start=1 if merges else 0):
                         parts = line.strip().split()
                         if len(parts) == 2: merges[(parts[0], parts[1])] = i
            except Exception as e:
                 raise IOError(f"Could not read merges file at {merges_file}: {e}")


        # --- 4. Instantiate components dynamically ---
        # Import modules dynamically - assumes they are in the same package
        from . import normalizers as norm_module
        from . import pre_tokenizers as pre_tok_module
        from . import processors as proc_module
        from . import models as model_module # Import the models module

        # Instantiate Normalizer
        normalizer_cls_name = config.get("normalizer")
        normalizer = None
        if normalizer_cls_name:
             try:
                 # TODO: Handle Sequence Normalizer loading properly if needed
                 NormalizerCls = getattr(norm_module, normalizer_cls_name)
                 normalizer = NormalizerCls()
             except AttributeError:
                 print(f"Warning: Normalizer class '{normalizer_cls_name}' not found.")


        # Instantiate PreTokenizer
        pre_tokenizer_cls_name = config.get("pre_tokenizer")
        pre_tokenizer = None
        if pre_tokenizer_cls_name:
             try:
                 PreTokCls = getattr(pre_tok_module, pre_tokenizer_cls_name)
                 pt_kwargs = config.get("pre_tokenizer_kwargs", {})
                 pre_tokenizer = PreTokCls(**pt_kwargs)
             except AttributeError:
                 print(f"Warning: PreTokenizer class '{pre_tokenizer_cls_name}' not found.")

        # Instantiate PostProcessor
        post_processor_cls_name = config.get("post_processor")
        post_processor = None
        if post_processor_cls_name:
             try:
                 ProcCls = getattr(proc_module, post_processor_cls_name)
                 pp_kwargs = config.get("post_processor_kwargs", {})
                 # Add pad token info needed by TemplateProcessor constructor
                 if post_processor_cls_name == "TemplateProcessor":
                     pad_token_str = config.get("pad_token", "[PAD]")
                     pad_token_id = vocab.get(pad_token_str)
                     if pad_token_id is None: print(f"Warning: Pad token '{pad_token_str}' not found.")
                     pp_kwargs["pad_token"] = pad_token_str
                     pp_kwargs["pad_token_id"] = pad_token_id if pad_token_id is not None else -1
                     # Convert templates back from string
                     pp_kwargs["template_single"] = pp_kwargs.get("template_single", "").strip()
                     pp_kwargs["template_pair"] = pp_kwargs.get("template_pair", "").strip()

                 post_processor = ProcCls(**pp_kwargs)
             except AttributeError:
                 print(f"Warning: PostProcessor class '{post_processor_cls_name}' not found.")


        # 5. Instantiate the main tokenizer
        # Get model kwargs from config
        model_kwargs = config.get("model_kwargs", {})

        return cls(
            vocab=vocab,
            merges=merges, # Pass loaded merges
            unk_token=config.get("unk_token", "[UNK]"),
            model_type=model_type,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            post_processor=post_processor,
            **model_kwargs # Pass model specific args from config
        )