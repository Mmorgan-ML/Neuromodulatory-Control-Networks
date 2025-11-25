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
from typing import List, Optional, Dict, Union, Tuple, Literal, Any
import torch # For return_tensors='pt'
import numpy as np # For return_tensors='np'
import re # For regular expression operations in decode cleanup

# Check for HuggingFace Transformers (Rust Backend)
try:
    from transformers import PreTrainedTokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import components using relative paths
from .normalizers import Normalizer
from .pre_tokenizers import PreTokenizer, ByteLevel
from .models import Model, BPEModel
from .processors import PostProcessor, TemplateProcessor

class Tokenizer:
    """
    Orchestrates the tokenization pipeline.
    
    Hybrid Architecture:
    1. Attempts to use HuggingFace 'transformers' (Rust) backend for high performance.
    2. Falls back to local Python implementation (Slow) if Fast backend fails or is unavailable.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping tokens to IDs.
        merges (Optional[Dict[Tuple[str, str], int]], optional): Merge rules for BPE.
        unk_token (str, optional): Unknown token string. Defaults to "[UNK]".
        model_type (str, optional): Type of the core sub-word model. Defaults to "BPE".
        normalizer (Optional[Normalizer], optional): Normalizer instance.
        pre_tokenizer (Optional[PreTokenizer], optional): Pre-tokenizer instance.
        post_processor (Optional[PostProcessor], optional): Post-processor instance.
        fast_tokenizer (Optional[Any], optional): Internal use - holds the Rust tokenizer instance.
        **kwargs: Additional kwargs passed to the underlying model.
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
        fast_tokenizer: Optional[Any] = None,
        **kwargs
    ):
        # --- Fast Backend Slot ---
        self.fast_tokenizer = fast_tokenizer
        
        # --- Python Backend Setup (Legacy/Fallback) ---
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.post_processor = post_processor
        self.unk_token = unk_token
        self._vocab_str_to_int = vocab
        self._vocab_int_to_str = {i: s for s, i in vocab.items()}
        self.model_type = model_type

        # Instantiate the core model based on type
        if model_type.upper() == "BPE":
            if merges is None and not self.fast_tokenizer:
                # Only raise if we don't have a fast tokenizer to rely on
                raise ValueError("Merges must be provided for BPEModel.")
            
            if merges is not None:
                self.model = BPEModel(vocab, merges, unk_token=unk_token, **kwargs)
            else:
                self.model = None # Fast tokenizer handling everything
        else:
            if not self.fast_tokenizer:
                raise ValueError(f"Unsupported model_type: {model_type}")
            self.model = None

        # Store common token IDs
        if self.fast_tokenizer:
            self.unk_token_id = self.fast_tokenizer.unk_token_id
            self.pad_token_id = self.fast_tokenizer.pad_token_id
            # Ensure pad token ID is valid (HF sometimes defaults to None)
            if self.pad_token_id is None:
                self.pad_token_id = self.fast_tokenizer.eos_token_id # Fallback
        else:
            self.unk_token_id = self.model.token_to_id(unk_token) if self.model and unk_token else None
            if post_processor and hasattr(post_processor, 'pad_token_id'):
                self.pad_token_id = post_processor.pad_token_id
            else:
                 pad_token_str = "[PAD]"
                 if post_processor and hasattr(post_processor, 'pad_token'):
                     pad_token_str = post_processor.pad_token
                 self.pad_token_id = self.model.token_to_id(pad_token_str) if self.model else None
                 if self.pad_token_id is None:
                     self.pad_token_id = -1

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        if self.fast_tokenizer:
            return self.fast_tokenizer.vocab_size
        return len(self._vocab_str_to_int)

    # --- Internal Python Methods (Used only if fast_tokenizer is None) ---
    def _normalize(self, text: str) -> str:
        return self.normalizer.normalize_str(text) if self.normalizer else text

    def _pre_tokenize(self, normalized_text: str) -> List[str]:
        return self.pre_tokenizer.pre_tokenize_str(normalized_text) if self.pre_tokenizer else [normalized_text]

    def _tokenize_sequence(self, sequence: str) -> List[str]:
        normalized = self._normalize(sequence)
        pre_tokenized = self._pre_tokenize(normalized)
        subword_tokens = []
        for word in pre_tokenized:
            subword_tokens.extend(self.model.tokenize(word))
        return subword_tokens

    # --- Public API (Delegates to Fast Backend if available) ---

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token or list of tokens to their corresponding IDs."""
        if self.fast_tokenizer:
            return self.fast_tokenizer.convert_tokens_to_ids(tokens)
            
        # Slow Fallback
        if isinstance(tokens, str):
            token_id = self.model.token_to_id(tokens)
            if token_id is None:
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
        if self.fast_tokenizer:
            return self.fast_tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

        # Slow Fallback
        if isinstance(ids, int):
            token = self.model.id_to_token(ids)
            is_special = self._is_special_token(token)
            if skip_special_tokens and is_special:
                return ""
            return token if token is not None else self.unk_token

        tokens = [self.model.id_to_token(id_) for id_ in ids]
        processed_tokens = []
        for token in tokens:
            is_special = self._is_special_token(token)
            if skip_special_tokens and is_special:
                continue
            processed_tokens.append(token if token is not None else self.unk_token)
        return processed_tokens

    def _is_special_token(self, token: Optional[str]) -> bool:
        """(Internal) Checks if a token is considered special."""
        if token is None: return False
        if self.post_processor and hasattr(self.post_processor, 'special_tokens'):
            if token in self.post_processor.special_tokens: return True
        if token == self.unk_token: return True
        pad_token_str = self.model.id_to_token(self.pad_token_id) if self.model and self.pad_token_id is not None and self.pad_token_id != -1 else None
        if token == pad_token_str: return True
        return False

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, Literal['longest', 'max_length']] = False,
        truncation: Union[bool, str, Literal['longest_first', 'only_first', 'only_second']] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Literal["pt", "np"]] = None,
        **kwargs
    ) -> Dict:
        """
        Encodes text into model inputs. Uses Rust backend if available.
        """
        # --- FAST PATH ---
        if self.fast_tokenizer:
            # Map arguments to HF signature
            # HF uses 'longest' or 'max_length' string, or True for default
            hf_padding = padding
            if padding is True: hf_padding = True
            elif padding is False: hf_padding = False
            
            hf_truncation = truncation
            if truncation is True: hf_truncation = True
            elif truncation is False: hf_truncation = False
            
            # Fast Tokenizer Call
            output = self.fast_tokenizer(
                text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=hf_padding,
                truncation=hf_truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs
            )
            
            # The output is a BatchEncoding object (dict-like), return as dict
            # or convert to specific return_tensors type if HF didn't do it automatically (it usually does)
            return dict(output)

        # --- SLOW PATH (Python) ---
        tokens_a = self._tokenize_sequence(text)
        tokens_b = None
        if text_pair is not None:
            tokens_b = self._tokenize_sequence(text_pair)

        ids_a = self.convert_tokens_to_ids(tokens_a)
        ids_b = self.convert_tokens_to_ids(tokens_b) if tokens_b is not None else None

        if self.post_processor:
            do_padding = bool(padding) or padding == "max_length" or padding == "longest"
            do_truncation = bool(truncation) or isinstance(truncation, str)
            trunc_strat = truncation if isinstance(truncation, str) and truncation in ['longest_first', 'only_first', 'only_second'] else "longest_first"

            output = self.post_processor.process(
                token_ids_a=ids_a,
                token_ids_b=ids_b,
                max_length=max_length,
                truncation=do_truncation,
                padding=do_padding,
                add_special_tokens=add_special_tokens,
                truncation_strategy=trunc_strat,
                **kwargs
            )
        else:
             if ids_b is not None:
                 input_ids = ids_a + ids_b
                 token_type_ids = ([0] * len(ids_a)) + ([1] * len(ids_b))
             else:
                 input_ids = ids_a
                 token_type_ids = [0] * len(ids_a)
             attention_mask = [1] * len(input_ids)

             if truncation and max_length is not None and len(input_ids) > max_length:
                 input_ids = input_ids[:max_length]
                 token_type_ids = token_type_ids[:max_length]
                 attention_mask = attention_mask[:max_length]

             if padding and max_length is not None and len(input_ids) < max_length:
                  pad_len = max_length - len(input_ids)
                  if self.pad_token_id is None or self.pad_token_id == -1:
                       raise ValueError("Padding requested but pad_token_id is not validly set.")
                  input_ids.extend([self.pad_token_id] * pad_len)
                  pad_type = self.post_processor.pad_token_type_id if self.post_processor and hasattr(self.post_processor, 'pad_token_type_id') else 0
                  token_type_ids.extend([pad_type] * pad_len)
                  attention_mask.extend([0] * pad_len)

             output = {
                 "input_ids": input_ids,
                 "token_type_ids": token_type_ids,
                 "attention_mask": attention_mask
             }

        if return_tensors == "pt":
            for key in output:
                output[key] = torch.tensor(output[key])
        elif return_tensors == "np":
            for key in output:
                output[key] = np.array(output[key])

        return output

    __call__ = encode

    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor, np.ndarray],
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True
        ) -> str:
        """
        Decodes token IDs into a string. Uses Rust backend if available.
        """
        # Handle Tensor/Numpy conversion
        if hasattr(token_ids, 'tolist'):
             ids_list = token_ids.tolist()
        else:
             ids_list = list(token_ids)
             
        # --- FAST PATH ---
        if self.fast_tokenizer:
            # HF decode expects a single list of IDs or a single int
            return self.fast_tokenizer.decode(
                ids_list, 
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )

        # --- SLOW PATH (Python) ---
        tokens = self.convert_ids_to_tokens(ids_list, skip_special_tokens=skip_special_tokens)

        if not tokens: return ""

        # Cleanup ByteLevel (Slow Path)
        if isinstance(self.pre_tokenizer, ByteLevel):
             try:
                 byte_values = []
                 inv_byte_map = {v: k for k, v in self.pre_tokenizer._byte_to_unicode.items()}
                 for token in tokens:
                      if len(token) == 1 and token in inv_byte_map:
                          byte_values.append(inv_byte_map[token])
                 if byte_values:
                      decoded_bytes = bytes(byte_values)
                      text = decoded_bytes.decode('utf-8', errors='replace')
                      return text.strip() if clean_up_tokenization_spaces else text
             except Exception:
                 pass 

        text = "".join(tokens)
        if clean_up_tokenization_spaces:
            if isinstance(self.model, BPEModel) and self.model.end_of_word_suffix:
                 text = text.replace(self.model.end_of_word_suffix, ' ')
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    # --- Saving and Loading ---

    def save_pretrained(self, save_directory: str):
        """Saves tokenizer files. Checks if Fast tokenizer handles saving."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # If we have a fast tokenizer, let it handle the heavy lifting of JSONs
        if self.fast_tokenizer:
            self.fast_tokenizer.save_pretrained(save_directory)
            # We still save our config wrapper so we know to try Fast load next time
            config = self._get_config()
            config_file = os.path.join(save_directory, "tokenizer_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return

        # Python implementation saving
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.model.vocab if hasattr(self.model, 'vocab') else self._vocab_str_to_int,
                      f, ensure_ascii=False, indent=2)

        if isinstance(self.model, BPEModel):
            merges_file = os.path.join(save_directory, "merges.txt")
            sorted_merges = sorted(self.model.merges.items(), key=lambda item: item[1])
            with open(merges_file, 'w', encoding='utf-8') as f:
                f.write("#version: 0.2\n")
                for (tok1, tok2), rank in sorted_merges:
                    f.write(f"{tok1} {tok2}\n")

        config = self._get_config()
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _get_config(self) -> Dict:
         """Helper to generate the config dictionary."""
         # Get pad token string
         if self.fast_tokenizer:
             pad_token_str = self.fast_tokenizer.pad_token if self.fast_tokenizer.pad_token else "[PAD]"
         else:
             pad_token_str = self.model.id_to_token(self.pad_token_id) if self.model and self.pad_token_id is not None and self.pad_token_id != -1 else "[PAD]"

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
         # Store internal args for Python recreation
         if self.model and isinstance(self.model, BPEModel):
             config["model_kwargs"]["end_of_word_suffix"] = self.model.end_of_word_suffix
         if self.pre_tokenizer and isinstance(self.pre_tokenizer, ByteLevel):
             config["pre_tokenizer_kwargs"]["add_prefix_space"] = self.pre_tokenizer.add_prefix_space
             config["pre_tokenizer_kwargs"]["use_regex"] = self.pre_tokenizer.use_regex
         if self.post_processor and isinstance(self.post_processor, TemplateProcessor):
             config["post_processor_kwargs"]["template_single"] = " ".join(self.post_processor.template_single)
             config["post_processor_kwargs"]["template_pair"] = " ".join(self.post_processor.template_pair)
             config["post_processor_kwargs"]["special_tokens"] = self.post_processor.special_tokens

         return config

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "Tokenizer":
        """
        Loads tokenizer. Attempts to load HuggingFace Fast Tokenizer first.
        """
        if not os.path.isdir(load_directory):
            raise EnvironmentError(f"Directory not found: {load_directory}")

        # --- ATTEMPT 1: FAST BACKEND ---
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try loading as a Fast tokenizer directly
                # This works if vocab.json/merges.txt or tokenizer.json exist
                fast_tok = PreTrainedTokenizerFast.from_pretrained(load_directory)
                
                # If successful, return a wrapper instance
                # We need to construct the wrapper with dummy values for the python parts
                # since the Fast tokenizer handles it all.
                
                # Extract basic info for the wrapper state
                vocab = fast_tok.vocab
                
                return cls(
                    vocab=vocab,
                    merges=None, # Fast tokenizer handles merges internally
                    unk_token=fast_tok.unk_token if fast_tok.unk_token else "[UNK]",
                    fast_tokenizer=fast_tok
                )
            except Exception as e:
                # print(f"Fast tokenizer load failed: {e}. Falling back to Python.")
                pass

        # --- ATTEMPT 2: PYTHON BACKEND (Fallback) ---
        
        # 1. Load configuration
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        if not os.path.exists(config_file):
             raise EnvironmentError(f"tokenizer_config.json not found in {load_directory}.")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 2. Load vocabulary
        vocab_file = os.path.join(load_directory, "vocab.json")
        if not os.path.exists(vocab_file):
             raise EnvironmentError(f"vocab.json not found in {load_directory}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # 3. Load merges
        merges = None
        model_type = config.get("model_type", "BPE")
        if model_type.upper() == "BPE":
            merges_file = os.path.join(load_directory, "merges.txt")
            if not os.path.exists(merges_file):
                 raise EnvironmentError(f"merges.txt not found in {load_directory}")
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
                 raise IOError(f"Could not read merges file: {e}")

        # 4. Instantiate Python components
        from . import normalizers as norm_module
        from . import pre_tokenizers as pre_tok_module
        from . import processors as proc_module

        # Normalizer
        normalizer_cls_name = config.get("normalizer")
        normalizer = None
        if normalizer_cls_name:
             try:
                 NormalizerCls = getattr(norm_module, normalizer_cls_name)
                 normalizer = NormalizerCls()
             except AttributeError: pass

        # PreTokenizer
        pre_tokenizer_cls_name = config.get("pre_tokenizer")
        pre_tokenizer = None
        if pre_tokenizer_cls_name:
             try:
                 PreTokCls = getattr(pre_tok_module, pre_tokenizer_cls_name)
                 pt_kwargs = config.get("pre_tokenizer_kwargs", {})
                 pre_tokenizer = PreTokCls(**pt_kwargs)
             except AttributeError: pass

        # PostProcessor
        post_processor_cls_name = config.get("post_processor")
        post_processor = None
        if post_processor_cls_name:
             try:
                 ProcCls = getattr(proc_module, post_processor_cls_name)
                 pp_kwargs = config.get("post_processor_kwargs", {})
                 if post_processor_cls_name == "TemplateProcessor":
                     pad_token_str = config.get("pad_token", "[PAD]")
                     pad_token_id = vocab.get(pad_token_str)
                     pp_kwargs["pad_token"] = pad_token_str
                     pp_kwargs["pad_token_id"] = pad_token_id if pad_token_id is not None else -1
                     pp_kwargs["template_single"] = pp_kwargs.get("template_single", "").strip()
                     pp_kwargs["template_pair"] = pp_kwargs.get("template_pair", "").strip()
                 post_processor = ProcCls(**pp_kwargs)
             except AttributeError: pass

        model_kwargs = config.get("model_kwargs", {})

        return cls(
            vocab=vocab,
            merges=merges,
            unk_token=config.get("unk_token", "[UNK]"),
            model_type=model_type,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            post_processor=post_processor,
            **model_kwargs
        )