"""Shared LLM model manager for classifier and repairer.

This module provides a single Qwen model instance that can be shared
between the classifier (Stage 2) and repairer (Stage 4) to avoid
loading the model twice.

Usage:
    manager = SharedModelManager(model_name="Qwen/Qwen2.5-32B-Instruct")
    classifier = LLMSpanClassifier(shared_model=manager)
    repairer = LLMNameRepairer(shared_model=manager)

    # Process documents...

    manager.cleanup()  # Unloads model from GPU
"""

from dataclasses import dataclass, field

import torch
from loguru import logger


@dataclass
class BatchStats:
    """Metrics for batched generation calls."""
    total_batches: int = 0
    total_prompts: int = 0
    oom_splits: int = 0
    padding_waste_tokens: int = 0


class SharedModelManager:
    """Manages a shared LLM model instance for multiple components.

    Benefits:
    - Loads Qwen 32B only once (~4 minutes saved)
    - Single memory footprint (~18GB instead of ~36GB)
    - Both classifier and repairer use same model
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 quantize_4bit: bool = True):
        """Initialize the shared model manager.

        Args:
            model_name: HuggingFace model name.
            device: Device to use ('cuda', 'cpu', or None for auto).
            quantize_4bit: Use 4-bit quantization for memory efficiency.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize_4bit = quantize_4bit

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self.batch_stats = BatchStats()

    @property
    def model(self):
        """Get the model, loading if necessary."""
        if not self._loaded:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer, loading if necessary."""
        if not self._loaded:
            self._load_model()
        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _validate_gpu(self):
        """Fail fast if GPU is not Tier 2 GPU-class for LLM workloads."""
        if not torch.cuda.is_available():
            return  # CPU mode, skip check

        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)

        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Compute capability: {capability[0]}.{capability[1]}")

        # Tier 2 GPU-class: compute capability >= 9.0 (Hopper)
        # Also allow A100 (8.0), L40s (8.9) for flexibility
        MIN_COMPUTE_CAPABILITY = 8.0
        if capability[0] + capability[1] / 10 < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"FATAL: GPU '{gpu_name}' (compute {capability[0]}.{capability[1]}) "
                f"is below minimum {MIN_COMPUTE_CAPABILITY} for LLM workloads. "
                f"Qwen 32B requires Tier 2 GPU/A100-class GPU. "
                f"Tier 1 GPU (compute 7.5) is only for XLM-R NER (--ner-only)."
            )

    def _load_model(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return

        self._validate_gpu()

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading shared LLM model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        # Flash attention 2 for faster generation (Tier 2 GPU/A100 support it)
        # Falls back at model load time if flash_attn not installed
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Requesting flash_attention_2")

        if self.quantize_4bit and self.device == "cuda":
            logger.info("Using 4-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except (ImportError, ValueError) as e:
            # Flash attention not available — fall back without it
            if "flash" in str(e).lower():
                logger.warning(f"Flash attention unavailable ({e}), falling back to default")
                model_kwargs.pop("attn_implementation", None)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                raise

        self._loaded = True
        logger.info("Shared LLM model loaded")

        # Log GPU + attention details for diagnostics
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")
            logger.info(f"Attention implementation: {getattr(self._model.config, '_attn_implementation', 'unknown')}")
            logger.info(f"Quantization: {'4-bit' if self.quantize_4bit else 'none'}")

            try:
                import flash_attn
                logger.info(f"flash_attn: v{flash_attn.__version__}")
            except ImportError:
                logger.warning("flash_attn: NOT INSTALLED")

    def generate(self, messages: list[dict], max_new_tokens: int = 2048) -> str:
        """Generate response from the model.

        Args:
            messages: Chat messages in OpenAI format.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        if not self._loaded:
            self._load_model()

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            )

        # Decode only new tokens
        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def generate_batch(self, messages_list: list[list[dict]], max_new_tokens: int = 2048) -> list[str]:
        """Generate responses for multiple independent prompts in a single batched call.

        Each element in messages_list is a complete, independent chat conversation.
        This does NOT mix documents — each prompt is processed independently but
        the GPU processes them in parallel for better utilization.

        Args:
            messages_list: List of chat message lists (each in OpenAI format).
            max_new_tokens: Maximum tokens to generate per response.

        Returns:
            List of generated text responses, one per input.
        """
        if not messages_list:
            return []

        # Degenerate case: single prompt, delegate to existing method
        if len(messages_list) == 1:
            return [self.generate(messages_list[0], max_new_tokens=max_new_tokens)]

        if not self._loaded:
            self._load_model()

        # Apply chat template to each conversation
        texts = [
            self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]

        # Left-pad for batched generation (required by transformers for correct decode)
        original_padding_side = self._tokenizer.padding_side
        original_pad_token_id = self._tokenizer.pad_token_id
        try:
            self._tokenizer.padding_side = 'left'
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to(self._model.device)

            # Compute per-item input lengths from attention mask
            # (left-padded: actual tokens = attention_mask.sum per row)
            input_lengths = inputs.attention_mask.sum(dim=1).tolist()

            # Track padding waste
            padded_input_len = inputs.input_ids.shape[1]
            total_tokens = inputs.input_ids.numel()
            non_pad_tokens = sum(input_lengths)
            self.batch_stats.padding_waste_tokens += total_tokens - non_pad_tokens
            self.batch_stats.total_batches += 1
            self.batch_stats.total_prompts += len(messages_list)

            try:
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                self.batch_stats.oom_splits += 1
                logger.warning(
                    f"OOM with batch size {len(messages_list)}, splitting in half"
                )
                torch.cuda.empty_cache()
                mid = len(messages_list) // 2
                left = self.generate_batch(messages_list[:mid], max_new_tokens=max_new_tokens)
                right = self.generate_batch(messages_list[mid:], max_new_tokens=max_new_tokens)
                return left + right

            # Decode: we slice at padded_input_len (= input_ids.shape[1])
            # because HF generate() returns sequences that include the full
            # padded input prefix. With LEFT-padding the layout is:
            #   [PAD...PAD, tok1...tokN, gen1...genM]
            # so padded_input_len is the universal cutoff for ALL items.
            # Do NOT use per-item attention_mask sums — that would include
            # tail-of-input tokens for shorter prompts.
            # Invariant: output[:, :padded_input_len] == input_ids
            assert outputs.shape[1] >= padded_input_len, (
                f"generate() output shorter than input: "
                f"{outputs.shape[1]} < {padded_input_len}"
            )
            results = []
            for i in range(len(messages_list)):
                new_tokens = outputs[i][padded_input_len:]
                response = self._tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                results.append(response.strip())

            return results

        finally:
            self._tokenizer.padding_side = original_padding_side
            if original_pad_token_id is not None:
                self._tokenizer.pad_token_id = original_pad_token_id

    def cleanup(self):
        """Unload model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Shared LLM model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
