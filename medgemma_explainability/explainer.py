"""
Main MedGemma Explainer class implementing Chefer et al. method.

This module provides the high-level API for generating explanations
for MedGemma vision-language model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
from dataclasses import dataclass, field

from .attention_hooks import AttentionHookManager, ManualAttentionComputer
from .relevancy import (
    propagate_relevancy,
    propagate_relevancy_additive,
    extract_token_relevancy,
    split_relevancy,
    compute_raw_attention_relevancy,
)
from .visualization import ExplanationResult
from .utils import (
    load_image,
    get_device,
    clear_gpu_memory,
    normalize_relevancy,
    reshape_image_relevancy,
)


class MedGemmaExplainer:
    """
    Explainability interface for MedGemma using Chefer et al. method.

    This class provides methods to:
    1. Load and configure MedGemma for explainability
    2. Generate explanations for image-text inputs
    3. Visualize relevancy maps

    Example:
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> model = AutoModelForImageTextToText.from_pretrained("google/medgemma-1.5-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
        >>> explainer = MedGemmaExplainer(model, processor)
        >>> result = explainer.explain(image, "What do you see in this image?")
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        num_query_heads: int = 8,
        num_kv_heads: int = 4,
        num_image_tokens: int = 256,
        image_start_idx: int = 6,
        num_language_layers: int = 34,
        local_window_size: int = 1024,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the explainer.

        Args:
            model: MedGemma model
            processor: HuggingFace processor for tokenization
            num_query_heads: Number of query heads (8 for MedGemma)
            num_kv_heads: Number of KV heads for GQA (4 for MedGemma)
            num_image_tokens: Number of image tokens (256 for MedGemma)
            image_start_idx: Index where image tokens start (6 for MedGemma)
            num_language_layers: Number of language model layers (34 for MedGemma)
            local_window_size: Window size for local attention
            device: Device to run on (auto-detected if None)
        """
        self.model = model
        self.processor = processor
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_image_tokens = num_image_tokens
        self.image_start_idx = image_start_idx
        self.num_language_layers = num_language_layers
        self.local_window_size = local_window_size
        self.device = device or get_device()

        # Initialize hook manager
        self.hook_manager = AttentionHookManager(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            expand_gqa=True,
        )

        # For manual attention computation fallback
        self.manual_computer = None

        # Cache for storing intermediate results
        self._attention_cache: Dict[int, torch.Tensor] = {}
        self._gradient_cache: Dict[int, torch.Tensor] = {}

    def explain(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        target_token_position: Optional[int] = None,
        use_additive_propagation: bool = False,
        include_raw_attention: bool = True,
        max_new_tokens: int = 100,
    ) -> ExplanationResult:
        """
        Generate explanation for an image-text input.

        IMPORTANT: For causal LM, logit at position i predicts token at i+1.
        To explain why token at position p was generated, we backprop from
        logit at position p-1 using the actual token id at position p.

        Args:
            image: Image path, URL, or PIL Image
            prompt: Text prompt/question
            target_token_position: Position of token to explain (None for last token)
            use_additive_propagation: Use additive R = R + Ā @ R (else multiplicative)
            include_raw_attention: Also compute raw attention baseline
            max_new_tokens: Maximum tokens to generate

        Returns:
            ExplanationResult containing relevancy maps and metadata
        """
        # Load image if needed
        if isinstance(image, str):
            image = load_image(image)

        # Clear previous state
        self._clear_caches()
        clear_gpu_memory()

        # Prepare inputs
        inputs = self._prepare_inputs(image, prompt)
        input_len = inputs['input_ids'].shape[1]

        # Generate response first
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        generated_ids = gen_outputs
        seq_len = generated_ids.shape[1]

        # Decode generated text
        generated_text = self.processor.decode(
            generated_ids[0], skip_special_tokens=True
        )

        # Default to last token if not specified
        if target_token_position is None:
            target_token_position = seq_len - 1

        # Get the actual token id at the target position
        target_token_id = generated_ids[0, target_token_position].item()

        # Backward pass with CORRECT target
        # Logit at p-1 predicts token at p
        R, attention_maps, logit_position = self._compute_token_relevancy(
            inputs, generated_ids, target_token_position, target_token_id
        )

        # Extract relevancy from the PREDICTING position (p-1)
        token_relevancy = extract_token_relevancy(R, logit_position)

        # Split into image and text
        image_relevancy, text_relevancy = split_relevancy(
            token_relevancy,
            num_image_tokens=self.num_image_tokens,
            image_start_idx=self.image_start_idx,
        )

        # Reshape image relevancy to 16x16
        image_relevancy_2d = reshape_image_relevancy(image_relevancy)

        # Normalize
        image_relevancy_2d = normalize_relevancy(image_relevancy_2d)
        text_relevancy = normalize_relevancy(text_relevancy)

        # Get token labels
        token_labels = self._get_token_labels(generated_ids)

        # Compute raw attention baseline if requested
        raw_image_relevancy = None
        raw_text_relevancy = None
        if include_raw_attention and attention_maps:
            raw_attn = compute_raw_attention_relevancy(attention_maps)
            raw_token_rel = raw_attn[logit_position, :]
            raw_img, raw_txt = split_relevancy(
                raw_token_rel, self.num_image_tokens, self.image_start_idx
            )
            raw_image_relevancy = normalize_relevancy(
                reshape_image_relevancy(raw_img)
            ).cpu().numpy()
            raw_text_relevancy = normalize_relevancy(raw_txt).cpu().numpy()

        # Build result - text tokens start after image section
        text_start_idx = self.image_start_idx + self.num_image_tokens
        result = ExplanationResult(
            image_relevancy=image_relevancy_2d.cpu().numpy(),
            text_relevancy=text_relevancy.cpu().numpy(),
            token_labels=token_labels[text_start_idx:],  # Text tokens only
            generated_text=generated_text,
            raw_image_relevancy=raw_image_relevancy,
            raw_text_relevancy=raw_text_relevancy,
            metadata={
                "seq_len": seq_len,
                "num_layers": len(attention_maps),
                "target_token_position": target_token_position,
                "logit_position": logit_position,
                "target_token": self.processor.decode([target_token_id]),
                "prompt": prompt,
                "input_len": input_len,
            },
        )

        return result

    def explain_keyword(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        keyword: str,
        max_new_tokens: int = 100,
    ) -> ExplanationResult:
        """
        Generate explanation for a specific keyword in the response.

        Finds the first occurrence of the keyword in generated tokens and
        explains that token.

        Args:
            image: Image path, URL, or PIL Image
            prompt: Text prompt/question
            keyword: Keyword to find and explain (case-insensitive)
            max_new_tokens: Maximum tokens to generate

        Returns:
            ExplanationResult for the keyword token
        """
        # Load image if needed
        if isinstance(image, str):
            image = load_image(image)

        # Prepare inputs and generate
        inputs = self._prepare_inputs(image, prompt)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        # Find keyword in generated tokens
        all_tokens = [self.processor.decode([t.item()]) for t in gen_outputs[0]]
        keyword_position = None

        for i in range(input_len, len(all_tokens)):
            if keyword.lower() in all_tokens[i].lower():
                keyword_position = i
                break

        if keyword_position is None:
            raise ValueError(f"Keyword '{keyword}' not found in generated response")

        # Now explain that specific token
        return self.explain(
            image=image,
            prompt=prompt,
            target_token_position=keyword_position,
            max_new_tokens=max_new_tokens,
        )

    def explain_answer_span(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 100,
    ) -> ExplanationResult:
        """
        Generate explanation for the entire answer by summing logits.

        This explains all generated tokens together by computing gradients
        from the sum of all answer token logits.

        Args:
            image: Image path, URL, or PIL Image
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate

        Returns:
            ExplanationResult for the entire answer
        """
        # Load image if needed
        if isinstance(image, str):
            image = load_image(image)

        # Clear previous state
        self._clear_caches()
        clear_gpu_memory()

        # Prepare inputs
        inputs = self._prepare_inputs(image, prompt)
        input_len = inputs['input_ids'].shape[1]

        # Generate response
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        generated_ids = gen_outputs
        seq_len = generated_ids.shape[1]
        answer_start = input_len
        answer_end = seq_len

        # Decode generated text
        generated_text = self.processor.decode(
            generated_ids[0], skip_special_tokens=True
        )

        # Backward from summed logits for all answer tokens
        R, attention_maps, pred_positions = self._backward_from_answer_span(
            inputs, generated_ids, answer_start, answer_end
        )

        # Average relevancy rows over predicting positions
        avg_relevancy = torch.zeros(seq_len, device=self.device)
        for pos in pred_positions:
            avg_relevancy += R[pos, :]
        avg_relevancy = avg_relevancy / len(pred_positions)

        # Split into image and text
        image_relevancy, text_relevancy = split_relevancy(
            avg_relevancy,
            num_image_tokens=self.num_image_tokens,
            image_start_idx=self.image_start_idx,
        )

        # Reshape and normalize
        image_relevancy_2d = normalize_relevancy(reshape_image_relevancy(image_relevancy))
        text_relevancy = normalize_relevancy(text_relevancy)

        # Get token labels
        token_labels = self._get_token_labels(generated_ids)
        text_start_idx = self.image_start_idx + self.num_image_tokens

        result = ExplanationResult(
            image_relevancy=image_relevancy_2d.cpu().numpy(),
            text_relevancy=text_relevancy.cpu().numpy(),
            token_labels=token_labels[text_start_idx:],
            generated_text=generated_text,
            raw_image_relevancy=None,
            raw_text_relevancy=None,
            metadata={
                "seq_len": seq_len,
                "num_layers": len(attention_maps),
                "answer_start": answer_start,
                "answer_end": answer_end,
                "num_answer_tokens": answer_end - answer_start,
                "prompt": prompt,
            },
        )

        return result

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare model inputs using the processor."""
        # Format prompt for instruction-tuned model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Use chat template if available
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

        # Move to device
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        return inputs

    def _register_attention_hooks(self) -> None:
        """Register hooks for attention capture."""
        self.hook_manager.register_hooks(self.model)

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        self.hook_manager.clear()

    def _clear_caches(self) -> None:
        """Clear attention and gradient caches."""
        self._attention_cache = {}
        self._gradient_cache = {}

    def _generate_with_attention(
        self,
        inputs: dict,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate text while capturing attention.

        Note: For explanation, we typically only need one forward pass,
        but we generate to get the full output sequence.
        """
        # Enable gradient computation for attention capture
        self.model.eval()

        # Generate with attention output enabled
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,  # Greedy for reproducibility
            )

        generated_ids = outputs.sequences

        # Decode generated text
        generated_text = self.processor.decode(
            generated_ids[0], skip_special_tokens=True
        )

        return generated_ids, generated_text

    def _compute_token_relevancy(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        target_token_position: int,
        target_token_id: int,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], int]:
        """
        Perform backward pass with CORRECT target for causal LM.

        For causal LM: logit at position i predicts token at i+1.
        To explain token at position p, we backprop from logit at p-1
        using the actual token id at position p.

        Args:
            inputs: Original model inputs (for pixel_values)
            generated_ids: Full generated sequence
            target_token_position: Position of token to explain (p)
            target_token_id: The actual token id at position p

        Returns:
            Tuple of (R matrix, attention_maps, logit_position)
        """
        seq_len = generated_ids.shape[1]

        # IMPORTANT: Keep model in eval mode, use torch.enable_grad() context
        self.model.eval()

        # Prepare full inputs for explanation
        full_inputs = {
            'input_ids': generated_ids,
            'attention_mask': torch.ones_like(generated_ids),
        }
        if 'pixel_values' in inputs:
            full_inputs['pixel_values'] = inputs['pixel_values']

        with torch.enable_grad():
            outputs = self.model(**full_inputs, output_attentions=True, return_dict=True)

            # Retain gradients on attention tensors
            for attn in outputs.attentions:
                if attn is not None:
                    attn.requires_grad_(True)
                    attn.retain_grad()

            # CORRECT TARGET: logit at position p-1 predicts token at p
            logit_position = target_token_position - 1

            # Get the logit for the ACTUAL token that was generated
            target_logit = outputs.logits[0, logit_position, target_token_id]

            # Backward from this specific target
            target_logit.backward(retain_graph=True)

            # Collect attention and gradients
            attention_maps = {}
            attention_grads = {}

            for i, attn in enumerate(outputs.attentions):
                if attn is not None:
                    attention_maps[i] = attn.detach().float()
                    if attn.grad is not None:
                        attention_grads[i] = attn.grad.detach().float()

        # Propagate relevancy (without extra normalization)
        R = torch.eye(seq_len, device=self.device, dtype=torch.float32)

        for layer_idx in sorted(attention_maps.keys()):
            if layer_idx not in attention_grads:
                continue

            A = attention_maps[layer_idx]
            grad_A = attention_grads[layer_idx]

            # Compute Abar: gradient-weighted attention
            weighted = A * grad_A
            weighted = torch.clamp(weighted, min=0)
            Abar = weighted.mean(dim=(0, 1))

            # Add identity for residual connection
            Abar = Abar + torch.eye(seq_len, device=self.device, dtype=torch.float32)

            # Propagate: R = Abar @ R
            R = torch.matmul(Abar, R)

        return R, attention_maps, logit_position

    def _backward_from_answer_span(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        answer_start: int,
        answer_end: int,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], List[int]]:
        """
        Perform backward pass from summed logits for entire answer span.

        This explains all answer tokens together by computing gradients
        from the sum of logits for all answer tokens.

        Args:
            inputs: Original model inputs (for pixel_values)
            generated_ids: Full generated sequence
            answer_start: Start position of answer tokens
            answer_end: End position of answer tokens (exclusive)

        Returns:
            Tuple of (R matrix, attention_maps, predicting_positions)
        """
        seq_len = generated_ids.shape[1]

        # Keep model in eval mode
        self.model.eval()

        # Prepare full inputs
        full_inputs = {
            'input_ids': generated_ids,
            'attention_mask': torch.ones_like(generated_ids),
        }
        if 'pixel_values' in inputs:
            full_inputs['pixel_values'] = inputs['pixel_values']

        with torch.enable_grad():
            outputs = self.model(**full_inputs, output_attentions=True, return_dict=True)

            for attn in outputs.attentions:
                if attn is not None:
                    attn.requires_grad_(True)
                    attn.retain_grad()

            # Sum logits for all answer tokens
            # Position p-1 predicts token at p
            total_logit = 0
            predicting_positions = []

            for p in range(answer_start, answer_end):
                logit_pos = p - 1
                token_id = generated_ids[0, p].item()
                total_logit = total_logit + outputs.logits[0, logit_pos, token_id]
                predicting_positions.append(logit_pos)

            # Backward from summed objective
            total_logit.backward(retain_graph=True)

            # Collect attention and gradients
            attention_maps = {}
            attention_grads = {}

            for i, attn in enumerate(outputs.attentions):
                if attn is not None:
                    attention_maps[i] = attn.detach().float()
                    if attn.grad is not None:
                        attention_grads[i] = attn.grad.detach().float()

        # Propagate relevancy
        R = torch.eye(seq_len, device=self.device, dtype=torch.float32)

        for layer_idx in sorted(attention_maps.keys()):
            if layer_idx not in attention_grads:
                continue

            A = attention_maps[layer_idx]
            grad_A = attention_grads[layer_idx]

            weighted = A * grad_A
            weighted = torch.clamp(weighted, min=0)
            Abar = weighted.mean(dim=(0, 1))
            Abar = Abar + torch.eye(seq_len, device=self.device, dtype=torch.float32)
            R = torch.matmul(Abar, R)

        return R, attention_maps, predicting_positions

    def _backward_from_target(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        target_token_idx: int,
    ) -> None:
        """
        DEPRECATED: Use _compute_token_relevancy instead.

        This old method used incorrect backprop targets.
        Kept for backwards compatibility but should not be used.
        """
        import warnings
        warnings.warn(
            "_backward_from_target is deprecated. Use _compute_token_relevancy instead.",
            DeprecationWarning
        )

        # Redirect to new method
        if target_token_idx == -1:
            target_token_idx = generated_ids.shape[1] - 1
        target_token_id = generated_ids[0, target_token_idx].item()

        R, attention_maps, logit_position = self._compute_token_relevancy(
            inputs, generated_ids, target_token_idx, target_token_id
        )

        # Store in caches for backwards compatibility
        self._attention_cache = attention_maps
        self._gradient_cache = {}

    def _get_token_labels(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to string labels."""
        labels = []
        for token_id in token_ids[0]:
            token_str = self.processor.decode([token_id.item()])
            labels.append(token_str)
        return labels

    def get_attention_stats(self) -> dict:
        """Get statistics about captured attention."""
        attention_maps = self.hook_manager.get_attention_weights()

        if not attention_maps:
            return {"error": "No attention captured"}

        stats = {
            "num_layers": len(attention_maps),
            "shapes": {},
            "value_ranges": {},
        }

        for layer_idx, attn in attention_maps.items():
            stats["shapes"][layer_idx] = tuple(attn.shape)
            stats["value_ranges"][layer_idx] = {
                "min": attn.min().item(),
                "max": attn.max().item(),
                "mean": attn.mean().item(),
            }

        return stats


def load_medgemma(
    model_name: str = "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",
) -> Tuple[nn.Module, any, torch.device]:
    """
    Load MedGemma model and processor.

    Args:
        model_name: HuggingFace model name
        torch_dtype: Data type for model weights
        device: Device to load on (auto-detected if None)
        trust_remote_code: Trust remote code in model
        attn_implementation: Attention implementation ("eager" required for
            attention output, "sdpa" for faster inference without attention)

    Returns:
        Tuple of (model, processor, device)
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    device = torch.device(device) if device else get_device()

    print(f"Loading {model_name} on {device}...")
    print(f"Using attention implementation: {attn_implementation}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=str(device),
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    print(f"Model loaded successfully")

    return model, processor, device


def create_explainer(
    model_name: str = "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device: Optional[str] = None,
    attn_implementation: str = "eager",
) -> MedGemmaExplainer:
    """
    Convenience function to create an explainer with default settings.

    Args:
        model_name: HuggingFace model name
        torch_dtype: Data type for model weights
        device: Device to load on
        attn_implementation: Attention implementation (must be "eager" for explainability)

    Returns:
        Configured MedGemmaExplainer
    """
    model, processor, device = load_medgemma(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        attn_implementation=attn_implementation,
    )

    return MedGemmaExplainer(model, processor, device=device)
