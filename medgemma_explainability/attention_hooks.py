"""
Attention hooks for capturing attention weights and gradients.

This module provides hook-based mechanisms to extract attention weights
and their gradients from transformer models during forward and backward passes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class AttentionCapture:
    """
    Container for captured attention data from a single layer.
    """
    layer_idx: int
    attention_weights: Optional[torch.Tensor] = None
    attention_gradients: Optional[torch.Tensor] = None
    query: Optional[torch.Tensor] = None
    key: Optional[torch.Tensor] = None


class AttentionHookManager:
    """
    Manages forward and backward hooks for attention extraction.

    This class registers hooks on attention modules to capture:
    1. Attention weights during forward pass
    2. Gradients of attention weights during backward pass

    Supports both standard multi-head attention and grouped-query attention (GQA).
    """

    def __init__(
        self,
        num_query_heads: int = 8,
        num_kv_heads: int = 4,
        expand_gqa: bool = True,
    ):
        """
        Initialize the hook manager.

        Args:
            num_query_heads: Number of query heads in the model
            num_kv_heads: Number of key-value heads (for GQA)
            expand_gqa: Whether to expand KV heads to match query heads
        """
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.expand_gqa = expand_gqa

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.attention_cache: Dict[int, AttentionCapture] = {}

        # For storing tensors that need gradients
        self._attention_tensors: Dict[int, torch.Tensor] = {}

    def _expand_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """Expand GQA attention to full query head dimension."""
        if not self.expand_gqa or attention.shape[1] == self.num_query_heads:
            return attention

        repeat_factor = self.num_query_heads // self.num_kv_heads
        return attention.repeat_interleave(repeat_factor, dim=1)

    def create_forward_hook(self, layer_idx: int) -> Callable:
        """
        Create a forward hook for capturing attention weights.

        Args:
            layer_idx: Index of the layer

        Returns:
            Hook function
        """
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                # Standard format: (hidden_states, attention_weights, ...)
                if len(output) >= 2 and output[1] is not None:
                    attn_weights = output[1]
                else:
                    # Attention weights might not be returned
                    return
            else:
                return

            # Expand GQA if needed
            attn_weights = self._expand_attention(attn_weights)

            # Store and enable gradient tracking
            if attn_weights.requires_grad:
                attn_weights.retain_grad()

            self._attention_tensors[layer_idx] = attn_weights

            if layer_idx not in self.attention_cache:
                self.attention_cache[layer_idx] = AttentionCapture(layer_idx=layer_idx)

            self.attention_cache[layer_idx].attention_weights = attn_weights.detach().clone()

        return hook

    def create_backward_hook(self, layer_idx: int) -> Callable:
        """
        Create a backward hook for capturing attention gradients.

        Note: This uses a tensor hook on the attention weights themselves.
        """
        def hook(grad):
            if layer_idx in self.attention_cache:
                # Expand gradient to match expanded attention
                grad_expanded = self._expand_attention(grad)
                self.attention_cache[layer_idx].attention_gradients = grad_expanded.detach().clone()
            return grad

        return hook

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on all attention layers.

        Args:
            model: The transformer model
        """
        self.clear()

        # Find attention modules - try different paths for MedGemma
        attention_modules = self._find_attention_modules(model)

        for layer_idx, attn_module in enumerate(attention_modules):
            # Register forward hook
            forward_hook = self.create_forward_hook(layer_idx)
            handle = attn_module.register_forward_hook(forward_hook)
            self.hooks.append(handle)

            # Initialize cache
            self.attention_cache[layer_idx] = AttentionCapture(layer_idx=layer_idx)

        print(f"Registered hooks on {len(attention_modules)} attention layers")

    def _find_attention_modules(self, model: nn.Module) -> List[nn.Module]:
        """
        Find attention modules in the model.

        Handles different model architectures.
        """
        attention_modules = []

        # Try MedGemma/PaliGemma structure first
        # Path: model.language_model.model.layers[i].self_attn
        if hasattr(model, 'language_model'):
            lm = model.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                for layer in lm.model.layers:
                    if hasattr(layer, 'self_attn'):
                        attention_modules.append(layer.self_attn)

        # Try direct model.model.layers path
        if not attention_modules and hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                for layer in model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        attention_modules.append(layer.self_attn)

        # Try model.layers directly
        if not attention_modules and hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'self_attn'):
                    attention_modules.append(layer.self_attn)

        return attention_modules

    def register_gradient_hooks(self) -> None:
        """
        Register gradient hooks on captured attention tensors.

        Call this after forward pass but before backward pass.
        """
        for layer_idx, attn_tensor in self._attention_tensors.items():
            if attn_tensor.requires_grad:
                hook = self.create_backward_hook(layer_idx)
                attn_tensor.register_hook(hook)

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Get all captured attention weights."""
        return {
            idx: capture.attention_weights
            for idx, capture in self.attention_cache.items()
            if capture.attention_weights is not None
        }

    def get_attention_gradients(self) -> Dict[int, torch.Tensor]:
        """Get all captured attention gradients."""
        return {
            idx: capture.attention_gradients
            for idx, capture in self.attention_cache.items()
            if capture.attention_gradients is not None
        }

    def clear(self) -> None:
        """Remove all hooks and clear caches."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.attention_cache = {}
        self._attention_tensors = {}

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.clear()


class ManualAttentionComputer:
    """
    Compute attention weights manually from Q, K projections.

    Use this as a fallback when output_attentions doesn't work.
    """

    def __init__(
        self,
        num_query_heads: int = 8,
        num_kv_heads: int = 4,
        head_dim: int = 256,
    ):
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.qk_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def create_qk_hook(self, layer_idx: int, proj_type: str) -> Callable:
        """Create hook to capture Q or K projections."""
        def hook(module, input, output):
            if layer_idx not in self.qk_cache:
                self.qk_cache[layer_idx] = {}
            self.qk_cache[layer_idx][proj_type] = output.detach().clone()
        return hook

    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks on Q and K projection layers."""
        self.clear()

        # Find Q and K projections
        layer_idx = 0
        for name, module in model.named_modules():
            if 'q_proj' in name:
                hook = self.create_qk_hook(layer_idx, 'query')
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
            elif 'k_proj' in name:
                hook = self.create_qk_hook(layer_idx, 'key')
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
                layer_idx += 1

    def compute_attention_from_qk(
        self,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute attention weights from cached Q and K.

        Args:
            layer_idx: Layer index
            attention_mask: Optional attention mask

        Returns:
            Attention weights of shape (batch, heads, seq, seq)
        """
        if layer_idx not in self.qk_cache:
            return None

        cache = self.qk_cache[layer_idx]
        if 'query' not in cache or 'key' not in cache:
            return None

        query = cache['query']
        key = cache['key']

        # Reshape to (batch, heads, seq, head_dim)
        batch_size, seq_len, _ = query.shape

        query = query.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        query = query.transpose(1, 2)  # (batch, heads, seq, head_dim)

        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        key = key.transpose(1, 2)

        # Expand KV heads to match query heads
        repeat_factor = self.num_query_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)

        return attn_weights

    def clear(self) -> None:
        """Remove hooks and clear cache."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.qk_cache = {}


def extract_attention_with_output_attentions(
    model: nn.Module,
    inputs: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Extract attention weights using output_attentions=True.

    This is the simplest approach but may not work with all models.

    Args:
        model: The model
        inputs: Input dictionary
        device: Device

    Returns:
        Tuple of (logits, list of attention tensors per layer)
    """
    # Enable attention output
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

    return outputs.logits, attentions


def verify_attention_extraction(
    attention_weights: Dict[int, torch.Tensor],
    expected_layers: int = 34,
    expected_heads: int = 8,
) -> dict:
    """
    Verify that attention extraction worked correctly.

    Args:
        attention_weights: Dictionary of layer_idx -> attention tensor
        expected_layers: Expected number of layers
        expected_heads: Expected number of attention heads

    Returns:
        Dictionary with verification results
    """
    results = {
        "num_layers_extracted": len(attention_weights),
        "expected_layers": expected_layers,
        "all_layers_present": len(attention_weights) == expected_layers,
        "shapes": {},
        "softmax_valid": {},
        "issues": [],
    }

    for layer_idx, attn in attention_weights.items():
        shape = tuple(attn.shape)
        results["shapes"][layer_idx] = shape

        # Check shape
        if len(shape) != 4:
            results["issues"].append(f"Layer {layer_idx}: Expected 4D tensor, got {len(shape)}D")
            continue

        # Check heads
        if shape[1] != expected_heads:
            results["issues"].append(
                f"Layer {layer_idx}: Expected {expected_heads} heads, got {shape[1]}"
            )

        # Check softmax (sum to 1 along last dim)
        row_sums = attn.sum(dim=-1)
        is_valid = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
        results["softmax_valid"][layer_idx] = is_valid

        if not is_valid:
            results["issues"].append(
                f"Layer {layer_idx}: Attention doesn't sum to 1 (softmax issue)"
            )

    results["success"] = len(results["issues"]) == 0

    return results
