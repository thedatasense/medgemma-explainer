"""
Relevancy propagation implementation for Chefer et al. method.

This module implements the core equations from:
    Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability
    Beyond Attention Visualization. CVPR 2021.

Key equations:
    - Equation 5: Ā = E_h((∇A ⊙ A)^+)
    - Equation 6: R = R + Ā @ R
"""

import torch
from typing import Dict, List, Optional, Tuple
from .utils import is_global_layer, create_local_attention_mask


def compute_Abar(
    attention: torch.Tensor,
    attention_grad: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute gradient-weighted attention (Equation 5 from Chefer et al.).

    Ā = E_h((∇A ⊙ A)^+)

    Where:
        - A is the attention weights
        - ∇A is the gradient of the loss w.r.t. attention
        - ⊙ is element-wise multiplication
        - (·)^+ denotes keeping only positive values
        - E_h denotes averaging over attention heads

    Args:
        attention: Attention weights, shape (batch, heads, seq, seq)
        attention_grad: Attention gradients, shape (batch, heads, seq, seq)
        normalize: Whether to normalize the result

    Returns:
        Gradient-weighted attention matrix, shape (seq, seq)
    """
    # Element-wise product: ∇A ⊙ A
    weighted = attention * attention_grad

    # Keep only positive values: (·)^+
    weighted = torch.clamp(weighted, min=0)

    # Average over batch and heads: E_h
    Abar = weighted.mean(dim=(0, 1))

    # Optional normalization to stabilize propagation
    if normalize and Abar.sum() > 0:
        Abar = Abar / Abar.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return Abar


def propagate_relevancy(
    attention_maps: Dict[int, torch.Tensor],
    attention_grads: Dict[int, torch.Tensor],
    seq_len: int,
    handle_local_attention: bool = True,
    local_window_size: int = 1024,
    normalize_Abar: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Propagate relevancy through all transformer layers (Equation 6).

    R = R + Ā @ R

    Starting with R = I (identity matrix), we iteratively update R
    for each layer from first to last.

    Args:
        attention_maps: Dict mapping layer_idx to attention tensor
        attention_grads: Dict mapping layer_idx to gradient tensor
        seq_len: Sequence length
        handle_local_attention: Whether to apply local attention masking
        local_window_size: Window size for local attention layers
        normalize_Abar: Whether to normalize Ā for each layer
        device: Device to create tensors on

    Returns:
        Final relevancy matrix R, shape (seq_len, seq_len)
    """
    if device is None:
        # Get device from first attention tensor
        first_attn = next(iter(attention_maps.values()))
        device = first_attn.device

    # Initialize R as identity matrix
    R = torch.eye(seq_len, device=device, dtype=torch.float32)

    # Sort layers to process in order
    layer_indices = sorted(attention_maps.keys())

    for layer_idx in layer_indices:
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx].float()  # Convert to float32
        grad_A = attention_grads[layer_idx].float()  # Convert to float32

        # Compute gradient-weighted attention
        Abar = compute_Abar(A, grad_A, normalize=normalize_Abar)

        # Handle local attention if needed
        if handle_local_attention and not is_global_layer(layer_idx):
            local_mask = create_local_attention_mask(
                seq_len, local_window_size, device=device
            )
            Abar = Abar * local_mask

        # Add identity to Abar (to preserve self-attention)
        Abar = Abar + torch.eye(seq_len, device=device, dtype=torch.float32)

        # Normalize rows to sum to 1 (optional but helps stability)
        Abar = Abar / Abar.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Update relevancy: R = R + Ā @ R
        # Note: Original paper uses R = Ā @ R, but we use R = R + Ā @ R
        # to accumulate relevancy across layers
        R = torch.matmul(Abar, R)

    return R


def propagate_relevancy_additive(
    attention_maps: Dict[int, torch.Tensor],
    attention_grads: Dict[int, torch.Tensor],
    seq_len: int,
    handle_local_attention: bool = True,
    local_window_size: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Alternative relevancy propagation with additive update.

    Uses: R = R + Ā @ R

    This variant accumulates relevancy additively, which can provide
    different interpretations of token importance.

    Args:
        attention_maps: Dict mapping layer_idx to attention tensor
        attention_grads: Dict mapping layer_idx to gradient tensor
        seq_len: Sequence length
        handle_local_attention: Whether to apply local attention masking
        local_window_size: Window size for local attention layers
        device: Device to create tensors on

    Returns:
        Final relevancy matrix R, shape (seq_len, seq_len)
    """
    if device is None:
        first_attn = next(iter(attention_maps.values()))
        device = first_attn.device

    # Initialize R as identity matrix
    R = torch.eye(seq_len, device=device, dtype=torch.float32)

    layer_indices = sorted(attention_maps.keys())

    for layer_idx in layer_indices:
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx].float()  # Convert to float32
        grad_A = attention_grads[layer_idx].float()  # Convert to float32

        # Compute gradient-weighted attention (without normalization for additive)
        Abar = compute_Abar(A, grad_A, normalize=False)

        # Handle local attention
        if handle_local_attention and not is_global_layer(layer_idx):
            local_mask = create_local_attention_mask(
                seq_len, local_window_size, device=device
            )
            Abar = Abar * local_mask

        # Additive update: R = R + Ā @ R
        R = R + torch.matmul(Abar, R)

    return R


def extract_token_relevancy(
    R: torch.Tensor,
    target_token_idx: int = -1,
) -> torch.Tensor:
    """
    Extract relevancy scores for input tokens w.r.t. a target token.

    Args:
        R: Relevancy matrix of shape (seq_len, seq_len)
        target_token_idx: Index of the target token (-1 for last token)

    Returns:
        1D relevancy vector for all input tokens
    """
    if target_token_idx == -1:
        target_token_idx = R.shape[0] - 1

    return R[target_token_idx, :]


def split_relevancy(
    relevancy: torch.Tensor,
    num_image_tokens: int = 256,
    image_start_idx: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split relevancy into image and text components.

    MedGemma token structure:
    - Positions 0-5: Prefix tokens (<bos>, <start_of_turn>, user, etc., <start_of_image>)
    - Positions 6-261: 256 image soft tokens
    - Position 262: <end_of_image>
    - Positions 263+: Text tokens (prompt and generated)

    Args:
        relevancy: 1D relevancy tensor
        num_image_tokens: Number of image tokens (256 for MedGemma)
        image_start_idx: Index where image tokens start (6 for MedGemma)

    Returns:
        Tuple of (image_relevancy, text_relevancy)
    """
    image_end_idx = image_start_idx + num_image_tokens

    image_relevancy = relevancy[image_start_idx:image_end_idx]
    text_relevancy = relevancy[image_end_idx:]

    return image_relevancy, text_relevancy


def compute_raw_attention_relevancy(
    attention_maps: Dict[int, torch.Tensor],
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Compute baseline relevancy using raw attention (without gradients).

    This serves as a comparison baseline for the Chefer method.

    Args:
        attention_maps: Dict mapping layer_idx to attention tensor
        layer_idx: Which layer to use (-1 for last layer)

    Returns:
        Attention-based relevancy matrix
    """
    if layer_idx == -1:
        layer_idx = max(attention_maps.keys())

    attention = attention_maps[layer_idx].float()  # Convert to float32

    # Average over batch and heads
    avg_attention = attention.mean(dim=(0, 1))

    return avg_attention


def aggregate_head_relevancy(
    attention: torch.Tensor,
    attention_grad: torch.Tensor,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Aggregate relevancy across attention heads with different strategies.

    Args:
        attention: Attention weights (batch, heads, seq, seq)
        attention_grad: Attention gradients (batch, heads, seq, seq)
        aggregation: Aggregation method ("mean", "max", "gradient_weighted")

    Returns:
        Aggregated attention matrix (seq, seq)
    """
    # Compute gradient-weighted attention per head
    weighted = attention * attention_grad
    weighted = torch.clamp(weighted, min=0)

    if aggregation == "mean":
        return weighted.mean(dim=(0, 1))
    elif aggregation == "max":
        return weighted.amax(dim=(0, 1))
    elif aggregation == "gradient_weighted":
        # Weight heads by their gradient magnitude
        head_importance = attention_grad.abs().mean(dim=(0, 2, 3))
        head_importance = head_importance / head_importance.sum()
        weighted_avg = (weighted[0] * head_importance.view(-1, 1, 1)).sum(dim=0)
        return weighted_avg
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_direct_token_attention(
    attention_maps: Dict[int, torch.Tensor],
    token_idx: int,
    num_image_tokens: int = 256,
    image_start_idx: int = 6,
    layers: str = "global",
    layer_list: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compute direct attention from a specific token to image patches.

    This is an improved method that bypasses gradient propagation and directly
    measures what image regions a response token attends to. Works better for
    object localization tasks.

    Args:
        attention_maps: Dict mapping layer_idx to attention tensor
        token_idx: Index of the token to analyze (e.g., the "remote" token)
        num_image_tokens: Number of image tokens (256 for MedGemma)
        image_start_idx: Index where image tokens start (6 for MedGemma)
        layers: Which layers to use ("global", "all", "last", or "custom")
        layer_list: Specific layer indices if layers="custom"

    Returns:
        1D tensor of attention weights to image tokens (256,)
    """
    from .utils import is_global_layer

    # Determine which layers to use
    if layers == "global":
        layer_indices = [i for i in attention_maps.keys() if is_global_layer(i)]
    elif layers == "all":
        layer_indices = list(attention_maps.keys())
    elif layers == "last":
        layer_indices = [max(attention_maps.keys())]
    elif layers == "custom" and layer_list is not None:
        layer_indices = layer_list
    else:
        layer_indices = list(attention_maps.keys())

    device = next(iter(attention_maps.values())).device

    # Aggregate attention across selected layers
    aggregated = torch.zeros(num_image_tokens, device=device, dtype=torch.float32)

    for layer_idx in layer_indices:
        attn = attention_maps[layer_idx].float()  # [batch, heads, seq, seq]
        # Get attention from token_idx to image tokens
        token_to_image_attn = attn[0, :, token_idx, image_start_idx:image_start_idx + num_image_tokens]
        # Average over heads
        token_to_image_attn = token_to_image_attn.mean(dim=0)
        aggregated += token_to_image_attn

    # Average across layers
    aggregated = aggregated / len(layer_indices)

    return aggregated


def compute_object_attention(
    attention_maps: Dict[int, torch.Tensor],
    token_indices: List[int],
    num_image_tokens: int = 256,
    image_start_idx: int = 6,
    layers: str = "global",
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Compute aggregated attention from multiple tokens to image patches.

    Useful for aggregating attention from all tokens that mention an object
    (e.g., all tokens containing "remote", "control", etc.)

    Args:
        attention_maps: Dict mapping layer_idx to attention tensor
        token_indices: List of token indices to aggregate
        num_image_tokens: Number of image tokens (256)
        image_start_idx: Index where image tokens start (6)
        layers: Which layers to use ("global", "all", "last")
        aggregation: How to aggregate across tokens ("mean", "max", "sum")

    Returns:
        1D tensor of attention weights to image tokens (256,)
    """
    device = next(iter(attention_maps.values())).device
    all_attentions = []

    for token_idx in token_indices:
        attn = compute_direct_token_attention(
            attention_maps, token_idx, num_image_tokens, image_start_idx, layers
        )
        all_attentions.append(attn)

    # Stack and aggregate
    stacked = torch.stack(all_attentions, dim=0)

    if aggregation == "mean":
        result = stacked.mean(dim=0)
    elif aggregation == "max":
        result = stacked.max(dim=0)[0]
    elif aggregation == "sum":
        result = stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return result


def visualize_relevancy_flow(
    attention_maps: Dict[int, torch.Tensor],
    attention_grads: Dict[int, torch.Tensor],
    seq_len: int,
    layers_to_show: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute relevancy at each layer for visualization.

    This allows visualizing how relevancy builds up across layers.

    Args:
        attention_maps: Dict mapping layer_idx to attention tensor
        attention_grads: Dict mapping layer_idx to gradient tensor
        seq_len: Sequence length
        layers_to_show: Specific layers to return (None for all)

    Returns:
        Dict mapping layer_idx to relevancy matrix at that point
    """
    device = next(iter(attention_maps.values())).device

    R = torch.eye(seq_len, device=device, dtype=torch.float32)
    relevancy_per_layer = {}

    layer_indices = sorted(attention_maps.keys())

    for layer_idx in layer_indices:
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx]
        grad_A = attention_grads[layer_idx]

        Abar = compute_Abar(A, grad_A, normalize=True)
        Abar = Abar + torch.eye(seq_len, device=device, dtype=Abar.dtype)
        Abar = Abar / Abar.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        R = torch.matmul(Abar, R)

        if layers_to_show is None or layer_idx in layers_to_show:
            relevancy_per_layer[layer_idx] = R.clone()

    return relevancy_per_layer
