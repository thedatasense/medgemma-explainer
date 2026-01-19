"""
Utility functions for MedGemma Explainability.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image
import requests
from io import BytesIO


def load_image(image_source: str) -> Image.Image:
    """
    Load an image from a file path or URL.

    Args:
        image_source: Local file path or URL

    Returns:
        PIL Image in RGB format
    """
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_source)

    return image.convert("RGB")


def is_global_layer(layer_idx: int) -> bool:
    """
    Check if a layer uses global attention in Gemma3.

    Gemma3 uses a 5:1 local:global ratio.
    Global layers are at indices 5, 11, 17, 23, 29 (0-indexed).

    Args:
        layer_idx: Zero-indexed layer number

    Returns:
        True if this layer uses global attention
    """
    return (layer_idx + 1) % 6 == 0


def get_attention_layers_info(num_layers: int = 34) -> dict:
    """
    Get information about attention layers.

    Returns:
        Dict with layer indices categorized by attention type
    """
    global_layers = [i for i in range(num_layers) if is_global_layer(i)]
    local_layers = [i for i in range(num_layers) if not is_global_layer(i)]

    return {
        "total_layers": num_layers,
        "global_layers": global_layers,
        "local_layers": local_layers,
        "num_global": len(global_layers),
        "num_local": len(local_layers),
    }


def expand_gqa_attention(
    attention: torch.Tensor,
    num_query_heads: int = 8,
    num_kv_heads: int = 4,
) -> torch.Tensor:
    """
    Expand Grouped-Query Attention weights to full query head dimension.

    In GQA, multiple query heads share the same key-value heads.
    This function expands the KV heads to match query heads.

    Args:
        attention: Tensor of shape (batch, kv_heads, seq, seq)
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads

    Returns:
        Tensor of shape (batch, query_heads, seq, seq)
    """
    if attention.shape[1] == num_query_heads:
        return attention

    repeat_factor = num_query_heads // num_kv_heads
    return attention.repeat_interleave(repeat_factor, dim=1)


def create_local_attention_mask(
    seq_len: int,
    window_size: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a mask for local attention (sliding window).

    Args:
        seq_len: Sequence length
        window_size: Size of the local attention window
        device: Device to create tensor on

    Returns:
        Binary mask tensor of shape (seq_len, seq_len)
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1
    return mask


def split_relevancy_by_modality(
    relevancy: torch.Tensor,
    num_image_tokens: int = 256,
    image_start_idx: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split relevancy scores into image and text components.

    MedGemma token structure:
    - Positions 0-5: Prefix tokens (<bos>, <start_of_turn>, user, etc., <start_of_image>)
    - Positions 6-261: 256 image soft tokens
    - Position 262: <end_of_image>
    - Positions 263+: Text tokens (prompt and generated)

    Args:
        relevancy: 1D relevancy tensor for all tokens
        num_image_tokens: Number of image tokens (default 256 for MedGemma)
        image_start_idx: Index where image tokens start (default 6 for MedGemma)

    Returns:
        Tuple of (image_relevancy, text_relevancy)
    """
    image_end_idx = image_start_idx + num_image_tokens
    image_relevancy = relevancy[image_start_idx:image_end_idx]
    text_relevancy = relevancy[image_end_idx:]
    return image_relevancy, text_relevancy


def reshape_image_relevancy(
    image_relevancy: torch.Tensor,
    grid_size: int = 16,
) -> torch.Tensor:
    """
    Reshape flat image relevancy to 2D grid.

    MedGemma uses 256 image tokens arranged in a 16x16 grid.

    Args:
        image_relevancy: 1D tensor of shape (256,)
        grid_size: Size of the square grid (default 16)

    Returns:
        2D tensor of shape (16, 16)
    """
    return image_relevancy.reshape(grid_size, grid_size)


def normalize_relevancy(relevancy: torch.Tensor) -> torch.Tensor:
    """
    Normalize relevancy scores to [0, 1] range.

    Args:
        relevancy: Relevancy tensor

    Returns:
        Normalized tensor
    """
    min_val = relevancy.min()
    max_val = relevancy.max()

    if max_val - min_val < 1e-8:
        return torch.zeros_like(relevancy)

    return (relevancy - min_val) / (max_val - min_val)


def print_model_architecture(model, max_depth: int = 3) -> None:
    """
    Print model architecture with limited depth.

    Args:
        model: PyTorch model
        max_depth: Maximum depth to print
    """
    def _print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return

        for name, child in module.named_children():
            print(f"{prefix}{name}: {child.__class__.__name__}")
            _print_module(child, prefix + "  ", depth + 1)

    print(f"Model: {model.__class__.__name__}")
    _print_module(model)


def get_device() -> torch.device:
    """
    Get the best available device.

    Returns:
        torch.device for CUDA if available, else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_gpu_memory():
    """
    Clear GPU memory cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
