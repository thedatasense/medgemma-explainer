"""
Visualization functions for explainability results.

This module provides functions to visualize:
- Image relevancy heatmaps
- Text token relevancy charts
- Comparison between raw attention and Chefer method
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class ExplanationResult:
    """
    Container for explanation results.
    """
    image_relevancy: np.ndarray      # (16, 16) grid
    text_relevancy: np.ndarray       # (num_tokens,)
    token_labels: List[str]          # Decoded tokens
    generated_text: str              # Full generated response
    raw_image_relevancy: Optional[np.ndarray] = None  # Raw attention baseline
    raw_text_relevancy: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


def visualize_explanation(
    image: Union[Image.Image, np.ndarray],
    result: ExplanationResult,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = "jet",
    alpha: float = 0.5,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive visualization of the explanation.

    Creates a figure with three panels:
    1. Original image
    2. Image with relevancy heatmap overlay
    3. Text token relevancy bar chart

    Args:
        image: Original input image
        result: ExplanationResult from the explainer
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap
        alpha: Transparency for heatmap overlay
        show_colorbar: Whether to show colorbar for heatmap

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Panel 1: Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2: Image with heatmap overlay
    _plot_image_heatmap(
        axes[1],
        image_np,
        result.image_relevancy,
        title="Image Relevancy",
        cmap=cmap,
        alpha=alpha,
        show_colorbar=show_colorbar,
    )

    # Panel 3: Text token relevancy
    _plot_text_relevancy(
        axes[2],
        result.text_relevancy,
        result.token_labels,
        title="Text Token Relevancy",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    return fig


def _plot_image_heatmap(
    ax: plt.Axes,
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Heatmap",
    cmap: str = "jet",
    alpha: float = 0.5,
    show_colorbar: bool = True,
) -> None:
    """
    Plot image with heatmap overlay.

    Args:
        ax: Matplotlib axes
        image: Original image (H, W, 3)
        heatmap: Relevancy heatmap (16, 16)
        title: Plot title
        cmap: Colormap
        alpha: Heatmap transparency
        show_colorbar: Whether to show colorbar
    """
    # Display original image
    ax.imshow(image)

    # Resize heatmap to image size
    h, w = image.shape[:2]
    heatmap_resized = _resize_heatmap(heatmap, (h, w))

    # Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
        heatmap_resized.max() - heatmap_resized.min() + 1e-8
    )

    # Overlay heatmap
    im = ax.imshow(heatmap_norm, cmap=cmap, alpha=alpha)

    ax.set_title(title)
    ax.axis("off")

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _resize_heatmap(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize heatmap to target size using bilinear interpolation.

    Args:
        heatmap: Input heatmap (e.g., 16x16)
        target_size: Target (height, width)

    Returns:
        Resized heatmap
    """
    from PIL import Image as PILImage

    # Convert to PIL for resizing
    heatmap_pil = PILImage.fromarray(heatmap.astype(np.float32))
    heatmap_resized = heatmap_pil.resize(
        (target_size[1], target_size[0]),
        PILImage.Resampling.BILINEAR
    )
    return np.array(heatmap_resized)


def _plot_text_relevancy(
    ax: plt.Axes,
    relevancy: np.ndarray,
    token_labels: List[str],
    title: str = "Text Relevancy",
    max_tokens: int = 30,
    color: str = "steelblue",
) -> None:
    """
    Plot text token relevancy as a horizontal bar chart.

    Args:
        ax: Matplotlib axes
        relevancy: Relevancy scores per token
        token_labels: Token labels
        title: Plot title
        max_tokens: Maximum tokens to display
        color: Bar color
    """
    # Limit number of tokens
    n_tokens = min(len(relevancy), max_tokens)
    relevancy = relevancy[:n_tokens]
    labels = token_labels[:n_tokens]

    # Clean up labels for display
    labels = [_clean_token_label(label) for label in labels]

    # Create bar chart
    y_pos = np.arange(n_tokens)
    ax.barh(y_pos, relevancy, color=color, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # Highest relevancy at top
    ax.set_xlabel("Relevancy Score")
    ax.set_title(title)

    # Add value labels
    for i, v in enumerate(relevancy):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)


def _clean_token_label(label: str) -> str:
    """Clean up token label for display."""
    # Handle special tokens
    label = label.replace("▁", " ")  # SentencePiece space
    label = label.replace("<", "[").replace(">", "]")  # Special tokens
    label = label.strip()

    # Truncate long labels
    if len(label) > 15:
        label = label[:12] + "..."

    return label if label else "[space]"


def visualize_comparison(
    image: Union[Image.Image, np.ndarray],
    result: ExplanationResult,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
    cmap: str = "jet",
) -> plt.Figure:
    """
    Create comparison visualization between raw attention and Chefer method.

    Creates a 2x3 figure:
    - Top row: Original, Raw attention heatmap, Raw text relevancy
    - Bottom row: Original, Chefer heatmap, Chefer text relevancy

    Args:
        image: Original input image
        result: ExplanationResult with both raw and Chefer results
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap for heatmaps

    Returns:
        matplotlib Figure object
    """
    if result.raw_image_relevancy is None:
        print("Warning: Raw attention not available, showing only Chefer method")
        return visualize_explanation(image, result, save_path, figsize[:1] + (5,), cmap)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Top row: Raw attention
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    _plot_image_heatmap(
        axes[0, 1],
        image_np,
        result.raw_image_relevancy,
        title="Raw Attention (Last Layer)",
        cmap=cmap,
    )

    _plot_text_relevancy(
        axes[0, 2],
        result.raw_text_relevancy,
        result.token_labels,
        title="Raw Attention - Text",
    )

    # Bottom row: Chefer method
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    _plot_image_heatmap(
        axes[1, 1],
        image_np,
        result.image_relevancy,
        title="Chefer Method (All Layers)",
        cmap=cmap,
    )

    _plot_text_relevancy(
        axes[1, 2],
        result.text_relevancy,
        result.token_labels,
        title="Chefer Method - Text",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison to {save_path}")

    return fig


def visualize_layer_progression(
    image: Union[Image.Image, np.ndarray],
    layer_relevancies: dict,
    num_image_tokens: int = 256,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "jet",
) -> plt.Figure:
    """
    Visualize how relevancy changes across layers.

    Args:
        image: Original input image
        layer_relevancies: Dict mapping layer_idx to relevancy matrix
        num_image_tokens: Number of image tokens
        save_path: Optional path to save figure
        figsize: Figure size (auto-computed if None)
        cmap: Colormap

    Returns:
        matplotlib Figure object
    """
    n_layers = len(layer_relevancies)
    cols = min(6, n_layers)
    rows = (n_layers + cols - 1) // cols

    if figsize is None:
        figsize = (3 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_layers > 1 else [axes]

    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    for idx, (layer_idx, relevancy) in enumerate(sorted(layer_relevancies.items())):
        if idx >= len(axes):
            break

        # Extract image relevancy from last token
        token_relevancy = relevancy[-1, :num_image_tokens]
        image_relevancy = token_relevancy.cpu().numpy().reshape(16, 16)

        _plot_image_heatmap(
            axes[idx],
            image_np,
            image_relevancy,
            title=f"Layer {layer_idx}",
            cmap=cmap,
            show_colorbar=False,
        )

    # Hide unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Relevancy Progression Across Layers", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved layer progression to {save_path}")

    return fig


def create_heatmap_overlay(
    image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> Image.Image:
    """
    Create a PIL Image with heatmap overlay.

    Args:
        image: Original image
        heatmap: Relevancy heatmap (16x16)
        alpha: Overlay transparency
        cmap: Colormap name

    Returns:
        PIL Image with overlay
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize heatmap to image size
    h, w = image.size[1], image.size[0]
    heatmap_resized = _resize_heatmap(heatmap, (h, w))

    # Normalize
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
        heatmap_resized.max() - heatmap_resized.min() + 1e-8
    )

    # Apply colormap
    colormap = plt.cm.get_cmap(cmap)
    heatmap_colored = colormap(heatmap_norm)[:, :, :3]  # Remove alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored)

    # Blend images
    blended = Image.blend(image.convert("RGB"), heatmap_pil, alpha)

    return blended


def plot_attention_heads(
    attention: torch.Tensor,
    layer_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Visualize attention patterns for all heads in a layer.

    Args:
        attention: Attention tensor (batch, heads, seq, seq)
        layer_idx: Layer index (for title)
        save_path: Optional save path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_heads = attention.shape[1]
    cols = 4
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    attention = attention[0].cpu().numpy()  # Remove batch dim

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        im = ax.imshow(attention[head_idx], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {head_idx}")
        ax.axis("off")

    # Hide unused
    for idx in range(n_heads, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Attention Patterns - Layer {layer_idx}", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_relevancy_stats(result: ExplanationResult, filepath: str) -> None:
    """
    Save relevancy statistics to a text file.

    Args:
        result: ExplanationResult
        filepath: Path to save stats
    """
    with open(filepath, "w") as f:
        f.write("Relevancy Statistics\n")
        f.write("=" * 50 + "\n\n")

        f.write("Image Relevancy:\n")
        f.write(f"  Shape: {result.image_relevancy.shape}\n")
        f.write(f"  Min: {result.image_relevancy.min():.6f}\n")
        f.write(f"  Max: {result.image_relevancy.max():.6f}\n")
        f.write(f"  Mean: {result.image_relevancy.mean():.6f}\n")
        f.write(f"  Std: {result.image_relevancy.std():.6f}\n\n")

        f.write("Text Relevancy:\n")
        f.write(f"  Num tokens: {len(result.text_relevancy)}\n")
        f.write(f"  Min: {result.text_relevancy.min():.6f}\n")
        f.write(f"  Max: {result.text_relevancy.max():.6f}\n")
        f.write(f"  Mean: {result.text_relevancy.mean():.6f}\n\n")

        f.write("Top 10 Text Tokens by Relevancy:\n")
        sorted_indices = np.argsort(result.text_relevancy)[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            f.write(f"  {i+1}. '{result.token_labels[idx]}': {result.text_relevancy[idx]:.6f}\n")

        f.write(f"\nGenerated Text: {result.generated_text}\n")

    print(f"Saved stats to {filepath}")
