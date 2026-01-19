"""
MedGemma Explainability - Chefer et al. Implementation

This package implements the gradient-weighted attention explainability method
from Chefer et al. (2021) for MedGemma 1.5 4B vision-language model.

Reference:
    Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability
    Beyond Attention Visualization. CVPR 2021.

Key insight for causal LM explainability:
    - Logit at position i predicts token at i+1
    - To explain token at position p, backprop from logit at p-1
    - Use actual token id, not argmax
    - Keep model in eval mode with torch.enable_grad() context
"""

from .explainer import MedGemmaExplainer, load_medgemma, create_explainer
from .relevancy import (
    compute_Abar,
    propagate_relevancy,
    extract_token_relevancy,
    split_relevancy,
    compute_direct_token_attention,
    compute_object_attention,
)
from .visualization import visualize_explanation
from .utils import normalize_relevancy, reshape_image_relevancy

__version__ = "0.3.0"
__all__ = [
    "MedGemmaExplainer",
    "load_medgemma",
    "create_explainer",
    "compute_Abar",
    "propagate_relevancy",
    "extract_token_relevancy",
    "split_relevancy",
    "compute_direct_token_attention",
    "compute_object_attention",
    "visualize_explanation",
    "normalize_relevancy",
    "reshape_image_relevancy",
]
