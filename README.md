# MedGemma Explainer

Implementation of the Chefer et al. (2021) gradient-weighted attention explainability method for MedGemma 1.5 4B vision-language model.

## Overview

This library generates **relevancy maps** that highlight which image regions and text tokens contribute to MedGemma's predictions. It implements the method from:

> Chefer, H., Gur, S., & Wolf, L. (2021). *Transformer Interpretability Beyond Attention Visualization.* CVPR 2021.

## Key Features

- **Gradient-weighted attention**: Combines attention patterns with gradient information
- **Token-specific explanations**: Explain why specific words were generated
- **Keyword search**: Automatically find and explain keywords in responses
- **Answer span explanation**: Explain the entire generated response
- **Medical imaging support**: Correct anatomical orientation for chest X-rays

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/medgemma-explainer.git
cd medgemma-explainer

# Install dependencies
pip install -r requirements.txt
```

Or in Google Colab:
```python
!git clone https://github.com/YOUR_USERNAME/medgemma-explainer.git
import sys
sys.path.insert(0, '/content/medgemma-explainer')

from medgemma_explainability import MedGemmaExplainer, load_medgemma
```

## Quick Start

```python
from medgemma_explainability import MedGemmaExplainer, load_medgemma

# Load model (must use eager attention)
model, processor, device = load_medgemma(
    "google/medgemma-1.5-4b-it",
    attn_implementation="eager"  # Required for attention output
)

# Create explainer
explainer = MedGemmaExplainer(model, processor, device=device)

# Generate explanation
result = explainer.explain(image, "What do you see in this image?")

# Or explain a specific keyword
result = explainer.explain_keyword(image, prompt, keyword="pneumonia")

# Or explain the entire answer
result = explainer.explain_answer_span(image, prompt)
```

## Critical Implementation Detail: Backprop Target

**This is the most important aspect of the implementation.**

For causal language models like MedGemma:

- **Logit at position i predicts token at position i+1**
- To explain why token at position `p` was generated:
  1. Backprop from logit at position `p-1`
  2. Use the **actual token id** at position `p` (not argmax)
  3. Extract relevancy from row `p-1` in the R matrix

### Why This Matters

A common mistake is to backprop from the last position using argmax:

```python
# WRONG - explains "what comes after the last token"
target_logit = logits[0, -1, logits[0, -1].argmax()]
```

The correct approach:

```python
# CORRECT - explains why token at position p was generated
logit_position = target_token_position - 1
target_token_id = input_ids[0, target_token_position]  # Actual token
target_logit = logits[0, logit_position, target_token_id]
# Extract from R[logit_position, :]
```

### Other Key Implementation Details

1. **Keep model in eval mode**: Use `torch.enable_grad()` context instead of `model.train()`
2. **Retain gradients**: Call `attn.requires_grad_(True)` and `attn.retain_grad()` on attention tensors
3. **Use eager attention**: MedGemma's default SDPA doesn't support `output_attentions=True`
4. **Convert to float32**: Attention tensors are bfloat16; convert for stable gradient computation

## Method: Chefer et al. Equations

**Equation 5: Gradient-Weighted Attention**
```
Ā = E_h[(∇A ⊙ A)⁺]
```

Where:
- `A` = attention weights
- `∇A` = gradient of loss w.r.t. attention
- `⊙` = element-wise multiplication
- `(·)⁺` = keep only positive values
- `E_h` = average over attention heads

**Equation 6: Relevancy Propagation**
```
R = Ā @ R
```

Starting with `R = I` (identity), propagate through each layer.

## MedGemma Architecture

- **Language Model**: 34 transformer layers
- **Attention**: 8 query heads, 4 KV heads (GQA)
- **Image Tokens**: 256 tokens (16×16 grid) at positions 6-261
- **Global Attention Layers**: 5, 11, 17, 23, 29 (5:1 local:global ratio)
- **Local Window**: 1024 tokens

### Token Structure

```
Position 0:       <bos>
Position 1:       <start_of_turn>
Position 2:       user
Position 3-5:     prefix tokens
Position 6-261:   256 IMAGE TOKENS (16×16 grid)
Position 262:     <end_of_image>
Position 263+:    Text prompt and generated response
```

## Medical Imaging Notes

For chest X-rays (PA view):
- **Left side of image = Patient's RIGHT side**
- **Right side of image = Patient's LEFT side**

The 16×16 relevancy grid maps to anatomical regions accordingly.

## API Reference

### MedGemmaExplainer

```python
class MedGemmaExplainer:
    def __init__(self, model, processor, device=None, ...):
        """Initialize explainer with MedGemma model."""

    def explain(self, image, prompt, target_token_position=None, ...):
        """Generate explanation for a specific token position."""

    def explain_keyword(self, image, prompt, keyword, ...):
        """Find and explain a keyword in the response."""

    def explain_answer_span(self, image, prompt, ...):
        """Explain the entire generated answer."""
```

### ExplanationResult

```python
@dataclass
class ExplanationResult:
    image_relevancy: np.ndarray      # 16x16 relevancy map
    text_relevancy: np.ndarray       # Text token relevancy scores
    token_labels: List[str]          # Decoded token strings
    generated_text: str              # Full generated response
    raw_image_relevancy: np.ndarray  # Raw attention baseline (optional)
    raw_text_relevancy: np.ndarray   # Raw attention baseline (optional)
    metadata: dict                   # Additional info
```

## File Structure

```
medgemma-explainer/
├── medgemma_explainability/  # Main package
│   ├── __init__.py           # Package exports
│   ├── explainer.py          # Main MedGemmaExplainer class
│   ├── relevancy.py          # Chefer method implementation
│   ├── visualization.py      # Plotting utilities
│   ├── attention_hooks.py    # Attention capture hooks
│   └── utils.py              # Helper functions
├── scripts/                  # Example scripts
├── notebooks/                # Tutorial notebooks
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

## Example Results

### Cat + Remote Control

When asked "Where is the remote control?", the model generates a response mentioning the remote. Explaining the "remote" token shows highest relevancy at the bottom-center of the image where the remote is located.

### Chest X-ray Pneumonia

When analyzing a chest X-ray with right middle lobe pneumonia, the relevancy map correctly highlights the patient's right lung field (left side of image).

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- Pillow
- matplotlib
- numpy

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{chefer2021transformer,
  title={Transformer interpretability beyond attention visualization},
  author={Chefer, Hila and Gur, Shir and Wolf, Lior},
  booktitle={CVPR},
  year={2021}
}
```

## License

This implementation is provided for research and educational purposes.
