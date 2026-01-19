#!/usr/bin/env python3
"""
Script to verify MedGemma model loading and explore architecture.

This script:
1. Loads MedGemma 1.5 4B instruction-tuned model
2. Prints the model architecture
3. Runs a test inference
4. Identifies attention modules for hook registration
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(size=(224, 224)):
    """Create a simple test image with colored regions."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    # Left half red, right half blue
    img[:, :size[1]//2, 0] = 255  # Red on left
    img[:, size[1]//2:, 2] = 255  # Blue on right
    return Image.fromarray(img)


def print_model_structure(model, max_depth=4):
    """Print model structure with limited depth."""
    def _print_children(module, prefix="", depth=0):
        if depth >= max_depth:
            return
        children = list(module.named_children())
        for i, (name, child) in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{name}: {child.__class__.__name__}")
            extension = "    " if is_last else "│   "
            _print_children(child, prefix + extension, depth + 1)

    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    _print_children(model)


def find_attention_modules(model):
    """Find all attention modules in the model."""
    attention_modules = []

    for name, module in model.named_modules():
        module_name = module.__class__.__name__.lower()
        if 'attention' in module_name or 'attn' in name.lower():
            attention_modules.append({
                'name': name,
                'class': module.__class__.__name__,
                'module': module
            })

    return attention_modules


def analyze_attention_module(module, name):
    """Analyze an attention module's structure."""
    print(f"\n  Attention module: {name}")
    print(f"  Class: {module.__class__.__name__}")

    # Check for common attributes
    attrs = ['num_heads', 'head_dim', 'num_key_value_heads', 'hidden_size']
    for attr in attrs:
        if hasattr(module, attr):
            print(f"  {attr}: {getattr(module, attr)}")

    # List child modules (projections)
    children = list(module.named_children())
    if children:
        print("  Submodules:")
        for child_name, child in children:
            if hasattr(child, 'weight'):
                print(f"    - {child_name}: {child.__class__.__name__} {tuple(child.weight.shape)}")
            else:
                print(f"    - {child_name}: {child.__class__.__name__}")


def main():
    print("MedGemma Model Loading and Architecture Exploration")
    print("=" * 60)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Import transformers
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print("\nTransformers imported successfully")
    except ImportError as e:
        print(f"Error importing transformers: {e}")
        print("Install with: pip install transformers")
        return

    # Model name
    model_name = "google/medgemma-1.5-4b-it"

    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes...")

    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print("Processor loaded")

        # Load model in bfloat16 to save memory
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        print("Model loaded")

    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you're logged in to HuggingFace: huggingface-cli login")
        print("2. Accept model license at: https://huggingface.co/google/medgemma-1.5-4b-it")
        print("3. Check you have enough GPU memory (need ~8GB for bfloat16)")
        return

    # Print model structure
    print_model_structure(model)

    # Find attention modules
    print("\n" + "=" * 60)
    print("Finding Attention Modules")
    print("=" * 60)

    attention_modules = find_attention_modules(model)
    print(f"\nFound {len(attention_modules)} attention-related modules")

    # Group by type
    language_attn = [m for m in attention_modules if 'language' in m['name'].lower()]
    vision_attn = [m for m in attention_modules if 'vision' in m['name'].lower()]
    other_attn = [m for m in attention_modules if m not in language_attn and m not in vision_attn]

    print(f"\nLanguage model attention modules: {len(language_attn)}")
    print(f"Vision model attention modules: {len(vision_attn)}")
    print(f"Other attention modules: {len(other_attn)}")

    # Analyze first language attention module
    if language_attn:
        print("\n" + "-" * 40)
        print("Sample Language Attention Module:")
        analyze_attention_module(language_attn[0]['module'], language_attn[0]['name'])

    # Analyze first vision attention module
    if vision_attn:
        print("\n" + "-" * 40)
        print("Sample Vision Attention Module:")
        analyze_attention_module(vision_attn[0]['module'], vision_attn[0]['name'])

    # Test inference
    print("\n" + "=" * 60)
    print("Test Inference")
    print("=" * 60)

    # Create test image
    test_image = create_test_image()
    print(f"\nTest image created: {test_image.size}")

    # Prepare prompt
    prompt = "Describe what you see in this image."

    # Format for instruction-tuned model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process inputs
    try:
        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, images=test_image, return_tensors="pt")
        else:
            inputs = processor(text=prompt, images=test_image, return_tensors="pt")

        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        print(f"\nInput shapes:")
        for k, v in inputs.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")

    except Exception as e:
        print(f"Error processing inputs: {e}")
        return

    # Run inference
    print("\nRunning inference...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{generated_text}")

    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Test attention output
    print("\n" + "=" * 60)
    print("Testing Attention Output")
    print("=" * 60)

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions
            print(f"\nAttentions available: Yes")
            print(f"Number of attention tensors: {len(attentions)}")
            if attentions:
                print(f"First attention shape: {attentions[0].shape}")
                print(f"Last attention shape: {attentions[-1].shape}")

                # Check values
                first_attn = attentions[0]
                print(f"\nFirst attention stats:")
                print(f"  Min: {first_attn.min().item():.6f}")
                print(f"  Max: {first_attn.max().item():.6f}")
                print(f"  Mean: {first_attn.mean().item():.6f}")

                # Check if softmax was applied (rows sum to 1)
                row_sums = first_attn[0, 0].sum(dim=-1)
                print(f"  Row sums (should be ~1): min={row_sums.min().item():.4f}, max={row_sums.max().item():.4f}")
        else:
            print("\nAttentions not directly available in outputs")
            print("Will need to use hooks to capture attention weights")

    except Exception as e:
        print(f"Error testing attention output: {e}")

    print("\n" + "=" * 60)
    print("Model Loading Verification Complete!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Language layers: {len(language_attn)}")
    print(f"  Vision layers: {len(vision_attn)}")
    print("  Inference: Working")

    return model, processor


if __name__ == "__main__":
    main()
