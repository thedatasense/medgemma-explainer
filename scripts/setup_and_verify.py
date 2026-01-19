#!/usr/bin/env python3
"""
Setup script to verify MedGemma model loading with HuggingFace authentication.

Run this in a Colab notebook cell:
    from google.colab import userdata
    import os
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

    %run scripts/setup_and_verify.py
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
    img[:, :size[1]//2, 0] = 255  # Red on left
    img[:, size[1]//2:, 2] = 255  # Blue on right
    return Image.fromarray(img)


def print_model_structure(model, max_depth=3):
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

    print(f"\nModel: {model.__class__.__name__}")
    print("-" * 50)
    _print_children(model)


def analyze_attention_config(model):
    """Analyze attention configuration from the model."""
    print("\n" + "=" * 60)
    print("Attention Configuration Analysis")
    print("=" * 60)

    # Get config
    config = model.config if hasattr(model, 'config') else None

    if config:
        print(f"\nModel config type: {type(config).__name__}")

        # Try to get text config
        text_config = getattr(config, 'text_config', None)
        if text_config:
            print(f"\nText/Language Model Config:")
            for attr in ['num_attention_heads', 'num_key_value_heads', 'num_hidden_layers',
                        'hidden_size', 'head_dim', 'attention_bias']:
                if hasattr(text_config, attr):
                    print(f"  {attr}: {getattr(text_config, attr)}")

        # Try to get vision config
        vision_config = getattr(config, 'vision_config', None)
        if vision_config:
            print(f"\nVision Model Config:")
            for attr in ['num_attention_heads', 'num_hidden_layers', 'hidden_size',
                        'image_size', 'patch_size', 'num_image_tokens']:
                if hasattr(vision_config, attr):
                    print(f"  {attr}: {getattr(vision_config, attr)}")


def find_and_count_attention_modules(model):
    """Find and categorize attention modules."""
    language_attn = []
    vision_attn = []
    other_attn = []

    for name, module in model.named_modules():
        module_class = module.__class__.__name__
        if 'Attention' in module_class or 'attn' in name.lower():
            if 'language' in name.lower() or 'llm' in name.lower():
                language_attn.append((name, module))
            elif 'vision' in name.lower() or 'vit' in name.lower() or 'siglip' in name.lower():
                vision_attn.append((name, module))
            else:
                other_attn.append((name, module))

    return language_attn, vision_attn, other_attn


def test_attention_output(model, processor, device):
    """Test if attention weights can be extracted."""
    print("\n" + "=" * 60)
    print("Testing Attention Extraction")
    print("=" * 60)

    # Create test inputs
    test_image = create_test_image()
    prompt = "What colors do you see?"

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
        return None

    # Test Approach A: output_attentions=True
    print("\n--- Approach A: output_attentions=True ---")
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

        attentions = getattr(outputs, 'attentions', None)

        if attentions is not None and len(attentions) > 0:
            print(f"SUCCESS! Got {len(attentions)} attention tensors")
            print(f"First attention shape: {attentions[0].shape}")
            print(f"Last attention shape: {attentions[-1].shape}")

            # Verify softmax
            first_attn = attentions[0]
            row_sums = first_attn[0, 0].sum(dim=-1)
            print(f"Row sums (should be ~1): min={row_sums.min().item():.4f}, max={row_sums.max().item():.4f}")

            return {
                'method': 'output_attentions',
                'attentions': attentions,
                'inputs': inputs
            }
        else:
            print("Attentions not available in outputs")

    except Exception as e:
        print(f"Error: {e}")

    # Test Approach B: Hook-based extraction
    print("\n--- Approach B: Forward hooks ---")
    try:
        attention_cache = {}
        hooks = []

        language_attn, _, _ = find_and_count_attention_modules(model)

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        attention_cache[layer_idx] = attn_weights.detach()
            return hook

        for idx, (name, module) in enumerate(language_attn):
            handle = module.register_forward_hook(make_hook(idx))
            hooks.append(handle)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Cleanup hooks
        for handle in hooks:
            handle.remove()

        if attention_cache:
            print(f"SUCCESS! Captured {len(attention_cache)} attention tensors via hooks")
            first_key = min(attention_cache.keys())
            print(f"First captured shape: {attention_cache[first_key].shape}")

            return {
                'method': 'hooks',
                'attentions': attention_cache,
                'inputs': inputs
            }
        else:
            print("No attention captured via hooks")

    except Exception as e:
        print(f"Error: {e}")

    return None


def main():
    print("=" * 60)
    print("MedGemma Model Loading and Verification")
    print("=" * 60)

    # Check for HF token
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
        print("\nWARNING: HF_TOKEN not found in environment!")
        print("Please set it before running this script:")
        print("  from google.colab import userdata")
        print("  import os")
        print("  os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')")
        return None

    print(f"\nHF_TOKEN: {'*' * 8}...{hf_token[-4:]}")

    # Login
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = "google/medgemma-1.5-4b-it"
    print(f"\nLoading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("Processor loaded")

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    print("Model loaded")

    # Print structure
    print_model_structure(model)

    # Analyze config
    analyze_attention_config(model)

    # Find attention modules
    language_attn, vision_attn, other_attn = find_and_count_attention_modules(model)
    print(f"\nFound attention modules:")
    print(f"  Language model: {len(language_attn)}")
    print(f"  Vision model: {len(vision_attn)}")
    print(f"  Other: {len(other_attn)}")

    if language_attn:
        print(f"\nSample language attention module: {language_attn[0][0]}")
        module = language_attn[0][1]
        for attr in ['num_heads', 'num_key_value_heads', 'head_dim']:
            if hasattr(module, attr):
                print(f"  {attr}: {getattr(module, attr)}")

    # Test attention extraction
    result = test_attention_output(model, processor, device)

    # Test generation
    print("\n" + "=" * 60)
    print("Test Generation")
    print("=" * 60)

    test_image = create_test_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "What colors do you see in this image?"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=test_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    generated = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated}")

    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)

    return model, processor, device


if __name__ == "__main__":
    result = main()
