#!/usr/bin/env python3
"""
End-to-end test of the MedGemma Chefer explainability implementation.

This script tests the full pipeline:
1. Load model
2. Run inference
3. Extract attention and gradients
4. Propagate relevancy
5. Visualize results
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medgemma_explainability.relevancy import (
    compute_Abar,
    propagate_relevancy,
    extract_token_relevancy,
    split_relevancy,
    compute_raw_attention_relevancy,
)
from medgemma_explainability.utils import (
    normalize_relevancy,
    reshape_image_relevancy,
    is_global_layer,
)
from medgemma_explainability.visualization import (
    ExplanationResult,
    visualize_explanation,
    visualize_comparison,
)


def create_test_image(pattern='split'):
    """Create test images with different patterns."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    if pattern == 'split':
        # Left red, right blue
        img[:, :112, 0] = 255
        img[:, 112:, 2] = 255
    elif pattern == 'quadrant':
        # Four colored quadrants
        img[:112, :112, 0] = 255  # Red top-left
        img[:112, 112:, 1] = 255  # Green top-right
        img[112:, :112, 2] = 255  # Blue bottom-left
        img[112:, 112:] = [255, 255, 0]  # Yellow bottom-right
    elif pattern == 'circle':
        # Circle in center
        y, x = np.ogrid[:224, :224]
        center = 112
        radius = 50
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        img[mask] = [255, 0, 0]  # Red circle
        img[~mask] = [0, 0, 255]  # Blue background

    return Image.fromarray(img)


def main():
    print("=" * 60)
    print("MedGemma Chefer Explainability - End-to-End Test")
    print("=" * 60)

    # Check HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        return

    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load model with eager attention
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = 'google/medgemma-1.5-4b-it'
    print(f"\nLoading {model_name} with eager attention...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    print("Model loaded")

    # Create test image
    test_image = create_test_image('split')
    prompt = "What colors do you see in this image?"

    print(f"\nTest prompt: {prompt}")

    # Prepare inputs
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': test_image},
        {'type': 'text', 'text': prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=test_image, return_tensors='pt').to(device)

    seq_len = inputs['input_ids'].shape[1]
    print(f"Sequence length: {seq_len}")

    # Step 1: Generate response
    print("\n--- Step 1: Generate Response ---")
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    generated_text = processor.decode(gen_outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text[-300:]}")

    # Step 2: Forward pass with attention
    print("\n--- Step 2: Extract Attention Weights ---")
    model.train()

    # Use generated sequence for full attention
    full_inputs = {
        'input_ids': gen_outputs,
        'attention_mask': torch.ones_like(gen_outputs),
        'pixel_values': inputs['pixel_values'],
    }

    outputs = model(**full_inputs, output_attentions=True, return_dict=True)

    attentions = outputs.attentions
    for attn in attentions:
        if attn is not None and attn.requires_grad:
            attn.retain_grad()

    print(f"Got {len(attentions)} attention tensors")
    print(f"Shape: {attentions[0].shape}")

    # Step 3: Backward pass for gradients
    print("\n--- Step 3: Compute Attention Gradients ---")

    logits = outputs.logits
    target_idx = -1  # Last token
    target_logit = logits[0, target_idx, logits[0, target_idx].argmax()]
    target_logit.backward(retain_graph=True)

    # Collect attention and gradients
    attention_maps = {}
    attention_grads = {}

    for i, attn in enumerate(attentions):
        if attn is not None:
            attention_maps[i] = attn.detach()
            if attn.grad is not None:
                attention_grads[i] = attn.grad.detach()

    print(f"Collected {len(attention_maps)} attention maps")
    print(f"Collected {len(attention_grads)} gradient maps")

    model.eval()

    # Step 4: Propagate relevancy
    print("\n--- Step 4: Propagate Relevancy (Chefer Method) ---")

    final_seq_len = gen_outputs.shape[1]

    R = propagate_relevancy(
        attention_maps,
        attention_grads,
        final_seq_len,
        handle_local_attention=True,
        local_window_size=1024,
        device=device,
    )

    print(f"Relevancy matrix shape: {R.shape}")

    # Step 5: Extract and split relevancy
    print("\n--- Step 5: Extract Token Relevancy ---")

    token_relevancy = extract_token_relevancy(R, target_token_idx=-1)

    # MedGemma token structure: image tokens start at position 6
    num_image_tokens = 256
    image_start_idx = 6
    image_relevancy, text_relevancy = split_relevancy(
        token_relevancy,
        num_image_tokens=num_image_tokens,
        image_start_idx=image_start_idx,
    )

    print(f"Image relevancy shape: {image_relevancy.shape}")
    print(f"Text relevancy shape: {text_relevancy.shape}")

    # Reshape and normalize image relevancy
    image_relevancy_2d = reshape_image_relevancy(image_relevancy)
    image_relevancy_2d = normalize_relevancy(image_relevancy_2d)
    text_relevancy = normalize_relevancy(text_relevancy)

    print(f"Image relevancy 2D shape: {image_relevancy_2d.shape}")
    print(f"Image relevancy range: [{image_relevancy_2d.min():.4f}, {image_relevancy_2d.max():.4f}]")
    print(f"Text relevancy range: [{text_relevancy.min():.4f}, {text_relevancy.max():.4f}]")

    # Step 6: Compute raw attention baseline
    print("\n--- Step 6: Compute Raw Attention Baseline ---")

    raw_attn = compute_raw_attention_relevancy(attention_maps)
    raw_token_rel = raw_attn[-1, :]
    raw_img, raw_txt = split_relevancy(raw_token_rel, num_image_tokens, image_start_idx)
    raw_image_2d = normalize_relevancy(reshape_image_relevancy(raw_img))
    raw_text = normalize_relevancy(raw_txt)

    print(f"Raw attention image range: [{raw_image_2d.min():.4f}, {raw_image_2d.max():.4f}]")

    # Step 7: Get token labels
    print("\n--- Step 7: Get Token Labels ---")

    token_labels = []
    for token_id in gen_outputs[0]:
        token_str = processor.decode([token_id.item()])
        token_labels.append(token_str)

    # Text tokens start after image section (position 262 = image_start_idx + num_image_tokens)
    text_start_idx = image_start_idx + num_image_tokens
    text_token_labels = token_labels[text_start_idx:]
    print(f"Total tokens: {len(token_labels)}")
    print(f"Text tokens: {len(text_token_labels)}")
    print(f"First few text tokens: {text_token_labels[:5]}")

    # Step 8: Create visualization
    print("\n--- Step 8: Create Visualizations ---")

    result = ExplanationResult(
        image_relevancy=image_relevancy_2d.cpu().numpy(),
        text_relevancy=text_relevancy.cpu().numpy(),
        token_labels=text_token_labels,
        generated_text=generated_text,
        raw_image_relevancy=raw_image_2d.cpu().numpy(),
        raw_text_relevancy=raw_text.cpu().numpy(),
        metadata={
            'seq_len': final_seq_len,
            'num_layers': len(attention_maps),
            'prompt': prompt,
        }
    )

    # Save basic visualization
    os.makedirs('outputs', exist_ok=True)

    fig = visualize_explanation(test_image, result, save_path='outputs/explanation.png')
    plt.close(fig)
    print("Saved: outputs/explanation.png")

    # Save comparison visualization
    fig = visualize_comparison(test_image, result, save_path='outputs/comparison.png')
    plt.close(fig)
    print("Saved: outputs/comparison.png")

    # Step 9: Analyze results
    print("\n--- Step 9: Analysis ---")

    # Check if left vs right image regions have different relevancy
    left_relevancy = image_relevancy_2d[:, :8].mean().item()
    right_relevancy = image_relevancy_2d[:, 8:].mean().item()

    print(f"\nImage region analysis:")
    print(f"  Left half (red) mean relevancy: {left_relevancy:.4f}")
    print(f"  Right half (blue) mean relevancy: {right_relevancy:.4f}")

    # Find top relevant text tokens
    text_rel_np = text_relevancy.cpu().numpy()
    top_indices = np.argsort(text_rel_np)[::-1][:10]

    print(f"\nTop 10 relevant text tokens:")
    for i, idx in enumerate(top_indices):
        if idx < len(text_token_labels):
            print(f"  {i+1}. '{text_token_labels[idx]}': {text_rel_np[idx]:.4f}")

    print("\n" + "=" * 60)
    print("End-to-End Test Complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = main()
