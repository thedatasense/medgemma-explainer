#!/usr/bin/env python3
"""
Cat Image Analysis with MedGemma Explainability.

Demonstrates the explainability method on a general image (cat).
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medgemma_explainability.relevancy import (
    propagate_relevancy,
    extract_token_relevancy,
    split_relevancy,
    compute_raw_attention_relevancy,
)
from medgemma_explainability.utils import (
    normalize_relevancy,
    reshape_image_relevancy,
)


def main():
    print("=" * 60)
    print("Cat Image Analysis with MedGemma Explainability")
    print("=" * 60)

    # Check HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        return

    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = 'google/medgemma-1.5-4b-it'
    print(f"\nLoading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    print("Model loaded!")

    # Load cat image
    image_path = 'cat_image.jpg'
    image = Image.open(image_path).convert('RGB')
    print(f"\nLoaded image: {image.size}")

    # Prompt about the cat
    prompt = "Describe this image. What do you see?"

    print(f"\nPrompt: {prompt}")

    # Prepare inputs
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors='pt').to(device)

    print(f"Input sequence length: {inputs['input_ids'].shape[1]}")

    # Generate response
    print("\n--- Generating Response ---")
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    generated_text = processor.decode(gen_outputs[0], skip_special_tokens=True)

    # Extract just the model's response
    if 'model' in generated_text:
        response = generated_text.split('model')[-1].strip()
    else:
        response = generated_text

    print(f"\nMedGemma Response:\n{response}")

    # Forward pass with attention
    print("\n--- Extracting Attention ---")
    model.train()

    full_inputs = {
        'input_ids': gen_outputs,
        'attention_mask': torch.ones_like(gen_outputs),
        'pixel_values': inputs['pixel_values'],
    }

    outputs = model(**full_inputs, output_attentions=True, return_dict=True)

    for attn in outputs.attentions:
        if attn is not None and attn.requires_grad:
            attn.retain_grad()

    # Backward pass
    print("--- Computing Gradients ---")
    logits = outputs.logits
    target_logit = logits[0, -1, logits[0, -1].argmax()]
    target_logit.backward(retain_graph=True)

    # Collect attention and gradients
    attention_maps = {}
    attention_grads = {}

    for i, attn in enumerate(outputs.attentions):
        if attn is not None:
            attention_maps[i] = attn.detach()
            if attn.grad is not None:
                attention_grads[i] = attn.grad.detach()

    print(f"Collected {len(attention_grads)} gradient maps")
    model.eval()

    # Propagate relevancy
    print("--- Propagating Relevancy ---")
    seq_len = gen_outputs.shape[1]

    NUM_IMAGE_TOKENS = 256
    IMAGE_START_IDX = 6

    R = propagate_relevancy(
        attention_maps,
        attention_grads,
        seq_len,
        handle_local_attention=True,
        local_window_size=1024,
        device=device,
    )

    # Extract relevancy
    token_relevancy = extract_token_relevancy(R, target_token_idx=-1)
    image_relevancy, text_relevancy = split_relevancy(
        token_relevancy, NUM_IMAGE_TOKENS, IMAGE_START_IDX
    )

    image_relevancy_2d = reshape_image_relevancy(image_relevancy)
    image_relevancy_2d = normalize_relevancy(image_relevancy_2d)
    text_relevancy = normalize_relevancy(text_relevancy)

    # Raw attention baseline
    raw_attn = compute_raw_attention_relevancy(attention_maps)
    raw_token_rel = raw_attn[-1, :]
    raw_img, raw_txt = split_relevancy(raw_token_rel, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    raw_image_2d = normalize_relevancy(reshape_image_relevancy(raw_img))

    # Get token labels
    text_start_idx = IMAGE_START_IDX + NUM_IMAGE_TOKENS
    token_labels = [processor.decode([t.item()]) for t in gen_outputs[0][text_start_idx:]]

    # Create visualizations
    print("\n--- Creating Visualizations ---")
    os.makedirs('outputs', exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Images
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Cat Image', fontsize=14)
    axes[0, 0].axis('off')

    # Resize heatmap
    heatmap = image_relevancy_2d.cpu().numpy()
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.BILINEAR))

    axes[0, 1].imshow(image)
    im1 = axes[0, 1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Chefer Method - Relevancy Map', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Raw attention
    raw_heatmap = raw_image_2d.cpu().numpy()
    raw_resized = np.array(Image.fromarray(raw_heatmap).resize(image.size, Image.BILINEAR))

    axes[0, 2].imshow(image)
    im2 = axes[0, 2].imshow(raw_resized, cmap='jet', alpha=0.5)
    axes[0, 2].set_title('Raw Attention (Last Layer)', fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Analysis
    # Heatmap alone
    axes[1, 0].imshow(heatmap, cmap='hot')
    axes[1, 0].set_title('Relevancy Heatmap (16x16 grid)', fontsize=14)
    axes[1, 0].set_xlabel('Image column')
    axes[1, 0].set_ylabel('Image row')
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046)

    # Region analysis (quadrants)
    regions = {
        'Top-Left': heatmap[:8, :8].mean(),
        'Top-Right': heatmap[:8, 8:].mean(),
        'Bottom-Left': heatmap[8:, :8].mean(),
        'Bottom-Right': heatmap[8:, 8:].mean(),
    }

    colors = ['steelblue'] * 4
    bars = axes[1, 1].barh(list(regions.keys()), list(regions.values()), color=colors)
    axes[1, 1].set_xlabel('Mean Relevancy')
    axes[1, 1].set_title('Relevancy by Image Quadrant', fontsize=14)

    for bar, (name, val) in zip(bars, regions.items()):
        axes[1, 1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=10)

    # Top text tokens
    text_rel_np = text_relevancy.cpu().numpy()
    n_tokens = min(15, len(text_rel_np))
    sorted_idx = np.argsort(text_rel_np)[::-1][:n_tokens]

    labels = [token_labels[i][:15].replace('\n', '\\n') for i in sorted_idx]
    values = [text_rel_np[i] for i in sorted_idx]

    axes[1, 2].barh(range(len(labels)), values, color='steelblue')
    axes[1, 2].set_yticks(range(len(labels)))
    axes[1, 2].set_yticklabels(labels, fontsize=9)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel('Relevancy Score')
    axes[1, 2].set_title('Top Text Token Relevancy', fontsize=14)

    plt.suptitle('MedGemma Explainability: Cat Image Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/cat_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/cat_analysis.png")

    # Print analysis
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nMedGemma's Description:\n{response}")
    print("\n" + "-" * 40)
    print("Relevancy by Image Quadrant:")
    for region, value in sorted(regions.items(), key=lambda x: -x[1]):
        print(f"  {region}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    return {
        'image_relevancy': image_relevancy_2d,
        'text_relevancy': text_relevancy,
        'response': response,
    }


if __name__ == "__main__":
    result = main()
