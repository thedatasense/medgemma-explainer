#!/usr/bin/env python3
"""
Diagnostic script to investigate attention extraction issues.

Tests multiple approaches:
1. Different target tokens (pneumonia-related vs last token)
2. Global layers only (bypass local attention windowing issue)
3. Attention rollout method
4. Layer-specific relevancy
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
    compute_Abar,
    extract_token_relevancy,
    split_relevancy,
)
from medgemma_explainability.utils import (
    normalize_relevancy,
    reshape_image_relevancy,
    is_global_layer,
)


def attention_rollout(attention_maps, add_residual=True):
    """
    Compute attention rollout (Abnar & Zuidema, 2020).

    This is an alternative to gradient-weighted attention.
    """
    layer_indices = sorted(attention_maps.keys())
    seq_len = attention_maps[layer_indices[0]].shape[-1]
    device = attention_maps[layer_indices[0]].device

    rollout = torch.eye(seq_len, device=device, dtype=torch.float32)

    for layer_idx in layer_indices:
        attn = attention_maps[layer_idx].float().mean(dim=(0, 1))  # Average over heads

        if add_residual:
            attn = 0.5 * attn + 0.5 * torch.eye(seq_len, device=device)
            attn = attn / attn.sum(dim=-1, keepdim=True)

        rollout = torch.matmul(attn, rollout)

    return rollout


def global_layers_only_relevancy(attention_maps, attention_grads, seq_len, device):
    """
    Compute relevancy using only global attention layers.

    Global layers (5, 11, 17, 23, 29) have full attention span,
    while local layers are limited to 1024 tokens.
    """
    R = torch.eye(seq_len, device=device, dtype=torch.float32)

    global_layers = [i for i in sorted(attention_maps.keys()) if is_global_layer(i)]
    print(f"Using global layers: {global_layers}")

    for layer_idx in global_layers:
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx].float()
        grad_A = attention_grads[layer_idx].float()

        Abar = compute_Abar(A, grad_A, normalize=True)
        Abar = Abar + torch.eye(seq_len, device=device, dtype=torch.float32)
        Abar = Abar / Abar.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        R = torch.matmul(Abar, R)

    return R


def find_keyword_tokens(token_labels, keywords):
    """Find indices of tokens that match keywords."""
    matches = {}
    for kw in keywords:
        kw_lower = kw.lower()
        for i, label in enumerate(token_labels):
            if kw_lower in label.lower():
                if kw not in matches:
                    matches[kw] = []
                matches[kw].append(i)
    return matches


def main():
    print("=" * 60)
    print("Attention Diagnostic Analysis")
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

    # Load chest X-ray
    image_path = 'chest_xray.jpg'
    image = Image.open(image_path).convert('RGB')
    print(f"\nLoaded image: {image.size}")

    prompt = "Analyze this chest X-ray. Is there evidence of pneumonia? If so, describe the location and appearance of the consolidation."

    # Prepare inputs
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors='pt').to(device)

    # Generate response
    print("\n--- Generating Response ---")
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    generated_text = processor.decode(gen_outputs[0], skip_special_tokens=True)
    if 'model' in generated_text:
        response = generated_text.split('model')[-1].strip()
    else:
        response = generated_text
    print(f"\nResponse:\n{response}")

    # Forward pass with attention
    print("\n--- Forward Pass with Attention ---")
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
    print("--- Backward Pass ---")
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

    print(f"Collected {len(attention_maps)} attention maps, {len(attention_grads)} gradient maps")
    model.eval()

    seq_len = gen_outputs.shape[1]
    NUM_IMAGE_TOKENS = 256
    IMAGE_START_IDX = 6

    # Get token labels
    text_start_idx = IMAGE_START_IDX + NUM_IMAGE_TOKENS
    token_labels = [processor.decode([t.item()]) for t in gen_outputs[0][text_start_idx:]]

    print(f"\nSequence length: {seq_len}")
    print(f"Number of text tokens: {len(token_labels)}")
    print(f"First 20 text tokens: {token_labels[:20]}")
    print(f"Last 20 text tokens: {token_labels[-20:]}")

    # Find keyword tokens
    keywords = ['pneumonia', 'right', 'lower', 'lobe', 'consolidation', 'opacity']
    keyword_matches = find_keyword_tokens(token_labels, keywords)
    print(f"\nKeyword token matches:")
    for kw, indices in keyword_matches.items():
        print(f"  '{kw}': positions {indices}")

    # === METHOD 1: Standard Chefer (all layers) ===
    print("\n" + "=" * 60)
    print("METHOD 1: Standard Chefer (all layers)")
    print("=" * 60)

    from medgemma_explainability.relevancy import propagate_relevancy
    R_standard = propagate_relevancy(
        attention_maps, attention_grads, seq_len,
        handle_local_attention=True, device=device
    )

    token_rel = extract_token_relevancy(R_standard, -1)
    img_rel_std, txt_rel_std = split_relevancy(token_rel, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    img_2d_std = normalize_relevancy(reshape_image_relevancy(img_rel_std))

    print(f"Image relevancy range: [{img_rel_std.min():.4f}, {img_rel_std.max():.4f}]")
    print(f"Text relevancy range: [{txt_rel_std.min():.4f}, {txt_rel_std.max():.4f}]")

    # === METHOD 2: Global Layers Only ===
    print("\n" + "=" * 60)
    print("METHOD 2: Global Layers Only")
    print("=" * 60)

    R_global = global_layers_only_relevancy(attention_maps, attention_grads, seq_len, device)

    token_rel_global = extract_token_relevancy(R_global, -1)
    img_rel_global, txt_rel_global = split_relevancy(token_rel_global, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    img_2d_global = normalize_relevancy(reshape_image_relevancy(img_rel_global))

    print(f"Image relevancy range: [{img_rel_global.min():.4f}, {img_rel_global.max():.4f}]")

    # === METHOD 3: Attention Rollout ===
    print("\n" + "=" * 60)
    print("METHOD 3: Attention Rollout (no gradients)")
    print("=" * 60)

    R_rollout = attention_rollout(attention_maps, add_residual=True)

    token_rel_rollout = extract_token_relevancy(R_rollout, -1)
    img_rel_rollout, txt_rel_rollout = split_relevancy(token_rel_rollout, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    img_2d_rollout = normalize_relevancy(reshape_image_relevancy(img_rel_rollout))

    print(f"Image relevancy range: [{img_rel_rollout.min():.4f}, {img_rel_rollout.max():.4f}]")

    # === METHOD 4: Keyword-Specific Tokens ===
    print("\n" + "=" * 60)
    print("METHOD 4: Keyword-Specific Token Relevancy")
    print("=" * 60)

    keyword_relevancies = {}
    if 'pneumonia' in keyword_matches and keyword_matches['pneumonia']:
        pneumonia_idx = keyword_matches['pneumonia'][0]
        # Adjust index to be relative to full sequence
        full_idx = text_start_idx + pneumonia_idx

        token_rel_pneumonia = extract_token_relevancy(R_standard, full_idx)
        img_rel_pn, _ = split_relevancy(token_rel_pneumonia, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
        img_2d_pneumonia = normalize_relevancy(reshape_image_relevancy(img_rel_pn))
        keyword_relevancies['pneumonia'] = img_2d_pneumonia
        print(f"'pneumonia' token at position {full_idx}")

    if 'right' in keyword_matches and keyword_matches['right']:
        right_idx = keyword_matches['right'][0]
        full_idx = text_start_idx + right_idx

        token_rel_right = extract_token_relevancy(R_standard, full_idx)
        img_rel_rt, _ = split_relevancy(token_rel_right, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
        img_2d_right = normalize_relevancy(reshape_image_relevancy(img_rel_rt))
        keyword_relevancies['right'] = img_2d_right
        print(f"'right' token at position {full_idx}")

    # === METHOD 5: Raw Attention at Best Layers ===
    print("\n" + "=" * 60)
    print("METHOD 5: Raw Attention (Layer 17 - highest image attention)")
    print("=" * 60)

    best_layer = 17  # From previous analysis
    attn_layer17 = attention_maps[best_layer].float().mean(dim=(0, 1))
    raw_token_rel = attn_layer17[-1, :]  # Last token
    raw_img, _ = split_relevancy(raw_token_rel, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    img_2d_raw17 = normalize_relevancy(reshape_image_relevancy(raw_img))

    print(f"Image attention range: [{raw_img.min():.4f}, {raw_img.max():.4f}]")

    # === Create comparison visualization ===
    print("\n--- Creating Diagnostic Visualization ---")
    os.makedirs('outputs', exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Chest X-ray', fontsize=12)
    axes[0, 0].axis('off')

    # Resize function
    def overlay_heatmap(ax, img, heatmap, title):
        heatmap_np = heatmap.cpu().numpy() if torch.is_tensor(heatmap) else heatmap
        heatmap_resized = np.array(Image.fromarray(heatmap_np.astype(np.float32)).resize(img.size, Image.BILINEAR))
        ax.imshow(img, cmap='gray')
        im = ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return im

    # Method 1: Standard Chefer
    overlay_heatmap(axes[0, 1], image, img_2d_std, 'Method 1: Standard Chefer\n(all layers)')

    # Method 2: Global layers only
    overlay_heatmap(axes[0, 2], image, img_2d_global, 'Method 2: Global Layers Only\n(layers 5,11,17,23,29)')

    # Method 3: Attention rollout
    overlay_heatmap(axes[1, 0], image, img_2d_rollout, 'Method 3: Attention Rollout\n(no gradients)')

    # Method 4a: Pneumonia token
    if 'pneumonia' in keyword_relevancies:
        overlay_heatmap(axes[1, 1], image, keyword_relevancies['pneumonia'],
                        "Method 4a: 'pneumonia' token\nrelevancy")
    else:
        axes[1, 1].text(0.5, 0.5, 'No pneumonia\ntoken found', ha='center', va='center')
        axes[1, 1].axis('off')

    # Method 4b: Right token
    if 'right' in keyword_relevancies:
        overlay_heatmap(axes[1, 2], image, keyword_relevancies['right'],
                        "Method 4b: 'right' token\nrelevancy")
    else:
        axes[1, 2].text(0.5, 0.5, 'No right\ntoken found', ha='center', va='center')
        axes[1, 2].axis('off')

    # Method 5: Raw attention layer 17
    overlay_heatmap(axes[2, 0], image, img_2d_raw17, 'Method 5: Raw Attention\n(Layer 17)')

    # Region analysis comparison
    methods = {
        'Standard Chefer': img_2d_std.cpu().numpy(),
        'Global Only': img_2d_global.cpu().numpy(),
        'Rollout': img_2d_rollout.cpu().numpy(),
    }

    region_data = []
    for method_name, heatmap in methods.items():
        regions = {
            'R-Upper': heatmap[:5, 8:].mean(),
            'R-Middle': heatmap[5:11, 8:].mean(),
            'R-Lower': heatmap[11:, 8:].mean(),
            'L-Upper': heatmap[:5, :8].mean(),
            'L-Middle': heatmap[5:11, :8].mean(),
            'L-Lower': heatmap[11:, :8].mean(),
        }
        region_data.append((method_name, regions))

    # Plot region comparison
    x = np.arange(6)
    width = 0.25
    region_names = list(region_data[0][1].keys())

    for i, (method_name, regions) in enumerate(region_data):
        values = [regions[r] for r in region_names]
        axes[2, 1].bar(x + i*width, values, width, label=method_name)

    axes[2, 1].set_xticks(x + width)
    axes[2, 1].set_xticklabels(region_names, rotation=45, ha='right', fontsize=9)
    axes[2, 1].set_ylabel('Mean Relevancy')
    axes[2, 1].set_title('Region Comparison by Method', fontsize=10)
    axes[2, 1].legend(fontsize=8)

    # Gradient statistics
    grad_stats = []
    for layer_idx in sorted(attention_grads.keys()):
        grad = attention_grads[layer_idx].float()
        grad_stats.append({
            'layer': layer_idx,
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'pos_frac': (grad > 0).float().mean().item(),
        })

    layers = [s['layer'] for s in grad_stats]
    pos_fracs = [s['pos_frac'] for s in grad_stats]

    axes[2, 2].bar(layers, pos_fracs)
    axes[2, 2].set_xlabel('Layer')
    axes[2, 2].set_ylabel('Fraction of Positive Gradients')
    axes[2, 2].set_title('Gradient Polarity by Layer', fontsize=10)
    axes[2, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    plt.suptitle('Attention Diagnostic: Multiple Methods Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/attention_diagnostic.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/attention_diagnostic.png")

    # === Print Analysis ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIC ANALYSIS RESULTS")
    print("=" * 60)

    print("\n1. Region Analysis (Right side should be higher for pneumonia):")
    for method_name, regions in region_data:
        right_avg = (regions['R-Upper'] + regions['R-Middle'] + regions['R-Lower']) / 3
        left_avg = (regions['L-Upper'] + regions['L-Middle'] + regions['L-Lower']) / 3
        print(f"\n  {method_name}:")
        print(f"    Right side avg: {right_avg:.4f}")
        print(f"    Left side avg: {left_avg:.4f}")
        print(f"    Ratio (R/L): {right_avg/left_avg:.2f}x")
        print(f"    Best region: {max(regions, key=regions.get)} ({max(regions.values()):.4f})")

    print("\n2. Gradient Statistics:")
    pos_layers = [s for s in grad_stats if s['pos_frac'] > 0.5]
    neg_layers = [s for s in grad_stats if s['pos_frac'] <= 0.5]
    print(f"   Layers with >50% positive gradients: {len(pos_layers)}")
    print(f"   Layers with <=50% positive gradients: {len(neg_layers)}")

    avg_pos_frac = sum(s['pos_frac'] for s in grad_stats) / len(grad_stats)
    print(f"   Average positive fraction: {avg_pos_frac:.3f}")

    print("\n3. Key Finding:")
    if avg_pos_frac < 0.4:
        print("   WARNING: Gradients are predominantly negative.")
        print("   The Chefer method clips negative values, which may explain weak signal.")
        print("   Consider using absolute gradients or attention rollout instead.")
    elif avg_pos_frac > 0.6:
        print("   Gradients are predominantly positive - Chefer method should work well.")
    else:
        print("   Gradients are balanced - results should be reasonable.")

    print("\n" + "=" * 60)
    print("Diagnostic Complete!")
    print("=" * 60)

    return {
        'standard': img_2d_std,
        'global_only': img_2d_global,
        'rollout': img_2d_rollout,
        'keyword_relevancies': keyword_relevancies,
        'raw_layer17': img_2d_raw17,
        'grad_stats': grad_stats,
    }


if __name__ == "__main__":
    results = main()
