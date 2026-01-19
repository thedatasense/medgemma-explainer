#!/usr/bin/env python3
"""
Chest X-ray Keyword Explanation with Chefer Method.

Demonstrates how to explain specific medical keywords in MedGemma's
diagnosis using correct backprop targets for causal language models.

Example: Analyzing a chest X-ray with pneumonia and explaining
why the model mentioned specific anatomical locations.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/content')

from medgemma_explainability.relevancy import (
    extract_token_relevancy,
    split_relevancy,
)
from medgemma_explainability.utils import (
    normalize_relevancy,
    reshape_image_relevancy,
)


def explain_token(model, full_inputs, target_token_position, target_token_id, seq_len, device):
    """
    Explain a specific token with correct backprop target.

    For causal LM: logit at position p-1 predicts token at position p.
    """
    model.eval()

    with torch.enable_grad():
        outputs = model(**full_inputs, output_attentions=True, return_dict=True)

        for attn in outputs.attentions:
            if attn is not None:
                attn.requires_grad_(True)
                attn.retain_grad()

        logit_position = target_token_position - 1
        target_logit = outputs.logits[0, logit_position, target_token_id]
        target_logit.backward(retain_graph=True)

        attention_maps = {}
        attention_grads = {}

        for i, attn in enumerate(outputs.attentions):
            if attn is not None:
                attention_maps[i] = attn.detach().float()
                if attn.grad is not None:
                    attention_grads[i] = attn.grad.detach().float()

    R = torch.eye(seq_len, device=device, dtype=torch.float32)

    for layer_idx in sorted(attention_maps.keys()):
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx]
        grad_A = attention_grads[layer_idx]

        weighted = A * grad_A
        weighted = torch.clamp(weighted, min=0)
        Abar = weighted.mean(dim=(0, 1))
        Abar = Abar + torch.eye(seq_len, device=device, dtype=torch.float32)
        R = torch.matmul(Abar, R)

    return R, attention_maps, logit_position


def explain_answer_span(model, full_inputs, answer_start, answer_end, input_ids, seq_len, device):
    """Explain the whole answer by summing logits across answer tokens."""
    model.eval()

    with torch.enable_grad():
        outputs = model(**full_inputs, output_attentions=True, return_dict=True)

        for attn in outputs.attentions:
            if attn is not None:
                attn.requires_grad_(True)
                attn.retain_grad()

        total_logit = 0
        predicting_positions = []

        for p in range(answer_start, answer_end):
            logit_pos = p - 1
            token_id = input_ids[0, p].item()
            total_logit = total_logit + outputs.logits[0, logit_pos, token_id]
            predicting_positions.append(logit_pos)

        total_logit.backward(retain_graph=True)

        attention_maps = {}
        attention_grads = {}

        for i, attn in enumerate(outputs.attentions):
            if attn is not None:
                attention_maps[i] = attn.detach().float()
                if attn.grad is not None:
                    attention_grads[i] = attn.grad.detach().float()

    R = torch.eye(seq_len, device=device, dtype=torch.float32)

    for layer_idx in sorted(attention_maps.keys()):
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx]
        grad_A = attention_grads[layer_idx]

        weighted = A * grad_A
        weighted = torch.clamp(weighted, min=0)
        Abar = weighted.mean(dim=(0, 1))
        Abar = Abar + torch.eye(seq_len, device=device, dtype=torch.float32)
        R = torch.matmul(Abar, R)

    return R, predicting_positions


def analyze_lung_regions(heatmap):
    """
    Analyze relevancy by anatomical lung regions.

    IMPORTANT: For chest X-ray PA view:
    - Left side of image = Patient's RIGHT side
    - Right side of image = Patient's LEFT side
    """
    hm = heatmap.cpu().numpy() if torch.is_tensor(heatmap) else heatmap

    regions = {
        'Pt RIGHT Upper\n(Left of image)': hm[:5, :8].mean(),
        'Pt RIGHT Middle\n(Left of image)': hm[5:11, :8].mean(),
        'Pt RIGHT Lower\n(Left of image)': hm[11:, :8].mean(),
        'Pt LEFT Upper\n(Right of image)': hm[:5, 8:].mean(),
        'Pt LEFT Middle\n(Right of image)': hm[5:11, 8:].mean(),
        'Pt LEFT Lower\n(Right of image)': hm[11:, 8:].mean(),
    }

    return regions


def main():
    print("=" * 60)
    print("Chest X-ray Keyword Explanation - Chefer Method")
    print("=" * 60)

    # Setup
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        return

    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

    device = 'cuda'
    NUM_IMAGE_TOKENS = 256
    IMAGE_START_IDX = 6

    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("Loading model...")
    processor = AutoProcessor.from_pretrained('google/medgemma-1.5-4b-it', trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        'google/medgemma-1.5-4b-it',
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    print("Model loaded!")

    # Load chest X-ray
    image = Image.open('/content/chest_xray.jpg').convert('RGB')
    prompt = "Analyze this chest X-ray. Is there evidence of pneumonia? If so, describe the location and appearance."

    print(f"\nPrompt: {prompt}")

    # Prepare inputs
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors='pt').to(device)

    input_len = inputs['input_ids'].shape[1]
    print(f"Input length (prompt): {input_len}")

    # Generate response
    print("\nGenerating response...")
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    seq_len = gen_outputs.shape[1]
    answer_start = input_len
    answer_end = seq_len

    # Decode response
    all_tokens = [processor.decode([t.item()]) for t in gen_outputs[0]]
    response_tokens = all_tokens[answer_start:]
    response_text = processor.decode(gen_outputs[0, answer_start:], skip_special_tokens=True)

    print(f"\nGenerated {answer_end - answer_start} answer tokens")
    print(f"\nMedGemma's Assessment:")
    print("-" * 40)
    print(response_text)
    print("-" * 40)

    # Find medical keywords in the answer
    keywords = {
        'pneumonia': [],
        'consolidation': [],
        'opacity': [],
        'right': [],
        'left': [],
        'lobe': [],
        'infiltrate': [],
    }

    for i, token in enumerate(response_tokens):
        for kw in keywords:
            if kw.lower() in token.lower():
                keywords[kw].append(answer_start + i)

    print(f"\nMedical keyword positions:")
    for kw, positions in keywords.items():
        if positions:
            print(f"  '{kw}': positions {positions}, tokens: {[all_tokens[p] for p in positions]}")

    # Prepare full inputs for explanation
    full_inputs = {
        'input_ids': gen_outputs,
        'attention_mask': torch.ones_like(gen_outputs),
        'pixel_values': inputs['pixel_values'],
    }

    # ==========================================
    # Explain key medical terms
    # ==========================================
    explanations = {}

    # Try to explain "pneumonia" or "consolidation" or "opacity"
    target_keywords = ['pneumonia', 'consolidation', 'opacity', 'right']

    for kw in target_keywords:
        if keywords[kw]:
            pos = keywords[kw][0]
            token_id = gen_outputs[0, pos].item()

            print(f"\n{'=' * 60}")
            print(f"Explaining '{kw}' token at position {pos}")
            print(f"{'=' * 60}")

            R, _, logit_pos = explain_token(
                model, full_inputs, pos, token_id, seq_len, device
            )

            token_rel = extract_token_relevancy(R, logit_pos)
            img_rel, _ = split_relevancy(token_rel, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
            img_2d = normalize_relevancy(reshape_image_relevancy(img_rel))

            explanations[kw] = img_2d

            # Analyze regions
            regions = analyze_lung_regions(img_2d)
            print(f"\n'{kw}' token relevancy by lung region:")
            for region, value in sorted(regions.items(), key=lambda x: -x[1]):
                marker = " <-- PNEUMONIA SIDE" if 'RIGHT' in region else ""
                print(f"  {region.split(chr(10))[0]}: {value:.4f}{marker}")

    # ==========================================
    # Explain whole answer
    # ==========================================
    print(f"\n{'=' * 60}")
    print("Explaining whole answer")
    print(f"{'=' * 60}")

    R_answer, pred_positions = explain_answer_span(
        model, full_inputs, answer_start, answer_end, gen_outputs, seq_len, device
    )

    avg_relevancy = torch.zeros(seq_len, device=device)
    for pos in pred_positions:
        avg_relevancy += R_answer[pos, :]
    avg_relevancy = avg_relevancy / len(pred_positions)

    img_rel_answer, _ = split_relevancy(avg_relevancy, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    img_2d_answer = normalize_relevancy(reshape_image_relevancy(img_rel_answer))

    explanations['whole_answer'] = img_2d_answer

    regions = analyze_lung_regions(img_2d_answer)
    print(f"\nWhole answer relevancy by lung region:")
    for region, value in sorted(regions.items(), key=lambda x: -x[1]):
        marker = " <-- PNEUMONIA SIDE" if 'RIGHT' in region else ""
        print(f"  {region.split(chr(10))[0]}: {value:.4f}{marker}")

    # ==========================================
    # Create visualization
    # ==========================================
    print("\n--- Creating visualization ---")

    # Determine layout based on how many keywords we found
    n_explanations = len([k for k in explanations if k != 'whole_answer'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    def show_heatmap(ax, hm_tensor, title, show_regions=True):
        hm = hm_tensor.cpu().numpy() if torch.is_tensor(hm_tensor) else hm_tensor
        hm_resized = np.array(Image.fromarray(hm).resize(image.size, Image.BILINEAR))
        ax.imshow(image, cmap='gray')
        im = ax.imshow(hm_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

        if show_regions:
            # Show key region values
            right_lung = hm[:, :8].mean()  # Patient's right (left of image)
            left_lung = hm[:, 8:].mean()   # Patient's left (right of image)
            ax.text(0.02, 0.02, f"Pt RIGHT lung: {right_lung:.3f}\nPt LEFT lung: {left_lung:.3f}",
                    transform=ax.transAxes, fontsize=9, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        return im

    # Original image with anatomical labels
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Chest X-ray (PA View)\nLeft of image = Patient RIGHT side', fontsize=11)
    axes[0, 0].axis('off')
    # Add anatomical labels
    axes[0, 0].text(0.15, 0.5, 'Patient\nRIGHT', transform=axes[0, 0].transAxes,
                    fontsize=12, color='red', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 0].text(0.85, 0.5, 'Patient\nLEFT', transform=axes[0, 0].transAxes,
                    fontsize=12, color='blue', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Whole answer explanation
    show_heatmap(axes[0, 1], img_2d_answer,
                 'Whole Answer Explanation\n(summed logits, averaged relevancy)')

    # Keyword explanations
    plot_idx = 2
    keyword_plots = []
    for kw in ['pneumonia', 'consolidation', 'opacity', 'right']:
        if kw in explanations and kw != 'whole_answer':
            row, col = divmod(plot_idx, 3)
            show_heatmap(axes[row, col], explanations[kw],
                        f'"{kw}" token explanation\n(backprop from position p-1)')
            keyword_plots.append(kw)
            plot_idx += 1
            if plot_idx >= 6:
                break

    # Region analysis bar chart
    if keyword_plots:
        main_kw = keyword_plots[0]
        regions = analyze_lung_regions(explanations[main_kw])

        # Simplify region names for plot
        simple_regions = {}
        for name, val in regions.items():
            simple_name = name.split('\n')[0]
            simple_regions[simple_name] = val

        # Color patient's right side (pneumonia) in red
        colors = ['red' if 'RIGHT' in k else 'steelblue' for k in simple_regions.keys()]

        row, col = divmod(plot_idx, 3)
        if plot_idx < 6:
            bars = axes[row, col].barh(list(simple_regions.keys()),
                                        list(simple_regions.values()), color=colors)
            axes[row, col].set_xlabel('Mean Relevancy')
            axes[row, col].set_title(f'Region Analysis ("{main_kw}" token)\nRed = Patient RIGHT (pneumonia side)',
                                     fontsize=10)
            for bar, (name, val) in zip(bars, simple_regions.items()):
                axes[row, col].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                                   f'{val:.3f}', va='center', fontsize=9)

    # Hide any unused subplots
    for idx in range(plot_idx + 1, 6):
        row, col = divmod(idx, 3)
        axes[row, col].axis('off')

    plt.suptitle('Chefer Method - Chest X-ray Pneumonia Analysis\n"Is there evidence of pneumonia?"',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/content/outputs/chest_xray_explanation.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/chest_xray_explanation.png")

    # ==========================================
    # Summary statistics
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nAnatomical Orientation Reminder:")
    print("  - Left side of image = Patient's RIGHT side (pneumonia location)")
    print("  - Right side of image = Patient's LEFT side")

    if 'whole_answer' in explanations:
        regions = analyze_lung_regions(explanations['whole_answer'])
        right_avg = np.mean([v for k, v in regions.items() if 'RIGHT' in k])
        left_avg = np.mean([v for k, v in regions.items() if 'LEFT' in k])

        print(f"\nWhole Answer - Lung Relevancy:")
        print(f"  Patient's RIGHT lung (pneumonia): {right_avg:.4f}")
        print(f"  Patient's LEFT lung (normal):     {left_avg:.4f}")
        print(f"  Ratio (RIGHT/LEFT): {right_avg/left_avg:.2f}x")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
