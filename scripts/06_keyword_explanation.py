#!/usr/bin/env python3
"""
Keyword-Specific Explanation with Chefer Method.

Demonstrates how to explain specific keywords in the model's response
using the correct backprop targets for causal language models.

Key principles:
1. For causal LM: logit at position i predicts token at i+1
   - To explain token at position p, backprop from logit at p-1
2. Use actual token id, not argmax
3. Keep model in eval mode, use torch.enable_grad() context
4. Each keyword needs its own backward pass
5. For whole answer: sum logits across answer span
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
    compute_Abar,
    extract_token_relevancy,
    split_relevancy,
)
from medgemma_explainability.utils import (
    normalize_relevancy,
    reshape_image_relevancy,
    is_global_layer,
)


def explain_token(model, full_inputs, target_token_position, target_token_id, seq_len, device):
    """
    Explain a specific token with correct backprop target.

    For causal LM: to explain token at position p,
    backprop from logit at position p-1 for token id at p.

    Args:
        model: The model
        full_inputs: Dict with input_ids, attention_mask, pixel_values
        target_token_position: Position of token to explain (p)
        target_token_id: The token id at position p
        seq_len: Sequence length
        device: Device

    Returns:
        R: Relevancy matrix
        attention_maps: Collected attention maps
        logit_position: The position used for backprop (p-1)
    """
    # Keep model in eval mode, but enable gradients
    model.eval()

    with torch.enable_grad():
        outputs = model(**full_inputs, output_attentions=True, return_dict=True)

        # Retain gradients on attention tensors
        for attn in outputs.attentions:
            if attn is not None:
                attn.requires_grad_(True)
                attn.retain_grad()

        # Logit at position p-1 predicts token at p
        logit_position = target_token_position - 1

        # Get the logit for the specific token that was generated
        target_logit = outputs.logits[0, logit_position, target_token_id]

        # Backward from this specific target
        target_logit.backward(retain_graph=True)

        # Collect attention and gradients
        attention_maps = {}
        attention_grads = {}

        for i, attn in enumerate(outputs.attentions):
            if attn is not None:
                attention_maps[i] = attn.detach().float()
                if attn.grad is not None:
                    attention_grads[i] = attn.grad.detach().float()

    # Propagate relevancy
    R = torch.eye(seq_len, device=device, dtype=torch.float32)

    for layer_idx in sorted(attention_maps.keys()):
        if layer_idx not in attention_grads:
            continue

        A = attention_maps[layer_idx]
        grad_A = attention_grads[layer_idx]

        # Compute Abar: gradient-weighted attention
        weighted = A * grad_A
        weighted = torch.clamp(weighted, min=0)
        Abar = weighted.mean(dim=(0, 1))

        # Add identity for residual connection
        Abar = Abar + torch.eye(seq_len, device=device, dtype=torch.float32)

        # Propagate
        R = torch.matmul(Abar, R)

    # Return relevancy matrix, attention maps, and the logit position
    return R, attention_maps, logit_position


def explain_answer_span(model, full_inputs, answer_start, answer_end, input_ids, seq_len, device):
    """
    Explain the whole answer by summing logits across answer tokens.

    Args:
        model: The model
        full_inputs: Dict with input_ids, attention_mask, pixel_values
        answer_start: Start position of answer tokens
        answer_end: End position of answer tokens (exclusive)
        input_ids: Full input_ids tensor
        seq_len: Sequence length
        device: Device

    Returns:
        R: Relevancy matrix
        predicting_positions: List of positions that predict answer tokens
    """
    model.eval()

    with torch.enable_grad():
        outputs = model(**full_inputs, output_attentions=True, return_dict=True)

        for attn in outputs.attentions:
            if attn is not None:
                attn.requires_grad_(True)
                attn.retain_grad()

        # Sum logits for all answer tokens
        # Position p-1 predicts token at p
        total_logit = 0
        predicting_positions = []

        for p in range(answer_start, answer_end):
            logit_pos = p - 1
            token_id = input_ids[0, p].item()
            total_logit = total_logit + outputs.logits[0, logit_pos, token_id]
            predicting_positions.append(logit_pos)

        # Backward from summed objective
        total_logit.backward(retain_graph=True)

        # Collect attention and gradients
        attention_maps = {}
        attention_grads = {}

        for i, attn in enumerate(outputs.attentions):
            if attn is not None:
                attention_maps[i] = attn.detach().float()
                if attn.grad is not None:
                    attention_grads[i] = attn.grad.detach().float()

    # Propagate relevancy
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


def main():
    print("=" * 60)
    print("Keyword-Specific Explanation - Chefer Method")
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

    # Load cat + remote image
    image = Image.open('/content/cat_tv_remote.jpg').convert('RGB')
    prompt = "Is there a remote control in this image? If so, where is it located?"

    # Prepare inputs
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors='pt').to(device)

    input_len = inputs['input_ids'].shape[1]
    print(f"\nInput length (prompt): {input_len}")

    # Generate response
    print("\nGenerating response...")
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    seq_len = gen_outputs.shape[1]
    answer_start = input_len
    answer_end = seq_len

    # Decode tokens for analysis
    all_tokens = [processor.decode([t.item()]) for t in gen_outputs[0]]
    print(f"\nGenerated {answer_end - answer_start} answer tokens")
    print(f"Answer tokens: {all_tokens[answer_start:]}")

    # Verify image token structure
    print(f"\n--- Token Structure Check ---")
    print(f"Positions 4-7: {all_tokens[4:8]}")
    print(f"Position {IMAGE_START_IDX + NUM_IMAGE_TOKENS} (should be end_of_image): {all_tokens[IMAGE_START_IDX + NUM_IMAGE_TOKENS]}")

    # Find keyword tokens in the answer
    keywords = {'remote': [], 'couch': [], 'center': [], 'image': []}
    for i, token in enumerate(all_tokens[answer_start:]):
        for kw in keywords:
            if kw.lower() in token.lower():
                keywords[kw].append(answer_start + i)

    print(f"\nKeyword positions in answer:")
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
    # TEST 1: Explain "remote" token specifically
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST 1: Explain 'remote' token")
    print("=" * 60)

    if keywords['remote']:
        remote_pos = keywords['remote'][0]
        remote_token_id = gen_outputs[0, remote_pos].item()

        print(f"Remote token at position {remote_pos}")
        print(f"Token id: {remote_token_id}, decoded: '{processor.decode([remote_token_id])}'")
        print(f"Backprop from logit at position {remote_pos - 1}")

        R_remote, _, logit_pos = explain_token(
            model, full_inputs, remote_pos, remote_token_id, seq_len, device
        )

        # Extract relevancy from the predicting position (p-1)
        token_rel = extract_token_relevancy(R_remote, logit_pos)
        img_rel, _ = split_relevancy(token_rel, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
        img_2d_remote = normalize_relevancy(reshape_image_relevancy(img_rel))

        hm = img_2d_remote.cpu().numpy()
        print(f"\n'remote' token relevancy:")
        print(f"  Top-Left:      {hm[:8, :8].mean():.4f}")
        print(f"  Top-Right:     {hm[:8, 8:].mean():.4f}")
        print(f"  Bottom-Left:   {hm[8:, :8].mean():.4f}")
        print(f"  Bottom-Center: {hm[10:, 5:11].mean():.4f} <-- REMOTE LOCATION")
        print(f"  Bottom-Right:  {hm[8:, 8:].mean():.4f}")

    # ==========================================
    # TEST 2: Explain whole answer span
    # ==========================================
    print("\n" + "=" * 60)
    print("TEST 2: Explain whole answer")
    print("=" * 60)

    R_answer, pred_positions = explain_answer_span(
        model, full_inputs, answer_start, answer_end, gen_outputs, seq_len, device
    )

    # Average relevancy rows over predicting positions
    avg_relevancy = torch.zeros(seq_len, device=device)
    for pos in pred_positions:
        avg_relevancy += R_answer[pos, :]
    avg_relevancy = avg_relevancy / len(pred_positions)

    img_rel_answer, _ = split_relevancy(avg_relevancy, NUM_IMAGE_TOKENS, IMAGE_START_IDX)
    img_2d_answer = normalize_relevancy(reshape_image_relevancy(img_rel_answer))

    hm = img_2d_answer.cpu().numpy()
    print(f"\nWhole answer relevancy:")
    print(f"  Top-Left:      {hm[:8, :8].mean():.4f}")
    print(f"  Top-Right:     {hm[:8, 8:].mean():.4f}")
    print(f"  Bottom-Left:   {hm[8:, :8].mean():.4f}")
    print(f"  Bottom-Center: {hm[10:, 5:11].mean():.4f} <-- REMOTE LOCATION")
    print(f"  Bottom-Right:  {hm[8:, 8:].mean():.4f}")

    # ==========================================
    # Create visualization
    # ==========================================
    print("\n--- Creating visualization ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def show_heatmap(ax, hm_tensor, title):
        hm = hm_tensor.cpu().numpy() if torch.is_tensor(hm_tensor) else hm_tensor
        hm_resized = np.array(Image.fromarray(hm).resize(image.size, Image.BILINEAR))
        ax.imshow(image)
        im = ax.imshow(hm_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

        # Show region values
        bc = hm[10:, 5:11].mean()
        tl = hm[:8, :8].mean()
        ax.text(0.02, 0.02, f"Bottom-center: {bc:.3f}\nTop-left: {tl:.3f}",
                transform=ax.transAxes, fontsize=9, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        return im

    # Original image with remote location marked
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image\n(Remote at bottom-center)', fontsize=12)
    axes[0, 0].axis('off')
    from matplotlib.patches import Rectangle
    rect = Rectangle((image.size[0]*0.3, image.size[1]*0.65),
                     image.size[0]*0.4, image.size[1]*0.3,
                     linewidth=3, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)

    # "remote" token explanation
    if keywords['remote']:
        show_heatmap(axes[0, 1], img_2d_remote,
                     '"remote" token explanation\n(backprop from position p-1)')

    # Whole answer explanation
    show_heatmap(axes[1, 0], img_2d_answer,
                 'Whole answer explanation\n(summed logits, averaged relevancy)')

    # Region comparison bar chart
    if keywords['remote']:
        regions_remote = {
            'Top-Left': img_2d_remote.cpu().numpy()[:8, :8].mean(),
            'Top-Right': img_2d_remote.cpu().numpy()[:8, 8:].mean(),
            'Bottom-Left': img_2d_remote.cpu().numpy()[8:, :8].mean(),
            'Bottom-Center\n(REMOTE)': img_2d_remote.cpu().numpy()[10:, 5:11].mean(),
            'Bottom-Right': img_2d_remote.cpu().numpy()[8:, 8:].mean(),
        }

        colors = ['steelblue'] * 3 + ['red'] + ['steelblue']
        bars = axes[1, 1].barh(list(regions_remote.keys()), list(regions_remote.values()), color=colors)
        axes[1, 1].set_xlabel('Mean Relevancy')
        axes[1, 1].set_title('Region Analysis\n("remote" token)', fontsize=11)
        for bar, (name, val) in zip(bars, regions_remote.items()):
            axes[1, 1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=10)

    plt.suptitle('Chefer Method - Keyword Explanation\nCat + Remote: "Where is the remote?"',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/content/outputs/cat_remote_explanation.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/cat_remote_explanation.png")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
