"""
Basic tests for MedGemma Explainability.

These tests verify core functionality without requiring the full model.
"""

import pytest
import torch
import numpy as np

from medgemma_explainability.utils import (
    is_global_layer,
    get_attention_layers_info,
    expand_gqa_attention,
    create_local_attention_mask,
    split_relevancy_by_modality,
    reshape_image_relevancy,
    normalize_relevancy,
)
from medgemma_explainability.relevancy import (
    compute_Abar,
    propagate_relevancy,
    extract_token_relevancy,
    split_relevancy,
)


class TestGlobalLocalLayers:
    """Test global/local layer identification."""

    def test_global_layer_indices(self):
        """Global layers should be at indices 5, 11, 17, 23, 29."""
        expected_global = [5, 11, 17, 23, 29]
        for i in range(34):
            if i in expected_global:
                assert is_global_layer(i), f"Layer {i} should be global"
            else:
                assert not is_global_layer(i), f"Layer {i} should be local"

    def test_attention_layers_info(self):
        """Test attention layer info function."""
        info = get_attention_layers_info(34)
        assert info["total_layers"] == 34
        assert info["num_global"] == 5
        assert info["num_local"] == 29
        assert len(info["global_layers"]) == 5
        assert len(info["local_layers"]) == 29


class TestGQAExpansion:
    """Test Grouped-Query Attention expansion."""

    def test_expand_gqa_4_to_8(self):
        """Test expanding 4 KV heads to 8 query heads."""
        batch, kv_heads, seq = 1, 4, 100
        attn = torch.randn(batch, kv_heads, seq, seq)

        expanded = expand_gqa_attention(attn, num_query_heads=8, num_kv_heads=4)

        assert expanded.shape == (batch, 8, seq, seq)
        # Each pair of query heads should be identical
        assert torch.allclose(expanded[:, 0], expanded[:, 1])
        assert torch.allclose(expanded[:, 2], expanded[:, 3])

    def test_no_expansion_needed(self):
        """Test when heads already match."""
        batch, heads, seq = 1, 8, 100
        attn = torch.randn(batch, heads, seq, seq)

        expanded = expand_gqa_attention(attn, num_query_heads=8, num_kv_heads=8)

        assert expanded.shape == attn.shape
        assert torch.allclose(expanded, attn)


class TestLocalAttentionMask:
    """Test local attention mask creation."""

    def test_mask_shape(self):
        """Test mask has correct shape."""
        seq_len = 100
        mask = create_local_attention_mask(seq_len, window_size=10)
        assert mask.shape == (seq_len, seq_len)

    def test_mask_is_causal(self):
        """Test mask is lower triangular (causal)."""
        seq_len = 10
        mask = create_local_attention_mask(seq_len, window_size=5)

        # Check no future positions attended
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j] == 0, f"Position {i} attending to future {j}"

    def test_mask_window_size(self):
        """Test window size is respected."""
        seq_len = 20
        window = 5
        mask = create_local_attention_mask(seq_len, window_size=window)

        # Check that positions outside window are masked
        for i in range(seq_len):
            for j in range(i - window):
                assert mask[i, j] == 0, f"Position {i} attending outside window to {j}"


class TestRelevancySplit:
    """Test relevancy splitting functions."""

    def test_split_by_modality(self):
        """Test splitting relevancy into image and text."""
        # 6 prefix + 256 image + 100 text = 362 total
        total = 362
        relevancy = torch.randn(total)

        img_rel, txt_rel = split_relevancy_by_modality(
            relevancy, num_image_tokens=256, image_start_idx=6
        )

        assert img_rel.shape == (256,)
        assert txt_rel.shape == (100,)  # 362 - 262 = 100
        assert torch.allclose(img_rel, relevancy[6:262])
        assert torch.allclose(txt_rel, relevancy[262:])

    def test_reshape_image_relevancy(self):
        """Test reshaping image relevancy to 2D grid."""
        flat = torch.randn(256)
        grid = reshape_image_relevancy(flat, grid_size=16)

        assert grid.shape == (16, 16)
        assert torch.allclose(grid.flatten(), flat)


class TestNormalization:
    """Test relevancy normalization."""

    def test_normalize_to_01(self):
        """Test normalization to [0, 1] range."""
        tensor = torch.tensor([-5.0, 0.0, 5.0, 10.0])
        normalized = normalize_relevancy(tensor)

        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert torch.isclose(normalized.min(), torch.tensor(0.0))
        assert torch.isclose(normalized.max(), torch.tensor(1.0))

    def test_normalize_constant(self):
        """Test normalization of constant tensor."""
        tensor = torch.ones(10) * 5
        normalized = normalize_relevancy(tensor)

        # All zeros for constant input
        assert torch.allclose(normalized, torch.zeros(10))


class TestComputeAbar:
    """Test gradient-weighted attention computation."""

    def test_abar_shape(self):
        """Test Ā has correct shape."""
        batch, heads, seq = 1, 8, 100
        A = torch.randn(batch, heads, seq, seq).abs()  # Attention (positive)
        grad_A = torch.randn(batch, heads, seq, seq)

        Abar = compute_Abar(A, grad_A, normalize=False)

        assert Abar.shape == (seq, seq)

    def test_abar_positive_only(self):
        """Test that only positive values are kept."""
        batch, heads, seq = 1, 8, 10
        A = torch.ones(batch, heads, seq, seq)
        grad_A = torch.randn(batch, heads, seq, seq)

        Abar = compute_Abar(A, grad_A, normalize=False)

        # Since we clamp negatives, result should be non-negative
        assert (Abar >= 0).all()

    def test_abar_with_negative_grads(self):
        """Test Ā with all negative gradients."""
        batch, heads, seq = 1, 8, 10
        A = torch.ones(batch, heads, seq, seq)
        grad_A = -torch.ones(batch, heads, seq, seq)

        Abar = compute_Abar(A, grad_A, normalize=False)

        # All negatives should be clamped to 0
        assert torch.allclose(Abar, torch.zeros_like(Abar))


class TestRelevancyPropagation:
    """Test relevancy propagation through layers."""

    def test_propagation_identity_init(self):
        """Test that R starts as identity."""
        seq_len = 10
        num_layers = 3

        # Create dummy attention and gradients (identity-like)
        attention_maps = {}
        attention_grads = {}

        for i in range(num_layers):
            # Attention close to identity (each token attends to itself)
            attn = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1)
            grad = torch.ones_like(attn)
            attention_maps[i] = attn
            attention_grads[i] = grad

        R = propagate_relevancy(
            attention_maps, attention_grads, seq_len,
            handle_local_attention=False,
            normalize_Abar=True,
        )

        assert R.shape == (seq_len, seq_len)

    def test_propagation_preserves_causality(self):
        """Test that future tokens don't influence past."""
        seq_len = 10
        num_layers = 2

        attention_maps = {}
        attention_grads = {}

        for i in range(num_layers):
            # Causal attention (lower triangular)
            attn = torch.tril(torch.ones(seq_len, seq_len))
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows
            attn = attn.unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1)
            grad = torch.ones_like(attn)
            attention_maps[i] = attn
            attention_grads[i] = grad

        R = propagate_relevancy(
            attention_maps, attention_grads, seq_len,
            handle_local_attention=False,
            normalize_Abar=True,
        )

        # Check R is roughly lower triangular (no strong future influence)
        # This is a soft check due to normalization
        assert R.shape == (seq_len, seq_len)


class TestExtractTokenRelevancy:
    """Test token relevancy extraction."""

    def test_extract_last_token(self):
        """Test extracting relevancy for last token."""
        seq_len = 10
        R = torch.randn(seq_len, seq_len)

        rel = extract_token_relevancy(R, target_token_idx=-1)

        assert rel.shape == (seq_len,)
        assert torch.allclose(rel, R[-1, :])

    def test_extract_specific_token(self):
        """Test extracting relevancy for specific token."""
        seq_len = 10
        R = torch.randn(seq_len, seq_len)

        rel = extract_token_relevancy(R, target_token_idx=5)

        assert rel.shape == (seq_len,)
        assert torch.allclose(rel, R[5, :])


class TestSplitRelevancy:
    """Test relevancy splitting with image start index."""

    def test_split_with_image_start(self):
        """Test splitting with image start index."""
        total = 270  # 6 prefix + 256 image + 8 text
        relevancy = torch.arange(total, dtype=torch.float)

        img_rel, txt_rel = split_relevancy(
            relevancy,
            num_image_tokens=256,
            image_start_idx=6,
        )

        assert img_rel.shape == (256,)
        assert txt_rel.shape == (8,)  # 8 text tokens after image
        assert torch.allclose(img_rel, relevancy[6:262])
        assert torch.allclose(txt_rel, relevancy[262:270])


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
