"""
Integration tests for MedGemma Explainability.

These tests require the model to be loaded and work with synthetic data
to verify the full pipeline.
"""

import pytest
import torch
import numpy as np

from medgemma_explainability.attention_hooks import (
    AttentionHookManager,
    verify_attention_extraction,
)
from medgemma_explainability.relevancy import (
    compute_Abar,
    propagate_relevancy,
    propagate_relevancy_additive,
    extract_token_relevancy,
    split_relevancy,
    visualize_relevancy_flow,
)
from medgemma_explainability.utils import (
    is_global_layer,
    expand_gqa_attention,
    normalize_relevancy,
    reshape_image_relevancy,
)


class TestFullPipeline:
    """Test the full relevancy propagation pipeline with synthetic data."""

    @pytest.fixture
    def synthetic_attention_data(self):
        """Create synthetic attention data mimicking MedGemma."""
        num_layers = 34
        batch_size = 1
        num_heads = 8  # After GQA expansion
        seq_len = 300  # 256 image + some text tokens

        attention_maps = {}
        attention_grads = {}

        for layer_idx in range(num_layers):
            # Create causal attention (lower triangular with some noise)
            attn = torch.tril(torch.ones(seq_len, seq_len))
            attn = attn + torch.randn(seq_len, seq_len) * 0.1
            attn = torch.clamp(attn, min=0)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows

            # Expand to full shape
            attn = attn.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

            # Create gradients (some positive, some negative)
            grad = torch.randn(batch_size, num_heads, seq_len, seq_len) * 0.1

            attention_maps[layer_idx] = attn
            attention_grads[layer_idx] = grad

        return {
            'attention_maps': attention_maps,
            'attention_grads': attention_grads,
            'seq_len': seq_len,
            'num_layers': num_layers,
        }

    def test_full_propagation(self, synthetic_attention_data):
        """Test full relevancy propagation through all layers."""
        R = propagate_relevancy(
            synthetic_attention_data['attention_maps'],
            synthetic_attention_data['attention_grads'],
            synthetic_attention_data['seq_len'],
            handle_local_attention=True,
        )

        assert R.shape == (300, 300)
        assert not torch.isnan(R).any()
        assert not torch.isinf(R).any()

    def test_additive_propagation(self, synthetic_attention_data):
        """Test additive propagation variant."""
        R = propagate_relevancy_additive(
            synthetic_attention_data['attention_maps'],
            synthetic_attention_data['attention_grads'],
            synthetic_attention_data['seq_len'],
            handle_local_attention=False,
        )

        assert R.shape == (300, 300)
        assert not torch.isnan(R).any()

    def test_relevancy_extraction_and_split(self, synthetic_attention_data):
        """Test extracting and splitting relevancy."""
        R = propagate_relevancy(
            synthetic_attention_data['attention_maps'],
            synthetic_attention_data['attention_grads'],
            synthetic_attention_data['seq_len'],
        )

        # Extract for last token
        token_rel = extract_token_relevancy(R, target_token_idx=-1)
        assert token_rel.shape == (300,)

        # Split into image and text (with image_start_idx=6)
        img_rel, txt_rel = split_relevancy(token_rel, num_image_tokens=256, image_start_idx=6)
        assert img_rel.shape == (256,)
        assert txt_rel.shape == (38,)  # 300 - 262 = 38

    def test_image_relevancy_reshape(self, synthetic_attention_data):
        """Test reshaping image relevancy to 2D grid."""
        R = propagate_relevancy(
            synthetic_attention_data['attention_maps'],
            synthetic_attention_data['attention_grads'],
            synthetic_attention_data['seq_len'],
        )

        token_rel = extract_token_relevancy(R)
        img_rel, _ = split_relevancy(token_rel, num_image_tokens=256, image_start_idx=6)

        # Reshape to 16x16 grid
        grid = reshape_image_relevancy(img_rel)
        assert grid.shape == (16, 16)

        # Normalize
        grid_norm = normalize_relevancy(grid)
        assert grid_norm.min() >= 0
        assert grid_norm.max() <= 1

    def test_layer_progression(self, synthetic_attention_data):
        """Test visualizing relevancy across layers."""
        layers_to_show = [0, 5, 11, 17, 23, 33]  # Mix of local and global

        layer_relevancies = visualize_relevancy_flow(
            synthetic_attention_data['attention_maps'],
            synthetic_attention_data['attention_grads'],
            synthetic_attention_data['seq_len'],
            layers_to_show=layers_to_show,
        )

        assert len(layer_relevancies) == len(layers_to_show)
        for layer_idx in layers_to_show:
            assert layer_idx in layer_relevancies
            assert layer_relevancies[layer_idx].shape == (300, 300)


class TestGQAHandling:
    """Test Grouped-Query Attention handling."""

    def test_gqa_expansion_in_pipeline(self):
        """Test that GQA expansion works correctly in the pipeline."""
        seq_len = 100
        batch_size = 1
        kv_heads = 4
        query_heads = 8

        # Create attention with KV heads
        attn_kv = torch.randn(batch_size, kv_heads, seq_len, seq_len).abs()
        attn_kv = attn_kv / attn_kv.sum(dim=-1, keepdim=True)

        # Expand to query heads
        attn_expanded = expand_gqa_attention(attn_kv, query_heads, kv_heads)

        assert attn_expanded.shape == (batch_size, query_heads, seq_len, seq_len)

        # Verify pairs are identical
        for i in range(0, query_heads, 2):
            assert torch.allclose(attn_expanded[:, i], attn_expanded[:, i+1])


class TestLocalGlobalAttention:
    """Test local vs global attention handling."""

    def test_global_layer_identification(self):
        """Verify correct identification of global layers."""
        # Global layers in Gemma3 are at positions 5, 11, 17, 23, 29 (5:1 ratio)
        global_layers = [i for i in range(34) if is_global_layer(i)]
        expected = [5, 11, 17, 23, 29]
        assert global_layers == expected

    def test_local_attention_masking_effect(self):
        """Test that local attention masking restricts attention span."""
        from medgemma_explainability.utils import create_local_attention_mask

        seq_len = 2000
        window = 1024

        mask = create_local_attention_mask(seq_len, window)

        # Position 1500 should only attend to positions [477, 1500]
        # (1500 - 1024 + 1 = 477)
        pos = 1500
        expected_start = max(0, pos - window + 1)

        # Check mask is 0 before window and 1 within window
        assert mask[pos, expected_start - 1].item() == 0 if expected_start > 0 else True
        assert mask[pos, expected_start].item() == 1
        assert mask[pos, pos].item() == 1


class TestNumericalStability:
    """Test numerical stability of the propagation."""

    def test_no_nan_with_zero_gradients(self):
        """Test propagation handles zero gradients."""
        seq_len = 100
        attention_maps = {
            0: torch.ones(1, 8, seq_len, seq_len) / seq_len
        }
        attention_grads = {
            0: torch.zeros(1, 8, seq_len, seq_len)
        }

        R = propagate_relevancy(attention_maps, attention_grads, seq_len)

        assert not torch.isnan(R).any()
        assert not torch.isinf(R).any()

    def test_no_nan_with_large_values(self):
        """Test propagation handles large values."""
        seq_len = 50
        attention_maps = {
            0: torch.ones(1, 8, seq_len, seq_len) * 1e-6
        }
        attention_grads = {
            0: torch.ones(1, 8, seq_len, seq_len) * 1e6
        }

        # Normalize attention
        attention_maps[0] = attention_maps[0] / attention_maps[0].sum(dim=-1, keepdim=True)

        R = propagate_relevancy(attention_maps, attention_grads, seq_len)

        assert not torch.isnan(R).any()
        assert not torch.isinf(R).any()


class TestAttentionVerification:
    """Test attention verification utilities."""

    def test_verify_valid_attention(self):
        """Test verification passes for valid attention."""
        seq_len = 50
        attention_weights = {}

        for i in range(34):
            attn = torch.randn(1, 8, seq_len, seq_len).abs()
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Proper softmax
            attention_weights[i] = attn

        result = verify_attention_extraction(attention_weights)

        assert result['all_layers_present']
        assert result['success']
        assert all(result['softmax_valid'].values())

    def test_verify_invalid_shape(self):
        """Test verification catches wrong shape."""
        attention_weights = {
            0: torch.randn(1, 4, 50, 50)  # Wrong number of heads
        }

        result = verify_attention_extraction(attention_weights, expected_layers=1)

        assert not result['success']
        assert len(result['issues']) > 0


class TestHookManager:
    """Test the attention hook manager."""

    def test_hook_manager_initialization(self):
        """Test hook manager initializes correctly."""
        manager = AttentionHookManager(num_query_heads=8, num_kv_heads=4)

        assert manager.num_query_heads == 8
        assert manager.num_kv_heads == 4
        assert len(manager.hooks) == 0
        assert len(manager.attention_cache) == 0

    def test_hook_manager_clear(self):
        """Test clearing hooks and caches."""
        manager = AttentionHookManager()
        manager.attention_cache[0] = "test"
        manager._attention_tensors[0] = torch.randn(1)

        manager.clear()

        assert len(manager.attention_cache) == 0
        assert len(manager._attention_tensors) == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
