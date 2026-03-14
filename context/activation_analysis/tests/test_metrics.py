"""Tests for similarity metrics (no GPU required)."""

import pytest
import torch

from src.metrics import (
    cosine_similarity_paired,
    l2_distance_paired,
    mean_pool,
    cosine_similarity_segment,
    linear_cka,
    compute_layer_metrics,
    compute_all_metrics,
)


class TestCosineSimilarityPaired:
    def test_identical_vectors(self):
        a = torch.randn(10, 128)
        result = cosine_similarity_paired(a, a)
        assert result.shape == (10,)
        assert torch.allclose(result, torch.ones(10), atol=1e-5)

    def test_orthogonal_vectors(self):
        a = torch.zeros(1, 4)
        b = torch.zeros(1, 4)
        a[0, 0] = 1.0
        b[0, 1] = 1.0
        result = cosine_similarity_paired(a, b)
        assert abs(result[0].item()) < 1e-5

    def test_opposite_vectors(self):
        a = torch.randn(5, 64)
        result = cosine_similarity_paired(a, -a)
        assert torch.allclose(result, -torch.ones(5), atol=1e-5)

    def test_output_range(self):
        a = torch.randn(20, 256)
        b = torch.randn(20, 256)
        result = cosine_similarity_paired(a, b)
        assert (result >= -1.0 - 1e-5).all()
        assert (result <= 1.0 + 1e-5).all()


class TestL2DistancePaired:
    def test_identical_vectors(self):
        a = torch.randn(10, 128)
        result = l2_distance_paired(a, a)
        assert torch.allclose(result, torch.zeros(10), atol=1e-5)

    def test_known_distance(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[3.0, 4.0]])
        result = l2_distance_paired(a, b)
        assert abs(result[0].item() - 5.0) < 1e-5

    def test_non_negative(self):
        a = torch.randn(20, 256)
        b = torch.randn(20, 256)
        result = l2_distance_paired(a, b)
        assert (result >= 0).all()


class TestMeanPool:
    def test_single_token(self):
        a = torch.randn(1, 128)
        result = mean_pool(a)
        assert torch.allclose(result, a[0])

    def test_multiple_tokens(self):
        a = torch.ones(5, 64)
        result = mean_pool(a)
        assert torch.allclose(result, torch.ones(64))


class TestCosineSegment:
    def test_identical(self):
        a = torch.randn(10, 128)
        result = cosine_similarity_segment(a, a)
        assert abs(result - 1.0) < 1e-4

    def test_different_token_counts(self):
        a = torch.randn(10, 128)
        b = torch.randn(20, 128)
        result = cosine_similarity_segment(a, b)
        assert -1.0 <= result <= 1.0


class TestLinearCKA:
    def test_identical(self):
        x = torch.randn(10, 64)
        result = linear_cka(x, x)
        assert abs(result - 1.0) < 1e-4

    def test_range(self):
        x = torch.randn(15, 64)
        y = torch.randn(20, 64)
        result = linear_cka(x, y)
        assert 0.0 <= result <= 1.0 + 1e-5

    def test_different_token_counts(self):
        x = torch.randn(10, 128)
        y = torch.randn(25, 128)
        result = linear_cka(x, y)
        assert isinstance(result, float)

    def test_constant_zero(self):
        x = torch.zeros(10, 64)
        y = torch.randn(10, 64)
        result = linear_cka(x, y)
        assert result == 0.0


class TestComputeLayerMetrics:
    def test_full_metrics(self):
        a = torch.randn(10, 128)
        b = torch.randn(10, 128)
        metrics = compute_layer_metrics(a, b, layer_index=4)
        assert metrics.layer_index == 4
        assert metrics.cosine_mean is not None
        assert metrics.l2_mean is not None
        assert metrics.cka is not None

    def test_segment_only(self):
        a = torch.randn(10, 128)
        b = torch.randn(20, 128)
        metrics = compute_layer_metrics(a, b, layer_index=8, token_level=False)
        assert metrics.cosine_per_token is None
        assert metrics.cosine_segment is not None
        assert metrics.cka is not None


class TestComputeAllMetrics:
    def test_multiple_layers(self):
        orig = {0: torch.randn(10, 64), 4: torch.randn(10, 64), 8: torch.randn(10, 64)}
        pert = {0: torch.randn(10, 64), 4: torch.randn(10, 64), 8: torch.randn(10, 64)}
        results = compute_all_metrics(orig, pert)
        assert len(results) == 3
        assert results[0].layer_index == 0
        assert results[1].layer_index == 4
        assert results[2].layer_index == 8

    def test_partial_overlap(self):
        orig = {0: torch.randn(10, 64), 4: torch.randn(10, 64)}
        pert = {4: torch.randn(10, 64), 8: torch.randn(10, 64)}
        results = compute_all_metrics(orig, pert)
        assert len(results) == 1
        assert results[0].layer_index == 4
