"""Tests for activation extractor (no GPU required - uses mock model)."""

import pytest
import torch
import torch.nn as nn

from src.activation_extractor import (
    ActivationResult,
    _make_hook,
    extract_activations,
)


class MockLayer(nn.Module):
    """A mock transformer layer that returns a dummy hidden state."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        # Need at least one parameter
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, **kwargs):
        # Return (hidden_states, ...) to match transformer output format
        return (hidden_states,)


class MockTransformerModel(nn.Module):
    """Mock model with model.model.layers structure."""

    def __init__(self, num_layers=4, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])
        self.model = nn.Module()
        self.model.layers = layers
        self.model.embed = nn.Embedding(1000, hidden_size)

    def forward(self, input_ids, **kwargs):
        # Simple forward: embed -> pass through layers
        hidden = self.model.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return (hidden,)

    def parameters(self):
        return self.model.embed.parameters()


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1


@pytest.fixture
def mock_model():
    model = MockTransformerModel(num_layers=4, hidden_size=64)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


class TestMakeHook:
    def test_hook_captures_selected_tokens(self):
        storage = {}
        token_indices = [0, 2, 4]
        hook = _make_hook(layer_idx=0, token_indices=token_indices, storage=storage, dtype=torch.float32)

        # Simulate a forward pass output
        hidden = torch.randn(1, 10, 64)  # batch=1, seq=10, hidden=64
        module = MockLayer()
        output = (hidden,)

        hook(module, None, output)

        assert 0 in storage
        assert storage[0].shape == (3, 64)
        assert storage[0].device == torch.device("cpu")

    def test_hook_stores_to_cpu(self):
        storage = {}
        hook = _make_hook(layer_idx=2, token_indices=[0], storage=storage, dtype=torch.float32)

        hidden = torch.randn(1, 5, 64)
        hook(MockLayer(), None, (hidden,))

        assert storage[2].device == torch.device("cpu")


class TestExtractActivations:
    def test_basic_extraction(self, mock_model, mock_tokenizer):
        token_ids = list(range(10, 30))  # 20 tokens
        token_indices = [0, 5, 10, 15]
        layer_indices = [0, 2]

        result = extract_activations(
            model=mock_model,
            tokenizer=mock_tokenizer,
            token_ids=token_ids,
            token_indices=token_indices,
            layer_indices=layer_indices,
        )

        assert isinstance(result, ActivationResult)
        assert result.num_layers == 2
        assert result.num_tokens == 4
        assert 0 in result.activations
        assert 2 in result.activations
        assert result.activations[0].shape == (4, 64)

    def test_empty_indices(self, mock_model, mock_tokenizer):
        result = extract_activations(
            model=mock_model,
            tokenizer=mock_tokenizer,
            token_ids=[10, 20, 30],
            token_indices=[],
            layer_indices=[0],
        )
        assert result.num_layers == 0

    def test_out_of_range_layer_skipped(self, mock_model, mock_tokenizer):
        result = extract_activations(
            model=mock_model,
            tokenizer=mock_tokenizer,
            token_ids=list(range(10, 20)),
            token_indices=[0, 1],
            layer_indices=[0, 99],  # layer 99 doesn't exist
        )
        assert 0 in result.activations
        assert 99 not in result.activations

    def test_hooks_cleaned_up(self, mock_model, mock_tokenizer):
        """Verify hooks are removed after extraction."""
        initial_hooks = sum(
            len(layer._forward_hooks) for layer in mock_model.model.layers
        )
        extract_activations(
            model=mock_model,
            tokenizer=mock_tokenizer,
            token_ids=list(range(10, 20)),
            token_indices=[0, 1],
            layer_indices=[0, 1, 2],
        )
        final_hooks = sum(
            len(layer._forward_hooks) for layer in mock_model.model.layers
        )
        assert final_hooks == initial_hooks
