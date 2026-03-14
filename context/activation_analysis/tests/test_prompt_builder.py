"""Tests for prompt construction and boundary tracking (no GPU required)."""

import pytest
from unittest.mock import MagicMock

from src.prompt_builder import (
    PromptPair,
    PromptTemplate,
    build_prompt_pair,
    _pad_or_trim_text,
    _place_segment,
    load_template,
)


class MockTokenizer:
    """Mock tokenizer that splits on whitespace for testing."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        if not text or text.isspace():
            return []
        tokens = text.split()
        # Map each word to a deterministic integer
        ids = [hash(w) % 10000 for w in tokens]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        # Inverse mapping not exact, but sufficient for testing
        return " ".join([f"tok_{i}" for i in ids])


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_template():
    return PromptTemplate(
        name="test",
        system_prompt="You are a helpful assistant.",
        prefix="This is the prefix section of the prompt.",
        segment="This is the segment that will be perturbed in the experiment.",
        subsequent="This is the subsequent text that follows the perturbed segment.",
        description="Test template",
    )


class TestPadOrTrimText:
    def test_pad_short_text(self, mock_tokenizer):
        result = _pad_or_trim_text("hello world", 20, mock_tokenizer)
        ids = mock_tokenizer.encode(result)
        assert len(ids) == 20

    def test_trim_long_text(self, mock_tokenizer):
        long_text = " ".join(["word"] * 100)
        result = _pad_or_trim_text(long_text, 10, mock_tokenizer)
        ids = mock_tokenizer.encode(result)
        assert len(ids) == 10

    def test_exact_length(self, mock_tokenizer):
        text = "one two three"
        target = len(mock_tokenizer.encode(text))
        result = _pad_or_trim_text(text, target, mock_tokenizer)
        ids = mock_tokenizer.encode(result)
        assert len(ids) == target


class TestPlaceSegment:
    def test_beginning_position(self, mock_tokenizer):
        prefix, seg, subsequent = _place_segment(
            "prefix text", "segment text", "subsequent text",
            "beginning", 50, mock_tokenizer,
        )
        assert prefix == ""
        assert seg == "segment text"

    def test_end_position(self, mock_tokenizer):
        prefix, seg, subsequent = _place_segment(
            "prefix text", "segment text", "subsequent text",
            "end", 50, mock_tokenizer,
        )
        assert subsequent == ""
        assert seg == "segment text"

    def test_middle_position(self, mock_tokenizer):
        prefix, seg, subsequent = _place_segment(
            "prefix text", "segment text", "subsequent text",
            "middle", 50, mock_tokenizer,
        )
        assert seg == "segment text"
        assert prefix != ""
        assert subsequent != ""


class TestBuildPromptPair:
    def test_type1_has_subsequent_ranges(self, mock_tokenizer, sample_template):
        pair = build_prompt_pair(
            template=sample_template,
            tokenizer=mock_tokenizer,
            perturbation_type="type1",
            target_length=100,
            perturbation_ratio=0.25,
            position="middle",
            original_segment="original segment text here",
            replacement_segment="replacement segment text different",
        )
        assert pair.perturbation_type == "type1"
        assert pair.subsequent_range_original is not None
        assert pair.subsequent_range_perturbed is not None

    def test_type2_no_subsequent_ranges(self, mock_tokenizer, sample_template):
        pair = build_prompt_pair(
            template=sample_template,
            tokenizer=mock_tokenizer,
            perturbation_type="type2",
            target_length=100,
            perturbation_ratio=0.25,
            position="middle",
            original_segment="original segment text here",
            replacement_segment="paraphrased segment text here",
        )
        assert pair.perturbation_type == "type2"
        assert pair.subsequent_range_original is None
        assert pair.subsequent_range_perturbed is None

    def test_token_ids_populated(self, mock_tokenizer, sample_template):
        pair = build_prompt_pair(
            template=sample_template,
            tokenizer=mock_tokenizer,
            perturbation_type="type1",
            target_length=100,
            perturbation_ratio=0.25,
            position="middle",
            original_segment="original text",
            replacement_segment="replacement text",
        )
        assert len(pair.original_token_ids) > 0
        assert len(pair.perturbed_token_ids) > 0

    def test_prefix_range_valid(self, mock_tokenizer, sample_template):
        pair = build_prompt_pair(
            template=sample_template,
            tokenizer=mock_tokenizer,
            perturbation_type="type1",
            target_length=100,
            perturbation_ratio=0.25,
            position="middle",
            original_segment="original text",
            replacement_segment="replacement text",
        )
        start, end = pair.prefix_range
        assert start == 0
        assert end >= 0
        assert end <= len(pair.original_token_ids)


class TestLoadTemplate:
    def test_load_tool_use_template(self, tmp_path):
        import yaml
        template_data = {
            "name": "test_tool",
            "system_prompt": "You are helpful.",
            "prefix": "Prefix text.",
            "segment": "Segment text.",
            "subsequent": "Subsequent text.",
            "description": "A test template.",
        }
        path = tmp_path / "test_template.yaml"
        with open(path, "w") as f:
            yaml.dump(template_data, f)

        template = load_template(path)
        assert template.name == "test_tool"
        assert template.system_prompt == "You are helpful."
