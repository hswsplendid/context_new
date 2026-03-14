"""Tests for token alignment logic (no GPU required)."""

import pytest

from src.prompt_builder import PromptPair
from src.token_aligner import align_type1, align_type2, compute_alignment, AlignmentMap


def _make_type1_pair(
    orig_ids: list[int],
    pert_ids: list[int],
    subseq_orig: tuple[int, int],
    subseq_pert: tuple[int, int],
) -> PromptPair:
    """Helper to create a Type1 PromptPair for testing."""
    return PromptPair(
        original_text="",
        perturbed_text="",
        original_token_ids=orig_ids,
        perturbed_token_ids=pert_ids,
        prefix_range=(0, 5),
        original_segment_range=(5, 10),
        perturbed_segment_range=(5, 10),
        subsequent_range_original=subseq_orig,
        subsequent_range_perturbed=subseq_pert,
        perturbation_type="type1",
    )


class TestAlignType1:
    def test_perfect_alignment(self):
        # Subsequent tokens are identical
        ids = [1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 200, 201, 202, 203, 204]
        pair = _make_type1_pair(
            orig_ids=ids,
            pert_ids=[1, 2, 3, 4, 5, 500, 501, 502, 503, 504, 200, 201, 202, 203, 204],
            subseq_orig=(10, 15),
            subseq_pert=(10, 15),
        )
        alignment = align_type1(pair)
        assert alignment.num_aligned == 5
        assert alignment.num_trimmed == 0
        assert alignment.original_indices == [10, 11, 12, 13, 14]
        assert alignment.perturbed_indices == [10, 11, 12, 13, 14]

    def test_offset_alignment(self):
        # B1 is shorter, so B2 starts at different position
        pair = _make_type1_pair(
            orig_ids=[1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 200, 201, 202],
            pert_ids=[1, 2, 3, 4, 5, 500, 501, 502, 200, 201, 202],
            subseq_orig=(10, 13),
            subseq_pert=(8, 11),
        )
        alignment = align_type1(pair)
        assert alignment.num_aligned == 3
        assert alignment.original_indices == [10, 11, 12]
        assert alignment.perturbed_indices == [8, 9, 10]

    def test_bpe_boundary_mismatch(self):
        # First token differs due to BPE, rest match
        pair = _make_type1_pair(
            orig_ids=[1, 2, 3, 4, 5, 100, 101, 102, 999, 200, 201, 202],
            pert_ids=[1, 2, 3, 4, 5, 500, 501, 888, 200, 201, 202],
            subseq_orig=(8, 12),
            subseq_pert=(7, 11),
        )
        alignment = align_type1(pair)
        # First tokens differ (999 vs 888), should be trimmed
        assert alignment.num_trimmed >= 1
        assert alignment.num_aligned >= 2

    def test_empty_subsequent(self):
        pair = _make_type1_pair(
            orig_ids=[1, 2, 3, 4, 5, 100, 101],
            pert_ids=[1, 2, 3, 4, 5, 500, 501],
            subseq_orig=(7, 7),
            subseq_pert=(7, 7),
        )
        alignment = align_type1(pair)
        assert alignment.num_aligned == 0

    def test_no_subsequent_range_raises(self):
        pair = PromptPair(
            original_text="",
            perturbed_text="",
            original_token_ids=[1, 2, 3],
            perturbed_token_ids=[1, 2, 3],
            prefix_range=(0, 1),
            original_segment_range=(1, 2),
            perturbed_segment_range=(1, 2),
            subsequent_range_original=None,
            subsequent_range_perturbed=None,
            perturbation_type="type1",
        )
        with pytest.raises(ValueError):
            align_type1(pair)


class TestAlignType2:
    def test_returns_segment_ranges(self):
        pair = PromptPair(
            original_text="",
            perturbed_text="",
            original_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            perturbed_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            prefix_range=(0, 3),
            original_segment_range=(3, 8),
            perturbed_segment_range=(3, 6),
            subsequent_range_original=None,
            subsequent_range_perturbed=None,
            perturbation_type="type2",
        )
        alignment = align_type2(pair)
        assert alignment.alignment_type == "type2"
        assert alignment.original_segment_range == (3, 8)
        assert alignment.perturbed_segment_range == (3, 6)


class TestComputeAlignment:
    def test_dispatches_type1(self):
        pair = _make_type1_pair(
            orig_ids=[1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 200, 201],
            pert_ids=[1, 2, 3, 4, 5, 500, 501, 502, 503, 504, 200, 201],
            subseq_orig=(10, 12),
            subseq_pert=(10, 12),
        )
        alignment = compute_alignment(pair)
        assert alignment.alignment_type == "type1"

    def test_dispatches_type2(self):
        pair = PromptPair(
            original_text="",
            perturbed_text="",
            original_token_ids=[1, 2, 3],
            perturbed_token_ids=[1, 2, 3],
            prefix_range=(0, 1),
            original_segment_range=(1, 2),
            perturbed_segment_range=(1, 2),
            subsequent_range_original=None,
            subsequent_range_perturbed=None,
            perturbation_type="type2",
        )
        alignment = compute_alignment(pair)
        assert alignment.alignment_type == "type2"

    def test_unknown_type_raises(self):
        pair = PromptPair(
            original_text="",
            perturbed_text="",
            original_token_ids=[1, 2, 3],
            perturbed_token_ids=[1, 2, 3],
            prefix_range=(0, 1),
            original_segment_range=(1, 2),
            perturbed_segment_range=(1, 2),
            subsequent_range_original=None,
            subsequent_range_perturbed=None,
            perturbation_type="type99",
        )
        with pytest.raises(ValueError):
            compute_alignment(pair)
