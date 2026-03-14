"""Token alignment for comparing activations between original and perturbed prompts."""

from dataclasses import dataclass
from typing import Optional

from .prompt_builder import PromptPair


@dataclass
class AlignmentMap:
    """Maps token indices between original and perturbed prompts for comparison.

    For Type 1: paired indices in the subsequent (A2/B2) regions.
    For Type 2: segment ranges for mean-pooling comparison.
    """
    # Type 1: aligned token index pairs in subsequent region
    original_indices: Optional[list[int]] = None  # indices into original tokens
    perturbed_indices: Optional[list[int]] = None  # indices into perturbed tokens
    num_aligned: int = 0
    num_trimmed: int = 0  # boundary tokens trimmed due to BPE mismatch

    # Type 2: segment ranges for pooled comparison
    original_segment_range: Optional[tuple[int, int]] = None
    perturbed_segment_range: Optional[tuple[int, int]] = None

    alignment_type: str = "type1"  # "type1" or "type2"


def align_type1(prompt_pair: PromptPair) -> AlignmentMap:
    """Align subsequent tokens (A2 vs B2) for Type 1 perturbation.

    A2 and B2 have identical text at different positions. We verify token IDs
    match and handle BPE boundary artifacts by trimming mismatched prefix tokens.
    """
    if prompt_pair.subsequent_range_original is None:
        raise ValueError("Type 1 prompt pair must have subsequent ranges")

    orig_range = prompt_pair.subsequent_range_original
    pert_range = prompt_pair.subsequent_range_perturbed

    orig_subseq = prompt_pair.original_token_ids[orig_range[0]: orig_range[1]]
    pert_subseq = prompt_pair.perturbed_token_ids[pert_range[0]: pert_range[1]]

    # Use the shorter length
    min_len = min(len(orig_subseq), len(pert_subseq))
    if min_len == 0:
        return AlignmentMap(
            original_indices=[],
            perturbed_indices=[],
            num_aligned=0,
            num_trimmed=0,
            alignment_type="type1",
        )

    # Find the first position where tokens match (handle BPE boundary mismatch)
    trim_count = 0
    for i in range(min_len):
        if orig_subseq[i] == pert_subseq[i]:
            break
        trim_count += 1
    else:
        # No matching tokens found at all
        return AlignmentMap(
            original_indices=[],
            perturbed_indices=[],
            num_aligned=0,
            num_trimmed=trim_count,
            alignment_type="type1",
        )

    # Verify remaining tokens match
    aligned_orig = []
    aligned_pert = []
    for i in range(trim_count, min_len):
        if orig_subseq[i] == pert_subseq[i]:
            aligned_orig.append(orig_range[0] + i)
            aligned_pert.append(pert_range[0] + i)

    return AlignmentMap(
        original_indices=aligned_orig,
        perturbed_indices=aligned_pert,
        num_aligned=len(aligned_orig),
        num_trimmed=trim_count,
        alignment_type="type1",
    )


def align_type2(prompt_pair: PromptPair) -> AlignmentMap:
    """Align segments (A1 vs B1) for Type 2 perturbation.

    Since tokens differ (paraphrase), we return segment ranges for
    mean-pooling or CKA comparison.
    """
    return AlignmentMap(
        original_segment_range=prompt_pair.original_segment_range,
        perturbed_segment_range=prompt_pair.perturbed_segment_range,
        alignment_type="type2",
    )


def compute_alignment(prompt_pair: PromptPair) -> AlignmentMap:
    """Compute alignment based on perturbation type."""
    if prompt_pair.perturbation_type == "type1":
        return align_type1(prompt_pair)
    elif prompt_pair.perturbation_type == "type2":
        return align_type2(prompt_pair)
    else:
        raise ValueError(f"Unknown perturbation type: {prompt_pair.perturbation_type}")
