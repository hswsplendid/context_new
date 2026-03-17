"""
Perturbation strategies.

Each strategy operates on a *token-id list* and modifies a contiguous span
[start, start+span_len).  The output is a new token-id list together with
metadata describing what changed.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Outcome of applying a perturbation."""

    original_ids: List[int]
    perturbed_ids: List[int]
    # Span in the *original* token list that was modified.
    original_span_start: int
    original_span_end: int
    # Span in the *perturbed* token list that replaced the original span.
    perturbed_span_start: int
    perturbed_span_end: int
    # The first token position *after* the perturbation in the perturbed seq.
    # Tokens before this position are either unchanged prefix or the new span.
    post_perturbation_start: int
    perturbation_type: str


# -----------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------
class PerturbationStrategy:
    """Abstract base for perturbation strategies."""

    name: str = "base"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.vocab_size = tokenizer.vocab_size

    def apply(
        self,
        token_ids: List[int],
        span_start: int,
        span_len: int,
    ) -> PerturbationResult:
        raise NotImplementedError


# -----------------------------------------------------------------------
# Concrete strategies
# -----------------------------------------------------------------------
class RandomReplace(PerturbationStrategy):
    """Replace the span with uniformly random token ids."""

    name = "random_replace"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        # Collect special token IDs to avoid inserting them.
        special = set(self.tokenizer.all_special_ids)
        replacements = []
        for _ in range(span_len):
            t = self.rng.randint(0, self.vocab_size - 1)
            while t in special:
                t = self.rng.randint(0, self.vocab_size - 1)
            replacements.append(t)
        perturbed = ids[:span_start] + replacements + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + span_len,
            post_perturbation_start=span_start + span_len,
            perturbation_type=self.name,
        )


class MaskReplace(PerturbationStrategy):
    """Replace every token in the span with *[UNK]* or a fixed mask token."""

    name = "mask"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        # Use unk_token_id if available, otherwise pick token id 0.
        mask_id = self.tokenizer.unk_token_id or 0
        replacements = [mask_id] * span_len
        perturbed = ids[:span_start] + replacements + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + span_len,
            post_perturbation_start=span_start + span_len,
            perturbation_type=self.name,
        )


class Delete(PerturbationStrategy):
    """Delete the span entirely. The perturbed sequence is shorter."""

    name = "delete"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        perturbed = ids[:span_start] + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start,  # zero-length span
            post_perturbation_start=span_start,
            perturbation_type=self.name,
        )


class Compress(PerturbationStrategy):
    """Compress the span to roughly half its length.

    Keeps every other token to simulate lossy compression while preserving
    some of the original content.
    """

    name = "compress"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        span = ids[span_start : span_start + span_len]
        # Keep every other token.
        compressed = span[::2]
        perturbed = ids[:span_start] + compressed + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + len(compressed),
            post_perturbation_start=span_start + len(compressed),
            perturbation_type=self.name,
        )


class Paraphrase(PerturbationStrategy):
    """Approximate paraphrasing by shuffling words within the span.

    True paraphrasing would require a generative model. This lightweight
    proxy preserves the bag of words (thus roughly preserving semantics)
    while altering surface form.
    """

    name = "paraphrase"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        span_text = self.tokenizer.decode(
            ids[span_start : span_start + span_len], skip_special_tokens=True
        )
        words = span_text.split()
        self.rng.shuffle(words)
        shuffled_text = " ".join(words)
        new_span_ids = self.tokenizer.encode(shuffled_text, add_special_tokens=False)
        perturbed = ids[:span_start] + new_span_ids + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + len(new_span_ids),
            post_perturbation_start=span_start + len(new_span_ids),
            perturbation_type=self.name,
        )


class SemanticChange(PerturbationStrategy):
    """Replace the span with different content from elsewhere in the vocab.

    Draws token ids from a *different* random passage to produce a span that
    is grammatically plausible but semantically unrelated.
    """

    name = "semantic_change"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 42,
        replacement_pool: Optional[List[int]] = None,
    ):
        super().__init__(tokenizer, seed)
        # If a pool of token ids is supplied (e.g. from a different passage),
        # replacements are drawn from it. Otherwise fall back to random ids.
        self._pool = replacement_pool

    def set_replacement_pool(self, pool: List[int]) -> None:
        self._pool = pool

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        if self._pool and len(self._pool) >= span_len:
            # Draw a contiguous chunk from the pool at a random offset.
            max_offset = len(self._pool) - span_len
            offset = self.rng.randint(0, max_offset)
            replacements = self._pool[offset : offset + span_len]
        else:
            # Fall back to random tokens (filtered of specials).
            special = set(self.tokenizer.all_special_ids)
            replacements = []
            for _ in range(span_len):
                t = self.rng.randint(0, self.vocab_size - 1)
                while t in special:
                    t = self.rng.randint(0, self.vocab_size - 1)
                replacements.append(t)
        perturbed = ids[:span_start] + replacements + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + span_len,
            post_perturbation_start=span_start + span_len,
            perturbation_type=self.name,
        )


# -----------------------------------------------------------------------
# Prompt-diff helper (not a strategy — takes two token lists)
# -----------------------------------------------------------------------

def _longest_common_prefix_len(ids_a: List[int], ids_b: List[int]) -> int:
    """Return the number of leading tokens shared by *ids_a* and *ids_b*."""
    lcp = 0
    for a, b in zip(ids_a, ids_b):
        if a != b:
            break
        lcp += 1
    else:
        lcp = min(len(ids_a), len(ids_b))
    return lcp


def build_prompt_diff_result(
    prev_ids: List[int],
    curr_ids: List[int],
) -> PerturbationResult:
    """Build a :class:`PerturbationResult` from the natural diff of two prompts.

    Finds the longest common prefix (LCP) between *prev_ids* and *curr_ids*
    and treats the diverging suffixes as the "perturbation".

    This is designed for consecutive agent prompts where ``prompt[i]`` is
    roughly a prefix of ``prompt[i+1]`` with new content appended.

    Returns
    -------
    PerturbationResult
        ``original_ids = prev_ids``, ``perturbed_ids = curr_ids``.
        The spans describe the differing suffixes in each sequence, and
        ``post_perturbation_start`` equals the LCP length (the divergence
        point).
    """
    lcp = _longest_common_prefix_len(prev_ids, curr_ids)

    return PerturbationResult(
        original_ids=prev_ids,
        perturbed_ids=curr_ids,
        original_span_start=lcp,
        original_span_end=len(prev_ids),
        perturbed_span_start=lcp,
        perturbed_span_end=len(curr_ids),
        post_perturbation_start=lcp,
        perturbation_type="prompt_diff",
    )


@dataclass
class TripletResult:
    """Outcome of building a triplet comparison from three consecutive prompts.

    Given three prefix-chain prompts P_k ⊂ P_{k+1} ⊂ P_{k+2}:
      - P_k     = [prefix]
      - P_{k+1} = [prefix][D1]
      - P_{k+2} = [prefix][D1][D2]

    We construct:
      - seq_full = P_{k+2}           = [prefix][D1][D2]  (D1 present)
      - seq_skip = [prefix] + [D2]                        (D1 removed)

    The "perturbation" is D1.  Forward measurement compares D2's hidden
    states between seq_full and seq_skip.
    """

    seq_full_ids: List[int]         # P_{k+2} = [prefix][D1][D2]
    seq_skip_ids: List[int]         # [prefix][D2]            (D1 removed)
    prefix_len: int                 # len(P_k) = |prefix|
    d1_len: int                     # len(D1) = len(P_{k+1}) - len(P_k)
    d2_len: int                     # len(D2) = len(P_{k+2}) - len(P_{k+1})
    # Where D2 starts in each sequence (= post-perturbation start):
    d2_start_full: int              # prefix_len + d1_len
    d2_start_skip: int              # prefix_len


def build_triplet_result(
    p_k_ids: List[int],
    p_k1_ids: List[int],
    p_k2_ids: List[int],
) -> Optional[TripletResult]:
    """Construct a triplet comparison from three consecutive prefix-chain prompts.

    Assumes P_k ⊂ P_{k+1} ⊂ P_{k+2} (each is a strict prefix of the next).
    Returns ``None`` if the prefix assumptions don't hold or D1/D2 are empty.
    """
    # Verify P_k is a prefix of P_{k+1}
    lcp_01 = _longest_common_prefix_len(p_k_ids, p_k1_ids)
    if lcp_01 != len(p_k_ids):
        logger.warning(
            "Triplet: P_k (%d tokens) is not a prefix of P_{k+1} (%d tokens). "
            "LCP=%d. Skipping.",
            len(p_k_ids), len(p_k1_ids), lcp_01,
        )
        return None

    # Verify P_{k+1} is a prefix of P_{k+2}
    lcp_12 = _longest_common_prefix_len(p_k1_ids, p_k2_ids)
    if lcp_12 != len(p_k1_ids):
        logger.warning(
            "Triplet: P_{k+1} (%d tokens) is not a prefix of P_{k+2} (%d tokens). "
            "LCP=%d. Skipping.",
            len(p_k1_ids), len(p_k2_ids), lcp_12,
        )
        return None

    prefix_len = len(p_k_ids)
    d1_len = len(p_k1_ids) - prefix_len
    d2_len = len(p_k2_ids) - len(p_k1_ids)

    if d1_len <= 0 or d2_len <= 0:
        logger.warning(
            "Triplet: D1=%d tokens, D2=%d tokens — both must be >0. Skipping.",
            d1_len, d2_len,
        )
        return None

    # seq_full = P_{k+2} = [prefix][D1][D2]
    seq_full_ids = list(p_k2_ids)

    # seq_skip = [prefix][D2] = P_k + D2 portion of P_{k+2}
    d2_tokens = p_k2_ids[len(p_k1_ids):]
    seq_skip_ids = list(p_k_ids) + list(d2_tokens)

    return TripletResult(
        seq_full_ids=seq_full_ids,
        seq_skip_ids=seq_skip_ids,
        prefix_len=prefix_len,
        d1_len=d1_len,
        d2_len=d2_len,
        d2_start_full=prefix_len + d1_len,
        d2_start_skip=prefix_len,
    )


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, type] = {
    "random_replace": RandomReplace,
    "mask": MaskReplace,
    "delete": Delete,
    "compress": Compress,
    "paraphrase": Paraphrase,
    "semantic_change": SemanticChange,
}


def get_strategy(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 42,
    **kwargs,
) -> PerturbationStrategy:
    """Instantiate a perturbation strategy by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown perturbation type '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(tokenizer=tokenizer, seed=seed, **kwargs)
