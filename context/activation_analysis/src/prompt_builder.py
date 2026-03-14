"""Prompt construction with segment boundary tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from transformers import PreTrainedTokenizer


@dataclass
class PromptPair:
    original_text: str
    perturbed_text: str
    original_token_ids: list[int]
    perturbed_token_ids: list[int]
    prefix_range: tuple[int, int]  # token index range [start, end)
    original_segment_range: tuple[int, int]  # A1 range
    perturbed_segment_range: tuple[int, int]  # B1 range
    subsequent_range_original: Optional[tuple[int, int]]  # A2 (Type1)
    subsequent_range_perturbed: Optional[tuple[int, int]]  # B2 (Type1)
    perturbation_type: str  # "type1" or "type2"


@dataclass
class PromptTemplate:
    """Loaded from YAML template files."""
    name: str
    system_prompt: str
    prefix: str
    segment: str
    subsequent: str
    description: str = ""


def load_template(path: str | Path) -> PromptTemplate:
    """Load a prompt template from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PromptTemplate(**data)


def load_replacements(path: str | Path) -> list[dict]:
    """Load Type1 replacement pairs from YAML.

    Each entry has 'original' and 'replacement' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["replacements"]


def _find_token_range(
    full_ids: list[int],
    text_before: str,
    segment_text: str,
    tokenizer: PreTrainedTokenizer,
) -> tuple[int, int]:
    """Find the token range of segment_text within the full tokenized sequence.

    Tokenizes the prefix (text_before) to find the start index, then tokenizes
    prefix+segment to find the end index.
    """
    prefix_ids = tokenizer.encode(text_before, add_special_tokens=False)
    prefix_plus_segment_ids = tokenizer.encode(
        text_before + segment_text, add_special_tokens=False
    )
    start = len(prefix_ids)
    end = len(prefix_plus_segment_ids)
    return (start, end)


def _pad_or_trim_text(
    text: str, target_tokens: int, tokenizer: PreTrainedTokenizer, pad_char: str = " "
) -> str:
    """Adjust text to approximately target token count by padding with filler."""
    current_ids = tokenizer.encode(text, add_special_tokens=False)
    current_len = len(current_ids)

    if current_len >= target_tokens:
        # Trim by decoding only the first target_tokens tokens
        trimmed_ids = current_ids[:target_tokens]
        return tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    # Pad with filler sentence repeated
    filler = " This is filler text for padding purposes."
    while current_len < target_tokens:
        text += filler
        current_ids = tokenizer.encode(text, add_special_tokens=False)
        current_len = len(current_ids)

    # Trim to exact count
    trimmed_ids = current_ids[:target_tokens]
    return tokenizer.decode(trimmed_ids, skip_special_tokens=True)


def _place_segment(
    prefix: str,
    segment: str,
    subsequent: str,
    position: str,
    target_length: int,
    tokenizer: PreTrainedTokenizer,
) -> tuple[str, str, str]:
    """Arrange prefix/segment/subsequent according to position, targeting token count.

    Returns (final_prefix, segment, final_subsequent) where total tokens ~ target_length.
    """
    seg_tokens = len(tokenizer.encode(segment, add_special_tokens=False))

    if position == "beginning":
        # Segment at start, subsequent fills the rest
        prefix_text = ""
        remaining = target_length - seg_tokens
        subsequent_text = _pad_or_trim_text(
            subsequent, max(remaining, 1), tokenizer
        )
    elif position == "end":
        # Prefix fills most, then segment at the end
        remaining = target_length - seg_tokens
        prefix_text = _pad_or_trim_text(prefix, max(remaining, 1), tokenizer)
        subsequent_text = ""
    else:  # middle
        remaining = target_length - seg_tokens
        half = remaining // 2
        prefix_text = _pad_or_trim_text(prefix, max(half, 1), tokenizer)
        subsequent_text = _pad_or_trim_text(
            subsequent, max(remaining - half, 1), tokenizer
        )

    return prefix_text, segment, subsequent_text


def build_prompt_pair(
    template: PromptTemplate,
    tokenizer: PreTrainedTokenizer,
    perturbation_type: str,
    target_length: int,
    perturbation_ratio: float,
    position: str,
    original_segment: Optional[str] = None,
    replacement_segment: Optional[str] = None,
) -> PromptPair:
    """Build an original/perturbed prompt pair with precise token boundaries.

    Args:
        template: Prompt template with prefix/segment/subsequent.
        tokenizer: Model tokenizer.
        perturbation_type: "type1" or "type2".
        target_length: Target total token count.
        perturbation_ratio: Fraction of tokens to perturb (controls segment length).
        position: Where to place the perturbed segment.
        original_segment: Original segment text (A1). Uses template default if None.
        replacement_segment: Replacement text (B1). Required.
    """
    seg_target = int(target_length * perturbation_ratio)
    orig_seg = original_segment or template.segment
    orig_seg = _pad_or_trim_text(orig_seg, seg_target, tokenizer)
    repl_seg = _pad_or_trim_text(replacement_segment, seg_target, tokenizer)

    # Arrange position
    prefix_text, orig_seg, subsequent_text = _place_segment(
        template.prefix + "\n" + template.system_prompt,
        orig_seg,
        template.subsequent,
        position,
        target_length,
        tokenizer,
    )

    # Build full texts
    original_text = prefix_text + orig_seg + subsequent_text
    perturbed_text = prefix_text + repl_seg + subsequent_text

    # Tokenize
    orig_ids = tokenizer.encode(original_text, add_special_tokens=False)
    pert_ids = tokenizer.encode(perturbed_text, add_special_tokens=False)

    # Find token boundaries
    prefix_end = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    prefix_range = (0, prefix_end)

    orig_prefix_plus_seg = tokenizer.encode(
        prefix_text + orig_seg, add_special_tokens=False
    )
    pert_prefix_plus_seg = tokenizer.encode(
        prefix_text + repl_seg, add_special_tokens=False
    )

    orig_seg_range = (prefix_end, len(orig_prefix_plus_seg))
    pert_seg_range = (prefix_end, len(pert_prefix_plus_seg))

    if perturbation_type == "type1":
        subseq_orig = (len(orig_prefix_plus_seg), len(orig_ids))
        subseq_pert = (len(pert_prefix_plus_seg), len(pert_ids))
    else:
        subseq_orig = None
        subseq_pert = None

    return PromptPair(
        original_text=original_text,
        perturbed_text=perturbed_text,
        original_token_ids=orig_ids,
        perturbed_token_ids=pert_ids,
        prefix_range=prefix_range,
        original_segment_range=orig_seg_range,
        perturbed_segment_range=pert_seg_range,
        subsequent_range_original=subseq_orig,
        subsequent_range_perturbed=subseq_pert,
        perturbation_type=perturbation_type,
    )
