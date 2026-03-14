"""LLM-based paraphrase generation for Type 2 perturbations."""

import logging
from pathlib import Path
from typing import Optional

import yaml
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

PARAPHRASE_SYSTEM_PROMPT = (
    "You are a paraphrasing assistant. Rewrite the following text with different "
    "wording but preserve the exact same meaning. Output ONLY the rewritten text, "
    "nothing else. Do not add explanations."
)


def generate_paraphrase(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a paraphrase of the input text using the model.

    Args:
        model: Language model for generation.
        tokenizer: Corresponding tokenizer.
        text: Original text to paraphrase.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        Paraphrased text string.
    """
    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = f"System: {PARAPHRASE_SYSTEM_PROMPT}\nUser: {text}\nAssistant:"

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    paraphrase = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return paraphrase


def batch_generate_paraphrases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    cache_path: Optional[str | Path] = None,
    **kwargs,
) -> list[str]:
    """Generate paraphrases for multiple texts, with optional caching.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        texts: List of original texts.
        cache_path: If provided, load/save paraphrases from/to this YAML file.
        **kwargs: Passed to generate_paraphrase.

    Returns:
        List of paraphrased texts.
    """
    cache = {}
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = yaml.safe_load(f) or {}
            logger.info(f"Loaded {len(cache)} cached paraphrases from {cache_path}")

    results = []
    new_entries = {}

    for text in texts:
        if text in cache:
            results.append(cache[text])
            logger.debug(f"Using cached paraphrase for: {text[:50]}...")
        else:
            paraphrase = generate_paraphrase(model, tokenizer, text, **kwargs)
            results.append(paraphrase)
            new_entries[text] = paraphrase
            logger.info(f"Generated paraphrase: {text[:50]}... -> {paraphrase[:50]}...")

    # Update cache
    if cache_path and new_entries:
        cache.update(new_entries)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            yaml.dump(cache, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved {len(new_entries)} new paraphrases to {cache_path}")

    return results
