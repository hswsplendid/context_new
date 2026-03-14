#!/usr/bin/env python3
"""
extract_prompts.py — Extract real agent prompts from a vLLM server log.

Parses the log produced by vLLM's request logger, pulls out the full prompt
text and token IDs for each request, and writes them to a JSON file that can
be consumed by the experiment framework (via `prompts.source: "jsonfile"`).

Usage:
    python extract_prompts.py --log prompt/agent.log --output prompt/extracted_prompts.json
    python extract_prompts.py --log prompt/agent.log --output prompt/extracted_prompts.json --pretty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# vLLM request logger line format:
#   ... Request <id> details: prompt: '<text>', prompt_token_ids: [<ids>], ...
PROMPT_RE = re.compile(
    r"Request\s+(\S+)\s+details:\s+prompt:\s+'(.*?)',\s+prompt_token_ids:\s+\[([\d,\s]+)\]",
)


def extract_from_log(log_path: str) -> list[dict]:
    """Parse a vLLM log file and return a list of prompt records."""
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = PROMPT_RE.search(line)
            if not m:
                continue

            request_id = m.group(1)
            prompt_text = m.group(2)
            token_ids_str = m.group(3)
            token_ids = [int(t.strip()) for t in token_ids_str.split(",") if t.strip()]

            # Unescape common escapes (\\n → \n etc.) from the log repr.
            prompt_text = (
                prompt_text
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\'", "'")
                .replace('\\"', '"')
            )

            records.append({
                "index": len(records),
                "request_id": request_id,
                "prompt_text": prompt_text,
                "prompt_token_ids": token_ids,
                "num_tokens": len(token_ids),
                "num_chars": len(prompt_text),
            })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract prompts from a vLLM server log into JSON.",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to the vLLM agent log file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: <log-dir>/extracted_prompts.json).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (larger file).",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else log_path.parent / "extracted_prompts.json"

    print(f"Parsing: {log_path}")
    records = extract_from_log(str(log_path))

    if not records:
        print("No prompts found in the log file.", file=sys.stderr)
        sys.exit(1)

    # Summary.
    print(f"Extracted {len(records)} prompts:")
    for r in records:
        print(f"  [{r['index']:2d}] request={r['request_id']}  "
              f"tokens={r['num_tokens']:,}  chars={r['num_chars']:,}")

    # Write.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=indent)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved to: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
