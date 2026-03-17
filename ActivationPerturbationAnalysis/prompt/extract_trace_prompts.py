"""
Extract subagent prefix chains from a Langfuse trace.json export.

Produces a JSON file consumable by PromptGenerator (source='jsonfile').
Each record contains a flattened prompt_text from the message list.

Subagent groups are identified by name pattern 'subagent-<id>-llm-call'.
Within each group, entries are sorted by startTime and verified to form
a strict prefix chain (each entry's messages are a prefix of the next).

Output format:
[
  {
    "prompt_text": "<flattened chat messages>",
    "chain_id": "subagent-440fbdce",
    "chain_index": 0,   # position within the chain
    "chain_length": 15,  # total entries in this chain
    "num_messages": 2,
    "trace_id": "...",
    "start_time": "..."
  },
  ...
]

Usage:
    python extract_trace_prompts.py [--trace trace.json] [--output extracted_prompts.json]
                                    [--min-chain-length 3] [--chat-template chatml]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List


# ── Message flattening ───────────────────────────────────────────────
# Convert a list of {role, content} messages into a single string
# using the ChatML format (used by Qwen, and many other models).

def flatten_messages_chatml(messages: List[Dict[str, str]]) -> str:
    """Flatten messages into ChatML format: <|im_start|>role\ncontent<|im_end|>"""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def flatten_messages_plain(messages: List[Dict[str, str]]) -> str:
    """Flatten messages into a simple role: content format."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


FLATTEN_FNS = {
    "chatml": flatten_messages_chatml,
    "plain": flatten_messages_plain,
}


# ── Extraction ───────────────────────────────────────────────────────

def extract_subagent_chains(
    data: List[Dict[str, Any]],
    min_chain_length: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group subagent entries into prefix chains, sorted by startTime."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        m = re.match(r"(subagent-[a-f0-9]+)-llm-call", entry.get("name", ""))
        if m:
            gid = m.group(1)
            groups.setdefault(gid, []).append(entry)

    # Sort each group by startTime and filter by min length.
    chains = {}
    for gid, entries in groups.items():
        entries.sort(key=lambda x: x["startTime"])
        if len(entries) >= min_chain_length:
            chains[gid] = entries

    return chains


def verify_prefix_chain(entries: List[Dict[str, Any]]) -> bool:
    """Verify that each entry's messages are a strict prefix of the next."""
    for i in range(len(entries) - 1):
        msgs_curr = entries[i]["input"]["messages"]
        msgs_next = entries[i + 1]["input"]["messages"]
        if len(msgs_next) <= len(msgs_curr):
            return False
        # Check that all messages in curr appear at the start of next.
        for j, msg in enumerate(msgs_curr):
            if json.dumps(msg, sort_keys=True) != json.dumps(msgs_next[j], sort_keys=True):
                return False
    return True


def build_records(
    chains: Dict[str, List[Dict[str, Any]]],
    flatten_fn,
    verify: bool = True,
) -> List[Dict[str, Any]]:
    """Convert chains into flat records for extracted_prompts.json."""
    records = []
    skipped_chains = 0

    for gid, entries in sorted(chains.items()):
        if verify and not verify_prefix_chain(entries):
            print(f"  WARNING: chain {gid} fails prefix verification, skipping.",
                  file=sys.stderr)
            skipped_chains += 1
            continue

        chain_len = len(entries)
        for idx, entry in enumerate(entries):
            messages = entry["input"]["messages"]
            prompt_text = flatten_fn(messages)
            records.append({
                "prompt_text": prompt_text,
                "chain_id": gid,
                "chain_index": idx,
                "chain_length": chain_len,
                "num_messages": len(messages),
                "trace_id": entry.get("traceId", ""),
                "start_time": entry.get("startTime", ""),
            })

    if skipped_chains:
        print(f"  Skipped {skipped_chains} chains that failed prefix verification.",
              file=sys.stderr)

    return records


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract subagent prefix chains from Langfuse trace.json"
    )
    parser.add_argument("--trace", default="trace.json",
                        help="Path to the Langfuse trace export (default: trace.json)")
    parser.add_argument("--output", default="extracted_prompts.json",
                        help="Output JSON path (default: extracted_prompts.json)")
    parser.add_argument("--min-chain-length", type=int, default=3,
                        help="Minimum chain length to include (default: 3)")
    parser.add_argument("--chat-template", default="chatml",
                        choices=list(FLATTEN_FNS.keys()),
                        help="Chat template for flattening messages (default: chatml)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip prefix-chain verification")
    args = parser.parse_args()

    print(f"Loading {args.trace}...")
    with open(args.trace, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} entries loaded.")

    chains = extract_subagent_chains(data, min_chain_length=args.min_chain_length)
    print(f"  {len(chains)} subagent chains (>={args.min_chain_length} entries).")

    chain_sizes = [len(v) for v in chains.values()]
    if chain_sizes:
        print(f"  Chain sizes: min={min(chain_sizes)}, max={max(chain_sizes)}, "
              f"total entries={sum(chain_sizes)}")

    flatten_fn = FLATTEN_FNS[args.chat_template]
    records = build_records(chains, flatten_fn, verify=not args.no_verify)
    print(f"  {len(records)} total prompt records.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Written to {args.output}")

    # Print summary by chain
    from collections import Counter
    chain_counts = Counter(r["chain_id"] for r in records)
    print(f"\nChains included: {len(chain_counts)}")
    for cid, cnt in sorted(chain_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {cid}: {cnt} prompts")
    if len(chain_counts) > 5:
        print(f"  ... and {len(chain_counts) - 5} more")


if __name__ == "__main__":
    main()
