"""
Extract prompt pairs from Feishu/OpenClaw logs to verify prefix-cache invalidation.

Given multiple API call logs from the SAME conversation, this script:
1. Extracts the full prompt text from each call
2. Finds the LCP (longest common prefix) between consecutive calls
3. Identifies where the cache breaks (= where Inbound Metadata changes)
4. Outputs prompts in a format consumable by the existing experiment code

This verifies the help7.md claim: dynamic Inbound Metadata before large
Project Context invalidates the entire Project Context cache.

Usage:
    python extract_feishu_for_cache_test.py --inputs call1.json call2.json call3.json \
        --output feishu_cache_test_prompts.json

    # Or if you only have one log (last call), simulate earlier rounds:
    python extract_feishu_for_cache_test.py --input feishu1.json --simulate-rounds \
        --output feishu_cache_test_prompts.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List


def get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content or "")


def load_call(path: str) -> Dict[str, Any]:
    """Load a single API call log and extract messages."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    req = json.loads(data["request"]) if isinstance(data["request"], str) else data["request"]
    return req["body"]


def flatten_prompt(messages: List[Dict], chat_template: str = "chatml") -> str:
    """Flatten messages into a single string using ChatML format."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = get_text(msg.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def find_lcp(a: str, b: str) -> int:
    """Find length of longest common prefix between two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def simulate_rounds_from_single(path: str) -> List[Dict[str, Any]]:
    """
    From a single API call log (which contains full history),
    simulate what each earlier round's prompt would have looked like.

    This works because OpenClaw sends full history each time:
    Round N's prompt = msgs[0:last_user_msg_index+1]
    """
    body = load_call(path)
    msgs = body["messages"]

    rounds = []
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            round_msgs = msgs[:i + 1]
            prompt_text = flatten_prompt(round_msgs)

            # Extract metadata that changes per round
            user_text = get_text(m.get("content", ""))
            msg_id = ""
            timestamp = ""
            match = re.search(r'"message_id":\s*"([^"]+)"', user_text)
            if match:
                msg_id = match.group(1)
            match = re.search(r'"timestamp":\s*"([^"]+)"', user_text)
            if match:
                timestamp = match.group(1)

            rounds.append({
                "prompt_text": prompt_text,
                "round_index": len(rounds),
                "num_messages": len(round_msgs),
                "char_count": len(prompt_text),
                "message_id": msg_id,
                "timestamp": timestamp,
            })

    return rounds


def analyze_cache_invalidation(rounds: List[Dict[str, Any]]):
    """Analyze where prefix cache breaks between consecutive rounds."""
    print(f"\n{'='*70}")
    print("PREFIX CACHE INVALIDATION ANALYSIS")
    print(f"{'='*70}\n")

    for i in range(len(rounds) - 1):
        a = rounds[i]["prompt_text"]
        b = rounds[i + 1]["prompt_text"]
        lcp_len = find_lcp(a, b)

        # What's at the LCP break point?
        context_before = a[max(0, lcp_len-50):lcp_len]
        context_after_a = a[lcp_len:lcp_len+100] if lcp_len < len(a) else "(end)"
        context_after_b = b[lcp_len:lcp_len+100] if lcp_len < len(b) else "(end)"

        pct_cached = lcp_len / len(b) * 100
        pct_wasted = (len(b) - lcp_len) / len(b) * 100

        print(f"Round {i} → Round {i+1}:")
        print(f"  Prompt A: {len(a):>6d} chars")
        print(f"  Prompt B: {len(b):>6d} chars")
        print(f"  LCP:      {lcp_len:>6d} chars ({pct_cached:.1f}% cached)")
        print(f"  Reprocess: {len(b)-lcp_len:>6d} chars ({pct_wasted:.1f}% wasted)")
        print(f"  Break point context: ...{repr(context_before[-30:])}|BREAK|{repr(context_after_b[:50])}...")
        print()


def build_experiment_records(
    rounds: List[Dict[str, Any]],
    chain_id: str = "feishu-main-agent",
) -> List[Dict[str, Any]]:
    """Convert rounds into records for the experiment framework."""
    records = []
    for r in rounds:
        records.append({
            "prompt_text": r["prompt_text"],
            "chain_id": chain_id,
            "chain_index": r["round_index"],
            "chain_length": len(rounds),
            "num_messages": r["num_messages"],
            "trace_id": r.get("message_id", ""),
            "start_time": r.get("timestamp", ""),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract Feishu prompts for cache test")
    parser.add_argument("--input", help="Single API call log (simulate earlier rounds)")
    parser.add_argument("--inputs", nargs="+", help="Multiple API call logs from same conversation")
    parser.add_argument("--output", default="feishu_cache_test_prompts.json")
    parser.add_argument("--simulate-rounds", action="store_true",
                        help="Simulate earlier rounds from a single log")
    args = parser.parse_args()

    if args.input and args.simulate_rounds:
        rounds = simulate_rounds_from_single(args.input)
    elif args.inputs:
        rounds = []
        for i, path in enumerate(args.inputs):
            body = load_call(path)
            prompt_text = flatten_prompt(body["messages"])
            rounds.append({
                "prompt_text": prompt_text,
                "round_index": i,
                "num_messages": len(body["messages"]),
                "char_count": len(prompt_text),
            })
    else:
        parser.error("Provide --input with --simulate-rounds, or --inputs with multiple files")

    print(f"Extracted {len(rounds)} rounds")
    for r in rounds:
        print(f"  Round {r['round_index']}: {r['char_count']} chars, {r['num_messages']} messages")

    # Analyze cache invalidation
    analyze_cache_invalidation(rounds)

    # Build experiment records
    records = build_experiment_records(rounds)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nWritten {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
