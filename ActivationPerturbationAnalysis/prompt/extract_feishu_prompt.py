"""
Extract and section-split the prompt from a Feishu/OpenClaw LLM request log.

Reads feishu1.json (a single LLM API call log) and outputs a structured
analysis of the prompt, split into logical sections for easy inspection.

Output: feishu1_sections.json — structured prompt breakdown
        feishu1_sections.md  — human-readable Markdown version

Usage:
    python extract_feishu_prompt.py [--input feishu1.json] [--output-prefix feishu1_sections]
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────

def get_text(content) -> str:
    """Extract plain text from OpenAI-style content (str or list-of-parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                url = p.get("image_url", {}).get("url", "")
                parts.append(f"[IMAGE: len={len(url)}]")
            else:
                parts.append(f"[{p.get('type', 'unknown')}]")
        return "\n".join(parts)
    return str(content or "")


def split_system_sections(text: str) -> List[Dict[str, Any]]:
    """Split system prompt into hierarchical sections by Markdown headers."""
    sections = []
    # Match # or ## headers
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [{"level": 0, "title": "(entire system prompt)", "content": text, "char_count": len(text)}]

    # Content before first header
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append({"level": 0, "title": "(preamble)", "content": pre, "char_count": len(pre)})

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append({
            "level": level,
            "title": title,
            "content": body,
            "char_count": len(body),
        })

    return sections


def extract_user_actual_text(text: str) -> str:
    """Extract the actual user message, stripping metadata wrappers."""
    lines = text.split("\n")
    # Find the [message_id: ...] line, take everything from that line onward
    for i, line in enumerate(lines):
        if line.strip().startswith("[message_id:"):
            # The actual user text starts on the next line (or same block)
            remaining = "\n".join(lines[i:]).strip()
            return remaining
    return text


def parse_tool_calls(msg: dict) -> List[Dict[str, str]]:
    """Extract tool call info from an assistant message."""
    calls = []
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        calls.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", ""),
        })
    return calls


# ── Main extraction ──────────────────────────────────────────────────

def extract(input_path: str) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    req = json.loads(data["request"]) if isinstance(data["request"], str) else data["request"]
    body = req["body"]
    msgs = body["messages"]

    # Parse response
    resp_text = ""
    if data.get("response"):
        resp = json.loads(data["response"]) if isinstance(data["response"], str) else data["response"]
        choices = resp.get("body", {}).get("choices", [])
        if choices:
            resp_text = choices[0].get("message", {}).get("content", "")

    # ── Build structured output ──
    result: Dict[str, Any] = {
        "metadata": {
            "model": data.get("model", ""),
            "request_id": data.get("request_id", ""),
            "uid": data.get("uid", ""),
            "total_messages": len(msgs),
            "parameters": {k: v for k, v in body.items() if k != "messages"},
        },
        "system_prompt": None,
        "conversation": [],
        "response": None,
    }

    # ── System prompt ──
    if msgs and msgs[0]["role"] == "system":
        sys_text = get_text(msgs[0]["content"])
        result["system_prompt"] = {
            "total_chars": len(sys_text),
            "sections": split_system_sections(sys_text),
        }

    # ── Conversation messages ──
    for i, m in enumerate(msgs):
        role = m.get("role", "")
        if role == "system":
            continue  # already handled

        entry: Dict[str, Any] = {
            "index": i,
            "role": role,
        }

        if role == "user":
            full_text = get_text(m["content"])
            entry["full_text"] = full_text
            entry["actual_user_text"] = extract_user_actual_text(full_text)
            entry["char_count"] = len(full_text)

        elif role == "assistant":
            text = get_text(m.get("content", ""))
            entry["text"] = text
            entry["char_count"] = len(text)
            tcs = parse_tool_calls(m)
            if tcs:
                entry["tool_calls"] = tcs

        elif role == "tool":
            text = get_text(m.get("content", ""))
            entry["tool_call_id"] = m.get("tool_call_id", "")
            entry["content"] = text
            entry["char_count"] = len(text)

        result["conversation"].append(entry)

    # ── Response ──
    if resp_text:
        result["response"] = {
            "char_count": len(resp_text),
            "text": resp_text,
        }

    return result


# ── Markdown renderer ────────────────────────────────────────────────

def render_markdown(result: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("# Feishu/OpenClaw Prompt 结构分析")
    lines.append("")

    # Metadata
    meta = result["metadata"]
    lines.append("## 1. 请求元数据")
    lines.append("")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 模型 | `{meta['model']}` |")
    lines.append(f"| request_id | `{meta['request_id']}` |")
    lines.append(f"| uid | `{meta['uid']}` |")
    lines.append(f"| 总消息数 | {meta['total_messages']} |")
    params = meta.get("parameters", {})
    if params:
        lines.append(f"| 推理参数 | `{json.dumps(params, ensure_ascii=False)}` |")
    lines.append("")

    # System prompt
    sp = result["system_prompt"]
    if sp:
        lines.append("## 2. System Prompt 结构")
        lines.append("")
        lines.append(f"**总长度**: {sp['total_chars']} 字符")
        lines.append("")

        # Section summary table
        lines.append("### 2.1 Section 概览")
        lines.append("")
        lines.append("| # | 层级 | Section 标题 | 字符数 | 占比 |")
        lines.append("|---|------|-------------|--------|------|")
        for idx, sec in enumerate(sp["sections"]):
            pct = sec["char_count"] / sp["total_chars"] * 100 if sp["total_chars"] > 0 else 0
            level_str = "#" * sec["level"] if sec["level"] > 0 else "-"
            lines.append(f"| {idx} | {level_str} | {sec['title']} | {sec['char_count']} | {pct:.1f}% |")
        lines.append("")

        # Top-level grouping
        lines.append("### 2.2 顶层分组（H1 级别）")
        lines.append("")
        h1_groups = []
        current_h1 = None
        current_h1_chars = 0
        for sec in sp["sections"]:
            if sec["level"] <= 1:
                if current_h1 is not None:
                    h1_groups.append((current_h1, current_h1_chars))
                current_h1 = sec["title"]
                current_h1_chars = sec["char_count"]
            else:
                current_h1_chars += sec["char_count"]
        if current_h1 is not None:
            h1_groups.append((current_h1, current_h1_chars))

        lines.append("| 分组 | 总字符数 | 占比 |")
        lines.append("|------|---------|------|")
        for title, chars in h1_groups:
            pct = chars / sp["total_chars"] * 100 if sp["total_chars"] > 0 else 0
            lines.append(f"| {title} | {chars} | {pct:.1f}% |")
        lines.append("")

        # Section details
        lines.append("### 2.3 各 Section 详细内容")
        lines.append("")
        for idx, sec in enumerate(sp["sections"]):
            level_prefix = "#" * (sec["level"] + 3) if sec["level"] > 0 else "####"
            lines.append(f"{level_prefix} Section {idx}: {sec['title']} ({sec['char_count']} chars)")
            lines.append("")
            content = sec["content"]
            if len(content) > 2000:
                lines.append("```")
                lines.append(content[:1000])
                lines.append(f"\n... (省略 {len(content) - 2000} 字符) ...\n")
                lines.append(content[-1000:])
                lines.append("```")
            else:
                lines.append("```")
                lines.append(content)
                lines.append("```")
            lines.append("")

    # Conversation
    lines.append("## 3. 对话流程")
    lines.append("")

    # Flow diagram
    lines.append("### 3.1 消息流")
    lines.append("")
    lines.append("```")
    for entry in result["conversation"]:
        role = entry["role"]
        idx = entry["index"]
        chars = entry.get("char_count", 0)
        if role == "user":
            actual = entry.get("actual_user_text", "")
            # Extract the core user question
            core_lines = [l for l in actual.split("\n") if not l.startswith("[") and not l.startswith("Sender") and not l.startswith("{") and not l.startswith("}") and l.strip()]
            core = " ".join(core_lines)[:100] if core_lines else "(empty)"
            lines.append(f"msg[{idx}] USER ({chars} chars): {core}")
        elif role == "assistant":
            tcs = entry.get("tool_calls", [])
            if tcs:
                tc_str = ", ".join(f"{tc['name']}(...)" for tc in tcs)
                lines.append(f"msg[{idx}] ASSISTANT ({chars} chars): [tool_call: {tc_str}]")
            else:
                preview = entry.get("text", "")[:80]
                lines.append(f"msg[{idx}] ASSISTANT ({chars} chars): {preview}")
        elif role == "tool":
            tcid = entry.get("tool_call_id", "")
            preview = entry.get("content", "")[:80]
            lines.append(f"msg[{idx}] TOOL ({chars} chars): {preview}")
    lines.append("```")
    lines.append("")

    # Each message detail
    lines.append("### 3.2 各消息详情")
    lines.append("")
    for entry in result["conversation"]:
        role = entry["role"]
        idx = entry["index"]
        lines.append(f"#### msg[{idx}]: {role.upper()} ({entry.get('char_count', 0)} chars)")
        lines.append("")

        if role == "user":
            lines.append("**实际用户输入**:")
            lines.append("```")
            lines.append(entry.get("actual_user_text", "")[:2000])
            lines.append("```")
        elif role == "assistant":
            tcs = entry.get("tool_calls", [])
            if tcs:
                for tc in tcs:
                    lines.append(f"**Tool Call**: `{tc['name']}` (id: `{tc['id']}`)")
                    args = tc.get("arguments", "")
                    if isinstance(args, str) and len(args) > 500:
                        lines.append(f"```json\n{args[:500]}\n...(truncated)\n```")
                    else:
                        lines.append(f"```json\n{args}\n```")
            text = entry.get("text", "")
            if text:
                if len(text) > 2000:
                    lines.append(f"\n**Text** ({len(text)} chars, truncated):")
                    lines.append("```")
                    lines.append(text[:2000])
                    lines.append("```")
                else:
                    lines.append(f"\n**Text**:")
                    lines.append("```")
                    lines.append(text)
                    lines.append("```")
        elif role == "tool":
            lines.append(f"**tool_call_id**: `{entry.get('tool_call_id', '')}`")
            content = entry.get("content", "")
            if len(content) > 3000:
                lines.append(f"\n**Content** ({len(content)} chars, truncated):")
                lines.append("```")
                lines.append(content[:1500])
                lines.append(f"\n... (省略 {len(content) - 3000} 字符) ...\n")
                lines.append(content[-1500:])
                lines.append("```")
            else:
                lines.append("```")
                lines.append(content)
                lines.append("```")
        lines.append("")

    # Response
    if result.get("response"):
        resp = result["response"]
        lines.append("## 4. 模型响应")
        lines.append("")
        lines.append(f"**长度**: {resp['char_count']} 字符")
        lines.append("")
        text = resp["text"]
        if len(text) > 3000:
            lines.append("```")
            lines.append(text[:1500])
            lines.append(f"\n... (省略 {len(text) - 3000} 字符) ...\n")
            lines.append(text[-1500:])
            lines.append("```")
        else:
            lines.append("```")
            lines.append(text)
            lines.append("```")
        lines.append("")

    # Summary analysis
    lines.append("## 5. Prompt 结构总结")
    lines.append("")

    # Compute sizes
    sys_chars = sp["total_chars"] if sp else 0
    conv_chars = sum(e.get("char_count", 0) for e in result["conversation"])
    total_chars = sys_chars + conv_chars
    tool_result_chars = sum(e.get("char_count", 0) for e in result["conversation"] if e["role"] == "tool")
    user_meta_chars = sum(e.get("char_count", 0) for e in result["conversation"] if e["role"] == "user")
    assistant_chars = sum(e.get("char_count", 0) for e in result["conversation"] if e["role"] == "assistant")

    lines.append("### 5.1 Token 预算分配（按字符数估算）")
    lines.append("")
    lines.append("| 组成部分 | 字符数 | 占比 |")
    lines.append("|---------|--------|------|")
    lines.append(f"| System Prompt | {sys_chars} | {sys_chars/total_chars*100:.1f}% |")
    lines.append(f"| User 消息（含元数据） | {user_meta_chars} | {user_meta_chars/total_chars*100:.1f}% |")
    lines.append(f"| Assistant 回复 | {assistant_chars} | {assistant_chars/total_chars*100:.1f}% |")
    lines.append(f"| Tool 返回结果 | {tool_result_chars} | {tool_result_chars/total_chars*100:.1f}% |")
    lines.append(f"| **总计** | **{total_chars}** | **100%** |")
    lines.append("")

    # System prompt breakdown by category
    if sp:
        lines.append("### 5.2 System Prompt 语义分类")
        lines.append("")

        categories = {
            "框架/运行时配置": ["Tooling", "Tool Call Style", "Model Aliases", "Workspace",
                          "Documentation", "Current Date", "Workspace Files", "Runtime",
                          "OpenClaw CLI Quick Reference"],
            "行为规则/安全": ["Safety", "Red Lines", "External vs Internal", "Reply Tags",
                        "Messaging", "Silent Replies"],
            "Skills/能力定义": ["Skills"],
            "Agent 身份/人格": ["SOUL.md", "Core Truths", "Boundaries", "Vibe", "Continuity",
                           "IDENTITY.md", "Make It Yours"],
            "用户/上下文": ["USER.md", "Context", "Group Chat", "Inbound Context"],
            "Memory/持久化": ["Memory", "MEMORY.md", "Preferences", "Notes"],
            "Workspace 文件": ["AGENTS.md", "TOOLS.md", "HEARTBEAT.md", "BOOTSTRAP.md",
                           "What Goes Here", "Examples", "Why Separate"],
            "启动/心跳": ["First Run", "Session Startup", "Heartbeats", "Group Chats",
                       "Tools", "The Conversation", "After You Know", "Connect", "When You're Done"],
        }

        cat_chars: Dict[str, int] = {c: 0 for c in categories}
        cat_chars["其他"] = 0
        for sec in sp["sections"]:
            matched = False
            for cat_name, keywords in categories.items():
                for kw in keywords:
                    if kw.lower() in sec["title"].lower():
                        cat_chars[cat_name] += sec["char_count"]
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                cat_chars["其他"] += sec["char_count"]

        lines.append("| 语义分类 | 字符数 | 占 System Prompt 比例 |")
        lines.append("|---------|--------|---------------------|")
        for cat, chars in sorted(cat_chars.items(), key=lambda x: -x[1]):
            if chars > 0:
                lines.append(f"| {cat} | {chars} | {chars/sp['total_chars']*100:.1f}% |")
        lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract and section-split Feishu prompt")
    parser.add_argument("--input", default="feishu1.json", help="Input JSON file")
    parser.add_argument("--output-prefix", default="feishu1_sections", help="Output file prefix")
    args = parser.parse_args()

    result = extract(args.input)

    # Write JSON
    json_path = f"{args.output_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Written: {json_path}")

    # Write Markdown
    md_path = f"{args.output_prefix}.md"
    md = render_markdown(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Written: {md_path}")

    # Print summary
    sp = result["system_prompt"]
    print(f"\nSummary:")
    print(f"  Messages: {result['metadata']['total_messages']}")
    print(f"  System prompt: {sp['total_chars']} chars, {len(sp['sections'])} sections")
    print(f"  Conversation: {len(result['conversation'])} messages")
    if result["response"]:
        print(f"  Response: {result['response']['char_count']} chars")


if __name__ == "__main__":
    main()
