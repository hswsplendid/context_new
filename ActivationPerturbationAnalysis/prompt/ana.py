import re
import difflib
import html

LOG_FILE = "agent.log"
OUTPUT_HTML = "prompt_diff.html"

# 提取 prompt
PROMPT_PATTERN = re.compile(r"prompt:\s*(.*?),\s*prompt_token_ids:")


def extract_prompts(path):
    prompts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = PROMPT_PATTERN.search(line)
            if m:
                prompts.append(m.group(1))

    return prompts


def longest_common_prefix(strings):
    shortest = min(strings, key=len)

    for i, c in enumerate(shortest):
        for s in strings:
            if s[i] != c:
                return shortest[:i], i

    return shortest, len(shortest)


def generate_html(prompts):

    p1, p2, p3 = prompts[:3]

    lcp, pos = longest_common_prefix([p1, p2, p3])

    diff = difflib.HtmlDiff(wrapcolumn=80)

    diff12 = diff.make_table(
        p1.split(),
        p2.split(),
        "Prompt1",
        "Prompt2",
        context=True
    )

    diff13 = diff.make_table(
        p1.split(),
        p3.split(),
        "Prompt1",
        "Prompt3",
        context=True
    )

    diff23 = diff.make_table(
        p2.split(),
        p3.split(),
        "Prompt2",
        "Prompt3",
        context=True
    )

    html_page = f"""
<html>
<head>
<meta charset="utf-8">
<title>Prompt Diff Report</title>

<style>
body {{
    font-family: Arial;
    margin: 40px;
}}

pre {{
    background: #f6f8fa;
    padding: 10px;
}}

table.diff {{
    font-size: 12px;
    border-collapse: collapse;
}}

.diff_add {{ background:#aaffaa; }}
.diff_chg {{ background:#ffff77; }}
.diff_sub {{ background:#ffaaaa; }}

</style>

</head>

<body>

<h1>Prompt Comparison Report</h1>

<h2>Longest Common Prefix</h2>

<p><b>Length:</b> {pos}</p>

<pre>{html.escape(lcp)}</pre>

<h2>First Difference</h2>

<table border="1" cellpadding="6">
<tr>
<th>Prompt</th>
<th>Different content (first 200 chars)</th>
</tr>

<tr>
<td>Prompt1</td>
<td>{html.escape(p1[pos:pos+200])}</td>
</tr>

<tr>
<td>Prompt2</td>
<td>{html.escape(p2[pos:pos+200])}</td>
</tr>

<tr>
<td>Prompt3</td>
<td>{html.escape(p3[pos:pos+200])}</td>
</tr>

</table>

<h2>Diff Prompt1 vs Prompt2</h2>
{diff12}

<h2>Diff Prompt1 vs Prompt3</h2>
{diff13}

<h2>Diff Prompt2 vs Prompt3</h2>
{diff23}

</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_page)


def main():

    prompts = extract_prompts(LOG_FILE)

    if len(prompts) < 3:
        print("Need at least 3 prompts in log")
        return

    generate_html(prompts[:3])

    print("Report generated:", OUTPUT_HTML)


if __name__ == "__main__":
    main()