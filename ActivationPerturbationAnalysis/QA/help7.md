Problem
When running local models (e.g. Qwen3-Coder-Next on MLX with an M3 Ultra), prefix caching is critical for interactive performance. However, the current system prompt layout in buildAgentSystemPrompt() interleaves dynamic content with large static blocks, which invalidates the prefix cache on nearly every request.

At ~65 tok/s prompt processing on our M3 Ultra (256GB), a full OpenClaw context (~50k tokens) takes ~12-13 minutes without cache hits. With effective prefix caching, the static prefix would be cached and only the dynamic tail (~500-5k tokens) would need processing — bringing prompt time down to under 1-2 seconds for repeat interactions.

Current Section Order
The system prompt is assembled in this order (from buildAgentSystemPrompt() in reply-Cx57rl6c.js):

Tooling (~2k tokens) — STATIC
Tool Call Style — STATIC
Safety — STATIC
CLI Quick Reference — STATIC
Skills (~1.5k tokens) — STATIC
Memory Recall — STATIC
Self-Update — STATIC
Model Aliases — STATIC
Workspace dir — STATIC
Documentation — STATIC
Authorized Senders — STATIC
Current Date & Time — STATIC (timezone only)
Workspace Files header — STATIC
Reply Tags — STATIC
Messaging — STATIC
Group Chat Context / Inbound Metadata — ⚠️ DYNAMIC (changes every request: chat_id, channel, provider, etc.)
Reactions — STATIC
Project Context (~25k tokens of workspace .md files) — SEMI-STATIC (changes only on file edits)
Silent Replies — STATIC
Heartbeats — STATIC
Runtime line — ⚠️ DYNAMIC (model, channel, capabilities, thinking level)
The Problem
The Inbound Metadata JSON (#16) changes on nearly every request (different chat_id, timestamps, etc.) and sits before the massive ~25k-token Project Context block (#18). This means the entire Project Context is reprocessed every request even when no workspace files have changed.

Similarly, the Runtime line (#21) is dynamic but sits after static sections like Silent Replies and Heartbeats — those static sections between two dynamic sections get no cache benefit either way.






/root/context/ActivationPerturbationAnalysis/QA/help7.md内容指出了问题：”传入的元数据 JSON ( #16 )   
在几乎每个请求中都发生变化（不同的 chat_id 、时间戳等），并且位于庞大的 ~25k 个 token 的项目上下文块 ( 
 #18 ) 之前。这意味着即使没有工作区文件发生变化，整个项目上下文也会在每次请求时重新处理。“如果我想通过 
合适的openclaw prompt，去验证是否存在此类的情况（必须注入大型 Project                                  
Context），我应该怎么操作？sub-agent 可以怎么触发这种效果吗？ 


现在数据已经很清楚了。让我直接回答你的两个问题。

  ---
  已验证：feishu1.json 确实存在 help7 描述的问题

  从实际数据看，System Prompt (30,898 chars) 的布局：

  [0      - 14,977]  STATIC   48.5%  ← Tooling, Safety, Skills 等（可缓存 ✅）
  [14,977 - 15,846]  DYNAMIC   2.8%  ← Inbound Metadata: chat_id, timestamps（每次变 ❌）
  [15,846 - 29,630]  STATIC   44.6%  ← Project Context: AGENTS/SOUL/TOOLS/IDENTITY/USER/MEMORY.md
  [29,630 - 30,510]  STATIC    2.8%  ← Silent Replies, Heartbeats
  [30,510 - 30,898]  DYNAMIC   1.3%  ← Runtime line

  Inbound Metadata 仅 ~850 chars（2.8%），但因为它夹在中间，导致后面 15,052 chars（48.7%）的 KV cache
  全部失效。

  ---
  如何操作来验证"大型 Project Context 被注入"的效果

  思路

  你需要做的是：收集同一对话中连续多次 LLM API 调用的完整
  prompt，然后对比哪些部分变了、哪些没变，来量化 prefix cache 的失效范围。

  具体操作方案

  方案 1：主 Agent 多轮对话（最简单，你已经有数据了）

  在飞书群里对 OpenClaw 连续发几条消息：

  你: @openclaw hello
  你: @openclaw 1+1等于多少
  你: @openclaw 今天天气怎么样

  然后从日志系统中按 conversation_label 拉取这 3 次调用的完整 request JSON。每次的 system prompt
  都包含完整的 Project Context (~25K tokens)，但 Inbound Metadata 中的 message_id 和 timestamp 每次不同
   → 可以精确定位 cache 断裂点。

  验证方法：对比两次调用的 system prompt，找到第一个不同字符的位置 → 这就是 prefix cache 失效的起点。

  方案 2：Sub-Agent（可以，但有区别）

  Sub-agent 的 system prompt 结构不同——它更短（~923 chars），不包含 Project Context 和 Inbound
  Metadata。参考之前 trace.json 的分析：

  # Sub-agent 的 system prompt 只有:
  "# Subagent\n## Current Time\n...\nYou are a subagent spawned by the main agent..."
  # 没有 AGENTS.md, SOUL.md 等 workspace 文件

  所以 sub-agent 无法直接复现 help7 的问题——它的 prompt 里没有大型 Project Context block。

  但你可以通过 sub-agent 验证另一个现象：对话历史中的 tool result
  作为"事实上的静态大块"被反复携带（类似 scheduler.py 的情况）。

  方案 3：构造对照实验（推荐的验证方案）

  如果你想用我们已有的实验框架来量化 help7 的影响，可以这样做：

  Step 1：收集同一个 conversation_label 下的 2-3 次连续主 Agent 调用的完整 request

  # 假设日志存储在某处，按 conversation_label 筛选
  # 每条记录就是一次完整的 API 调用，包含完整 messages 列表

  Step 2：从每次调用中提取完整的 prompt text（system + all messages 拼接）

  Step 3：用我们已有的实验代码做 triplet 或 prompt_diff 比较——但这次比较的不是"删除
  D1"，而是"比较两次调用中 Project Context 部分的 hidden states 是否因为前面的 Inbound Metadata
  变化而不同"

  我来写一个提取脚本：

● Write(extract_feishu_for_cache_test.py)
  ⎿  Wrote 193 lines to extract_feishu_for_cache_test.py
     """
     Extract prompt pairs from Feishu/OpenClaw logs to verify prefix-cache invalidation.

     Given multiple API call logs from the SAME conversation, this script:
     1. Extracts the full prompt text from each call
     2. Finds the LCP (longest common prefix) between consecutive calls
     3. Identifies where the cache breaks (= where Inbound Metadata changes)
     4. Outputs prompts in a format consumable by the existing experiment code

     This verifies the help7.md claim: dynamic Inbound Metadata before large
     … +183 lines (ctrl+o to expand)

● Bash(python3 extract_feishu_for_cache_test.py --input feishu1.json --simulate-rounds)
  ⎿  Extracted 6 rounds
       Round 0: 32003 chars, 2 messages
       Round 1: 34383 chars, 4 messages
     … +31 lines (ctrl+o to expand)

● 这个结果非常有意思！让我解释清楚。 
                                                                                          
  ---
  关键发现：这个对话的 cache 效率其实很高（90%+）                                                      
   
  从上面的分析看，同一对话内的连续轮次，LCP（最长公共前缀）都在 90% 以上。这是因为：                   
                                                                                                     
  - System prompt 在同一对话内不变（包括 Inbound Metadata 里的 chat_id 也没变）                        
  - 每轮只在末尾追加新 messages
  - 所以每轮 prompt 都是上一轮的严格前缀 + 新的 assistant/tool/user 消息

  help7 描述的问题不在"同一对话的连续轮次"之间，而是在"不同对话"之间！

  同一对话内：Round 1 → Round 2 → Round 3
    System prompt 完全相同 ✅ → prefix cache 有效 (90%+)

  不同对话间：Chat A 的 Round 1 → Chat B 的 Round 1
    System prompt 中 Inbound Metadata 不同 ❌ → cache 在 48.5% 处断裂
    后面的 Project Context (44.6%) 全部失效

  回到你的问题：如何验证

  你需要的数据

  要验证 help7 的问题，你需要的不是"同一对话的多轮"，而是来自不同对话/不同群聊的两次调用。对比它们的
  system prompt：

  Call from Chat A:  [...STATIC...][chat_id=A, ts=19:20][...Project Context 25K...]
  Call from Chat B:  [...STATIC...][chat_id=B, ts=19:21][...Project Context 25K...]
                                    ↑ 这里不同，后面 25K 全部 cache miss

  操作步骤

  1. 在两个不同的飞书群分别 @OpenClaw 发一条消息（比如都发 "hello"）
  2. 从日志中拉取这两次调用的 request JSON
  3. 用我写的脚本对比 system prompt 的 LCP 断裂点

  Sub-Agent 能触发吗？

  不能直接触发 help7 的问题，原因：

  1. Sub-agent 的 system prompt 很短（~923 chars），不包含 Project Context
  2. Sub-agent 没有 Inbound Metadata 注入

  但 sub-agent 可以间接帮助验证一个相关问题：如果你在 sub-agent 的 system prompt
  中手动注入大块静态内容（通过 workspace 文件），再在前面加入每次变化的 metadata，就可以人为复现 cache
  失效。不过这需要修改 OpenClaw 的 agent 配置。

  最实际的验证方案

  不需要 sub-agent。直接在两个不同群聊各发一条消息就够了：

  # 拿到两个不同群的调用日志
  python extract_feishu_for_cache_test.py \
      --inputs chat_a_call.json chat_b_call.json \
      --output cross_chat_cache_test.json

  这会直接展示：两个 prompt 在 Inbound Metadata 处（~48.5% 位置）就开始不同，后面 51.5%
  的内容（包括完全相同的 25K Project Context）全部需要重新处理。



  