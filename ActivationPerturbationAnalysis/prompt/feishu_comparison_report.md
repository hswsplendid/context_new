# feishu1.json vs feishu2.json Prompt 对比分析报告

## 1. 数据概览

| 属性 | Call 0 (feishu1) | Call 1 (feishu2-obj0) | Call 2 (feishu2-obj1) |
|------|-----------------|----------------------|----------------------|
| 来源文件 | feishu1.json | feishu2.json #0 | feishu2.json #1 |
| 模型 | qwen-flash | qwen-flash | qwen-flash |
| 总消息数 | 18 | 20 | 22 |
| 对话消息数 | 17 | 19 | 21 |
| System Prompt 长度 | 30,898 chars | 30,898 chars | 30,898 chars |
| 全 prompt 长度(ChatML) | 60,249 chars | 65,582 chars | 65,729 chars |
| 最后用户消息时间 | 19:43 GMT+8 | 22:05 GMT+8 | 22:05 GMT+8 |
| 最后用户请求 | "总结缺点" | "指定一个初学者可学习说明书" | (同左，Call 2 是 Call 1 的 tool 调用继续) |

> **feishu2.json 包含 2 个 JSON 对象**：代表同一用户请求触发的 2 次连续 API 调用（Call 1 = 收到用户消息，Call 2 = assistant 调用 tool 后的继续）。

## 2. System Prompt 对比

### 结论：三次调用的 System Prompt **完全相同** ✅

| 对比 | 结果 |
|------|------|
| Call 0 vs Call 1 | **IDENTICAL** (30,898 chars, 逐字符匹配) |
| Call 0 vs Call 2 | **IDENTICAL** |
| Call 1 vs Call 2 | **IDENTICAL** |

70 个 section 的**标题顺序**和**内容**全部一致，无任何差异。

### 原因分析

这三次调用来自**同一对话** (`conversation_label` 相同)。OpenClaw 的 system prompt 中 Inbound Metadata 包含 `chat_id` 等字段，但在同一对话内这些字段不变。因此：

- **同一对话内**：System Prompt 完全一致 → prefix cache 对 system prompt 部分 100% 有效
- **不同对话间**（help7 描述的场景）：Inbound Metadata 中的 `chat_id`/`timestamp` 会变 → cache 在 ~48.5% 处断裂

## 3. 对话历史对比

### 严格前缀关系 ✅

```
Call 0: [system][msg1..msg17]                         (18 msgs)
Call 1: [system][msg1..msg17][+assistant][+user]      (20 msgs) = Call 0 + 2
Call 2: [system][msg1..msg19][+assistant][+tool]      (22 msgs) = Call 1 + 2
```

| 对比 | 关系 | 新增消息 |
|------|------|---------|
| Call 0 → Call 1 | 严格前缀 | +assistant (4,208 chars 缺点分析) + user (1,064 chars 新请求) |
| Call 1 → Call 2 | 严格前缀 | +assistant (0 chars + 1 tool_call) + tool (86 chars 写文件结果) |

### 新增消息内容

**Call 0 → Call 1 新增**:
1. `assistant`: scheduler.py 缺点分析的完整回复 (4,208 chars)
2. `user`: 新请求 "指定一个初学者可学习说明书" (msg_id: `om_x100b54b4ae0c04b8`, ts: 22:05)

**Call 1 → Call 2 新增**:
1. `assistant`: 发起 tool_call（写文件操作）
2. `tool`: 写入 scheduler_tutorial.md (3,839 bytes) 的结果确认 (86 chars)

## 4. Prefix Cache 效率分析

### 全 Prompt LCP（最长公共前缀）

| 对比 | Prompt A | Prompt B | LCP | 缓存率 | 需重处理 |
|------|----------|----------|-----|--------|---------|
| Call 0 → Call 1 | 60,249 chars | 65,582 chars | 60,249 chars | **91.9%** | 5,333 chars (8.1%) |
| Call 1 → Call 2 | 65,582 chars | 65,729 chars | 65,582 chars | **99.8%** | 147 chars (0.2%) |

### 解读

1. **Call 0 → Call 1 (91.9% cached)**：Call 1 的 prompt 是 Call 0 的完整内容 + 新的 assistant 回复 + 新的 user 消息。缓存断裂点恰好在 Call 0 prompt 的末尾 → **最优情况**，只有新增内容需要处理。

2. **Call 1 → Call 2 (99.8% cached)**：Call 2 只比 Call 1 多了一个空 assistant 消息（tool_call）和 86 chars 的 tool 结果。缓存几乎 100% 命中。

## 5. 与 help7 问题的关联

### help7 描述的问题

> Inbound Metadata（DYNAMIC）位于 Project Context（~25K tokens STATIC）之前，导致不同对话间即使 Project Context 完全相同，也无法利用 prefix cache。

### 本次数据的验证结果

| 场景 | cache 效率 | 说明 |
|------|-----------|------|
| 同一对话连续轮次 (本数据) | **91.9% - 99.8%** | System prompt 不变，对话历史严格前缀追加 |
| 不同对话（help7 场景） | **~48.5%** | Inbound Metadata 中 chat_id 不同，后面 ~15K chars Project Context 全部失效 |

**本数据无法直接验证 help7 的跨对话 cache 失效问题**，因为 feishu1.json 和 feishu2.json 属于同一对话。

### 要验证 help7 需要的数据

需要来自**不同飞书群/不同对话**的两次 API 调用日志。对比它们的 system prompt 即可定位 Inbound Metadata 导致的 cache 断裂点。

## 6. 关键发现总结

### 6.1 同一对话内的 prompt 特征

1. **System Prompt 100% 不变**：70 个 section，30,898 chars，逐字符一致
2. **对话历史严格前缀追加**：每次新调用 = 上次完整内容 + 新 assistant 回复 + 新 user/tool 消息
3. **Prefix cache 高效**：91.9% ~ 99.8% 的内容可缓存
4. **scheduler.py 代码确实作为静态内容重复携带**：每次调用都包含之前 tool 返回的 9,022 chars 代码（msg[11]），占总 prompt 的 ~14%

### 6.2 单次用户请求可触发多次 API 调用

feishu2.json 证实：用户发送一条消息 → OpenClaw 先生成回复（Call 1, 20 msgs），发起 tool_call → 收到 tool 结果后再次调用 LLM（Call 2, 22 msgs）。这是典型的 **ReAct (Reason+Act)** 模式。

### 6.3 token 预算增长趋势

```
Call 0:  60,249 chars  (base)
Call 1:  65,582 chars  (+8.8%)
Call 2:  65,729 chars  (+0.2%)
```

对话越长，每次调用的 prompt 越大。但由于严格前缀特性，增量部分相对较小。

## 7. 对实验设计的启示

### 可以用这些数据做什么

1. **Triplet 实验**：3 次调用构成完美的 triplet chain（Call 0 ⊂ Call 1 ⊂ Call 2）
   - 可以测量删除 Call 1 新增的 assistant 回复对 Call 2 的 user 消息隐藏状态的影响
   - 验证：大段 assistant 回复（4,208 chars 缺点分析）是否显著改变后续内容的表征

2. **静态内容影响**：scheduler.py 代码（9,022 chars）从 Call 0 开始就存在于 prompt 中
   - 可以构造对照实验：移除 scheduler.py 代码后，后续对话的隐藏状态是否发生显著变化

### 还需要什么数据

- **跨对话对比**：两个不同 `chat_id` 的调用日志 → 验证 help7 的 Inbound Metadata cache 失效
- **更长对话链**：>10 轮对话 → 观察 prompt 膨胀和 cache 效率的长期趋势
