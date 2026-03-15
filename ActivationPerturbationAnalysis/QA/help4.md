Plan to implement                                                                               │
│                                                                                                 │
│ Plan: Natural Prompt-Diff Perturbation Mode                                                     │
│                                                                                                 │
│ Context                                                                                         │
│                                                                                                 │
│ Currently, experiments apply artificial perturbations (random_replace, mask, delete, etc.) to a │
│  single prompt and measure how hidden states change after the perturbation point. When using    │
│ real agent prompts from extracted_prompts.json, consecutive prompts are progressively longer —  │
│ each is roughly a prefix of the next with new content appended. The user wants to treat the     │
│ natural difference between consecutive prompts as the perturbation, eliminating artificial      │
│ perturbation types entirely for this mode.                                                      │
│                                                                                                 │
│ Key insight: since prompt[i] is generally a prefix of prompt[i+1], the "perturbation" is the    │
│ appended content. We need to measure how this appended content changes the hidden states of the │
│  shared prefix tokens (backward influence), not post-perturbation tokens (forward propagation). │
│                                                                                                 │
│ Changes                                                                                         │
│                                                                                                 │
│ 1. src/metrics.py — Add compute_prefix_distance_buckets()                                       │
│                                                                                                 │
│ New function that compares hidden states of the shared prefix between two sequences, with       │
│ buckets measured by distance backward from the divergence point:                                │
│                                                                                                 │
│ - immediate: last 10 tokens before divergence (closest to the change)                           │
│ - short: last 100 tokens before divergence                                                      │
│ - all_prefix: all shared prefix tokens (0 to K)                                                 │
│ - periodic_N: 10-token windows at offsets 0, 100, 200... from divergence going backward         │
│ - per_token: individual token measurements for last 10 tokens before divergence                 │
│                                                                                                 │
│ Parameters: original_states, perturbed_states, common_prefix_len, plus window size configs.     │
│                                                                                                 │
│ 2. src/perturbations.py — Add build_prompt_diff_result()                                        │
│                                                                                                 │
│ New standalone function (not a PerturbationStrategy subclass, since it takes two token lists    │
│ rather than one list + span params):                                                            │
│                                                                                                 │
│ def build_prompt_diff_result(                                                                   │
│     prev_ids: List[int],                                                                        │
│     curr_ids: List[int],                                                                        │
│ ) -> PerturbationResult:                                                                        │
│                                                                                                 │
│ - Finds the longest common prefix (LCP) between prev_ids and curr_ids                           │
│ - Returns PerturbationResult where:                                                             │
│   - original_ids = prev_ids, perturbed_ids = curr_ids                                           │
│   - original_span_start = LCP, original_span_end = len(prev_ids)                                │
│   - perturbed_span_start = LCP, perturbed_span_end = len(curr_ids)                              │
│   - post_perturbation_start = LCP (divergence point)                                            │
│   - perturbation_type = "prompt_diff"                                                           │
│                                                                                                 │
│ 3. src/experiment.py — Add run_prompt_diff_experiment()                                         │
│                                                                                                 │
│ New function alongside the existing run_single_experiment():                                    │
│                                                                                                 │
│ def run_prompt_diff_experiment(                                                                 │
│     model, tokenizer, prev_prompt: str, curr_prompt: str,                                       │
│     measurement_cfg: dict, prompt_pair_index: int,                                              │
│ ) -> List[Dict]:                                                                                │
│                                                                                                 │
│ Steps:                                                                                          │
│ 1. Tokenize both prompts                                                                        │
│ 2. Call build_prompt_diff_result() to find common prefix                                        │
│ 3. Forward pass on both full sequences                                                          │
│ 4. Call compute_prefix_distance_buckets() to compare shared prefix hidden states                │
│ 5. Return metric records                                                                        │
│                                                                                                 │
│ 4. src/sweep.py — Modify run_sweep() for prompt_diff mode                                       │
│                                                                                                 │
│ Add logic in run_sweep():                                                                       │
│ - When config has perturbation_types: ["prompt_diff"] and source: "jsonfile":                   │
│   - Build jobs as consecutive pairs (prompt[0], prompt[1]), (prompt[1], prompt[2]), ... — total │
│  N-1 jobs for N prompts                                                                         │
│   - No need for perturbation_positions or perturbation_span_lengths (they come naturally from   │
│ the diff)                                                                                       │
│   - Each job's metadata records the pair index, actual diff position, and actual diff length    │
│ - Add a build_prompt_diff_jobs() helper that generates these pair-based jobs                    │
│ - In the execution loop, detect prompt_diff jobs and call run_prompt_diff_experiment() instead  │
│ of run_single_experiment()                                                                      │
│                                                                                                 │
│ 5. src/storage.py — Extend schema                                                               │
│                                                                                                 │
│ Add new optional columns to RESULT_COLUMNS:                                                     │
│ - prompt_pair_index — which consecutive pair (0 = prompts 0→1, 1 = prompts 1→2, ...)            │
│ - common_prefix_length — number of shared prefix tokens                                         │
│                                                                                                 │
│ Existing columns perturbation_position_abs and perturbation_span_length naturally capture the   │
│ diff location and size.                                                                         │
│                                                                                                 │
│ 6. config/agent_prompts_llama3.yaml — Add prompt_diff config                                    │
│                                                                                                 │
│ Add prompt_diff to perturbation_types (or create a new config variant):                         │
│ sweep:                                                                                          │
│   context_lengths: []                                                                           │
│   perturbation_positions: []       # not used for prompt_diff                                   │
│   perturbation_span_lengths: []    # not used for prompt_diff                                   │
│   perturbation_types:                                                                           │
│     - "prompt_diff"                                                                             │
│                                                                                                 │
│ Verification                                                                                    │
│                                                                                                 │
│ 1. Run with mock model/tokenizer to verify consecutive pair iteration and diff detection        │
│ 2. Check that build_prompt_diff_result() correctly finds common prefixes using the extracted    │
│ prompts                                                                                         │
│ 3. Verify metric buckets are computed relative to divergence point (backward direction)         │
│ 4. Confirm results are stored correctly with prompt_pair_index metadata                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────

Changes Made

  1. src/metrics.py — Added compute_prefix_distance_buckets() (lines 189–296)
    - Compares hidden states of the shared prefix between two sequences
    - Buckets measured backward from divergence point: immediate, short, all_prefix, periodic_N,
  and per_token
    - Same GPU-optimized pattern as the existing compute_distance_buckets()
  2. src/perturbations.py — Added build_prompt_diff_result() (lines 248–292)
    - Standalone function (not a strategy subclass) that takes two token lists
    - Finds longest common prefix (LCP) and returns a PerturbationResult with
  perturbation_type="prompt_diff"
  3. src/experiment.py — Added run_prompt_diff_experiment() (lines 125–209)
    - Tokenizes both prompts, calls build_prompt_diff_result() to find LCP
    - Runs forward passes on both full sequences
    - Calls compute_prefix_distance_buckets() for backward-influence metrics
  4. src/sweep.py — Added prompt_diff mode support
    - build_prompt_diff_jobs() helper: generates N-1 consecutive pair jobs from N prompts
    - run_sweep() now detects perturbation_types: ["prompt_diff"] + source: "jsonfile" and
  dispatches to run_prompt_diff_experiment() with proper metadata (including prompt_pair_index,
  common_prefix_length)
  5. src/storage.py — Extended schema with two new optional columns:
    - prompt_pair_index — which consecutive pair (0, 1, 2, ...)
    - common_prefix_length — number of shared prefix tokens
  6. config/prompt_diff_llama3.yaml — New config file for prompt_diff mode with empty
  perturbation_positions/perturbation_span_lengths lists

请你分析新改动部分处理agent的prompt代码逻辑，并输出。并给出运行指令。                            
                                                                        
● Read 6 files (ctrl+o to expand)                                                                  
                                                                                                   
● 以下是完整的代码逻辑分析。     
                                                                                                   
  ---                                                       
  端到端数据流分析                                                                                 
                                                                                                   
  整体流程图                                                                                       
                                                                                                   
  config YAML                    extracted_prompts.json     
       │                                │
       ▼                                ▼
  run_sweep.py ──► run_sweep() ──► PromptGenerator(source="jsonfile")
                       │                     │
                       │          _passages = [prompt_text_0, prompt_text_1, ..., prompt_text_N-1]
                       │                     │
                       ▼                     ▼
                检测 prompt_diff 模式      统计每条 prompt 的 token 数
                (perturbation_types == ["prompt_diff"] && source == "jsonfile")
                       │
                       ▼
                build_prompt_diff_jobs()    ← 生成 N-1 个连续对任务
                       │
                       ▼
                执行循环: 逐 job 执行
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     prompt_diff 分支        标准扰动分支
            │                (原有逻辑不变)
            ▼
     prompt_gen.generate(index=i)   ← 取 prev_prompt
     prompt_gen.generate(index=i+1) ← 取 curr_prompt
            │
            ▼
     run_prompt_diff_experiment()
            │
            ├─► tokenize 两条 prompt
            ├─► build_prompt_diff_result()  ← 求 LCP
            ├─► extract_hidden_states(prev) ← 模型前向传播
            ├─► extract_hidden_states(curr) ← 模型前向传播
            ├─► compute_prefix_distance_buckets() ← 比较共享前缀的隐藏状态
            │
            ▼
     store.add_records(metadata含 prompt_pair_index, common_prefix_length)
            │
            ▼
     results_part_XXXXXX.parquet

  ---
  第一阶段：配置加载与 Prompt 读取

  入口: run_sweep.py → run_sweep(config)

  # config/prompt_diff_llama3.yaml 关键配置:
  sweep:
    perturbation_types: ["prompt_diff"]   # 触发新模式的标记
    perturbation_positions: []            # prompt_diff 模式不使用
    perturbation_span_lengths: []         # prompt_diff 模式不使用
  prompts:
    source: "jsonfile"                    # 从 JSON 文件加载真实 agent prompt
    jsonfile_path: "prompt/extracted_prompts.json"

  PromptGenerator 加载 JSON (src/prompts.py:93-95):
  if source == "jsonfile":
      self._json_records = self._load_jsonfile(jsonfile_path)  # 读取 JSON 文件
      self._passages = [r["prompt_text"] for r in self._json_records]
      # _passages = [prompt_0_text, prompt_1_text, ..., prompt_N-1_text]

  extracted_prompts.json 格式是一个列表，每个元素包含 prompt_text、prompt_token_ids、num_tokens
  等字段。这些 prompt 是从 vLLM 服务器日志中提取的真实 agent 请求，按时间顺序排列，每条 prompt
  是上一条的前缀加上新追加的内容。

  ---
  第二阶段：模式检测与 Job 构建

  sweep.py:184-191 — 模式检测:
  sweep = config["sweep"]
  is_prompt_diff = (
      sweep["perturbation_types"] == ["prompt_diff"] and is_jsonfile
  )
  # 只有同时满足两个条件才进入 prompt_diff 模式:
  # 1. perturbation_types 列表恰好只有 "prompt_diff" 一个元素
  # 2. prompt 来源是 jsonfile（真实 agent prompts）

  build_prompt_diff_jobs() (sweep.py:80-102) — 生成连续对任务:
  # 输入: prompt_token_counts = [512, 1200, 2800, 4100, ...]  (N 条 prompt 的 token 数)
  # 输出: N-1 个 job，每个 job 描述一个相邻 prompt 对

  for i in range(len(prompt_token_counts) - 1):
      params = {
          "perturbation_type": "prompt_diff",
          "prompt_pair_index": i,          # 第几个对 (0=prompt0→1, 1=prompt1→2, ...)
          "prev_prompt_index": i,          # 前一条 prompt 的索引
          "curr_prompt_index": i + 1,      # 后一条 prompt 的索引
          "prev_token_count": prompt_token_counts[i],
          "curr_token_count": prompt_token_counts[i + 1],
      }

  与标准模式的 Cartesian product（context_length × position × span × type）不同，prompt_diff 模式的
   job 数量固定为 N-1（N 条 prompt 产生 N-1 个连续对）。

  ---
  第三阶段：执行循环中的 prompt_diff 分支

  sweep.py:212-259 — 按 perturbation_type == "prompt_diff" 路由:

  if params.get("perturbation_type") == "prompt_diff":
      # 1. 从 PromptGenerator 取出两条完整的原始 prompt 文本
      prev_prompt = prompt_gen.generate(target_length=0, index=prev_idx)
      curr_prompt = prompt_gen.generate(target_length=0, index=curr_idx)
      # target_length=0 + source="jsonfile" → 返回完整原始文本，不做截断

      # 2. 调用新的 experiment 函数
      records = run_prompt_diff_experiment(
          model, tokenizer, prev_prompt, curr_prompt,
          measurement_cfg, prompt_pair_index=pair_idx,
      )

      # 3. 重新计算 LCP 用于 metadata（这里有重复 tokenize，但只影响 metadata）
      prev_ids = tokenizer.encode(prev_prompt, add_special_tokens=False)
      curr_ids = tokenizer.encode(curr_prompt, add_special_tokens=False)
      lcp = ...  # 同样的 LCP 逻辑

      # 4. 构建 metadata，写入存储
      metadata = {
          "context_length": len(curr_ids),                # 较长 prompt 的 token 数
          "perturbation_position_abs": lcp,               # 分歧点位置 = LCP 长度
          "perturbation_span_length": len(curr_ids) - lcp,# 新增内容的 token 数
          "prompt_pair_index": pair_idx,                  # 新增字段
          "common_prefix_length": lcp,                    # 新增字段
          ...
      }

  ---
  第四阶段：run_prompt_diff_experiment() 核心逻辑

  experiment.py:124-206:

  prev_prompt = "system\nYou are...\nuser\nHello"       (较短)
  curr_prompt = "system\nYou are...\nuser\nHello\nassistant\nHi!\nuser\nTell me..."  (较长)

  Step 1: Tokenize
    prev_ids = [101, 202, 303, 404, 505]           (len=5)
    curr_ids = [101, 202, 303, 404, 505, 606, 707] (len=7)

  Step 2: build_prompt_diff_result()
    LCP = 5 (前 5 个 token 完全相同)
    → PerturbationResult:
        original_ids = prev_ids
        perturbed_ids = curr_ids
        original_span_start = 5,  original_span_end = 5   (prev 在 LCP 后没有额外内容)
        perturbed_span_start = 5, perturbed_span_end = 7   (curr 在 LCP 后多了 2 个 token)
        post_perturbation_start = 5  (分歧点)

  Step 3: 两次前向传播
    prev_hidden = model(prev_ids) → shape (num_layers, 5, hidden_dim)
    curr_hidden = model(curr_ids) → shape (num_layers, 7, hidden_dim)
    # 关键：虽然 token [0:5] 的 id 一样，但因为 curr 后面多了 token，
    # Transformer 的 causal attention 不会让后续 token 影响前面 —— 但是！
    # 这里 prev 和 curr 长度不同，所以同一个 prefix token 看到的 attention mask 不同：
    # prev 中 token[4] 的 attention 范围是 [0:5]
    # curr 中 token[4] 的 attention 范围也是 [0:5]（causal mask 只看左边）
    # → 所以共享前缀部分的 hidden states 理论上应该完全相同！
    #
    # 但实际上由于实现细节（如 RoPE position encoding 可能因总序列长度不同
    # 而产生差异，或者 flash attention 的数值精度差异），可能会有微小差异。
    # 这正是我们要度量的：后续内容是否通过模型的某些机制反向影响了前缀的表示。

  Step 4: compute_prefix_distance_buckets()
    比较 prev_hidden[:, :5, :] 和 curr_hidden[:, :5, :] 的 cosine similarity
    按"距分歧点的距离"分桶（向后看）

  ---
  第五阶段：compute_prefix_distance_buckets() 度量逻辑

  metrics.py:188-311 — 与旧的 compute_distance_buckets() 的关键区别:

  ┌─────────────┬────────────────────────────────┬────────────────────────────────────┐
  │             │  旧 compute_distance_buckets   │ 新 compute_prefix_distance_buckets │
  ├─────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ 度量方向    │ 从扰动点向前（forward）        │ 从分歧点向后（backward）           │
  ├─────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ 比较区域    │ 扰动点之后的 token             │ 共享前缀内的 token（分歧点之前）   │
  ├─────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ 物理意义    │ 扰动如何向后传播影响后续 token │ 追加内容是否反向影响了前缀 token   │
  ├─────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ bucket 定义 │ offset 0 = 扰动后第 1 个 token │ offset 0 = 分歧前最后 1 个 token   │
  └─────────────┴────────────────────────────────┴────────────────────────────────────┘

  Token 位置:   0    1    2    ...   K-10  K-9  ...  K-1  |  K   K+1  ...
                ←────── shared prefix (K tokens) ──────→   |  ← divergence
                                                           |
                ←─── all_prefix ────────────────────────→  |
                                      ←── immediate(10) ─→ |
                          ←────────── short(100) ────────→ |
                                                           |
  periodic:     |..10..|              |..10..|   |..10..|  |
                periodic_200          periodic_100  periodic_0

  per_token:                                    ↑  ↑  ... ↑
                                             K-10 K-9    K-1
                                       (逐 token 度量，从 K-1 向回走)

  桶的具体定义：
  - immediate: [K-10, K) — 紧邻分歧点前的 10 个 token（受影响最大的区域）
  - short: [K-100, K) — 分歧点前 100 个 token
  - all_prefix: [0, K) — 整个共享前缀
  - periodic_0: [K-10, K) — 从分歧点回退 0 步，取 10 token 窗口
  - periodic_100: [K-110, K-100) — 从分歧点回退 100 步，取 10 token 窗口
  - per_token: 从 K-1 开始逐 token 向后，共 10 个独立度量

  ---
  第六阶段：存储

  storage.py 新增两列：

  RESULT_COLUMNS = [
      ...原有 16 列...,
      "prompt_pair_index",      # 第几个连续对 (0, 1, 2, ...)
      "common_prefix_length",   # 共享前缀 token 数 (= LCP = 分歧点位置)
  ]

  每条 metric record 同时携带原有字段（复用 perturbation_position_abs = LCP,
  perturbation_span_length = 追加长度）和新增的 prompt_pair_index、common_prefix_length。

  ---
  关于 causal model 的一个重要说明

  在标准 causal (autoregressive) Transformer 中，由于使用了 causal attention mask（每个 token 只能
  attend 到它左边的 token），理论上追加的新 token 不会改变共享前缀的 hidden states。所以
  compute_prefix_distance_buckets 测量到的 cosine similarity 对于大多数位置应该接近 1.0。

  但以下情况可能导致差异：
  1. 数值精度：不同序列长度的 flash attention kernel 可能有微小浮点差异
  2. Position encoding 实现细节：某些 RoPE 实现可能因总长度不同而产生差异
  3. KV cache / padding 策略：批处理时 padding 位置不同可能引入差异

  这个实验正是为了量化这种"理论上应该为零但实际可能非零"的反向影响。

  ---
  运行指令

  cd /root/context/ActivationPerturbationAnalysis

  # 使用新的 prompt_diff 配置运行
  python run_sweep.py --config config/prompt_diff_llama3.yaml

  前提条件：
  1. prompt/extracted_prompts.json 里必须有数据（当前看起来是空的）。如果需要先生成它：
  # 从 vLLM 日志中提取 prompt
  python extract_prompts.py --log prompt/agent.log --output prompt/extracted_prompts.json
  2. 模型路径 /root/share/models/Meta-Llama-3-8B-Instruct 必须存在且可访问
  3. GPU 需要至少 20GB 空闲显存（按配置）

  结果将写入 ./Llama-3-8B-prompt-diff-results/ 目录，格式为 Parquet 文件。


  
● 一切正常。下面做个完整总结。                                                                     
                                                                                      
  ---                                                                                              
  修改总结                                                                                         
   
  改动只涉及两个文件：                                                                             
                                                                                                 
  src/metrics.py

  - 删除了 compute_prefix_distance_buckets()（反向度量函数）
  - 不需要新增任何函数，直接复用已有的 compute_distance_buckets()

  src/experiment.py

  - 修改了 run_prompt_diff_experiment()：
    - 调用 compute_distance_buckets（向前度量），而非之前错误的
  compute_prefix_distance_buckets（向后度量）
    - 两个序列的 post_perturbation_start 均设为 LCP（分歧点），从分歧点开始向前对齐比较
    - 当其中一边分歧点后没有 token 时（纯前缀追加），输出警告并跳过（因为无法做前向比较）

  逻辑对比

  ┌────────────┬─────────────────────────────────┬──────────────────────────────────────────────┐
  │            │         旧（错误）实现          │                新（修正）实现                │
  ├────────────┼─────────────────────────────────┼──────────────────────────────────────────────┤
  │ 度量方向   │ 从分歧点向后看共享前缀          │ 从分歧点向前看后续 token                     │
  ├────────────┼─────────────────────────────────┼──────────────────────────────────────────────┤
  │ 调用函数   │ compute_prefix_distance_buckets │ compute_distance_buckets（复用原有）         │
  ├────────────┼─────────────────────────────────┼──────────────────────────────────────────────┤
  │ post_start │ 不适用                          │ orig=LCP, pert=LCP                           │
  ├────────────┼─────────────────────────────────┼──────────────────────────────────────────────┤
  │ 纯前缀追加 │ 有数据（比较前缀）              │ 无数据（正确跳过，因为 prev 在 LCP 后没有    │
  │            │                                 │ token）                                      │
  ├────────────┼─────────────────────────────────┼──────────────────────────────────────────────┤
  │ 中间分歧   │ 不测量后续 token                │ 正确测量两边在分歧点后的 hidden states       │
  └────────────┴─────────────────────────────────┴──────────────────────────────────────────────┘

  运行指令不变

  cd /root/context/ActivationPerturbationAnalysis
  python run_sweep.py --config config/prompt_diff_llama3.yaml

  注意：对于 agent prompt 常见的"纯前缀追加"情况（prompt[i] 是 prompt[i+1] 的完整前缀），prev
  在分歧点后没有 token，min(post_prev, post_curr) =
  0，该对会被跳过。只有分歧发生在中间（两边分歧点后都有内容）的 prompt 对才会产生有效度量。