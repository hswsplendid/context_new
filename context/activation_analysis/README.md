# Agent Prompt Perturbation - Intermediate Activation Analysis

分析大语言模型在 Agent 任务中，当 Prompt 局部发生变化时，各层中间激活（Hidden States）如何传播这些变化。通过系统性的实验框架，量化不同扰动条件对模型内部表征的影响。

## 研究问题

在 Agent 场景下（如工具调用、多轮对话），当 Prompt 中某一部分发生改变（例如替换工具描述、改写用户指令），模型各层的隐藏状态会如何变化？这些变化如何随层深度传播到后续 Token？

## 目标模型

**Qwen3-30B-A3B**（MoE 架构，48 层，hidden_size=2048，128 专家 / 8 活跃）

硬件要求：1–4 张 NVIDIA A800-SXM4-80GB（通过 `device_map="auto"` 自动多卡分片）

---

## 项目结构

```
activation_analysis/
├── configs/                         # 实验配置文件
│   ├── default.yaml                 # 默认配置（Type 1）
│   ├── experiment_type1.yaml        # 内容替换实验
│   └── experiment_type2.yaml        # 语义改写实验
├── src/                             # 核心源码
│   ├── config.py                    # 配置 Dataclass + YAML 加载
│   ├── model_loader.py              # 模型 / Tokenizer 加载
│   ├── prompt_builder.py            # Prompt 构建 + 分段边界追踪
│   ├── token_aligner.py             # Token 对齐（处理 BPE 边界）
│   ├── activation_extractor.py      # 基于 Hook 的选择性激活提取
│   ├── metrics.py                   # Cosine / L2 / CKA 相似度指标
│   ├── paraphrase_generator.py      # LLM 自动改写生成（Type 2）
│   ├── experiment_runner.py         # 单次实验流水线
│   ├── batch_runner.py              # 参数网格扫描
│   ├── storage.py                   # 结果持久化（CSV / Tensor / YAML）
│   └── visualization.py            # 可视化（热力图 / 折线图 / 柱状图）
├── prompts/                         # Prompt 模板与扰动定义
│   ├── agent_templates/
│   │   ├── tool_use_agent.yaml      # 工具调用 Agent 模板
│   │   └── multi_turn_agent.yaml    # 多轮对话 Agent 模板
│   └── perturbations/
│       └── type1_replacements.yaml  # 内容替换对
├── scripts/                         # 命令行入口
│   ├── run_experiment.py            # 运行单次实验
│   ├── run_sweep.py                 # 运行完整参数扫描
│   └── visualize_results.py         # 从已保存结果生成图表
├── tests/                           # 单元测试（无需 GPU）
├── results/                         # 运行时自动生成
└── requirements.txt
```

---

## 核心概念

### 两种扰动类型

| 类型 | 做法 | 比较对象 | 指标粒度 |
|------|------|----------|----------|
| **Type 1**（内容替换） | 将 Prompt 中一段文本替换为语义不同的内容（如把"文件读取"工具替换为"数据库查询"工具） | 后续区域 A2 vs B2（相同文本、不同上下文） | Token 级 + Segment 级 |
| **Type 2**（语义改写） | 用 LLM 生成同义改写替换原文段 | 被改写区域 A1 vs B1 本身 | Segment 级（CKA / 余弦） |

### Prompt 结构

每个 Prompt 由三部分拼接：

```
[Prefix] + [Segment（被扰动部分）] + [Subsequent（后续部分）]
```

- **Prefix**：不变的上下文前缀
- **Segment**：原始版本 A1 / 扰动版本 B1
- **Subsequent**：后续文本（Type 1 中 A2 和 B2 文本相同，但前面的上下文不同）

### 实验变量

| 变量 | 取值 | 作用 |
|------|------|------|
| 上下文长度 | 512, 1024, 2048, 4096 tokens | 考察长度对激活传播的影响 |
| 扰动比例 | 10%, 25%, 50% | 考察改变量对下游的影响 |
| 扰动位置 | beginning, middle, end | 考察位置效应 |
| 层深度 | 13 个采样层（每隔 4 层） | 浅层 vs 深层激活差异 |
| 扰动类型 | Type1（替换）, Type2（改写） | 不同类型的变化 |

---

## 安装

```bash
cd activation_analysis
pip install -r requirements.txt
```

依赖列表：
- `torch >= 2.1.0`
- `transformers >= 4.40.0`
- `pyyaml`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `numpy`, `scikit-learn`

---

## 使用方法

### 1. 运行单次实验

适合调试或快速验证单个配置点：

```bash
python scripts/run_experiment.py \
    --config configs/default.yaml \
    --context-length 1024 \
    --ratio 0.25 \
    --position middle \
    --pair-index 0 \
    --experiment-id my_test
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/default.yaml` | 实验配置文件路径 |
| `--context-length` | `1024` | 目标 Prompt 总 Token 数 |
| `--ratio` | `0.25` | 扰动段占总长度的比例 |
| `--position` | `middle` | 扰动段位置：`beginning` / `middle` / `end` |
| `--pair-index` | `0` | 使用第几对替换文本（见 `type1_replacements.yaml`） |
| `--output-dir` | 来自配置 | 覆盖输出目录 |
| `--experiment-id` | `single_exp` | 实验标识符 |

输出保存至 `results/{experiment-id}/metrics.csv`

### 2. 运行完整参数扫描

遍历所有实验变量组合（上下文长度 × 扰动比例 × 位置 × 替换对）：

```bash
# Type 1 实验（内容替换）
python scripts/run_sweep.py --config configs/experiment_type1.yaml

# Type 2 实验（语义改写）
python scripts/run_sweep.py --config configs/experiment_type2.yaml

# 不生成图表（仅保存指标数据）
python scripts/run_sweep.py --config configs/default.yaml --no-plots
```

默认配置下的实验网格：4 长度 × 3 比例 × 3 位置 × 3 替换对 = **108 次实验**。

模型只加载一次，所有实验复用。单次实验遇到 GPU OOM 会自动清理缓存并跳过，不影响后续实验。

输出目录结构：

```
results/sweep_{8位UUID}/
├── config.yaml              # 冻结的实验配置
├── metrics.csv              # 所有实验的指标数据
├── paraphrase_cache.yaml    # （Type 2）缓存的改写结果
└── plots/                   # 可视化图表
    ├── similarity_vs_depth_by_ratio.png
    ├── similarity_vs_depth_by_length.png
    ├── shallow_vs_deep.png
    ├── position_effect.png
    ├── context_length_effect.png
    └── heatmap_{length}_{ratio}_{position}.png
```

### 3. 从已保存结果生成图表

无需重新运行实验，直接从 CSV 生成可视化：

```bash
# 生成全部图表
python scripts/visualize_results.py results/sweep_xxxx/metrics.csv

# 只生成特定类型的图表
python scripts/visualize_results.py results/sweep_xxxx/metrics.csv \
    --plot-type depth \
    --metric cosine_mean

# 生成指定条件的热力图
python scripts/visualize_results.py results/sweep_xxxx/metrics.csv \
    --plot-type heatmap \
    --context-length 2048 \
    --ratio 0.25 \
    --position middle
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `metrics_path`（必填） | metrics.csv 文件路径 |
| `--output-dir` | 图表输出目录（默认与 CSV 同目录） |
| `--metric` | 绘图指标：`cosine_mean` / `cosine_segment` / `l2_mean` / `cka` |
| `--plot-type` | 图表类型：`all` / `heatmap` / `depth` / `bar` / `position` / `context` |

---

## 配置文件说明

配置采用 YAML 格式，所有字段均有默认值，可按需覆盖。示例（`configs/default.yaml`）：

```yaml
model:
  model_path: "Qwen/Qwen3-30B-A3B"   # HuggingFace 模型路径或本地路径
  torch_dtype: "bfloat16"              # 推理精度
  device_map: "auto"                   # 多卡自动分片
  max_gpu_count: null                  # 限制使用的 GPU 数量（null=不限制）

extraction:
  layer_indices: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  # 采样层
  save_full_hidden: false              # 是否保存完整隐藏状态张量
  use_hooks: true                      # 使用 Hook 提取（推荐）

perturbation:
  type: "type1"                        # "type1"（替换）或 "type2"（改写）
  context_lengths: [512, 1024, 2048, 4096]
  ratios: [0.10, 0.25, 0.50]
  positions: ["beginning", "middle", "end"]

metrics:
  cosine: true                         # 余弦相似度
  cka: true                            # 线性 CKA
  l2: true                             # L2 距离
  granularity: ["token", "segment"]    # 比较粒度

output_dir: "results"
seed: 42
template_path: "prompts/agent_templates/tool_use_agent.yaml"
replacements_path: "prompts/perturbations/type1_replacements.yaml"
```

---

## Prompt 模板格式

模板文件定义 Agent 场景的 Prompt 结构（YAML 格式）：

```yaml
name: "tool_use_agent"
description: "工具调用 Agent 模板"
system_prompt: |
  You are a helpful AI assistant with access to the following tools.
prefix: |
  ### Available Tools
  1. **web_search**: ...
  2. **calculator**: ...
segment: |
  3. **file_reader**: Read contents of a file...    # ← 被扰动的部分
subsequent: |
  4. **code_executor**: ...
  ### Conversation
  User: Can you help me...
```

替换对文件（`type1_replacements.yaml`）定义具体的扰动内容：

```yaml
replacements:
  - original: |
      3. **file_reader**: Read contents of a file...
    replacement: |
      3. **database_query**: Execute SQL queries...
```

项目内置两个 Agent 模板和三组替换对，可自行扩展。

---

## 核心实现原理

### Hook 提取机制

不使用 `output_hidden_states=True`（会在 GPU 上同时保存所有 49 层），而是在目标层注册 PyTorch Forward Hook：

```python
def hook_fn(module, input, output):
    hidden = output[0]                    # (1, seq_len, 2048)
    selected = hidden[0, token_indices, :] # 只取感兴趣的 Token
    storage[layer_idx] = selected.cpu()    # 立即转移到 CPU
```

- 仅提取感兴趣的 Token（非全序列），内存开销极小
- 13 层 × 1000 Token × 2048 维 × 2 字节 ≈ 53MB CPU 内存
- 每次前向传播后自动清理 Hook

### Token 对齐策略

**Type 1**：A2 和 B2 文本相同但绝对位置不同（因为 B1 可能比 A1 长或短）。逐个比对 Token ID，跳过因 BPE 分词导致的首部不匹配 Token。

**Type 2**：A1 和 B1 Token 完全不同（改写），无法逐 Token 对齐。使用 Mean Pooling 后的余弦相似度，或直接用 CKA（天然支持不同长度矩阵）。

### 相似度指标

| 指标 | 适用粒度 | 公式 | 适用类型 |
|------|----------|------|----------|
| Cosine Similarity | Token 级 | `cos(a, b)` per token | Type 1 |
| L2 Distance | Token 级 | `‖a - b‖₂` per token | Type 1 |
| Cosine (Segment) | Segment 级 | `cos(mean(A), mean(B))` | 两种 |
| Linear CKA | Segment 级 | `‖XᵀY‖²_F / (‖XᵀX‖_F · ‖YᵀY‖_F)` | 两种 |

---

## 可视化图表

| 图表 | 内容 | 文件名 |
|------|------|--------|
| 相似度热力图 | X=层索引，Y=Token位置，颜色=相似度 | `heatmap_{length}_{ratio}_{pos}.png` |
| 相似度-层深度折线图 | 不同扰动比例/长度下，相似度随层深度变化 | `similarity_vs_depth_by_*.png` |
| 浅层/中层/深层柱状图 | 三组层的平均相似度对比 | `shallow_vs_deep.png` |
| 位置效应折线图 | beginning/middle/end 在各层的相似度 | `position_effect.png` |
| 上下文长度效应折线图 | 不同总长度在各层的相似度 | `context_length_effect.png` |

所有图表使用英文标注，由 matplotlib + seaborn 渲染。

---

## 运行测试

单元测试无需 GPU，使用 Mock 模型和 CPU 张量：

```bash
pip install pytest
python -m pytest tests/ -v
```

测试覆盖：
- `test_metrics.py`：余弦相似度、L2 距离、CKA 的正确性与边界条件
- `test_token_aligner.py`：Token 对齐、BPE 边界处理、空序列处理
- `test_prompt_builder.py`：Prompt 构建、长度控制、边界追踪
- `test_activation_extractor.py`：Hook 注册/清理、张量形状、设备转移

---

## GPU 控制

通过环境变量限制使用的 GPU：

```bash
# 只使用第 0、1 号 GPU
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_sweep.py --config configs/default.yaml

# 或在配置文件中设置
# model:
#   max_gpu_count: 2
```

---

## 自定义扩展

### 添加新的 Agent 模板

在 `prompts/agent_templates/` 下创建 YAML 文件，包含 `name`、`system_prompt`、`prefix`、`segment`、`subsequent` 字段，然后在配置文件中指定 `template_path`。

### 添加新的替换对

在 `prompts/perturbations/` 下创建 YAML 文件，格式为 `replacements` 列表，每项包含 `original` 和 `replacement`，然后在配置文件中指定 `replacements_path`。

### 添加新的指标

在 `src/metrics.py` 中实现指标函数，在 `compute_layer_metrics()` 中集成，在 `LayerMetrics` 中添加对应字段，最后在 `src/storage.py` 的 `_metrics_to_row()` 中添加 CSV 列。

---

## 典型工作流

```
1. 选择/编写 Prompt 模板和替换对
2. 编写或修改 YAML 配置文件
3. 运行参数扫描：python scripts/run_sweep.py --config your_config.yaml
4. 查看 results/sweep_xxx/plots/ 下的图表
5. 如需调整可视化：python scripts/visualize_results.py results/sweep_xxx/metrics.csv --metric cka
```
