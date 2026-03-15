---
  改动总结                                                                                     
   
  新增文件                                                                                     
                                                                                             
  ┌──────────────────────────────────┬────────────────────────────────────────────────┐
  │               文件               │                      作用                      │
  ├──────────────────────────────────┼────────────────────────────────────────────────┤
  │ extract_prompts.py               │ 从 vLLM 日志中提取 prompt，输出为 JSON         │
  ├──────────────────────────────────┼────────────────────────────────────────────────┤
  │ prompt/extracted_prompts.json    │ 已提取的 12 条实测 prompt（7.5k~12.4k tokens） │
  ├──────────────────────────────────┼────────────────────────────────────────────────┤
  │ config/agent_prompts_llama3.yaml │ Llama-3-8B + 实测 prompt 的配置                │
  ├──────────────────────────────────┼────────────────────────────────────────────────┤
  │ config/agent_prompts_phi3.yaml   │ Phi-3.5-MoE + 实测 prompt 的配置               │
  └──────────────────────────────────┴────────────────────────────────────────────────┘

  修改文件

  ┌────────────────┬────────────────────────────────────────────────────────────────────────┐
  │      文件      │                                  改动                                  │
  ├────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ src/prompts.py │ 新增 source="jsonfile" 模式，从 JSON 文件加载 prompt                   │
  ├────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ src/sweep.py   │ 支持 jsonfile 模式下按 prompt 真实 token 数生成实验任务，正确映射      │
  │                │ prompt 索引                                                            │
  └────────────────┴────────────────────────────────────────────────────────────────────────┘

  完全向后兼容

  原有 source="dataset" 和 source="template" 模式不受影响。

  ---
  使用方式

  第一步：提取 prompt（已完成）

  python extract_prompts.py --log prompt/agent.log --output prompt/extracted_prompts.json

  输出 12 条 prompt，token 数从 7,506 到 12,358 不等。

  第二步：用不同模型跑实验

  cd /root/context/ActivationPerturbationAnalysis

  # Llama-3-8B（8B 模型，单卡可跑）
  CUDA_VISIBLE_DEVICES=3 python run_sweep.py --config config/agent_prompts_llama3.yaml

  # Phi-3.5-MoE（较大 MoE，需多卡）
  CUDA_VISIBLE_DEVICES=4,5,6 python run_sweep.py --config config/agent_prompts_phi3.yaml

  每个模型将生成 1080 个实验（12 prompt × 5 位置 × 3 跨度 × 6 扰动类型）。

  第三步：绘图

  # 通用分析图
  python run_analysis.py --results ./Llama-3-8B-agent-results
  python run_analysis.py --results ./Phi-3.5-MoE-agent-results

  # 高质量热力图
  python plot_heatmaps.py --data-dir Llama-3-8B-agent-results --auto-layers
  python plot_heatmaps.py --data-dir Phi-3.5-MoE-agent-results --auto-layers

  配置文件中的关键字段

  prompts:
    source: "jsonfile"                          # 新模式
    jsonfile_path: "prompt/extracted_prompts.json"  # 指向提取的文件

  sweep.context_lengths 在 jsonfile 模式下被忽略，实验自动使用每条 prompt 的真实 token 数作为
  context_length 维度。

当我用prompt/extracted_prompts.json的真实agent的prompt跑测试时，要对比的就是这一条和上一条的prompt改变，这里的扰动就不需要人为规定了。扰动的其实就是这一个prompt相较于上一个prompt的有差别的地方。请你理解我的含义,设计plan，修改代码。