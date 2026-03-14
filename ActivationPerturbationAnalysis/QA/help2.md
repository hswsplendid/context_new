请你分析绘图的代码应该如何执行，已知数据在/root/context/ActivationPerturbationAnalysis/Llama-3- 
8B-results，和/root/context/ActivationPerturbationAnalysis/Phi-3.5-MoE-results下，输出进行绘图的  
具体指令，和到底绘制什么图的详细说明。                                                            

● Read 3 files (ctrl+o to expand)

● 现在让我确认一下两个结果目录中是否有数据文件。

● Read 1 file (ctrl+o to expand)    
                                                               
● 两个目录都有数据。下面是完整分析：
                                                                                                  
  ---        
  绘图系统说明                                                                                    
                                                                                          
  项目有 两套绘图脚本，功能不同：                                                                 
                                                                                                
  ---                                                                                             
  脚本一：run_analysis.py — 通用分析套件（7 类图）                                                
                                                                                                  
  通过 --results 指定数据目录，--figures 指定输出目录。                                           
                                                                                                  
  具体指令                                                                                        
                                                                                                  
  cd /root/context/ActivationPerturbationAnalysis                                                 

  # Llama-3-8B
  python run_analysis.py --results ./Llama-3-8B-results --figures ./Llama-3-8B-results/figures

  # Phi-3.5-MoE
  python run_analysis.py --results ./Phi-3.5-MoE-results --figures ./Phi-3.5-MoE-results/figures

  生成的 7 类图详细说明

  #: 1
  图名: Similarity vs Distance 曲线
  文件名模式: similarity_vs_distance.png / similarity_vs_distance_{type}.png /
    similarity_vs_distance_ctx{N}.png
  具体说明: X轴=扰动点之后的 token
    偏移距离，Y轴=余弦相似度。选取代表性层绘制多条线。揭示扰动效果如何随距离衰减恢复。会生成：1张
  总图
     + 6张按扰动类型拆分 + 5张按上下文长度拆分 = 约12张
  ────────────────────────────────────────
  #: 2
  图名: 层级热力图
  文件名模式: heatmap_{bucket}.png
  具体说明: X轴=扰动类型，Y轴=层编号，颜色=平均余弦相似度。展示每层对不同扰动的敏感程度。生成 3
    张：immediate/short/all_remaining
  ────────────────────────────────────────
  #: 3
  图名: 上下文长度对比
  文件名模式: context_length_{bucket}.png
  具体说明: X轴=上下文长度(256~4096,
    log2)，Y轴=平均余弦相似度，每条线=一种扰动类型。回答：序列越长，扰动影响是否被稀释？ 生成 3 张
  ────────────────────────────────────────
  #: 4
  图名: 扰动位置效应
  文件名模式: position_effect_{bucket}.png
  具体说明:
  X轴=扰动位置(10%~90%)，Y轴=平均余弦相似度。回答：在序列前部/中部/后部扰动，影响有何不同？
    生成 3 张
  ────────────────────────────────────────
  #: 5
  图名: 扰动跨度效应
  文件名模式: span_length_{bucket}.png
  具体说明: X轴=扰动 span 长度(8/32/128 tokens,
    log2)，Y轴=平均余弦相似度。回答：扰动范围越大，影响是否线性增长？ 生成 3 张
  ────────────────────────────────────────
  #: 6
  图名: 逐 Token 相似度曲线
  文件名模式: per_token_{type}.png
  具体说明: X轴=扰动后第0~9个 token，Y轴=余弦相似度，每条线=一层。展示扰动后最近10个 token
    的细粒度恢复过程。生成 6 张（每种扰动类型一张）
  ────────────────────────────────────────
  #: 7
  图名: 相似度分布箱线图
  文件名模式: distribution_{bucket}.png
  具体说明: X轴=扰动类型，Y轴=余弦相似度箱线图。展示每种扰动类型的结果分散程度和离群值。生成 3 张

  每个模型约生成 ~33 张图。

  ---
  脚本二：plot_heatmaps.py — 高质量热力图专版（4 类图）

  这个脚本是为发表论文设计的，使用 1 -
  cosine_similarity（扰动影响度）作为颜色值，带数值标注，更精美。

  注意： 此脚本的数据路径和采样层号是硬编码的，需要修改后才能用。

  需要修改的地方

  对于 Llama-3-8B（32层模型）：
  DATA_DIR = "Llama-3-8B-results"                              # 改数据目录
  SAMPLED_LAYERS = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32]       # 改采样层（Llama-3-8B 共33层 =
  embedding + 32 transformer layers）

  对于 Phi-3.5-MoE（层数需确认）：
  DATA_DIR = "Phi-3.5-MoE-results"
  SAMPLED_LAYERS = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32]       # 根据实际层数调整

  具体指令

  cd /root/context/ActivationPerturbationAnalysis

  # 修改 plot_heatmaps.py 中的 DATA_DIR 和 SAMPLED_LAYERS 后：
  python plot_heatmaps.py

  生成的 4 类图详细说明

  ┌─────┬──────────────┬──────────────────┬────────────────────────────────────────────────────┐
  │  #  │     图名     │      文件名      │                      具体说明                      │
  ├─────┼──────────────┼──────────────────┼────────────────────────────────────────────────────┤
  │     │              │                  │ 1×3 布局：immediate/short/all_remaining            │
  │ 1   │ 主热力图三联 │ heatmap_triptych │ 三个窗口并排。Y轴=6种扰动类型，X轴=采样层，颜色=扰 │
  │     │ 画           │ .png             │ 动影响度(1-similarity)，带数值标注。共用           │
  │     │              │                  │ colorbar。一图总览全局                             │
  ├─────┼──────────────┼──────────────────┼────────────────────────────────────────────────────┤
  │     │ 按位置分层热 │ heatmap_by_posit │ 5×3 布局（5个扰动位置 × 3个窗口）。每个子图: Y轴=  │
  │ 2   │ 力图         │ ion.png          │ 扰动类型，X轴=层。揭示扰动位置与层的交互效应。共   │
  │     │              │                  │ 15 个子图                                          │
  ├─────┼──────────────┼──────────────────┼────────────────────────────────────────────────────┤
  │ 3   │ 位置效应热力 │ heatmap_position │ 1×3 布局：Y轴=5个扰动位置(10%~90%)，X轴=层，颜色=  │
  │     │ 图           │ _effect.png      │ 影响度。跨扰动类型平均后，看位置×层的纯交互        │
  ├─────┼──────────────┼──────────────────┼────────────────────────────────────────────────────┤
  │ 4   │ 按扰动类型的 │ heatmap_position │ 6×3 布局（6种扰动 × 3个窗口）。每个子图:           │
  │     │ 位置热力图   │ _per_type.png    │ Y轴=位置，X轴=层。最详细的三维交叉分析             │
  └─────┴──────────────┴──────────────────┴────────────────────────────────────────────────────┘

  此外还会在终端打印一份文字分析报告，包含：最具破坏性的扰动类型、影响峰值所在层、恢复模式、位置敏
  感性等。

  ---
  总结：执行清单

  cd /root/context/ActivationPerturbationAnalysis

  # ========== 通用分析图 ==========
  # Llama-3-8B (约33张图)
  python run_analysis.py --results ./Llama-3-8B-results --figures ./Llama-3-8B-results/figures

  # Phi-3.5-MoE (约33张图)
  python run_analysis.py --results ./Phi-3.5-MoE-results --figures ./Phi-3.5-MoE-results/figures

  # ========== 高质量热力图 ==========
  # 需先修改 plot_heatmaps.py 中的 DATA_DIR 和 SAMPLED_LAYERS，然后：
  python plot_heatmaps.py

  如果你希望，我可以直接帮你修改 plot_heatmaps.py 使其支持命令行参数传入
  DATA_DIR，避免每次手动改代码。


  ● 改好了。新增了以下命令行参数：   
                                                                                                  
  ┌────────────────────┬────────────────────────────────────────────────┐                         
  │        参数        │                      说明                      │                         
  ├────────────────────┼────────────────────────────────────────────────┤                         
  │ --data-dir         │ 数据目录路径（替代硬编码的 DATA_DIR）          │                         
  ├────────────────────┼────────────────────────────────────────────────┤                         
  │ --out-dir          │ 输出目录（默认 <data-dir>/figures）            │                         
  ├────────────────────┼────────────────────────────────────────────────┤                         
  │ --layers 0 4 8 ... │ 手动指定采样层列表                             │                         
  ├────────────────────┼────────────────────────────────────────────────┤                         
  │ --auto-layers      │ 自动从数据中均匀选取层（优先于 --layers）      │
  ├────────────────────┼────────────────────────────────────────────────┤
  │ --num-layers N     │ 配合 --auto-layers 使用，控制选几层（默认 12） │
  └────────────────────┴────────────────────────────────────────────────┘

  使用方式

  # Llama-3-8B (32层) — 手动指定层
  python plot_heatmaps.py --data-dir Llama-3-8B-results --layers 0 4 8 12 16 20 24 28 32

  # Phi-3.5-MoE — 自动选层（不用查模型有几层）
  python plot_heatmaps.py --data-dir Phi-3.5-MoE-results --auto-layers
  python plot_heatmaps.py --data-dir Llama-3-8B-results --auto-layers

  # 自动选层但只选 8 个
  python plot_heatmaps.py --data-dir Phi-3.5-MoE-results --auto-layers --num-layers 8

  # 不传参数 — 行为和原来完全一样（用默认 Qwen2.5-32B-results + 硬编码层）
  python plot_heatmaps.py

