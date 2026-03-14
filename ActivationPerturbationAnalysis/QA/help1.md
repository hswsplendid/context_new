（1）请你分析，如果我需要使用/root/context/ActivationPerturbationAnalysis下的代码，测试不同系列的模型进行测试，可能涉及到的模型包括/root/share/models/Meta-Llama-3-8B-Instruct和/root/share/models/Phi-3.5-MoE-instruct，每次测试我应该如何操作？请你给出解决方法。
（2）原来的代码设计，测试是如何安排GPU的，每次应 该安排几张卡需要凡在指令里吗/还是在代码里？                                                           


现在我对整个代码库有了完整的了解，下面是详细的分析：

  ---
  (1) 如何切换不同模型进行测试

  核心：只需修改 config/default.yaml 中的 model.path 和 storage.output_dir

  每次测试不同模型，你需要做以下操作：

  方案 A：手动修改配置文件（最简单）

  # 测试 Llama-3-8B-Instruct 时：
  model:
    path: "/root/share/models/Meta-Llama-3-8B-Instruct"
    dtype: "bfloat16"
    max_memory_per_gpu_gb: 75
    trust_remote_code: false    # Llama 不需要 trust_remote_code

  storage:
    output_dir: "./Llama-3-8B-results"   # 改输出目录，避免覆盖

  # 测试 Phi-3.5-MoE-instruct 时：
  model:
    path: "/root/share/models/Phi-3.5-MoE-instruct"
    dtype: "bfloat16"
    max_memory_per_gpu_gb: 75
    trust_remote_code: true     # Phi-3.5 MoE 需要 trust_remote_code

  storage:
    output_dir: "./Phi-3.5-MoE-results"

  然后运行：
  python run_sweep.py --config config/default.yaml

  方案 B：为每个模型创建独立配置文件（推荐）

  cp config/default.yaml config/llama3-8b.yaml
  cp config/default.yaml config/phi3.5-moe.yaml

  分别修改各自的 model.path、trust_remote_code 和 storage.output_dir，然后：

  python run_sweep.py --config config/llama3-8b.yaml
  python run_sweep.py --config config/phi3.5-moe.yaml

  注意事项

  ┌──────────────────────────┬────────────┬───────────────────┬─────────────────────┐
  │           模型           │ 大约参数量 │ trust_remote_code │ 预估显存需求 (bf16) │
  ├──────────────────────────┼────────────┼───────────────────┼─────────────────────┤
  │ Meta-Llama-3-8B-Instruct │ 8B         │ false             │ ~16 GB（1张卡够）   │
  ├──────────────────────────┼────────────┼───────────────────┼─────────────────────┤
  │ Phi-3.5-MoE-instruct     │ ~42B (MoE) │ true              │ ~84 GB（需多卡）    │
  ├──────────────────────────┼────────────┼───────────────────┼─────────────────────┤
  │ Qwen3-235B-A22B (原配置) │ 235B (MoE) │ true              │ 非常大（需多卡）    │
  └──────────────────────────┴────────────┴───────────────────┴─────────────────────┘

  对于 8B 的 Llama，可以把 max_memory_per_gpu_gb 调小，甚至可以用 --parallel 模式在多张卡上并行跑多个实验。

  ---
  (2) GPU 分配机制分析

  原始代码的 GPU 管理设计

  代码有两种模式，GPU 分配方式完全不同：

  顺序模式（默认，不加 --parallel）

  python run_sweep.py --config config/default.yaml

  - GPU 数量由环境变量 CUDA_VISIBLE_DEVICES 控制，不在配置文件里指定用几张卡
  - 代码调用 torch.cuda.device_count() 获取可见 GPU 数量
  - 使用 device_map="auto"（HuggingFace Accelerate），自动把模型分片到所有可见 GPU
  - max_memory_per_gpu_gb 限制每张卡最多用多少显存
  - 模型只加载一次，所有实验串行共用

  你需要在启动命令前设置环境变量来控制用几张卡：

  # 用 0,1 号两张卡
  CUDA_VISIBLE_DEVICES=0,1 python run_sweep.py --config config/default.yaml

  # 只用 0 号卡
  CUDA_VISIBLE_DEVICES=0 python run_sweep.py --config config/default.yaml

  # 用全部 8 张卡（不设置就是默认全部）
  python run_sweep.py --config config/default.yaml

  并行模式（加 --parallel）

  python run_sweep.py --config config/default.yaml --parallel

  - 代码通过 pynvml 自动检测所有 GPU 的空闲显存
  - 排除空闲显存低于 min_free_memory_gb（配置中为 40GB）的卡
  - 每个实验分配 gpus_per_experiment=1 张卡（硬编码为 1）
  - 每个 worker 通过设置 CUDA_VISIBLE_DEVICES 独占自己的 GPU
  - 每个 worker 独立加载一份模型，适合模型小到单卡能放下的情况

  总结

  ┌──────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │     问题     │                                        答案                                         │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ GPU 数量在哪 │ 顺序模式：通过命令行前的 CUDA_VISIBLE_DEVICES 环境变量控制                          │
  │ 里控制？     │                                                                                     │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │              │ 并行模式：代码自动检测可用 GPU，通过 min_free_memory_gb 过滤                        │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 需要写在代码 │ 不需要。GPU 数量不在代码或配置文件里硬编码                                          │
  │ 里吗？       │                                                                                     │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ config.yaml  │ max_memory_per_gpu_gb=每卡显存上限；min_free_memory_gb=并行模式下筛选卡的门槛；max_ │
  │ 中 GPU 相关  │ parallel_experiments=并行实验数上限                                                 │
  │ 参数的作用？ │                                                                                     │
  └──────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  推荐的完整启动命令

  # Llama-3-8B（小模型，单卡即可，可用并行模式）
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sweep.py --config config/llama3-8b.yaml --parallel

  # Phi-3.5-MoE（较大MoE，需要多卡跑一个模型，用顺序模式）
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sweep.py --config config/phi3.5-moe.yaml


  # 最终指令  配置文件我写两个就可以测不同模型
  CUDA_VISIBLE_DEVICES=5,6 python run_sweep.py --config config/llama3-8b.yaml
  CUDA_VISIBLE_DEVICES=1,2,3,4 python run_sweep.py --config config/phi3.5-moe.yaml


  # 注意让ai写代码时，计算都在gpu上，不要拉满CPU;目前6（2，4）这个指令显存利用率对phi是较低的，可以少用几张卡

  # CPU利用率还是很高，可以改进