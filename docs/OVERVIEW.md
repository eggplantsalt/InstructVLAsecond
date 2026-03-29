# InstructVLA 技术总览（OVERVIEW）

本文档面向新成员与二次开发者，基于仓库静态分析整理。目标是让你在最短时间内完成：
1. 架构理解
2. 训练与评测跑通
3. 关键模块定位与扩展开发

> 说明：若某些内容依赖私有数据路径、集群环境或外部资产，本文件会标注 [需补充]。

---

## 第一部分：项目架构分析

### 1. 项目整体架构

#### 1.1 目录树（tree）

~~~text
InstructVLA/
├── README.md
├── pip_requirements.txt
├── conda_requirements.txt
├── train_libero_10.sh
├── train_libero_goal.sh
├── train_libero_obj.sh
├── train_libero_spatial.sh
├── conf/
│   ├── vla.py
│   ├── action_expert_backbone_small.json
│   └── action_expert_backbone_large.json
├── vla/
│   ├── instructvla_eagle_dual_sys_v2_meta_query_v2.py
│   ├── instructvla_eagle_dual_sys_v2_meta_query_v2_state.py
│   ├── instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist.py
│   ├── action_head.py
│   ├── film_vit.py
│   ├── eagle_utils.py
│   └── modeling_eagle_chat.py
├── scripts/
│   ├── train_eagle_dual_v2_action_only_meta_query_v2.py
│   ├── train_eagle_dual_v2_action_only_meta_query_v2_libero_wrist.py
│   └── unload_lora.py
├── training/
│   ├── materialize.py
│   ├── metrics.py
│   └── strategies/
│       ├── base_strategy_cogact.py
│       ├── fsdp.py
│       ├── ddp.py
│       └── _zeros.py
├── deploy/
│   ├── robot_utils.py
│   ├── openvla_utils.py
│   ├── instructvla_utils.py
│   └── libero/
│       ├── run_libero_eval.py
│       ├── libero_utils.py
│       ├── regenerate_libero_dataset.py
│       └── libero_requirements.txt
├── data_pipeline/
│   ├── annotate_instruction_api_batch.py
│   ├── primitive_fractal.py
│   ├── data_loading_example.ipynb
│   └── real_data_script/
├── mm_dataset/
│   └── data_utils.py
├── mm_evaluation/
│   ├── README.md
│   ├── Evaluator.py
│   ├── VLA_IT_InstructVLA.py
│   └── vlmeval/
├── prismatic/
│   ├── models/
│   ├── training/
│   ├── util/
│   └── vla/
└── SimplerEnv/
    ├── README.md
    ├── setup.py
    ├── simpler_env/
    └── scripts_self/
~~~

#### 1.2 核心模块与职责

- 配置层
  - [conf/vla.py](../conf/vla.py)
  - 用 draccus 注册 VLA 变体（数据混合、基础 VLM、分布式规模、优化超参）。
- 模型层
  - [vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py)
  - [vla/instructvla_eagle_dual_sys_v2_meta_query_v2_state.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2_state.py)
  - [vla/instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist.py)
  - 负责 InstructVLA 主体、Stage1/Stage2（LoRA/X-LoRA）训练逻辑、动作头融合。
- 动作建模层
  - [vla/action_head.py](../vla/action_head.py)
  - FiLMed DINO 视觉特征 + 认知 token + Flow Matching 动作序列建模。
- 数据层
  - [mm_dataset/data_utils.py](../mm_dataset/data_utils.py)
  - 多模态监督数据集（Bunny、pointing、point-detection）及 collator。
  - prismatic 下 RLDS/OXE 数据加载（训练脚本通过 get_vla_dataset_and_collator 间接调用）。
- 训练编排层
  - [scripts/train_eagle_dual_v2_action_only_meta_query_v2.py](../scripts/train_eagle_dual_v2_action_only_meta_query_v2.py)
  - [scripts/train_eagle_dual_v2_action_only_meta_query_v2_libero_wrist.py](../scripts/train_eagle_dual_v2_action_only_meta_query_v2_libero_wrist.py)
  - [training/materialize.py](../training/materialize.py)
  - [training/strategies/base_strategy_cogact.py](../training/strategies/base_strategy_cogact.py)
  - [training/strategies/fsdp.py](../training/strategies/fsdp.py)
  - 负责初始化分布式策略、训练循环、保存 checkpoint、记录指标。
- 部署评测层
  - [deploy/instructvla_utils.py](../deploy/instructvla_utils.py)
  - [deploy/robot_utils.py](../deploy/robot_utils.py)
  - [deploy/libero/run_libero_eval.py](../deploy/libero/run_libero_eval.py)
  - 负责动作推理、动作集成（ensemble）、LIBERO 评测回放与日志统计。
- 数据工具与基准
  - [deploy/libero/regenerate_libero_dataset.py](../deploy/libero/regenerate_libero_dataset.py)
  - [mm_evaluation/README.md](../mm_evaluation/README.md)
  - [SimplerEnv](../SimplerEnv)

#### 1.3 模块依赖关系图

~~~mermaid
flowchart LR
    A[conf/vla.py 配置注册] --> B[训练脚本 scripts/train_*]
    B --> C[模型层 vla/instructvla_*]
    B --> D[数据层 prismatic RLDS + mm_dataset]
    B --> E[训练策略层 training/strategies]
    E --> F[指标层 training/metrics]

    C --> G[action_head.py]
    C --> H[eagle_utils.py]
    C --> I[ckpt/Eagle2-2B 与 LoRA/X-LoRA]

    J[deploy/libero/run_libero_eval.py] --> K[deploy/robot_utils.py]
    K --> L[deploy/instructvla_utils.py]
    K --> M[deploy/openvla_utils.py]
    J --> N[deploy/libero/libero_utils.py]

    O[data_pipeline 与 mm_evaluation] --> B
    O --> J
~~~

#### 1.4 数据流图（训练 + 评测）

~~~mermaid
flowchart LR
    subgraph Train[训练数据流]
      T1[OXE/RLDS 数据 + Bunny 等 mm 数据] --> T2[Dataset/Collator]
      T2 --> T3[InstructVLA forward]
      T3 --> T4[语言损失 + 动作损失]
      T4 --> T5[FSDP/DDP 优化更新]
      T5 --> T6[Checkpoint + 指标日志]
    end

    subgraph Eval[评测数据流]
      E1[LIBERO 环境观测 agentview/wrist/state] --> E2[预处理 resize/crop]
      E2 --> E3[InstructVLAServer.get_vla_action]
      E3 --> E4[动作后处理 gripper normalize/invert]
      E4 --> E5[环境 step + 成功率统计 + 视频保存]
    end
~~~

#### 1.5 每模块技术栈与关键依赖

- 深度学习框架
  - PyTorch 2.2.0（含 torch.distributed + FSDP）
  - transformers 4.51.0
  - peft 0.13.0（LoRA/X-LoRA）
- 视觉与数据
  - timm（DINOv2 backbone）
  - TensorFlow（部署中图像 crop/resize 工具）
  - datasets（HuggingFace 数据集）
- 训练与配置
  - draccus（配置注入）
  - wandb、jsonlines（实验追踪）
- 机器人与仿真
  - LIBERO、robosuite、mujoco、sapien
  - SimplerEnv + ManiSkill2_real2sim

💡 提示：生产部署时建议将训练依赖与评测依赖分环境隔离。仓库已建议 LIBERO 评测单独 conda 环境。

⚠️ 警告：部分脚本中存在集群路径与私有绝对路径（例如 /mnt/...）；迁移环境需统一替换为本地路径。

**Checklist**
- [ ] 已识别配置、模型、数据、训练、部署五层结构
- [ ] 已理解训练与评测两条主数据流
- [ ] 已确认关键外部依赖（FSDP、peft、LIBERO、SimplerEnv）

---

### 2. 模块详细说明

#### 2.1 配置模块

- 关键文件
  - [conf/vla.py](../conf/vla.py)
- 功能描述
  - 定义 VLAConfig（ChoiceRegistry），通过 vla_id 注册不同实验配方。
  - 将数据混合、优化器参数、分布式规模、冻结策略统一抽象。
- 接口与调用
  - 训练脚本中 TrainConfig.vla 默认读取 VLARegistry 的某一项。
  - 启动时通过命令行参数覆盖（例如 --vla.type、--vla.learning_rate）。
- 关键参数
  - vla_id、base_vlm、data_mix
  - expected_world_size、global_batch_size、per_device_batch_size
  - learning_rate、lr_scheduler_type、warmup_ratio

#### 2.2 模型模块（InstructVLA）

- 关键文件
  - [vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py)
  - [vla/instructvla_eagle_dual_sys_v2_meta_query_v2_state.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2_state.py)
  - [vla/instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2_libero_wrist.py)
- 功能描述
  - 在 VLM 基础上增加动作生成分支，支持 Stage1（LoRA + 新 token）与 Stage2（X-LoRA）。
  - forward 同时支持语言建模与动作建模模式。
- 调用关系
  - 由训练脚本 load/load_vla 初始化后，交给 training strategy 训练。
- 关键类与函数
  - InstructVLA
  - load、load_vla、get_vla_dataset_and_collator（同文件内定义）
  - predict_action（部署调用）

#### 2.3 动作头模块

- 关键文件
  - [vla/action_head.py](../vla/action_head.py)
- 功能描述
  - ActionModel：使用 FiLMed DINO 视觉特征 + 语言潜变量 + Flow Matching 训练动作序列。
  - ActionModelWithState：在 ActionModel 基础上追加 proprio state embedding。
- 关键结构
  - FiLMedDinoVisionBackbone
  - SinusoidalPosEmb
  - ActionEncoder
- 输出行为
  - 训练：输出 flow matching MSE loss
  - 推理：sampling 逐步积分生成未来动作 chunk

#### 2.4 数据模块

- 关键文件
  - [mm_dataset/data_utils.py](../mm_dataset/data_utils.py)
- 功能描述
  - LazySupervisedDataset：Bunny 对话-图像监督数据。
  - LazyPointingDataset：pointing+解释风格数据。
  - LazyPointDetectionDataset：点检测与计数数据。
  - DataCollatorForSupervisedDataset：padding 与 image_flags 打包。
- 调用关系
  - 训练脚本 use_mm=True 时注入 mm_dataset 与 mm_collator。

#### 2.5 训练策略模块

- 关键文件
  - [training/materialize.py](../training/materialize.py)
  - [training/strategies/base_strategy_cogact.py](../training/strategies/base_strategy_cogact.py)
  - [training/strategies/fsdp.py](../training/strategies/fsdp.py)
  - [training/metrics.py](../training/metrics.py)
- 功能描述
  - materialize 按字符串选择 FSDP/DDP。
  - base_strategy_cogact 提供通用训练循环与 VLA 训练循环。
  - fsdp 封装 mixed precision、checkpoint、optimizer/scheduler。
  - metrics 统一 JSONL/W&B 指标上报。

#### 2.6 部署与评测模块

- 关键文件
  - [deploy/instructvla_utils.py](../deploy/instructvla_utils.py)
  - [deploy/robot_utils.py](../deploy/robot_utils.py)
  - [deploy/libero/run_libero_eval.py](../deploy/libero/run_libero_eval.py)
  - [deploy/libero/libero_utils.py](../deploy/libero/libero_utils.py)
- 功能描述
  - InstructVLAServer 封装模型加载、动作 chunk 管理、可选动作集成。
  - robot_utils 统一 get_model/get_action 与 gripper 归一化逻辑。
  - run_libero_eval 执行 LIBERO 多任务评测，输出成功率与回放视频。

#### 2.7 配置文件作用与参数说明（高频）

- 训练配置入口
  - [scripts/train_eagle_dual_v2_action_only_meta_query_v2.py](../scripts/train_eagle_dual_v2_action_only_meta_query_v2.py)
  - [scripts/train_eagle_dual_v2_action_only_meta_query_v2_libero_wrist.py](../scripts/train_eagle_dual_v2_action_only_meta_query_v2_libero_wrist.py)
- 高频参数
  - --vla.base_vlm：基础视觉语言模型目录
  - --vla.type：配置注册 ID
  - --vla.data_mix：数据混合标识（如 libero_goal_no_noops）
  - --stage：stage1 或 stage2
  - --future_action_window_size、--past_action_window_size：动作窗口
  - --use_mm、--with_pointing：是否混入通用多模态数据
  - --image_aug：训练时是否启用图像增强

💡 提示：对齐 checkpoint 与配置时，优先检查 stage、action_dim、future_action_window_size 三个参数是否一致。

⚠️ 警告：Stage2 默认 adapter 映射里含 empty_language_adapter，需要按注释替换为 Stage1 导出的 LoRA 模块，否则性能会明显受限。

**Checklist**
- [ ] 已掌握每个核心目录的功能边界
- [ ] 已明确训练脚本到模型、策略、数据的调用关系
- [ ] 已识别 Stage1/Stage2 差异与风险点

---

### 3. 代码组织逻辑

#### 3.1 命名规范与组织原则

- 脚本命名
  - train_*.py：训练入口
  - run_*_eval.py：评测入口
  - *_utils.py：部署或数据辅助函数
- 模型命名
  - instructvla_eagle_dual_sys_v2_meta_query_v2*.py：按输入模态差异拆分变体
- 组织原则
  - 配置与实现解耦（conf 与 scripts）
  - 训练策略与模型解耦（training/strategies 与 vla）
  - 评测与训练解耦（deploy 与 scripts）

#### 3.2 使用到的设计模式

- 策略模式
  - training/materialize.py 动态选择 FSDP/DDP。
- 工厂/注册模式
  - conf/vla.py 使用 ChoiceRegistry + Enum 注册配置。
- 门面模式
  - deploy/instructvla_utils.py 用 InstructVLAServer 对外暴露统一动作接口。

#### 3.3 数据流与控制流

- 控制流（训练）
  - CLI -> draccus 解析 -> 加载模型/数据 -> 构建策略 -> 训练循环 -> 保存 checkpoint/日志
- 数据流（训练）
  - RLDS batch + mm batch -> forward -> loss 聚合 -> backward + step -> metric push
- 控制流（评测）
  - 任务枚举 -> 环境 reset -> observation -> 推理动作 -> step -> done 统计

**Checklist**
- [ ] 能解释为何配置、策略、模型分层
- [ ] 能画出训练入口到保存 checkpoint 的流程
- [ ] 能定位评测动作从观测到执行的全链路

---

## 第二部分：快速上手指南

### 1. 环境准备

#### 1.1 系统要求

- OS：Linux 优先（脚本与依赖按 Linux 组织）
- Python：3.10
- CUDA：12.1（推荐，与 README 安装示例一致）
- GPU：支持 BF16（FSDP mixed precision 路径默认开启）

> Windows 可做代码阅读与部分推理调试；完整训练/仿真评测建议 Linux。Windows 全流程支持状态：[需补充]

#### 1.2 依赖安装步骤

~~~bash
conda create -n instructvla python=3.10 -y
conda activate instructvla

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.0 accelerate==1.3.0 peft==0.13.0
pip install numpy==1.26.4 packaging ninja

# 可选：训练提速
pip install flash-attn==2.5.5 --no-build-isolation

pip install -r pip_requirements.txt
~~~

LIBERO 评测建议单独环境：

~~~bash
conda create --name instructvla_libero --clone instructvla
conda activate instructvla_libero

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git libero
cd libero
pip install -e .
cd ../deploy/libero
pip install -r libero_requirements.txt
~~~

SimplerEnv 安装：

~~~bash
conda activate instructvla
cd SimplerEnv
git clone https://github.com/YangS03/my_maniskill.git ManiSkill2_real2sim
cd ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
~~~

#### 1.3 常见问题与解决方案

- 问题：启动时报 cuDNN 动态库符号错误
  - 方案：参考训练 shell 脚本设置 LD_LIBRARY_PATH 与 LD_PRELOAD。
- 问题：FSDP 启动断言 world size 不匹配
  - 方案：调整 --vla.expected_world_size 与实际 GPU 数一致，或临时 --debug True（仅调试）。
- 问题：推理速度异常慢（Stage2）
  - 方案：确认已应用 eagle_utils 中针对 X-LoRA hook 清理逻辑。

💡 提示：首次运行先用最小 batch 和单机 1 卡 smoke test，再放大到多卡。

⚠️ 警告：训练脚本里默认会读取本地或集群数据路径，不存在会直接失败。

**Checklist**
- [ ] 已创建并激活正确 conda 环境
- [ ] 已安装主依赖与可选加速依赖
- [ ] 已完成至少一次最小化 smoke test

---

### 2. 项目启动流程

#### 2.1 训练启动命令序列（基础版）

~~~bash
# 1) 激活环境
conda activate instructvla

# 2) 启动分布式训练（示例：单机8卡）
python -m torch.distributed.run \
  --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_port 29323 \
  scripts/train_eagle_dual_v2_action_only_meta_query_v2.py \
  --vla.base_vlm ckpt/Eagle2-2B \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix oxe_magic_soup_plus_minus \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.train_strategy fsdp-full-shard \
  --run_root_dir ./outputs \
  --run_id instructvla_stage1_demo \
  --stage stage1
~~~

#### 2.2 LIBERO 训练（wrist 变体）

可直接参考：
- [train_libero_goal.sh](../train_libero_goal.sh)
- [train_libero_10.sh](../train_libero_10.sh)
- [train_libero_obj.sh](../train_libero_obj.sh)
- [train_libero_spatial.sh](../train_libero_spatial.sh)

示例（goal）：

~~~bash
bash train_libero_goal.sh
~~~

#### 2.3 LIBERO 评测

~~~bash
conda activate instructvla_libero
python deploy/libero/run_libero_eval.py \
  --model_family instruct_vla \
  --pretrained_checkpoint /path/to/checkpoint.pt \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task 50 \
  --use_wandb False
~~~

#### 2.4 SimplerEnv 评测

可参考：
- [scripts_test_SimplerEnv/eval_instruct_vla_1.sh](../scripts_test_SimplerEnv/eval_instruct_vla_1.sh)
- [scripts_test_SimplerEnv/evaluate_single_checkpoint.sh](../scripts_test_SimplerEnv/evaluate_single_checkpoint.sh)

#### 2.5 参数含义与可选值（启动相关）

- --stage
  - 可选：stage1, stage2
  - 影响：stage1 用 LoRA；stage2 用 X-LoRA 与多专家融合
- --vla.data_mix
  - 示例：libero_goal_no_noops, libero_10_no_noops, oxe_magic_soup_plus_minus
  - 影响：训练数据分布与泛化范围
- --image_aug
  - 可选：True/False
  - 影响：部署时 center_crop 是否必须打开

💡 提示：若 checkpoint 名称中包含 image_aug，评测建议强制 center_crop=True。

⚠️ 警告：不要同时启用 --load_in_8bit 与 --load_in_4bit。

**Checklist**
- [ ] 能独立启动一次训练任务
- [ ] 能独立完成一次 LIBERO 评测
- [ ] 已理解 stage/data_mix/image_aug 对行为的影响

---

### 3. 训练参数配置

> 本节基于 TrainConfig + VLAConfig 静态整理。

#### 3.1 参数列表（核心）

| 参数 | 默认值 | 取值建议 | 影响 |
|---|---:|---|---|
| vla.expected_world_size | 8/16（依配置项） | 与实际 GPU 数一致 | 决定是否通过启动断言 |
| vla.global_batch_size | 256 | 64-512 | 影响稳定性与吞吐 |
| vla.per_device_batch_size | 16/32 | 显存允许范围内最大 | 影响显存占用 |
| vla.learning_rate | 2e-5（配置）/5e-5（LIBERO脚本） | 1e-5 到 1e-4 | 收敛速度与震荡 |
| vla.lr_scheduler_type | constant | constant 或 linear-warmup+cosine-decay | 影响训练后期质量 |
| future_action_window_size | 15（TrainConfig默认） | 7-15 常见 | 预测动作 chunk 长度 |
| past_action_window_size | 0 | 0-8 | 历史动作上下文 |
| action_dim | 7 | 固定7（当前实现） | 与环境 action 维度一致 |
| use_mm | True | True/False | 是否混入通用多模态监督 |
| with_pointing | True | True/False | 是否混入 pointing/detection 数据 |
| disable_instruction | False | True/False | 消融指令输入 |
| num_of_meta_query | 64 | 32-128 [需补充] | 影响认知 token 容量 |

#### 3.2 常用参数模板

Stage1（机器人为主，关闭 mm）：

~~~bash
--stage stage1 \
--use_mm False \
--vla.train_strategy fsdp-full-shard \
--future_action_window_size 7 \
--past_action_window_size 0
~~~

Stage2（语言泛化增强，开启 mm）：

~~~bash
--stage stage2 \
--use_mm True \
--with_pointing True \
--vla.train_strategy fsdp-full-shard
~~~

State 变体（SimplerEnv 强相关）：

~~~bash
# 使用包含 state 的模型入口脚本/模型定义
# 对应文件：vla/instructvla_eagle_dual_sys_v2_meta_query_v2_state.py
~~~

#### 3.3 调优建议与最佳实践

- 先固定数据混合，再调学习率与 batch，不要同时改太多变量。
- 先跑 stage1 收敛，再切 stage2；并确认 adapter 替换正确。
- 训练中定期评测，避免只看训练 loss。
- 为每个实验记录：代码 commit、配置、数据版本、checkpoint 路径。

💡 提示：若显存紧张，优先降低 per_device_batch_size，再调 global_batch_size 与梯度累积策略。[需补充]

⚠️ 警告：错误的 unnorm_key 会在推理动作反归一化阶段造成异常或动作失真。

**Checklist**
- [ ] 已掌握核心参数及默认值
- [ ] 已准备 stage1/stage2 两套参数模板
- [ ] 已建立最小可复现实验记录习惯

---

## 第三部分：深入学习路线

### 1. 代码阅读顺序

#### 1.1 推荐阅读路径

1. [README.md](../README.md)
2. [conf/vla.py](../conf/vla.py)
3. [scripts/train_eagle_dual_v2_action_only_meta_query_v2.py](../scripts/train_eagle_dual_v2_action_only_meta_query_v2.py)
4. [training/materialize.py](../training/materialize.py)
5. [training/strategies/base_strategy_cogact.py](../training/strategies/base_strategy_cogact.py)
6. [vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py](../vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py)
7. [vla/action_head.py](../vla/action_head.py)
8. [deploy/instructvla_utils.py](../deploy/instructvla_utils.py)
9. [deploy/libero/run_libero_eval.py](../deploy/libero/run_libero_eval.py)

#### 1.2 必读与可选文件

- 必读
  - conf/vla.py
  - scripts/train_*.py
  - vla/instructvla_*.py
  - vla/action_head.py
  - training/strategies/*.py
- 可选
  - mm_dataset/data_utils.py
  - deploy/openvla_utils.py
  - deploy/libero/regenerate_libero_dataset.py
  - scripts/unload_lora.py

#### 1.3 分阶段学习目标

- 第1阶段：跑通训练
  - 目标：理解配置如何注入训练脚本并成功起训
- 第2阶段：理解模型
  - 目标：看懂 forward 的语言/动作双损失计算
- 第3阶段：理解部署
  - 目标：能从环境 observation 追踪到 action 输出
- 第4阶段：开始二开
  - 目标：新增一个数据混合策略或动作头变体

**Checklist**
- [ ] 已按路径完成第一轮通读
- [ ] 已能解释训练与部署的关键函数入口
- [ ] 已具备单点改动并验证的能力

---

### 2. 核心概念理解

#### 2.1 关键术语

- VLA：Vision-Language-Action 模型
- RLDS/OXE：机器人轨迹数据格式/数据集合
- LoRA/X-LoRA：参数高效微调方法
- Flow Matching：通过速度场拟合实现连续生成
- unnorm_key：动作反归一化使用的数据统计键

#### 2.2 核心算法与业务逻辑

- 业务主线
  - 视觉与语言输入 -> 生成认知 token -> 动作头预测未来动作序列
- 动作头逻辑
  - 视觉特征（DINO）与认知特征拼接，结合时间嵌入和噪声动作，优化 flow matching 损失

Flow Matching 目标可理解为：

$$
\mathcal{L}=\mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(\psi_t)-\frac{d\psi_t}{dt}\|^2\right]
$$

其中 $x_1$ 为真实动作，$x_0$ 为噪声动作，$\psi_t$ 为两者插值轨迹。

#### 2.3 理论基础与参考资料

- InstructVLA 论文主页与 arXiv（见 README）
- PEFT 文档（LoRA/X-LoRA）
- PyTorch FSDP 官方文档
- LIBERO benchmark 说明

💡 提示：阅读 action_head.py 时优先画出 tensor shape 变化，理解速度更快。

**Checklist**
- [ ] 能清楚解释 LoRA 与 X-LoRA 在项目中的作用差异
- [ ] 能解释动作头输入输出及损失来源
- [ ] 能说明 unnorm_key 与部署动作反归一化的关系

---

### 3. 二次开发指南

#### 3.1 可扩展点与自定义方法

- 扩展点 A：新增 VLA 配置
  - 修改 [conf/vla.py](../conf/vla.py)
  - 新增 dataclass + 注册到 VLARegistry
- 扩展点 B：新增动作头结构
  - 修改 [vla/action_head.py](../vla/action_head.py)
  - 在对应 InstructVLA 变体中替换 action_model
- 扩展点 C：新增数据来源
  - 修改 [mm_dataset/data_utils.py](../mm_dataset/data_utils.py)
  - 在训练脚本拼接到 mm_dataset

#### 3.2 贡献规范与开发流程

~~~text
建议流程：
1) 新建分支
2) 小步提交（配置、模型、脚本分开）
3) 每步附最小复现实验日志
4) 提交 PR 前至少跑通一个训练 smoke test + 一个评测样例
~~~

#### 3.3 调试技巧与测试方法

- 调试技巧
  - 用 --debug True 临时放开 world size 断言
  - 在 deploy 路径打印 task_description 与 action shape
  - 对 stage2 关注 hook 清理是否生效
- 测试建议
  - 单元级：[需补充]（仓库暂无统一 pytest 用例组织）
  - 集成级：最小数据子集跑 100-500 step 验证 loss 与 checkpoint 生成
  - 评测级：固定 seed 跑 1-2 个 LIBERO 任务做回归

#### 3.4 二开示例

示例1：新增一个数据混合配置

~~~python
# 文件：conf/vla.py
# 目标：添加新数据混合项并可通过 --vla.type 选择
from dataclasses import dataclass

@dataclass
class Exp_Custom_Mix(Exp_SigLIP_224px_Bridge):
    vla_id: str = "custom-mix-v1"
    data_mix: str = "custom_mix_no_noops"
    expected_world_size: int = 8
    global_batch_size: int = 128
    per_device_batch_size: int = 16
~~~

示例2：在训练脚本中切换到 state 变体

~~~python
# 文件：scripts/train_eagle_dual_v2_action_only_meta_query_v2.py
# 目标：导入 state 版本模型并传入 proprios
# from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import ...
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2_state import load, load_vla, get_vla_dataset_and_collator
~~~

示例3：扩展部署端动作后处理

~~~python
# 文件：deploy/robot_utils.py
# 目标：在不同环境下切换 gripper 后处理策略
if cfg.model_family == "instruct_vla":
    action = server.get_vla_action(...)
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
~~~

💡 提示：二开优先遵循现有分层，不要把数据预处理逻辑直接塞进模型 forward。

⚠️ 警告：修改 action_dim、窗口长度、state 输入维度时，训练与部署两端必须同步。

**Checklist**
- [ ] 已确定至少一个可扩展点
- [ ] 已准备最小改动 + 最小验证方案
- [ ] 已定义回滚策略（保留可运行 baseline 配置）

---

## 附录：常用命令速查

### A. 导出 Stage1 LoRA 并卸载

参考 [scripts/unload_lora.py](../scripts/unload_lora.py)

~~~bash
python scripts/unload_lora.py
~~~

### B. 语言评测资产准备

参考 [mm_evaluation/README.md](../mm_evaluation/README.md)

~~~text
mm_evaluation/
├── language_evaluation/
│   └── images/
└── language_evaluation_with_gt.json
~~~

### C. 快速定位关键入口

- 训练入口：scripts/train_eagle_dual_v2_action_only_meta_query_v2.py
- LIBERO 评测入口：deploy/libero/run_libero_eval.py
- 模型主体：vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py
- 动作头：vla/action_head.py

**Checklist**
- [ ] 已能通过命令速查定位脚本
- [ ] 已知道 Stage1/Stage2 导出链路
- [ ] 已知道语言评测资产目录结构
