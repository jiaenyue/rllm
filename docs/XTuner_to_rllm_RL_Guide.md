# 论文分类模型微调指南 - 从 XTuner SFT 到 rllm RL

本文档旨在为“书生大模型实战营——论文分类打榜赛”提供一份详尽的、端到端的模型微调指南。我们将使用 `rllm` 框架，基于 `InternLM-1.8B` 模型，完成从数据准备到模型训练和优化的全过程。

---

## 阶段一：数据准备

此阶段的目标是将原始的 arXiv 论文数据转换为模型微调所需的、包含多样化 Prompt 的 SFT (Supervised Fine-Tuning) 格式。

### 1. 准备工作

- **下载数据**: 从比赛官方渠道下载 `arxiv-metadata-oai-snapshot.json` 数据集。
- **放置数据**: 将下载的文件放置在项目根目录下的 `data/` 文件夹中。最终路径应为：`data/arxiv-metadata-oai-snapshot.json`。

### 2. 配置文件 (`config.json`)

数据处理的行为由 `examples/paper_classification/config.json` 文件控制。您可以根据需求调整其中的参数：

- `input_filepath`: 原始数据文件路径 (默认已设置为 `data/arxiv-metadata-oai-snapshot.json`)。
- `sft_train_filepath` / `sft_validation_filepath`: 处理后输出的训练集和验证集路径。
- `processing_percentage`: 希望处理的原始数据百分比。在开发阶段，可以将其设置为较小的值（如 `0.1`，即 10%）以加快处理速度。
- `val_split_ratio`: 从处理后的数据中划分出多少作为验证集。
- `prompt_templates`: 用于生成 SFT 数据的 Prompt 模板列表。测试集包含20种不同的模板，因此使用多样化的模板进行训练有助于提升模型的泛化能力。

### 3. 执行数据处理

运行以下命令来启动数据处理脚本：

```bash
python3 examples/paper_classification/process_arxiv_data.py
```

脚本执行完毕后，您将在 `data/` 目录下看到新生成的 `sft_train.jsonl` 和 `sft_validation.jsonl` 文件，它们将用于下一阶段的模型训练。

---

## 阶段二：SFT 微调 (使用 XTuner)

根据您的策略，我们推荐使用比赛官方 Baseline 提供的专业 SFT 框架（如 XTuner）进行第一阶段的微调，以获得一个高质量的初始模型。

1.  **遵循 XTuner 指南**: 请参考比赛官方提供的 XTuner 微调 Baseline 文档，使用我们生成的 `sft_train.jsonl` 和 `sft_validation.jsonl` 作为训练数据。
2.  **获取模型产物**: 训练完成后，XTuner 会生成一个包含完整模型权重和配置文件的目录。请记下这个目录的路径，例如：`/path/to/your/xtuner_sft_model`。

---

## 阶段三：RL 进阶训练 (使用 rllm)

此阶段的目标是�� XTuner 训练出的优秀 SFT 模型，作为 `rllm` 框架中强化学习（RL）训练的起点，以求获得超越单纯 SFT 的性能上限。

### 1. 准备 RL 组件

为了进行 RL 训练，我们需要为 `rllm` 框架创建相应的环境和奖励函数。这些脚本已经为您准备好。

- **RL 环境**: `rllm/environments/paper_classification_env.py`
- **奖励函数**: `rllm/rewards/paper_classification_reward.py`

### 2. 创建 RL 训练配置文件

在 `rllm/trainer/config/` 目录下创建一个新的配置文件 `rl_paper_classifier.yaml`，内容如下：

```yaml
# rllm/trainer/config/rl_paper_classifier.yaml

defaults:
  - rl_trainer_accelerate # 继承基础的 RL 训练器配置

# 1. 指定我们的 XTuner SFT 模型路径 (稍后会通过命令行覆盖)
model:
  partial_pretrain: /path/to/your/xtuner_sft_model

# 2. 配置 RL 训练器参数
train:
  total_max_steps: 1000 # RL 训练的总步数
  # 根据需要调整其他 RL 参数，如 PPO 的 epochs, clip_eps 等

# 3. 配置环境
env:
  cls: rllm.environments.paper_classification_env.PaperClassificationEnv
  config:
    # 使用 SFT 的验证集作为 RL 训练的环境数据源
    dataset_path: data/sft_validation.jsonl 

# 4. 配置奖励函数
reward:
  cls: rllm.rewards.paper_classification_reward.PaperClassificationReward

# 根据需要配置其他参数，如 tokenizer, optimizer 等
```

### 3. 启动 RL 训练

使用 `rllm` 的主 RL 训练脚本 `agent_trainer.py` 来启动训练。

```bash
python3 rllm/trainer/agent_trainer.py \
    --config-path ./rllm/trainer/config \
    --config-name=rl_paper_classifier \
    model.partial_pretrain=/path/to/your/xtuner_sft_model \
    output_dir=outputs/paper_classifier_xtuner_rl
```

**关键命令解释**:

- `--config-path`: 指定 `hydra` 配置文件的搜索路径。
- `--config-name`: 加载我们刚刚创建的 `rl_paper_classifier` 配置文件。
- `model.partial_pretrain`: **这是最关键的一步**。我们通过命令行覆盖此参数，指向您在 XTuner 中训练产出的**完整模型目录**。`rllm` 将从此路径加载模型和分词器，作为 Agent 的初始状态。
- `output_dir`: RL 训练产物（新的、更强的模型权重）的输出目录。

### 4. 评估与提交

训练完成后，最终的模型权重将保存在 `outputs/paper_classifier_xtuner_rl` 目录中。您可以按照比赛要求，提交这套经过 RL 进一步优化的模型权重。
