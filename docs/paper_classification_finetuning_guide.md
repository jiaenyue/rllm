# 论文分类模型微调指南

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

## 阶段二：模型微调

此阶段我们将使用上一阶段生成的 SFT 数据集，对 `InternLM-1.8B` 模型进行微调。

### 1. 训练脚本

我们使用专门的训练脚本 `examples/paper_classification/train_paper_classifier.py` 来启动训练任务。该脚本基于 `hydra` 进行配置，允许我们通过命令行灵活地覆盖训练参数。

### 2. 执行训练

运行以下命令来启动模型微调。请注意，您需要根据实际情况修改部分参数：

```bash
python3 examples/paper_classification/train_paper_classifier.py \
    --config-name=sft_trainer_accelerate \
    model.partial_pretrain=/path/to/your/internlm-1.8b/model \
    data.train_files='[data/sft_train.jsonl]' \
    data.val_files='[data/sft_validation.jsonl]' \
    output_dir=outputs/paper_classifier_internlm_1.8b \
    optimizer.lr=5e-5 \
    scheduler.num_warmup_steps=100 \
    train.total_max_steps=2000 \
    train.eval_interval=200 \
    train.save_interval=500 \
    train.batch_size_per_device=4 \
    train.gradient_accumulation_steps=8
```

**关键参数说明**:

- `model.partial_pretrain`: **必须修改**。请将其替换为您的 `InternLM-1.8B` 模型权重所在的**绝对路径**。
- `data.train_files` / `data.val_files`: 指定训练和验证数据集。
- `output_dir`: 指定模型检查点 (checkpoints) 和训练日志的输出目录。
- `optimizer.lr`: 学习率，`5e-5` 是一个常用的初始值。
- `train.total_max_steps`: 总训练步数。您可以根据数据量和期望的训练轮数进行调整。
- `train.batch_size_per_device`: 每个 GPU 的批处理大小。如果遇到显存不足 (OOM) 的问题，请减小此值。
- `train.gradient_accumulation_steps`: 梯度累积步数。可用于在不增加显存占用的情况下，实现等效的更大 batch size (Effective Batch Size = `batch_size_per_device` * `num_gpus` * `gradient_accumulation_steps`)。

---

## 阶段三：评估与提交

训练完成后，您可以在 `output_dir` 中找到保存的模型权重（LoRA 适配器）。

- **评估**: 比赛使用 OpenCompass 进行最终评测。在本地，您可以通过观察训练过程中的 `validation loss` 来初步判断模型性能。
- **提交**: 根据比赛要求，将 `output_dir` 目录中保存的适配器权重（如 `adapter_model.safetensors` 等文件）上传到指定的 Git 仓库。

---

## 阶段四：迭代与优化

为了在排行榜上取得更好的成绩，您可以尝试以下优化策略：

1.  **调整数据**: 修改 `config.json`，使用更大比例 (`processing_percentage`) 的数据进行训练。
2.  **调整超参数**: 尝试不同的学习率 (`lr`)、训练步数 (`total_max_steps`) 和 batch size。
3.  **优化 Prompt**: 在 `config.json` 的 `prompt_templates` 中尝试更有效、更多样化的 Prompt 模板。
4.  **完整数据训练**: 在找到最优超参数后，设置 `generate_full_sft_dataset: true`，并使用 `sft_full.jsonl` 对模型进行最终训练，不划分验证集。

```