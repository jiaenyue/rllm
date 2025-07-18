# RL训练环境安装指南

## 系统要求
- Linux系统 (推荐Ubuntu 20.04+)
- Python 3.8-3.10
- CUDA 11.7+ (如需GPU支持)
- NVIDIA驱动 >= 515.65.01 (如需GPU支持)

## 基础安装

1. 克隆仓库：
```bash
git clone https://github.com/your-repo/rllm.git
cd rllm
```

2. 创建conda环境：
```bash
conda create -n rllm python=3.10
conda activate rllm
```

## 依赖安装

### 核心依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Flash Attention替代方案
由于硬件限制无法安装flash-attention，我们使用xformers作为替代：

```bash
pip install xformers
```

## 配置修改

已对以下文件进行必要修改：
1. `rllm/trainer/verl/verl/workers/fsdp_workers.py`:
   - 将`flash_attn.bert_padding`替换为`xformers.ops.fmha.bert_padding`
   - 移除NPU相关代码
   - 将`attn_implementation="flash_attention_2"`改为`attn_implementation="eager"`

## 环境验证

运行测试脚本验证安装：
```bash
python tests/sanity_check.py
```

## 常见问题

### 1. CUDA版本不匹配
解决方案：
```bash
conda install cuda -c nvidia/label/cuda-11.7.0
```

### 2. xformers安装失败
尝试从源码编译：
```bash
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -e .
```

### 3. 内存不足
降低训练batch size：
```yaml
# 修改ppo_trainer.yaml
data:
  train_batch_size: 16  # 默认1024
```

## 环境变量配置

```bash
# 设置PyTorch使用CUDA
export CUDA_VISIBLE_DEVICES=0  # 指定GPU设备

# 启用xformers优化
export XFORMERS_FORCE_DISABLE_TRITON=0
```

## Docker支持

```dockerfile
FROM nvidia/cuda:11.7.1-base

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git

# 克隆项目
RUN git clone https://github.com/your-repo/rllm.git /app
WORKDIR /app

# 安装依赖
RUN pip install -r requirements.txt
RUN pip install xformers

# 设置环境变量
ENV PYTHONPATH=/app
```

## 性能调优

1. 启用混合精度训练：
```yaml
# ppo_trainer.yaml
actor_rollout_ref:
  model:
    dtype: bfloat16
```

2. 优化数据加载：
```python
# 设置num_workers为CPU核心数的70%
DataLoader(..., num_workers=multiprocessing.cpu_count()*0.7)
```

## 开发建议

1. 使用VS Code远程开发
2. 推荐扩展：
   - Python
   - Pylance
   - Docker
3. 调试配置参考`.vscode/launch.json`
