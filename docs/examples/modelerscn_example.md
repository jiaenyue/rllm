# Modelers.cn Example

This example demonstrates how to use a model from `modelers.cn`.

## Step 1: Run the training script

You can use the `modelerscn::` schema to specify a model from `modelers.cn`. The script will automatically download the model to the local cache.

```bash
python -m rllm.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=modelerscn::my-awesome-model \
    ...
```
