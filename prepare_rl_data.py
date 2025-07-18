import json
import pandas as pd
from rllm.data.dataset import DatasetRegistry, Dataset
import os

def prepare_rl_data():
    jsonl_path = "data/rl_env_data.jsonl"
    parquet_path = "data/rl_env_data.parquet"
    dataset_name = "rl_env_data"
    split_name = "train" # Using 'train' for both for simplicity in this initial run

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    # Read JSONL file
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to Pandas DataFrame and then to Parquet
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {jsonl_path} to {parquet_path}")

    # Register the dataset
    # DatasetRegistry.register_dataset expects a list of dicts, not a path
    # So, we need to load it back or pass the original list of dicts
    registered_dataset = DatasetRegistry.register_dataset(dataset_name, data, split_name)
    print(f"Registered dataset '{dataset_name}' split '{split_name}' with {len(data)} examples.")

    # For validation, register the same data under a 'test' split
    DatasetRegistry.register_dataset(dataset_name, data, "test")
    print(f"Registered dataset '{dataset_name}' split 'test' with {len(data)} examples.")


if __name__ == "__main__":
    prepare_rl_data()