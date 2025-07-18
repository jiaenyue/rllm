import os
import pandas as pd
import json
from modelscope.hub.snapshot_download import snapshot_download

def download_model(model_id, local_dir):
    """
    Downloads a model from ModelScope hub to a local directory.
    """
    print(f"Downloading model '{model_id}' to '{local_dir}'...")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    snapshot_download(model_id=model_id, cache_dir=local_dir)
    print("Model download complete.")

def convert_csv_to_jsonl(csv_path, jsonl_path):
    """
    Converts the official validation CSV to the JSONL format required by our RL environment.
    The official CSV has columns: 'question', 'A', 'B', ..., 'J', 'answer'.
    """
    print(f"Converting '{csv_path}' to '{jsonl_path}'...")
    if not os.path.exists(os.path.dirname(jsonl_path)):
        os.makedirs(os.path.dirname(jsonl_path))
        
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("Please make sure the official validation set CSV is in the 'data/' directory.")
        return

    instruction_text = "你是个优秀的论文分类师"
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # The 'question' column contains the full prompt including the options.
            # We can directly use it as the 'input' for our SFT format.
            input_text = row["question"]
            
            # The 'answer' column contains the correct letter.
            output_text = row["answer"]
            
            record = {
                "instruction": instruction_text,
                "input": input_text,
                "output": output_text
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Conversion complete. Saved {len(df)} records to '{jsonl_path}'.")


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_ID = "jiaenyue/L1G4_internlm2_5-1_8b-chat_0_5"
    MODEL_SAVE_DIR = "models/xtuner_sft_model"
    
    OFFICIAL_VALIDATION_CSV = "data/newformat_sft_test_data.csv"
    RL_ENV_DATA_JSONL = "data/rl_env_data.jsonl"

    # --- Execution ---
    download_model(model_id=MODEL_ID, local_dir=MODEL_SAVE_DIR)
    convert_csv_to_jsonl(csv_path=OFFICIAL_VALIDATION_CSV, jsonl_path=RL_ENV_DATA_JSONL)
    
    print("\nRL assets are ready.")
    print(f" - SFT Model: '{MODEL_SAVE_DIR}'")
    print(f" - RL Environment Data: '{RL_ENV_DATA_JSONL}'")