# rllm/environments/paper_classification_env.py

import json
import random
from typing import Any, Dict, List, Tuple

from rllm.environments.base.env import Env
from rllm.types import Observation, Action

class PaperClassificationEnv(Env):
    """
    A simple paper classification environment for RL training in rllm.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dataset_path = config.get("dataset_path", "data/sft_validation.jsonl")
        self.max_steps = config.get("max_steps", 1) # Each episode is a single step
        self.dataset = self._load_dataset()
        self.current_step = 0
        self.current_sample = None

    def _load_dataset(self) -> List[Dict[str, str]]:
        """Loads the SFT dataset to be used as the data source for the environment."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def reset(self) -> Observation:
        """
        Resets the environment by randomly selecting a new data sample.
        """
        self.current_step = 0
        self.current_sample = random.choice(self.dataset)
        # The observation is the input prompt from the SFT data
        return {"observation": self.current_sample["input"]}

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Takes a step. The agent's action is its predicted classification label.
        """
        if self.current_step >= self.max_steps:
            raise ValueError("Episode has already ended.")

        self.current_step += 1
        
        predicted_label = str(action.get("action", "")).strip().upper()
        true_label = self.current_sample["output"].strip().upper()

        # Define the reward
        reward = 1.0 if predicted_label == true_label else -1.0
        
        done = self.current_step >= self.max_steps
        
        # The next observation can be empty as the episode is over
        next_observation = {"observation": ""}
        
        info = {
            "predicted_label": predicted_label,
            "true_label": true_label,
            "is_correct": (reward > 0),
            "reward": reward # Pass the reward in the info dict
        }
        
        return next_observation, reward, done, info
