import hydra
from omegaconf import DictConfig
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.data.dataset import Dataset # Import Dataset class
from rllm.agents import ToolAgent # Using ToolAgent as a general purpose agent
from rllm.environments import ToolEnvironment # Using ToolEnvironment as a general purpose environment

@hydra.main(config_path="rllm/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    # Load the RL environment data
    train_dataset = Dataset.load_data("data/rl_env_data.jsonl")
    val_dataset = Dataset.load_data("data/rl_env_data.jsonl") # Using the same data for validation for now

    # Initialize AgentTrainer
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()