# rllm/rewards/paper_classification_reward.py

from rllm.rewards.reward_fn import RewardFn
from rllm.types import Trajectory

class PaperClassificationReward(RewardFn):
    """
    A simple reward function that extracts the reward value directly 
    from the info dictionary of the last step in a trajectory.
    
    The actual reward logic is handled within the PaperClassificationEnv.
    """
    def __call__(self, trajectory: Trajectory) -> Trajectory:
        if not trajectory.steps:
            trajectory.reward = 0.0
            return trajectory

        # The reward is pre-calculated by the environment and stored in the info dict
        last_step_info = trajectory.steps[-1].info
        reward = last_step_info.get("reward", 0.0)
        
        trajectory.reward = float(reward)
        return trajectory
