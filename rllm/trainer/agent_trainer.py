from typing import Any, Optional
import ray
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, ResourcePoolManager
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, AsyncActorRolloutRefWorker
from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from pprint import pprint

from rllm.data import Dataset


class AgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.
    """

    def __init__(
        self,
        agent_class: type,
        env_class: type,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
    ):
        """
        Initialize the AgentTrainer.

        Args:
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
            train_dataset: Optional train dataset to use
            val_dataset: Optional validation dataset to use
            agent_args: Optional arguments to pass to the agent class
            env_args: Optional arguments to pass to the environment class
        """
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.config = config

        # Load tokenizer
        model_path = self.config.actor_rollout_ref.model.path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=self.config.actor_rollout_ref.model.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'function_call' %}{{ '<|im_start|>tool_code\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'tool_code' %}{{ '<|im_start|>tool_code\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'tool_code_result' %}{{ '<|im_start|>tool_code_result\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% else %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% if loop.last %}{{ '<|im_end|>' }}{% else %}{{ '\n' }}{% endif %}{% endfor %}"

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):
        if not ray.is_initialized():
            ray.init(
                runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN",
                 "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
                num_cpus=self.config.ray_init.num_cpus,
            )

        # print initial config
        pprint(OmegaConf.to_container(self.config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(self.config)

        # download the checkpoint from hdfs (if applicable)
        local_path = copy_to_local(self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get('use_shm', False))

        # instantiate tokenizer and processor
        processor = hf_processor(local_path, trust_remote_code=self.config.actor_rollout_ref.model.trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if self.config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            actor_rollout_cls = AsyncActorRolloutRefWorker if self.config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif self.config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            # from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker # Already imported at top

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if self.config.reward_model.enable:
            if self.config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif self.config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if self.config.algorithm.use_kl_in_reward or self.config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(self.config, self.tokenizer, num_examine=0, **self.config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(self.config, self.tokenizer, num_examine=1, **self.config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Use the datasets passed to AgentTrainer's __init__
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        train_sampler = None # RayPPOTrainer will create default sampler if needed

        trainer = RayPPOTrainer(
            config=self.config,
            tokenizer=self.tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=self.config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()
