# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import inspect
import json
import os
import textwrap
import time
import warnings
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any
import torch.nn.functional as F

import datasets
import pandas as pd
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.callbacks import SyncRefModelCallback
from custom_coex_config import CoExConfig
from rewards.diversity import trace_jaccard_diversity_reward
from rewards.diversity import compute_one_minus_bleu_rewards_for_adapter, trace_jaccard_diversity_reward
from rewards.main_weak_correctness import (
    align_main_correct_rate_to_local_rows,
    compute_group_coverage_stats,
    compute_main_correct_rate_by_prompt,
    compute_main_weak_correctness_bonus,
)
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

from transformers.training_args import OptimizerNames
from accelerate.utils import (
    DistributedType,
)
from transformers.utils import (
    is_datasets_available,
    is_liger_kernel_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
    
import math
import torch
from accelerate.utils import DistributedType

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model_state_dict

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    import tempfile
    from vllm.lora.request import LoRARequest

if is_wandb_available():
    import wandb

if is_trackio_available():
    import trackio


logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]

# What we call a rollout function is a callable that takes prompts (list), args (CoExConfig), and processing_class as
# parameters and returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and
# "logprobs" fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], Any, Any], dict[str, Any]]


class LoRAIntegrityLoggingCallback(TrainerCallback):
    """Observe LoRA optimizer boundaries without changing the training path."""

    def __init__(self, trainer):
        self._trainer_ref = weakref.ref(trainer)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self._trainer_ref()
        if trainer is not None:
            trainer._log_lora_optimizer_boundary("pre_optimizer_step")
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        trainer = self._trainer_ref()
        if trainer is not None:
            trainer._log_lora_optimizer_boundary("post_optimizer_step")
        return control


class CoExTrainer(BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (CoEx) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import CoExTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = CoExTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`CoExConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It must take prompts, args, and processing_class as parameters
            and return a dict with `"prompt_ids"`, `"completion_ids"`, and `"logprobs"` fields. Any other fields that
            are forwarded to the reward functions. This feature is experimental and may change or be removed at any
            time without prior notice.
    """

    _tag_names = ["trl", "coex"]
    _name = "CoEx"
    _POLICY_REPULSION_REWARD_TYPES = {
        "policy_repulsion_margin",
        "policy_repulsion_margin_barrier",
    }
    _TRACE_JACCARD_REWARD_TYPES = {"trace_jaccard", "trace_jaccard3"}
    _ONE_MINUS_BLEU_REWARD_TYPES = {"one_minus_bleu", "one_minus_bleu_score", "1-bleu"}
    _MAIN_WEAK_CORRECTNESS_REWARD_TYPES = {"main_weak_correctness_bonus"}
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs_correctness: RewardFunc | list[RewardFunc],
        reward_funcs_diversity: RewardFunc | list[RewardFunc],
        args: CoExConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        diversity_reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
        completion_store_path: str = None,
        trainer_output: str = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = CoExConfig(f"{model_name}-CoEx")
            
        # for proposed
        self.all_adapter_names = ["default"]
        self.all_adapter_num_completions = [args.num_completion_main_adapter]
        
        if args.num_diversity_adapters > 0 and args.num_completion_per_diversity_adapter > 0:
            self.all_adapter_names += [f"diversity_{i}" for i in range(args.num_diversity_adapters)]
            self.all_adapter_num_completions += [args.num_completion_per_diversity_adapter] * args.num_diversity_adapters
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `CoExConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = get_config_model_id(model.config)
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `CoExConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # For Sleep mode in vLLM
        self._vllm_slept = False
        self._vllm_lora_hash_results = {}
        self._source_trace_jsonl_path = None
        self._source_trace_write_count = 0

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config), truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        reward_funcs = reward_funcs_correctness
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes
        
        diversity_reward_funcs = reward_funcs_diversity
        # Reward functions
        if not isinstance(diversity_reward_funcs, list):
            diversity_reward_funcs = [diversity_reward_funcs]
        self.diversity_reward_func_names = []
        for i, reward_func in enumerate(diversity_reward_funcs):
            if isinstance(reward_func, str):
                diversity_reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(diversity_reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.diversity_reward_func_names.append(get_config_model_id(diversity_reward_funcs[i].config).split("/")[-1])
            else:
                self.diversity_reward_func_names.append(diversity_reward_funcs[i].__name__)
        self.diversity_reward_funcs = diversity_reward_funcs
        
        # Diversity reward weights
        if args.diversity_reward_weights is not None:
            if len(args.diversity_reward_weights) != len(diversity_reward_funcs):
                raise ValueError(
                    f"Number of diversity reward weights ({len(args.diversity_reward_weights)}) must match number of "
                    f"diversity reward functions ({len(diversity_reward_funcs)})"
                )
            self.diversity_reward_weights = torch.tensor(args.diversity_reward_weights, dtype=torch.float32)
        else:
            self.diversity_reward_weights = torch.ones(len(diversity_reward_funcs), dtype=torch.float32)
            
        # Diversity reward processing class
        if diversity_reward_processing_classes is None:
            diversity_reward_processing_classes = [None] * len(diversity_reward_funcs)
        elif not isinstance(diversity_reward_processing_classes, list):
            diversity_reward_processing_classes = [diversity_reward_processing_classes]
        if len(diversity_reward_processing_classes) != len(diversity_reward_funcs):
            raise ValueError(
                f"The number of diversity reward processing classes ({len(diversity_reward_processing_classes)}) must match the number of "
                f"diversity reward functions ({len(diversity_reward_funcs)})."
            )
        
        for i, (diversity_reward_processing_class, reward_func) in enumerate(
            zip(diversity_reward_processing_classes, diversity_reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if diversity_reward_processing_class is None:
                    diversity_reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if diversity_reward_processing_class.pad_token_id is None:
                    diversity_reward_processing_class.pad_token = diversity_reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = diversity_reward_processing_class.pad_token_id
                diversity_reward_processing_classes[i] = diversity_reward_processing_class
        self.diversity_reward_processing_classes = diversity_reward_processing_classes
        
        

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are importing from 'rollout_func', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the CoEx paper
        self.num_generations = args.num_generations  # = G in the CoEx paper
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_kernel = args.use_liger_kernel
        self.loss_type = args.loss_type
        self.dmpo_base_loss_type = args.dmpo_base_loss_type
        self.dmpo_beta = args.dmpo_beta
        self.dmpo_temperature = args.dmpo_temperature
        self.dmpo_skip_zero_advantage_groups = args.dmpo_skip_zero_advantage_groups
        self.dmpo_candidate_scope = args.dmpo_candidate_scope
        self.dmpo_log_metrics = args.dmpo_log_metrics
        self.dmpo_sanity_check = args.dmpo_sanity_check
        self._dmpo_sanity_printed = False
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.correctness_gated = args.correctness_gated
        self.correctness_threshold = args.correctness_threshold

        self.diversity_reward_type = args.diversity_reward_type
        self.policy_repulsion_target = args.policy_repulsion_target
        self.diversity_comparison_scope = args.diversity_comparison_scope
        self.policy_repulsion_batch_size = args.policy_repulsion_batch_size
        self.policy_repulsion_gate_by_correctness = args.policy_repulsion_gate_by_correctness
        self.policy_repulsion_gate_threshold = args.policy_repulsion_gate_threshold

        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_kernel and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in CoExTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the CoEx paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None
        self._last_loaded_step_per_adapter = {}
        self._lora_switch_count = 0
        self._lora_pre_optimizer_state = None
        self._lora_integrity_callback_added = False


        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in CoEx, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in CoEx
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )
        self.enable_all_lora_grads(self.model)
        if hasattr(self.model, "_set_static_graph"):
            self.model._set_static_graph()

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(cast_outputs_to_original_dtype)

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Liger loss
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the CoEx loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_coex_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # # Keep logs sized to the generation batch to record only outputs from the latest model update.
        # self._logs = {
        #     "images": deque(maxlen=args.generation_batch_size),
        #     "prompt": deque(maxlen=args.generation_batch_size),
        #     "completion": deque(maxlen=args.generation_batch_size),
        #     "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
        #     "advantages": deque(maxlen=args.generation_batch_size),
        # }
        self._logs = {}

        for adapter_name in self.all_adapter_names:
            self._logs[f"{adapter_name}_images"] = deque(maxlen=args.generation_batch_size)
            self._logs[f"{adapter_name}/prompt"] = deque(maxlen=args.generation_batch_size)
            self._logs[f"{adapter_name}/correctness_completion"] = deque(maxlen=args.generation_batch_size)
            self._logs[f"{adapter_name}/correctness_rewards"] = defaultdict(lambda: deque(maxlen=args.generation_batch_size))
            self._logs[f"{adapter_name}/correctness_advantages"] = deque(maxlen=args.generation_batch_size)
            
            if adapter_name != "default":
                self._logs[f"{adapter_name}/diversity_completion"] = deque(maxlen=args.generation_batch_size)
                self._logs[f"{adapter_name}/diversity_rewards"] = defaultdict(lambda: deque(maxlen=args.generation_batch_size))
                self._logs[f"{adapter_name}/diversity_advantages"] = deque(maxlen=args.generation_batch_size)


        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if completion_store_path is not None:
            self.completion_store_path = completion_store_path
        else:
            self.completion_store_path = f"completion_storage/{self.args.run_name}_outputs"
        os.makedirs(self.completion_store_path, exist_ok=True)

        if trainer_output is not None:
            self.trainer_output = trainer_output
        else:
            self.trainer_output = f"trainer_output/{self.args.run_name}"
        os.makedirs(self.trainer_output, exist_ok=True)

        self.args.output_dir = self.trainer_output

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None

                prompt_batch = self.args.generation_batch_size // self.num_generations 
                max_seqs_needed = prompt_batch * max(self.all_adapter_num_completions) 
                
                # breakpoint()
                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    # max_num_seqs=self.args.per_device_train_batch_size
                    max_num_seqs = max_seqs_needed 
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                    enable_lora=True if is_peft_model(model) else False, # for LoRA models,
                    max_lora_rank=args.lora_r if hasattr(args, 'lora_r') else 64,
                )
                # if self.args.vllm_enable_sleep_mode:
                #     self.llm.sleep(level=2)
                    
                import tempfile
                self.lora_temp_dir = tempfile.mkdtemp(prefix="vllm_lora_cache_")
                
                self.lora_modules = None
                if is_peft_model(model):
                    self.lora_modules = []
                    for adapter_index, adapter_name in enumerate(self.all_adapter_names):
                        adapter_export_dir = os.path.join(self.lora_temp_dir, f"{adapter_name}_adapter")
                        adapter_info = {
                            "name": adapter_name,
                            "export_dir": adapter_export_dir,
                            "path": self._vllm_adapter_lora_path(adapter_export_dir, adapter_name),
                            "id": adapter_index + 1,
                        }
                        self._save_adapter_for_vllm(
                            model,
                            adapter_info,
                            phase="vllm_init_export",
                            optimizer_step=0,
                        )
                        self.lora_modules.append(adapter_info)
                    print(f"  >> [vLLM Init] LoRA modules saved to temporary cache directory: {self.lora_temp_dir} | Total adapters: {len(self.lora_modules)}")
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )
        
        for i, reward_func in enumerate(self.diversity_reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.diversity_reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.diversity_reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )
                    
        self.add_callback(LoRAIntegrityLoggingCallback(self))
        self._lora_integrity_callback_added = True
        self._log_lora_integrity_snapshot("trainer_init", optimizer_step=0)

    def enable_all_lora_grads(self, model):
        """
        Force-enable gradients for all LoRA adapter layers in a PEFT model.
        This overrides PEFT's set_adapter() behavior that disables inactive adapters.
        """
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def _assert_all_lora_in_optimizer(self):
        if self.optimizer is None:
            return

        lora_params = []
        for name, p in self.model.named_parameters():
            if p is not None and "lora_" in name:
                lora_params.append(p)

        opt_params = set()
        for g in self.optimizer.param_groups:
            for p in g["params"]:
                opt_params.add(id(p))

        missing = [p for p in lora_params if id(p) not in opt_params]
        if len(missing) > 0:
            raise RuntimeError(
                f"[Optimizer] Missing {len(missing)}/{len(lora_params)} LoRA params in optimizer param_groups. "
                "Some adapters will never learn."
            )

    def create_optimizer(self):
        self.enable_all_lora_grads(self.model)
        super().create_optimizer()
        self._assert_all_lora_in_optimizer()
        self._log_lora_integrity_snapshot("after_optimizer_create", optimizer_step=0)

    import math
    
    def _match_adapter_param(self, name: str, adapter_name: str) -> bool:
        if adapter_name == "default":
            return (".default." in name) or ("_default_" in name) or (".default_" in name) or ("_default." in name)

        tag1 = f".{adapter_name}."
        tag2 = f"_{adapter_name}_"
        tag3 = f".{adapter_name}_"
        tag4 = f"_{adapter_name}."
        return (tag1 in name) or (tag2 in name) or (tag3 in name) or (tag4 in name)

    def _is_main_lora_logging_process(self) -> bool:
        accelerator = getattr(self, "accelerator", None)
        return accelerator is None or accelerator.is_main_process

    @staticmethod
    def _logging_active_adapter(model) -> str:
        active = getattr(model, "active_adapter", None)
        if active is None:
            active = getattr(model, "active_adapters", None)
        if isinstance(active, (list, tuple)):
            return ",".join(str(name) for name in active)
        return str(active)

    def _set_adapter_with_logging(self, model, adapter_name, phase: str):
        """Call PEFT set_adapter unchanged, then report the resulting state."""
        before = self._logging_active_adapter(model)
        output = model.set_adapter(adapter_name)
        after = self._logging_active_adapter(model)
        self._lora_switch_count += 1

        if (
            bool(getattr(self.args, "log_adapter_switches", True))
            and self._is_main_lora_logging_process()
        ):
            active_trainable = 0
            all_lora_trainable = 0
            for name, param in model.named_parameters():
                if "lora_" not in name or not param.requires_grad:
                    continue
                all_lora_trainable += param.numel()
                if self._match_adapter_param(name, str(adapter_name)):
                    active_trainable += param.numel()

            print(
                "[LORA_SWITCH] "
                f"seq={self._lora_switch_count} "
                f"global_step={getattr(self.state, 'global_step', 0)} "
                f"micro_step={getattr(self, '_step', 0)} "
                f"phase={phase} before={before} requested={adapter_name} after={after} "
                f"changed={before != after} "
                f"active_trainable_numel={active_trainable} "
                f"all_lora_trainable_numel={all_lora_trainable}",
                flush=True,
            )
        return output

    def _should_log_lora_integrity(self, optimizer_step: int) -> bool:
        interval = int(getattr(self.args, "adapter_sanity_check_steps", 0) or 0)
        if interval <= 0:
            return False
        return optimizer_step in {0, 1} or optimizer_step % interval == 0

    @torch.no_grad()
    def _collect_lora_integrity(
        self, adapter_names: Optional[list[str]] = None
    ) -> dict[str, dict[str, Any]]:
        adapter_names = list(adapter_names or self.all_adapter_names)
        optimizer_ids = set()
        if self.optimizer is not None:
            optimizer_ids = {
                id(param)
                for group in self.optimizer.param_groups
                for param in group["params"]
            }

        stats = {
            adapter: {
                "tensor_count": 0,
                "numel": 0,
                "trainable_numel": 0,
                "optimizer_numel": 0,
                "grad_numel": 0,
                "a_norm_sum": None,
                "b_norm_sum": None,
                "weight_sum": None,
                "weight_norm_sq": None,
                "grad_norm_sq": None,
                "samples": [],
            }
            for adapter in adapter_names
        }

        def add_scalar(container, key, value):
            current = container[key]
            container[key] = value if current is None else current + value

        for name, param in self.model.named_parameters():
            if param is None or "lora_" not in name:
                continue

            adapter_name = next(
                (
                    adapter
                    for adapter in adapter_names
                    if self._match_adapter_param(name, adapter)
                ),
                None,
            )
            if adapter_name is None:
                continue

            stat = stats[adapter_name]
            stat["tensor_count"] += 1
            stat["numel"] += param.numel()
            if param.requires_grad:
                stat["trainable_numel"] += param.numel()
            if id(param) in optimizer_ids:
                stat["optimizer_numel"] += param.numel()

            value = param.detach().float()
            value_norm = torch.linalg.vector_norm(value)
            add_scalar(stat, "weight_sum", value.sum())
            add_scalar(stat, "weight_norm_sq", value_norm * value_norm)
            if "lora_A" in name:
                add_scalar(stat, "a_norm_sum", value_norm)
            elif "lora_B" in name:
                add_scalar(stat, "b_norm_sum", value_norm)

            flat = value.reshape(-1)
            if flat.numel() > 0:
                stride = max(1, flat.numel() // 4)
                stat["samples"].append(flat[::stride][:4])

            if param.grad is not None:
                grad = param.grad.detach().float()
                grad_norm = torch.linalg.vector_norm(grad)
                add_scalar(stat, "grad_norm_sq", grad_norm * grad_norm)
                stat["grad_numel"] += grad.numel()

        result = {}
        for adapter_name, stat in stats.items():
            sample = (
                torch.cat(stat["samples"]).detach().cpu()
                if stat["samples"]
                else torch.empty(0)
            )
            result[adapter_name] = {
                "tensor_count": stat["tensor_count"],
                "numel": stat["numel"],
                "trainable_numel": stat["trainable_numel"],
                "optimizer_numel": stat["optimizer_numel"],
                "grad_numel": stat["grad_numel"],
                "lora_A_norm_sum": (
                    float(stat["a_norm_sum"].item())
                    if stat["a_norm_sum"] is not None
                    else 0.0
                ),
                "lora_B_norm_sum": (
                    float(stat["b_norm_sum"].item())
                    if stat["b_norm_sum"] is not None
                    else 0.0
                ),
                "weight_sum": (
                    float(stat["weight_sum"].item())
                    if stat["weight_sum"] is not None
                    else 0.0
                ),
                "weight_norm": (
                    float(torch.sqrt(stat["weight_norm_sq"]).item())
                    if stat["weight_norm_sq"] is not None
                    else 0.0
                ),
                "grad_norm": (
                    float(torch.sqrt(stat["grad_norm_sq"]).item())
                    if stat["grad_norm_sq"] is not None
                    else 0.0
                ),
                "_sample": sample,
            }
        return result

    def _print_lora_integrity(
        self,
        tag: str,
        optimizer_step: int,
        summary: dict[str, dict[str, Any]],
        baseline: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        active = self._logging_active_adapter(self.model)
        print(
            f"[LORA_INTEGRITY] tag={tag} optimizer_step={optimizer_step} "
            f"global_step={getattr(self.state, 'global_step', 0)} active={active}",
            flush=True,
        )

        for adapter_name in summary:
            stat = summary[adapter_name]
            previous = baseline.get(adapter_name) if baseline is not None else None
            sample_delta = None
            weight_sum_delta = None
            weight_norm_delta = None
            updated = None
            if previous is not None:
                current_sample = stat["_sample"]
                previous_sample = previous["_sample"]
                if current_sample.shape == previous_sample.shape:
                    sample_delta = float(
                        torch.linalg.vector_norm(
                            current_sample.float() - previous_sample.float()
                        ).item()
                    )
                weight_sum_delta = stat["weight_sum"] - previous["weight_sum"]
                weight_norm_delta = stat["weight_norm"] - previous["weight_norm"]
                updated = bool(
                    (sample_delta is not None and sample_delta > 0.0)
                    or abs(weight_sum_delta) > 0.0
                    or abs(weight_norm_delta) > 0.0
                )

            print(
                "  [LORA_ADAPTER] "
                f"name={adapter_name} tensors={stat['tensor_count']} "
                f"numel={stat['numel']} trainable_numel={stat['trainable_numel']} "
                f"optimizer_numel={stat['optimizer_numel']} "
                f"grad_numel={stat['grad_numel']} "
                f"A_norm_sum={stat['lora_A_norm_sum']:.8e} "
                f"B_norm_sum={stat['lora_B_norm_sum']:.8e} "
                f"weight_norm={stat['weight_norm']:.8e} "
                f"grad_norm={stat['grad_norm']:.8e} "
                f"sample_delta_l2={sample_delta if sample_delta is not None else 'NA'} "
                f"weight_sum_delta={weight_sum_delta if weight_sum_delta is not None else 'NA'} "
                f"weight_norm_delta={weight_norm_delta if weight_norm_delta is not None else 'NA'} "
                f"updated={updated if updated is not None else 'NA'}",
                flush=True,
            )

    def _log_lora_integrity_snapshot(
        self,
        tag: str,
        optimizer_step: int,
        baseline: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        if (
            not self._is_main_lora_logging_process()
            or not self._should_log_lora_integrity(optimizer_step)
        ):
            return None
        summary = self._collect_lora_integrity()
        self._print_lora_integrity(tag, optimizer_step, summary, baseline)
        return summary

    def _log_lora_optimizer_boundary(self, tag: str) -> None:
        optimizer_step = int(getattr(self.state, "global_step", 0)) + 1
        if (
            not self._is_main_lora_logging_process()
            or not self._should_log_lora_integrity(optimizer_step)
        ):
            return

        summary = self._collect_lora_integrity()
        if tag == "pre_optimizer_step":
            self._lora_pre_optimizer_state = summary
            self._print_lora_integrity(tag, optimizer_step, summary)
        elif tag == "post_optimizer_step":
            self._print_lora_integrity(
                tag,
                optimizer_step,
                summary,
                baseline=self._lora_pre_optimizer_state,
            )
            self._lora_pre_optimizer_state = None



    def _vllm_adapter_lora_path(self, export_dir: str, adapter_name: str) -> str:
        """Return the directory that vLLM should load as the LoRA root."""
        if adapter_name == "default":
            return export_dir
        return os.path.join(export_dir, adapter_name)

    def _get_lora_module_info(self, adapter_name: str) -> dict[str, Any] | None:
        for module in self.lora_modules or []:
            if module.get("name") == adapter_name:
                return module
        return None

    def _should_check_vllm_lora_hash(self, optimizer_step: int | None = None) -> bool:
        if not bool(getattr(self.args, "vllm_lora_hash_check", False)):
            return False
        interval = int(getattr(self.args, "vllm_lora_hash_check_interval", 1) or 1)
        step = int(optimizer_step if optimizer_step is not None else getattr(self.state, "global_step", 0))
        return step in {0, 1} or step % interval == 0

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def _tensor_manifest_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        entries = {}
        for key in sorted(state_dict):
            tensor = state_dict[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            entries[key] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": self._tensor_sha256(tensor),
            }
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {
            "manifest": hashlib.sha256(payload).hexdigest(),
            "num_tensors": len(entries),
            "entries": entries,
        }

    @torch.no_grad()
    def _training_lora_manifest(self, model: nn.Module, adapter_name: str) -> dict[str, Any]:
        state = get_peft_model_state_dict(model, adapter_name=adapter_name)
        return self._tensor_manifest_from_state_dict(state)

    def _export_lora_manifest(self, lora_path: str) -> dict[str, Any]:
        from safetensors.torch import load_file

        safetensors_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"Expected exported LoRA safetensors at {safetensors_path}. "
                "vLLM lora_local_path must point at a directory containing adapter_model.safetensors."
            )
        state = load_file(safetensors_path, device="cpu")
        manifest = self._tensor_manifest_from_state_dict(state)
        manifest["safetensors_path"] = safetensors_path
        return manifest

    def _compare_lora_manifests(
        self,
        training_manifest: dict[str, Any],
        export_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        train_entries = training_manifest["entries"]
        export_entries = export_manifest["entries"]
        train_keys = set(train_entries)
        export_keys = set(export_entries)
        missing = sorted(train_keys - export_keys)
        unexpected = sorted(export_keys - train_keys)
        mismatched = sorted(
            key
            for key in train_keys & export_keys
            if train_entries[key] != export_entries[key]
        )
        mismatch_count = len(missing) + len(unexpected) + len(mismatched)
        return {
            "mismatch_count": mismatch_count,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "mismatched_keys": mismatched,
            "status": "PASS" if mismatch_count == 0 else "FAIL",
        }

    def _check_vllm_lora_export_hash(
        self,
        model: nn.Module,
        adapter_name: str,
        lora_path: str,
        optimizer_step: int | None = None,
    ) -> dict[str, Any] | None:
        if not self._should_check_vllm_lora_hash(optimizer_step):
            return None
        if not self._is_main_lora_logging_process():
            return None

        training_manifest = self._training_lora_manifest(model, adapter_name)
        export_manifest = self._export_lora_manifest(lora_path)
        comparison = self._compare_lora_manifests(training_manifest, export_manifest)
        total = max(training_manifest["num_tensors"], export_manifest["num_tensors"])
        result = {
            "adapter": adapter_name,
            "requested_path": lora_path,
            "training_manifest": training_manifest["manifest"],
            "export_manifest": export_manifest["manifest"],
            "training_num_tensors": training_manifest["num_tensors"],
            "export_num_tensors": export_manifest["num_tensors"],
            **comparison,
        }
        self._vllm_lora_hash_results[adapter_name] = result

        tag = "VLLM_LORA_HASH" if comparison["status"] == "PASS" else "VLLM_LORA_HASH_ERROR"
        print(
            f"[{tag}] adapter={adapter_name} requested_path={lora_path} "
            f"training_manifest={training_manifest['manifest']} "
            f"export_manifest={export_manifest['manifest']} "
            f"mismatch={comparison['mismatch_count']}/{total} status={comparison['status']}",
            flush=True,
        )
        if comparison["status"] != "PASS":
            print(
                f"[{tag}_DETAIL] adapter={adapter_name} "
                f"missing_first={comparison['missing_keys'][:5]} "
                f"unexpected_first={comparison['unexpected_keys'][:5]} "
                f"mismatched_first={comparison['mismatched_keys'][:5]}",
                flush=True,
            )
            if bool(getattr(self.args, "vllm_lora_hash_check_strict", True)):
                raise RuntimeError(
                    f"vLLM LoRA export hash check failed for adapter={adapter_name}: "
                    f"{comparison['mismatch_count']}/{total} mismatched at {lora_path}"
                )
        return result

    def _save_adapter_for_vllm(
        self,
        model: nn.Module,
        adapter_info: dict[str, Any],
        phase: str,
        optimizer_step: int | None = None,
    ) -> dict[str, Any] | None:
        adapter_name = adapter_info["name"]
        export_dir = adapter_info.get("export_dir") or adapter_info.get("path")
        if export_dir is None:
            export_dir = os.path.join(self.lora_temp_dir, f"{adapter_name}_adapter")
        os.makedirs(export_dir, exist_ok=True)

        self._set_adapter_with_logging(model, adapter_name, phase)
        self.enable_all_lora_grads(model)
        model.save_pretrained(
            export_dir,
            save_adapter=True,
            save_config=True,
            safe_serialization=True,
            selected_adapters=[adapter_name],
        )
        lora_path = self._vllm_adapter_lora_path(export_dir, adapter_name)
        adapter_info["export_dir"] = export_dir
        adapter_info["path"] = lora_path
        hash_result = self._check_vllm_lora_export_hash(
            model,
            adapter_name,
            lora_path,
            optimizer_step=optimizer_step,
        )
        if hash_result is not None:
            adapter_info["training_manifest"] = hash_result["training_manifest"]
            adapter_info["export_manifest"] = hash_result["export_manifest"]
            adapter_info["adapter_hash_match"] = hash_result["status"] == "PASS"
        return hash_result



    def _source_trace_enabled(self) -> bool:
        return bool(getattr(self.args, "source_owned_trace", False))

    def _should_log_source_trace(self) -> bool:
        if not self._source_trace_enabled() or not self._is_main_lora_logging_process():
            return False
        interval = int(getattr(self.args, "source_owned_trace_log_steps", 1) or 1)
        step = int(getattr(self.state, "global_step", 0)) + 1
        return step in {0, 1} or step % interval == 0

    @staticmethod
    def _json_safe_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        try:
            if isinstance(value, (float, int, str, bool)):
                return value
            if hasattr(value, "item"):
                return value.item()
        except Exception:
            pass
        return value

    @staticmethod
    def _mean_float(values) -> float | None:
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            if values.numel() == 0:
                return None
            return float(values.detach().float().mean().cpu().item())
        if len(values) == 0:
            return None
        return float(sum(float(v) for v in values) / len(values))

    def _problem_id_from_input(self, example: dict[str, Any]) -> Any:
        for key in ("problem_id", "id", "idx", "sample_id", "question_id"):
            if key in example:
                return self._json_safe_scalar(example.get(key))
        return None

    def _build_source_trace_metadata(
        self,
        input_example: dict[str, Any],
        source_adapter_name: str,
        source_adapter_idx: int,
        original_index: int,
        completion_index_within_adapter: int,
        completion_ids: list[int],
        sampling_logps,
    ) -> dict[str, Any]:
        adapter_info = self._get_lora_module_info(source_adapter_name) or {}
        hash_result = self._vllm_lora_hash_results.get(source_adapter_name, {})
        adapter_hash_match = hash_result.get("status") == "PASS"
        backend = "vllm" if self.use_vllm else ("transformers_paged" if self.use_transformers_paged else "transformers")
        prompt_index = int(original_index // max(1, self.num_generations))
        sample_id = (
            f"step{int(getattr(self.state, 'global_step', 0))}::prompt{prompt_index}::"
            f"{source_adapter_name}::sample{completion_index_within_adapter}"
        )
        return {
            "sample_id": sample_id,
            "global_step": int(getattr(self.state, "global_step", 0)),
            "prompt_index": prompt_index,
            "problem_id": self._problem_id_from_input(input_example),
            "source_adapter_name": source_adapter_name,
            "source_adapter_idx": int(source_adapter_idx),
            "completion_index_within_adapter": int(completion_index_within_adapter),
            "original_flat_index": int(original_index),
            "is_main": source_adapter_name == "default",
            "is_diversity": source_adapter_name != "default",
            "generation_backend": backend,
            "requested_source_adapter": source_adapter_name,
            "vllm_lora_name": adapter_info.get("name") if self.use_vllm else None,
            "vllm_lora_int_id": adapter_info.get("id") if self.use_vllm else None,
            "vllm_lora_local_path": adapter_info.get("path") if self.use_vllm else None,
            "expected_adapter_manifest": hash_result.get("training_manifest"),
            "exported_adapter_manifest": hash_result.get("export_manifest"),
            "adapter_hash_match": adapter_hash_match if hash_result else None,
            "verified_generation_policy": source_adapter_name if (not self.use_vllm or adapter_hash_match) else "unverified",
            "sampling_logprobs_available": sampling_logps is not None,
            "sampling_logprob_source": "vllm" if (self.use_vllm and sampling_logps is not None) else None,
            "sampling_logprob_adapter": source_adapter_name if sampling_logps is not None else None,
            "sampling_logprob_mean": self._mean_float(sampling_logps),
            "completion_length": len(completion_ids),
            "completion_mask_sum": len(completion_ids),
            # Populated by correctness/diversity scoring below; defaulted here
            # so every record -- including main/default rows, which never go
            # through _score_completions_diversity -- has these keys present
            # for easy diagnostics rather than silently missing.
            "is_correct": None,
            "answer_correct": None,
            "answer_correct_float": None,
            "main_correct_rate": None,
            "main_weak_factor": None,
            "main_weak_correctness_bonus": 0.0,
            "aux_correct_advantage": None,
            "main_weak_correctness_advantage": 0.0,
        }

    def _row_masked_mean_list(self, values: torch.Tensor | None, mask: torch.Tensor) -> list[float | None]:
        batch = int(mask.shape[0])
        if values is None:
            return [None] * batch
        detached = values.detach().float()
        if detached.ndim == 2 and detached.shape == mask.shape:
            denom = mask.detach().float().sum(dim=-1).clamp(min=1.0)
            return ((detached * mask.detach().float()).sum(dim=-1) / denom).cpu().tolist()
        if detached.ndim == 2 and detached.shape[0] == batch and detached.shape[1] == 1:
            return detached[:, 0].cpu().tolist()
        if detached.ndim == 1 and detached.shape[0] == batch:
            return detached.cpu().tolist()
        return [None] * batch

    def _masked_flat_stats(self, values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        detached = values.detach().float()
        if detached.ndim == 2 and detached.shape == mask.shape:
            flat = detached[mask.bool()]
        elif detached.ndim == 2 and detached.shape[1] == 1:
            flat = detached[:, 0]
        else:
            flat = detached.reshape(-1)
        if flat.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        }

    def _write_source_trace_records(self, records: list[dict[str, Any]]) -> None:
        if not records or not self._source_trace_enabled() or not self._is_main_lora_logging_process():
            return
        if self._source_trace_jsonl_path is None:
            output_dir = Path(getattr(self.args, "output_dir", "trainer_output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            self._source_trace_jsonl_path = str(output_dir / "source_owned_rollout_metadata.jsonl")
        with open(self._source_trace_jsonl_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._source_trace_write_count += len(records)

    def _expected_update_scope(self, adapter_name: str) -> str:
        if adapter_name == "default":
            return "collective_correctness"
        return "source_owned_adapter"

    def _expected_advantage_source(self, adapter_name: str) -> str:
        if adapter_name == "default":
            return "correctness_only_collective"
        if bool(getattr(self.args, "no_div", False)):
            return "correctness_only_source_owned"
        if bool(getattr(self.args, "no_correctness", False)):
            return "diversity_only_source_owned"
        return "correctness_plus_diversity_reward_source_owned"

    def _annotate_and_verify_update_semantics(
        self,
        generation_batch: dict[str, Any],
        adapter_name: str,
    ) -> None:
        expected_update_scope = self._expected_update_scope(adapter_name)
        advantage_source = self._expected_advantage_source(adapter_name)
        generation_batch["expected_update_scope"] = expected_update_scope
        generation_batch["update_scope"] = expected_update_scope
        generation_batch["advantage_source"] = advantage_source

        advantages = generation_batch.get("advantages")
        correctness_advantages = generation_batch.get("correctness_advantages")
        diversity_advantages = generation_batch.get("diversity_advantages")
        verified = True

        if isinstance(advantages, torch.Tensor):
            if adapter_name == "default":
                reference = correctness_advantages
            elif bool(getattr(self.args, "no_div", False)):
                reference = correctness_advantages
            elif bool(getattr(self.args, "no_correctness", False)):
                reference = diversity_advantages
            else:
                if correctness_advantages is None or diversity_advantages is None:
                    reference = None
                else:
                    reference = (
                        float(getattr(self.args, "correctness_weight_specialist", 0.0)) * correctness_advantages
                        + float(getattr(self.args, "diversity_weight_specialist", 0.0)) * diversity_advantages
                    )

            if isinstance(reference, torch.Tensor):
                verified = torch.allclose(
                    advantages.detach().float(),
                    reference.detach().float(),
                    rtol=1e-5,
                    atol=1e-6,
                )
                if not bool(verified):
                    max_abs_diff = (
                        advantages.detach().float() - reference.detach().float()
                    ).abs().max().item()
                    raise AssertionError(
                        f"Advantage source mismatch for adapter={adapter_name}: "
                        f"advantage_source={advantage_source}, max_abs_diff={max_abs_diff}"
                    )

        generation_batch["advantage_source_verified"] = bool(verified)

        source_trace_metadata = generation_batch.get("source_trace_metadata")
        if isinstance(source_trace_metadata, list):
            if adapter_name != "default":
                bad_sources = [
                    record.get("source_adapter_name")
                    for record in source_trace_metadata
                    if isinstance(record, dict) and record.get("source_adapter_name") != adapter_name
                ]
                if bad_sources:
                    raise AssertionError(
                        f"Diversity adapter update received non-source-owned samples: "
                        f"adapter={adapter_name}, bad_sources={bad_sources[:10]}"
                    )

            for record in source_trace_metadata:
                if isinstance(record, dict):
                    record["planned_update_scope"] = expected_update_scope
                    record["planned_advantage_source"] = advantage_source
                    record["planned_advantage_source_verified"] = bool(verified)

    def _emit_ratio_trace_and_metadata(
        self,
        inputs: dict[str, Any],
        adapter_name: str,
        per_token_logps: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        old_was_provided: bool,
        log_ratio: torch.Tensor,
        ratio: torch.Tensor,
        clip_mask: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        if not self._source_trace_enabled():
            return

        old_source_type = "hf_recomputed_adapter" if old_was_provided else "current_detach"
        old_adapter = adapter_name if old_was_provided else f"{adapter_name}_current_detach"
        old_denominator_policy = adapter_name
        expected_update_scope = inputs.get("expected_update_scope") or self._expected_update_scope(adapter_name)
        update_scope = inputs.get("update_scope") or expected_update_scope
        update_scope_ok = update_scope == expected_update_scope
        source_owned_required = adapter_name != "default"
        advantage_source = inputs.get("advantage_source") or self._expected_advantage_source(adapter_name)
        advantage_source_verified = bool(inputs.get("advantage_source_verified", False))
        metadata = inputs.get("source_trace_metadata") or []
        if isinstance(metadata, tuple):
            metadata = list(metadata)

        old_means = self._row_masked_mean_list(old_per_token_logps, completion_mask)
        current_means = self._row_masked_mean_list(per_token_logps, completion_mask)
        log_ratio_means = self._row_masked_mean_list(log_ratio, completion_mask)
        ratio_means = self._row_masked_mean_list(ratio, completion_mask)
        mask_sums = completion_mask.detach().float().sum(dim=-1).cpu().tolist()
        advantages_list = advantages.detach().float().cpu().tolist()
        correctness_rewards = inputs.get("correctness_reward_per_sample")
        diversity_rewards = inputs.get("diversity_reward_per_sample")
        correctness_list = (
            correctness_rewards.detach().float().cpu().tolist()
            if isinstance(correctness_rewards, torch.Tensor)
            else [None] * len(advantages_list)
        )
        diversity_list = (
            diversity_rewards.detach().float().cpu().tolist()
            if isinstance(diversity_rewards, torch.Tensor)
            else [None] * len(advantages_list)
        )

        updated_records = []
        for i, record in enumerate(metadata):
            if not isinstance(record, dict):
                continue
            behavior_policy = record.get("source_adapter_name")
            behavior_equals_old = behavior_policy == old_denominator_policy
            old_equals_update = old_denominator_policy == adapter_name
            raw_source_owned_ok = bool(behavior_equals_old and old_equals_update and behavior_policy == adapter_name)
            source_owned_ok = raw_source_owned_ok if source_owned_required else None
            if source_owned_required and not raw_source_owned_ok:
                raise AssertionError(
                    f"Source-owned update invariant failed for adapter={adapter_name}: "
                    f"behavior_policy={behavior_policy}, old_denominator_policy={old_denominator_policy}, "
                    f"update_adapter={adapter_name}, sample_id={record.get('sample_id')}"
                )
            corr = correctness_list[i] if i < len(correctness_list) else None
            div = diversity_list[i] if i < len(diversity_list) else None
            if adapter_name == "default" or div is None:
                combined_reward = corr
            elif corr is None:
                combined_reward = div
            else:
                combined_reward = (
                    float(getattr(self.args, "correctness_weight_specialist", 0.0)) * corr
                    + float(getattr(self.args, "diversity_weight_specialist", 0.0)) * div
                )
            record.update(
                {
                    "loss_record_id": f"{record.get('sample_id')}::update::{adapter_name}",
                    "old_logprob_source_type": old_source_type,
                    "old_logprob_adapter": old_adapter,
                    "old_logprob_denominator_policy": old_denominator_policy,
                    "current_logprob_adapter": adapter_name,
                    "update_adapter": adapter_name,
                    "expected_update_scope": expected_update_scope,
                    "update_scope": update_scope,
                    "update_scope_ok": update_scope_ok,
                    "source_owned_update_required": source_owned_required,
                    "source_owned_update_ok": source_owned_ok,
                    "raw_source_owned_update_ok": raw_source_owned_ok,
                    "behavior_equals_old_denominator": behavior_equals_old,
                    "old_equals_update_source": old_equals_update,
                    "advantage_source": advantage_source,
                    "advantage_source_verified": advantage_source_verified,
                    "ratio_expected_near_one": old_source_type == "current_detach",
                    "old_logprob_mean": old_means[i] if i < len(old_means) else None,
                    "current_logprob_mean": current_means[i] if i < len(current_means) else None,
                    "log_ratio_mean": log_ratio_means[i] if i < len(log_ratio_means) else None,
                    "ratio_mean": ratio_means[i] if i < len(ratio_means) else None,
                    "completion_mask_sum": mask_sums[i] if i < len(mask_sums) else None,
                    "correctness_reward": corr,
                    "diversity_reward": div,
                    "combined_reward": combined_reward,
                    "advantage": advantages_list[i] if i < len(advantages_list) else None,
                }
            )
            updated_records.append(record)

        self._write_source_trace_records(updated_records)

        if not self._should_log_source_trace():
            return
        source_policies = sorted({str(r.get("source_adapter_name")) for r in metadata if isinstance(r, dict)})
        requested_policies = sorted({str(r.get("requested_source_adapter")) for r in metadata if isinstance(r, dict)})
        verified_policies = sorted({str(r.get("verified_generation_policy")) for r in metadata if isinstance(r, dict)})
        behavior_policy = source_policies[0] if len(source_policies) == 1 else f"mixed({','.join(source_policies)})"
        requested_policy = requested_policies[0] if len(requested_policies) == 1 else f"mixed({','.join(requested_policies)})"
        verified_policy = verified_policies[0] if len(verified_policies) == 1 else f"mixed({','.join(verified_policies)})"
        if source_owned_required:
            batch_source_owned = all(r.get("source_owned_update_ok") is True for r in updated_records) if updated_records else False
        else:
            batch_source_owned = "NA_collective_correctness"
        log_ratio_stats = self._masked_flat_stats(log_ratio, completion_mask)
        ratio_stats = self._masked_flat_stats(ratio, completion_mask)
        clip_values = clip_mask.detach().float()
        clip_frac = self._masked_flat_stats(clip_values, completion_mask)["mean"]
        adv = advantages.detach().float()
        adv_std = float(adv.std(unbiased=False).item()) if adv.numel() > 1 else 0.0
        nonzero_adv = float((adv.abs() > 0).float().mean().item()) if adv.numel() > 0 else 0.0
        corr_tensor = correctness_rewards.detach().float() if isinstance(correctness_rewards, torch.Tensor) else None
        div_tensor = diversity_rewards.detach().float() if isinstance(diversity_rewards, torch.Tensor) else None
        corr_mean = float(corr_tensor.mean().item()) if corr_tensor is not None and corr_tensor.numel() > 0 else None
        div_mean = float(div_tensor.mean().item()) if div_tensor is not None and div_tensor.numel() > 0 else None
        div_std = float(div_tensor.std(unbiased=False).item()) if div_tensor is not None and div_tensor.numel() > 1 else 0.0
        combined_values = [r.get("combined_reward") for r in updated_records if r.get("combined_reward") is not None]
        combined_mean = sum(combined_values) / len(combined_values) if combined_values else None
        print(
            "[RATIO_TRACE] "
            f"step={getattr(self.state, 'global_step', 0)} adapter={adapter_name} num_samples={len(metadata)} "
            f"behavior_policy={behavior_policy} requested_generation_policy={requested_policy} "
            f"verified_generation_policy={verified_policy} old_logprob_source_type={old_source_type} "
            f"old_logprob_adapter={old_adapter} current_logprob_adapter={adapter_name} update_adapter={adapter_name} "
            f"expected_update_scope={expected_update_scope} update_scope={update_scope} update_scope_ok={update_scope_ok} "
            f"source_owned_update_required={source_owned_required} source_owned_update_ok={batch_source_owned} "
            f"advantage_source={advantage_source} advantage_source_verified={advantage_source_verified} "
            f"behavior_equals_old_denominator={behavior_policy == old_denominator_policy} "
            f"old_equals_update_source={old_denominator_policy == adapter_name} "
            f"old_logps_mean={self._masked_flat_stats(old_per_token_logps, completion_mask)['mean']:.6e} "
            f"current_logps_mean={self._masked_flat_stats(per_token_logps, completion_mask)['mean']:.6e} "
            f"log_ratio_mean={log_ratio_stats['mean']:.6e} log_ratio_std={log_ratio_stats['std']:.6e} "
            f"ratio_mean={ratio_stats['mean']:.6e} ratio_std={ratio_stats['std']:.6e} "
            f"ratio_min={ratio_stats['min']:.6e} ratio_max={ratio_stats['max']:.6e} "
            f"clip_frac={clip_frac:.6e} advantage_mean={float(adv.mean().item()) if adv.numel() else 0.0:.6e} "
            f"advantage_std={adv_std:.6e} completion_mask_sum={float(completion_mask.sum().item()):.1f}",
            flush=True,
        )
        print(
            "[REWARD_ADV_TRACE] "
            f"step={getattr(self.state, 'global_step', 0)} adapter={adapter_name} "
            f"diversity_reward_mean={div_mean} diversity_reward_std={div_std} "
            f"correctness_reward_mean={corr_mean} combined_reward_mean={combined_mean} "
            f"advantage_mean={float(adv.mean().item()) if adv.numel() else 0.0:.6e} "
            f"advantage_std={adv_std:.6e} advantage_abs_mean={float(adv.abs().mean().item()) if adv.numel() else 0.0:.6e} "
            f"nonzero_advantage_ratio={nonzero_adv:.6e} "
            f"completion_mask_sum_mean={float(completion_mask.detach().float().sum(dim=-1).mean().item()):.6e} "
            f"loss={float(loss.detach().float().cpu().item()):.6e} grad_norm=NA_pre_backward",
            flush=True,
        )



    @torch.no_grad()
    def _lora_fingerprint(self, model: nn.Module, adapter_name: str) -> torch.Tensor:
        s1 = torch.zeros((), device="cpu")
        s2 = torch.zeros((), device="cpu")
        smax = torch.zeros((), device="cpu")
        subs = torch.zeros((), device="cpu")
        n = 0

        matched_any = False
        for name, p in model.named_parameters():
            if p is None or "lora_" not in name:
                continue
            if not self._match_adapter_param(name, adapter_name):
                continue
            matched_any = True

            x = p.detach().float().cpu() 
            s1 += x.sum()
            s2 += (x * x).sum()
            smax = torch.maximum(smax, x.abs().max())

            flat = x.view(-1)
            stride = max(1, flat.numel() // 2048)
            subs += flat[::stride].sum()

            n += x.numel()

        if (adapter_name != "default") and (not matched_any):
            raise RuntimeError(
                f"[LoRA FP] No LoRA params matched for adapter='{adapter_name}'. "
                "Check adapter creation/naming in PEFT."
            )

        return torch.stack([s1, s2, smax, subs, torch.tensor(float(n), device="cpu")])
        
    def _fp_to_scalar_dict(self, fp, prefix: str) -> dict[str, float]:
        if isinstance(fp, torch.Tensor):
            fp_cpu = fp.detach().float().cpu().tolist()
        elif isinstance(fp, (list, tuple)):
            fp_cpu = list(fp)
        else:
            raise TypeError(f"[LoRA FP] fp must be Tensor/list, got {type(fp)}")

        if isinstance(fp_cpu, (float, int)):
            raise ValueError(f"[LoRA FP] fp is scalar ({fp_cpu}). Expected length-5 vector.")

        if len(fp_cpu) != 5:
            raise ValueError(f"[LoRA FP] fp length={len(fp_cpu)}. Expected 5.")

        return {
            f"{prefix}/lora_fp_sum": float(fp_cpu[0]),
            f"{prefix}/lora_fp_sumsq": float(fp_cpu[1]),
            f"{prefix}/lora_fp_maxabs": float(fp_cpu[2]),
            f"{prefix}/lora_fp_subsum": float(fp_cpu[3]),
            f"{prefix}/lora_fp_numel": float(fp_cpu[4]),
        }

    @torch.no_grad()
    def _log_lora_fingerprints(self, mode: str):
        if not hasattr(self, "_prev_lora_fp"):
            self._prev_lora_fp = {}

        for adapter_name in self.all_adapter_names:
            fp_local = self._lora_fingerprint(self.model, adapter_name).detach().cpu().tolist()

            if dist.is_available() and dist.is_initialized():
                obj_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, fp_local)   # obj_list: [fp_rank0, fp_rank1, ...]
                fp_all = obj_list
            else:
                fp_all = [fp_local]
                
            fp_mean = torch.tensor(fp_all, dtype=torch.float32).mean(dim=0)  # (5,)

            for k, v in self._fp_to_scalar_dict(fp_mean, f"{adapter_name}").items():
                self._metrics[mode][k].append(v)

            prev = self._prev_lora_fp.get(adapter_name, None)
            if prev is not None:
                d = fp_mean[:4] - prev[:4]
                delta = torch.sqrt((d * d).sum()).item()
                self._metrics[mode][f"{adapter_name}/lora_fp_delta_l2"].append(float(delta))

            self._prev_lora_fp[adapter_name] = fp_mean.detach()

                    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In CoExTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to CoEx, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )


    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 로컬 helper로 두면 NameError 절대 안 남
        def _slice_if_batched(v, start, end, batch_size):
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == batch_size:
                return v[start:end]
            return v

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if not isinstance(inputs, dict):
                inputs = {"base": inputs}

            total_loss = []

            for adapter_name, adapter_inputs in inputs.items():
                self._set_adapter_with_logging(self.model, adapter_name, "training_backward")
                self.enable_all_lora_grads(self.model)
                adapter_inputs["_adapter_name"] = adapter_name

                if self.loss_type == "dapo" or (
                    self.loss_type == "dmpo" and self.dmpo_base_loss_type == "dapo"
                ):
                    # Local Completion tokens on this rank
                    local_token = adapter_inputs["completion_mask"].sum().to(torch.float32)

                    # global completion tokens (sum across all ranks)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(local_token, op=dist.ReduceOp.SUM)
                    global_token = local_token

                    # Store Scalar Normalizer Source (global tokens)
                    adapter_inputs["num_items_in_batch"] = global_token

                if is_sagemaker_mp_enabled():
                    loss_mb = smp_forward_backward(model, adapter_inputs, self.args.gradient_accumulation_steps)
                    return loss_mb.reduce_mean().detach().to(self.args.device)

                current_batch_size = next(
                    (t.shape[0] for t in adapter_inputs.values()
                    if isinstance(t, torch.Tensor) and t.ndim >= 1),
                    None,
                )

                if (
                    self.loss_type in {"dmpo", "pure_dmpo"}
                    and current_batch_size is not None
                    and self.args.mini_batch_size is not None
                    and current_batch_size > self.args.mini_batch_size
                ):
                    raise ValueError(
                        "DMPO requires each prompt rollout group to stay in one loss call. "
                        "Disable mini-batch splitting or set mini_batch_size >= the adapter batch size."
                    )

                # --- mini-batch split 경로 ---
                if (
                    current_batch_size is not None
                    and self.args.mini_batch_size is not None
                    and current_batch_size > self.args.mini_batch_size
                ):
                    mini_bs = self.args.mini_batch_size
                    num_chunks = math.ceil(current_batch_size / mini_bs)
                    accumulated_loss = 0.0

                    for start in range(0, current_batch_size, mini_bs):
                        end = min(start + mini_bs, current_batch_size)

                        chunk_inputs = {
                            k: _slice_if_batched(v, start, end, current_batch_size)
                            for k, v in adapter_inputs.items()
                        }

                        with self.compute_loss_context_manager():
                            loss = self.compute_loss(model, chunk_inputs, num_items_in_batch=None)

                        if self.loss_type != "dapo":
                            # Scale loss to account for gradient accumulation
                            loss = loss / num_chunks

                        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                            self.accelerator.backward(loss, scale_wrt_gas=False)
                        else:
                            self.accelerator.backward(loss)

                        accumulated_loss += loss.detach()

                    total_loss.append(accumulated_loss)

                # --- 일반 경로 ---
                else:
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, adapter_inputs, num_items_in_batch=num_items_in_batch)

                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                        self.accelerator.backward(loss, scale_wrt_gas=False)
                    else:
                        self.accelerator.backward(loss)

                    total_loss.append(loss.detach())

            loss = torch.stack(total_loss).mean()
            return loss.detach()
        

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    def _get_unwrapped_policy_model(self, model):
        return self.accelerator.unwrap_model(model)

    def _get_unwrapped_causal_lm(self, model):
        unwrapped_model = self._get_unwrapped_policy_model(model)
        if is_peft_model(unwrapped_model):
            # This is the same LoRA-injected CausalLM, not an adapter-free copy.
            return unwrapped_model.get_base_model()
        return unwrapped_model

    @staticmethod
    def _active_adapter_name(model) -> str:
        active_adapter = getattr(model, "active_adapter", None)
        if active_adapter is None:
            active_adapter = getattr(model, "active_adapters", None)
        if isinstance(active_adapter, (list, tuple)):
            return ",".join(str(name) for name in active_adapter)
        return str(active_adapter)

    @staticmethod
    def _match_lm_head_dtype(lm_head, hidden_states):
        lm_head_weight = getattr(lm_head, "weight", None)
        if (
            lm_head_weight is not None
            and hidden_states.dtype != lm_head_weight.dtype
        ):
            hidden_states = hidden_states.to(lm_head_weight.dtype)
        return hidden_states

    def _project_selected_logps_chunk(
        self,
        lm_head,
        hidden_states,
        target_ids,
    ):
        hidden_states = self._match_lm_head_dtype(
            lm_head,
            hidden_states,
        )
        logits = lm_head(hidden_states)
        if logits.dtype in (torch.float16, torch.bfloat16):
            # Match Accelerate's output conversion for one token chunk only.
            logits = logits.float()
        logits = logits / self.temperature
        selected_logps = selective_log_softmax(logits, target_ids)
        del logits
        return selected_logps

    def _chunked_selected_logps_from_hidden_states(
        self,
        lm_head,
        hidden_states,
        target_ids,
        compute_entropy=False,
        chunk_size=None,
        use_checkpoint=True,
    ):
        chunk_size = chunk_size or self.args.logprob_token_chunk_size
        if chunk_size <= 0:
            raise ValueError("logprob_token_chunk_size must be positive")
        if hidden_states.shape[:2] != target_ids.shape:
            raise ValueError(
                "Hidden-state and target-token shapes differ: "
                f"{tuple(hidden_states.shape)} vs {tuple(target_ids.shape)}"
            )

        logps_chunks = []
        entropy_chunks = []
        keep_graph = torch.is_grad_enabled() and hidden_states.requires_grad

        def project_chunk(hidden_chunk, target_chunk):
            return self._project_selected_logps_chunk(
                lm_head,
                hidden_chunk,
                target_chunk,
            )

        for token_start in range(0, hidden_states.size(1), chunk_size):
            token_end = min(token_start + chunk_size, hidden_states.size(1))
            hidden_chunk = hidden_states[:, token_start:token_end, :]
            target_chunk = target_ids[:, token_start:token_end]

            if keep_graph and use_checkpoint:
                # Checkpointing prevents autograd from retaining every
                # [B, chunk_T, V] softmax intermediate until backward.
                logps_chunk = checkpoint(
                    project_chunk,
                    hidden_chunk,
                    target_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

                if compute_entropy:
                    # Entropy is metric/masking data and does not need a graph.
                    with torch.no_grad():
                        entropy_hidden = self._match_lm_head_dtype(
                            lm_head,
                            hidden_chunk.detach(),
                        )
                        entropy_logits = lm_head(entropy_hidden)
                        if entropy_logits.dtype in (
                            torch.float16,
                            torch.bfloat16,
                        ):
                            entropy_logits = entropy_logits.float()
                        entropy_logits = entropy_logits / self.temperature
                        entropy_chunks.append(
                            entropy_from_logits(entropy_logits)
                        )
                        del entropy_logits
                        del entropy_hidden
            else:
                # Old/ref/reward-only callers reach this branch under no_grad.
                projection_hidden = self._match_lm_head_dtype(
                    lm_head,
                    hidden_chunk,
                )
                logits_chunk = lm_head(projection_hidden)
                if logits_chunk.dtype in (torch.float16, torch.bfloat16):
                    logits_chunk = logits_chunk.float()
                logits_chunk = logits_chunk / self.temperature
                logps_chunk = selective_log_softmax(
                    logits_chunk,
                    target_chunk,
                )
                if compute_entropy:
                    entropy_chunks.append(entropy_from_logits(logits_chunk))
                del logits_chunk
                del projection_hidden

            logps_chunks.append(logps_chunk)

        per_token_logps = torch.cat(logps_chunks, dim=1)
        entropies = (
            torch.cat(entropy_chunks, dim=1)
            if compute_entropy
            else None
        )
        return per_token_logps, entropies

    @torch.no_grad()
    def _maybe_run_chunked_logprob_sanity_check(
        self,
        policy_model,
        model_inputs,
        target_ids,
    ):
        enabled = self.args.logprob_sanity_check or os.environ.get(
            "COEX_LOGPROB_SANITY_CHECK",
            "0",
        ).lower() in {"1", "true", "yes"}
        if not enabled:
            return

        unwrapped_peft_model = self._get_unwrapped_policy_model(policy_model)
        if not is_peft_model(unwrapped_peft_model):
            return
        if "logits_to_keep" not in self.model_kwarg_keys:
            raise RuntimeError(
                "PEFT logprob sanity check requires logits_to_keep support."
            )

        active_adapter = self._active_adapter_name(unwrapped_peft_model)
        checked_adapters = getattr(
            self,
            "_chunked_logprob_sanity_checked_adapters",
            set(),
        )
        if active_adapter in checked_adapters:
            return

        debug_tokens = min(
            int(os.environ.get("COEX_LOGPROB_SANITY_MAX_TOKENS", "16")),
            target_ids.size(1),
        )
        if debug_tokens <= 0:
            return

        causal_lm = unwrapped_peft_model.get_base_model()
        base_active_adapter = self._active_adapter_name(
            getattr(unwrapped_peft_model, "base_model", causal_lm)
        )
        if base_active_adapter != active_adapter:
            raise RuntimeError(
                "PEFT active-adapter mismatch: "
                f"wrapper={active_adapter}, base={base_active_adapter}"
            )

        lm_head = causal_lm.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("PEFT policy has no output LM head")

        input_batch_size = model_inputs["input_ids"].size(0)
        sanity_model_inputs = {}
        for key, value in model_inputs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.size(0) == input_batch_size
            ):
                sanity_model_inputs[key] = value[:1]
            else:
                sanity_model_inputs[key] = value
        sanity_model_inputs.pop("logits_to_keep", None)
        sanity_model_inputs["use_cache"] = False
        sample_targets = target_ids[:1, -debug_tokens:]

        was_training = unwrapped_peft_model.training
        unwrapped_peft_model.eval()
        try:
            with self.compute_loss_context_manager():
                # Reference path: the original PEFT wrapper forward.
                full_outputs = unwrapped_peft_model(
                    **sanity_model_inputs,
                    logits_to_keep=debug_tokens + 1,
                )
                full_logits = full_outputs.logits[:, :-1, :]
                full_logits = full_logits[:, -debug_tokens:, :]
                if full_logits.dtype in (torch.float16, torch.bfloat16):
                    full_logits = full_logits.float()
                full_logits = full_logits / self.temperature
                full_shape = tuple(full_logits.shape)
                full_logps = selective_log_softmax(
                    full_logits,
                    sample_targets,
                )

                # Candidate path: the same LoRA-injected backbone and head.
                direct_hidden_states = causal_lm.model(
                    **sanity_model_inputs
                ).last_hidden_state
                direct_hidden_states = direct_hidden_states[:, :-1, :]
                direct_hidden_states = direct_hidden_states[
                    :,
                    -debug_tokens:,
                    :,
                ]
                debug_chunk_size = max(
                    1,
                    min(
                        self.args.logprob_token_chunk_size,
                        max(1, debug_tokens // 2),
                    ),
                )
                chunked_logps, _ = (
                    self._chunked_selected_logps_from_hidden_states(
                        lm_head,
                        direct_hidden_states,
                        sample_targets,
                        compute_entropy=False,
                        chunk_size=debug_chunk_size,
                        use_checkpoint=False,
                    )
                )
        finally:
            unwrapped_peft_model.train(was_training)

        absolute_difference = (full_logps - chunked_logps).abs()
        max_abs_diff = absolute_difference.max().item()
        mean_abs_diff = absolute_difference.mean().item()
        print(
            "[LOGPROB_SANITY] "
            f"active_adapter={active_adapter} "
            f"full_logits_shape={full_shape} "
            f"chunk_size={debug_chunk_size} "
            f"max_abs_diff={max_abs_diff:.8e} "
            f"mean_abs_diff={mean_abs_diff:.8e}"
        )

        checked_adapters.add(active_adapter)
        self._chunked_logprob_sanity_checked_adapters = checked_adapters

        del full_outputs
        del full_logits
        del full_logps
        del direct_hidden_states
        del chunked_logps
        del absolute_difference

        if max_abs_diff > 1e-2:
            raise RuntimeError(
                "Chunked logprob sanity check failed for active adapter "
                f"{active_adapter}: max_abs_diff={max_abs_diff:.8e}"
            )

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute selected token log-probs without full [B, T, V] logits."""
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        causal_lm = self._get_unwrapped_causal_lm(model)
        lm_head = causal_lm.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("Model does not expose an output LM head")
        if not hasattr(causal_lm, "model"):
            raise RuntimeError(
                "Chunked logprob computation requires a `.model` backbone"
            )

        run_sanity_check = torch.is_grad_enabled()
        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "use_cache": False,
            }
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack(
                    [sample_rows.sum() for sample_rows in rows_per_sample]
                )
                cum_rows = torch.cat(
                    [
                        torch.tensor([0], device=rows_per_sample.device),
                        rows_per_sample.cumsum(0),
                    ]
                )
                row_start, row_end = (
                    cum_rows[start].item(),
                    cum_rows[end].item(),
                )
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor(
                    [0] + list(num_images),
                    device=image_grid_thw.device,
                ).cumsum(0)
                img_start, img_end = (
                    cum_imgs[start].item(),
                    cum_imgs[end].item(),
                )
                model_inputs["image_grid_thw"] = image_grid_thw[
                    img_start:img_end
                ]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[
                    start:end
                ]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start:end]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start:end]

            with self.compute_loss_context_manager():
                # get_base_model() retains the injected LoRA modules and the
                # adapter selected by PeftModel.set_adapter().
                hidden_states = causal_lm.model(
                    **model_inputs
                ).last_hidden_state
                hidden_states = hidden_states[:, :-1, :]
                hidden_states = hidden_states[:, -logits_to_keep:, :]
                completion_ids = input_ids_batch[:, -logits_to_keep:]

                if torch.is_grad_enabled() and not hidden_states.requires_grad:
                    raise RuntimeError(
                        "Current-policy hidden states do not require gradients"
                    )

                logps, entropies = (
                    self._chunked_selected_logps_from_hidden_states(
                        lm_head,
                        hidden_states,
                        completion_ids,
                        compute_entropy=compute_entropy,
                    )
                )

            if run_sanity_check:
                self._maybe_run_chunked_logprob_sanity_check(
                    model,
                    model_inputs,
                    completion_ids,
                )

            if logps.shape != completion_ids.shape:
                raise RuntimeError(
                    "Chunked selected-logprob shape mismatch: "
                    f"{tuple(logps.shape)} vs {tuple(completion_ids.shape)}"
                )

            all_logps.append(logps)
            if compute_entropy:
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = (
            torch.cat(all_entropies, dim=0)
            if compute_entropy
            else None
        )
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: list[str] | None = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    def _reload_lora_in_vllm_colocate(self, adapter_info: dict):
        # adapter_info: {"name": str, "export_dir": str, "path": str, "id": int}; path is the actual vLLM LoRA root.
        if not (self.use_vllm and self.vllm_mode == "colocate"):
            return
        eng = getattr(self.llm, "llm_engine", None)
        if eng is None:
            return

        # remove old (ignore errors)
        try:
            eng.remove_lora(adapter_info["id"])
        except Exception:
            pass

        # add new
        try:
            from vllm.lora.request import LoRARequest
            lora_path = adapter_info["path"]
            print(
                f"[VLLM_LORA_REQUEST] adapter={adapter_info['name']} id={adapter_info['id']} "
                f"lora_local_path={lora_path}",
                flush=True,
            )
            ok = eng.add_lora(
                LoRARequest(
                    lora_name=adapter_info["name"],
                    lora_int_id=adapter_info["id"],
                    lora_local_path=lora_path,
                )
            )
            if not ok:
                logger.warning(f"[vLLM] add_lora returned False for {adapter_info['name']}")
        except Exception as e:
            logger.warning(f"[vLLM] LoRA reload failed for {adapter_info['name']}: {e}")

        # prefix cache reset (stale KV 방지)
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass
    
    @profiling_decorator
    def _move_model_to_vllm(self, adapter_name: Optional[str] = None):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model) and self.lora_modules:
            adapters_to_process = [adapter_name] if adapter_name else self.all_adapter_names

            for adapter in adapters_to_process:
                print(f"  >> [vLLM Sync] Syncing Adapter '{adapter}' to vLLM...")

                adapter_info = next((module for module in self.lora_modules if module["name"] == adapter), None)

                if not adapter_info:
                    print(f"  >> [Warning] Adapter info for '{adapter}' not found in lora_modules.")
                    continue

                optimizer_step = int(getattr(self.state, "global_step", 0)) + 1
                sync_before = None
                if (
                    self._is_main_lora_logging_process()
                    and self._should_log_lora_integrity(optimizer_step)
                ):
                    sync_before = self._collect_lora_integrity([adapter])
                    self._print_lora_integrity(
                        "pre_vllm_sync", optimizer_step, sync_before
                    )

                # DeepSpeed ZeRO-3의 경우 전체 파라미터를 gather
                with gather_if_zero3(list(self.model.parameters())):
                    if self.accelerator.is_main_process:
                        print(f"  >> [vLLM Sync] Merging Adapter '{adapter}'...")
                        self._save_adapter_for_vllm(
                            self.model,
                            adapter_info,
                            phase="vllm_sync_export",
                            optimizer_step=optimizer_step,
                        )
                        print(f"  >> [vLLM Sync] Adapter '{adapter}' saved to '{adapter_info['path']}'.")
                    
                    # Synchronize with barrier for DeepSpeed
                    if zero_stage_3:
                        torch.distributed.barrier()
                
                self._reload_lora_in_vllm_colocate(adapter_info)
                if sync_before is not None:
                    sync_after = self._collect_lora_integrity([adapter])
                    self._print_lora_integrity(
                        "post_vllm_sync",
                        optimizer_step,
                        sync_after,
                        baseline=sync_before,
                    )

        else:
            # Non-PEFT models
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
    
    # @profiling_decorator
    # def _move_model_to_vllm(self):
    #     # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
    #     deepspeed_plugin = self.accelerator.state.deepspeed_plugin
    #     zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
    #     if zero_stage_3:
    #         import deepspeed

    #         gather_if_zero3 = deepspeed.zero.GatheredParameters
    #     else:
    #         gather_if_zero3 = nullcontext

    #     if is_peft_model(self.model):
    #         # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
    #         # merging adapters in a sharded manner is not supported.
    #         # TODO: does this work with FSDP?
    #         with gather_if_zero3(list(self.model.parameters())):
    #             self.model.merge_adapter()

    #             # Update vLLM weights while parameters are gathered
    #             if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
    #                 # Update vLLM weights while parameters are gathered
    #                 # For PEFT with FSDP we need to use the memory efficient post-order traversal
    #                 fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
    #                 fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
    #                 if fsdp_version == 1:
    #                     self._sync_fsdp1_params_to_vllm(
    #                         self.model
    #                     )  # use memory-efficient post-order traversal for FSDP
    #                 elif fsdp_version == 2:
    #                     self._sync_fsdp2_params_to_vllm(self.model)
    #             else:
    #                 # DeepSpeed ZeRO-3 with PEFT
    #                 for name, param in self.model.named_parameters():
    #                     # When using PEFT, we need to recover the original parameter name and discard some parameters
    #                     name = name.removeprefix("base_model.model.").replace(".base_layer", "")
    #                     if self.model.prefix in name:
    #                         continue
    #                     # When module to save, remove its prefix and discard the original module
    #                     if "original_module" in name:
    #                         continue
    #                     name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

    #                     if self.vllm_mode == "server" and self.accelerator.is_main_process:
    #                         self.vllm_client.update_named_param(name, param.data)
    #                     elif self.vllm_mode == "colocate":
    #                         llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    #                         llm_model.load_weights([(name, param.data)])
    #             # Unmerge adapters while parameters are still gathered
    #             self.model.unmerge_adapter()
    #             # Parameters will automatically be repartitioned when exiting the context
    #     else:
    #         # For non-PEFT models, simply gather (if needed) and update each parameter individually.
    #         if self.is_fsdp_enabled:
    #             fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
    #             fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
    #             if fsdp_version == 1:
    #                 self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
    #             elif fsdp_version == 2:
    #                 self._sync_fsdp2_params_to_vllm(self.model)
    #         else:
    #             for name, param in self.model.named_parameters():
    #                 name = self._fix_param_name_to_vllm(name)
    #                 with gather_if_zero3([param]):
    #                     if self.vllm_mode == "server" and self.accelerator.is_main_process:
    #                         self.vllm_client.update_named_param(name, param.data)
    #                     elif self.vllm_mode == "colocate":
    #                         llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    #                         llm_model.load_weights([(name, param.data)])

    #     # Reset cache on vLLM
    #     if self.vllm_mode == "server" and self.accelerator.is_main_process:
    #         self.vllm_client.reset_prefix_cache()
    #     elif self.vllm_mode == "colocate":
    #         self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                
                assert self.num_generations == sum(self.all_adapter_num_completions)

                adapter_to_indices = {name: [] for name in self.all_adapter_names}
                num_data_per_batch = len(generation_batch) // self.num_generations
                # breakpoint()
                """
                (Pdb) adapter_to_indices.keys()
                dict_keys(['default', 'diversity_0', 'diversity_1'])
                (Pdb) num_data_per_batch
                1
                """
                # gen 축에서의 시작 위치 (offset)
                gen_offset = 0
                for name, n_comp in zip(self.all_adapter_names, self.all_adapter_num_completions):
                    # 이 어댑터가 차지하는 gen_idx 구간: [gen_offset, gen_offset + n_comp)
                    for b in range(num_data_per_batch):
                        for g in range(gen_offset, gen_offset + n_comp):
                            flat_idx = b * self.num_generations + g
                            adapter_to_indices[name].append(flat_idx)
                    gen_offset += n_comp
                # breakpoint()
                """
                (Pdb) adapter_to_indices
                {'default': [0, 1, 2, 3], 'diversity_0': [4, 5, 6], 'diversity_1': [7, 8, 9]}
                """
                
                generation_batch_per_adapter, pass_data_per_adapter, forward_kwargs_per_adapter = self._generate_completions(generation_batch, adapter_to_indices)
                # breakpoint()
                """
                (Pdb) generation_batch_per_adapter["default"]["prompt_ids"].shape
                torch.Size([10, 163])
                (Pdb) generation_batch_per_adapter["diversity_0"]["prompt_ids"].shape
                torch.Size([3, 163])
                """
                for adapter_name, num_completions in zip(self.all_adapter_names, self.all_adapter_num_completions):
                    current_num_completions = num_completions
                    # breakpoint()
                    """
                    (Pdb) self.all_adapter_names
                    ['default', 'diversity_0', 'diversity_1']
                    (Pdb) self.all_adapter_num_completions
                    [4, 3, 3]
                    (Pdb) current_num_completions
                    4
                    """
                    if adapter_name == "default":
                        current_num_completions = self.num_generations # self.num_generations = 10
                    generation_batch_per_adapter[adapter_name]= self._score_completions_correctness(generation_batch_per_adapter[adapter_name], 
                                                                                                    pass_data_per_adapter[adapter_name], 
                                                                                                    forward_kwargs_per_adapter[adapter_name], 
                                                                                                    adapter_name,
                                                                                                    current_num_completions,
                                                                                                    adapter_to_indices[adapter_name])
                    if adapter_name == "default":
                        generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["correctness_advantages"]

                        # main_correct_rate(x) per prompt, from PURE answer correctness
                        # only (never the mixed correctness_reward scalar). default's own
                        # correctness pass above already scored the FULL rollout pool
                        # (current_num_completions == self.num_generations), so every row's
                        # answer_correct_float/prompt_index/source_adapter_name is already
                        # populated in source_trace_metadata at this point -- this filters
                        # down to "default"-sourced rows only, robust to any number of
                        # diversity adapters and to multi-prompt batches.
                        main_correct_rate_by_prompt = compute_main_correct_rate_by_prompt(
                            generation_batch_per_adapter[adapter_name]["source_trace_metadata"]
                        )
                        self._log_main_correct_rate_and_group_coverage(
                            generation_batch_per_adapter[adapter_name]["source_trace_metadata"],
                            main_correct_rate_by_prompt,
                        )
                    else:
                        # no_div 모드: diversity adapter도 correctness만으로 학습
                        if self.args.no_div:
                            generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["correctness_advantages"]
                            generation_batch_per_adapter[adapter_name]["diversity_advantages"] = torch.zeros_like(
                                generation_batch_per_adapter[adapter_name]["correctness_advantages"]
                            )
                        else:
                            # diversity reward comparison pool.
                            # Default: compare only within this diversity adapter's rollout group,
                            # excluding the current rollout inside the reward function.
                            all_pass_data = pass_data_per_adapter["default"]
                            source_indices = list(adapter_to_indices[adapter_name])
                            source_index_set = set(source_indices)
                            if self.diversity_comparison_scope == "intra_adapter":
                                comparison_indices = source_indices
                            # pass_data_per_adapter["default"] holds all completions across
                            # every adapter (it is never sliced — see _generate_completions).
                            all_pass_data = pass_data_per_adapter["default"]
                            source_indices = list(adapter_to_indices[adapter_name])
                            source_index_set = set(source_indices)

                            index_to_adapter = {}
                            for rollout_adapter, rollout_indices in adapter_to_indices.items():
                                for index in rollout_indices:
                                    index_to_adapter[index] = rollout_adapter

                            if self.diversity_comparison_scope == "intra_adapter":
                                comparison_indices = source_indices
                                # Reference group = the adapter's own completions (self-exclusion
                                # is handled inside the reward helper per exclude_self flag).
                                reference_groups = {
                                    adapter_name: [
                                        all_pass_data["completions"][i] for i in source_indices
                                    ]
                                }
                                source_group_name = adapter_name
                            elif self.diversity_comparison_scope == "all_other":
                                comparison_indices = [
                                    index
                                    for index in range(len(all_pass_data["completions"]))
                                    if index not in source_index_set
                                ]
                                # Build per-source reference groups: main/default + every other
                                # diversity adapter, each keyed by adapter name.
                                reference_groups = {}
                                for ref_adapter, ref_indices in adapter_to_indices.items():
                                    if ref_adapter != adapter_name:
                                        reference_groups[ref_adapter] = [
                                            all_pass_data["completions"][i] for i in ref_indices
                                        ]
                                # source is never in reference_groups → no self-exclusion needed
                                source_group_name = None
                            else:
                                raise ValueError(
                                    f"Unknown diversity_comparison_scope: {self.diversity_comparison_scope}"
                                )
                                index_to_adapter = {}
                            for rollout_adapter, rollout_indices in adapter_to_indices.items():
                                for index in rollout_indices:
                                    index_to_adapter[index] = rollout_adapter
                            other_data = {
                                key: [all_pass_data[key][index] for index in comparison_indices]
                                for key in ["prompts", "completions", "completion_ids_list"]
                            }
                            other_data["adapter_names"] = [
                                index_to_adapter[index] for index in comparison_indices
                            ]
                            other_data["candidate_indices"] = source_indices
                            other_data["comparison_indices"] = comparison_indices
                            other_data["exclude_self"] = self.diversity_comparison_scope == "intra_adapter"
                            other_data["comparison_scope"] = self.diversity_comparison_scope
                            # New: structured reference groups for principled BLEU reward
                            other_data["reference_groups"] = reference_groups
                            other_data["source_group_name"] = source_group_name
                            
                            generation_batch_per_adapter[adapter_name] = self._score_completions_diversity(
                                generation_batch_per_adapter[adapter_name],
                                pass_data_per_adapter[adapter_name],
                                forward_kwargs_per_adapter[adapter_name],
                                other_data,
                                adapter_name,
                                current_num_completions,
                                main_correct_rate_by_prompt=main_correct_rate_by_prompt,
                            )

                            if self.correctness_gated is True and self.args.diversity_reward_type not in self._MAIN_WEAK_CORRECTNESS_REWARD_TYPES:
                                # Gate on PURE answer correctness only -- never the mixed
                                # correctness_reward scalar (0.0/0.5/1.0/1.5), which would
                                # let a wrong-but-think-tagged sample (reward 0.5) slip
                                # through an 0.5 threshold ungated.
                                #
                                # main_weak_correctness_bonus is deliberately EXCLUDED from
                                # this gate: its diversity_advantages already come from
                                # Norm[answer_correct_float] (see _score_completions_diversity),
                                # where a correct sample gets positive advantage and a wrong
                                # sample gets NEGATIVE advantage by design -- that contrast is
                                # the intended training signal. Zeroing wrong samples here
                                # would remove the negative half of the contrast and weaken
                                # the signal, so this reward type manages its own
                                # correctness-based shaping internally instead.
                                answer_correct_float = generation_batch_per_adapter[adapter_name]["answer_correct_float"]
                                mask = answer_correct_float < self.correctness_threshold
                                generation_batch_per_adapter[adapter_name]["diversity_advantages"][mask] = 0.0

                            # no_correctness 모드: diversity만으로 학습
                            if self.args.no_correctness:
                                generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["diversity_advantages"]
                            else:
                                generation_batch_per_adapter[adapter_name]["advantages"] = \
                                    self.args.correctness_weight_specialist * generation_batch_per_adapter[adapter_name]["correctness_advantages"] + \
                                    self.args.diversity_weight_specialist * generation_batch_per_adapter[adapter_name]["diversity_advantages"]

                    self._annotate_and_verify_update_semantics(
                        generation_batch_per_adapter[adapter_name],
                        adapter_name,
                    )

                    # if adapter_name == "default":
                    #     generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["correctness_advantages"]
                    # else:
                    #     # no_div 모드: diversity adapter도 correctness만으로 학습
                    #     if self.args.no_div:
                    #         generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["correctness_advantages"]
                    #         generation_batch_per_adapter[adapter_name]["diversity_advantages"] = torch.zeros_like(
                    #             generation_batch_per_adapter[adapter_name]["correctness_advantages"]
                    #         )
                    #     else:
                    #         other_data = {}
                    #         for key in ["completions", "completion_ids_list"]:
                    #             other_data[key] = pass_data_per_adapter["default"][key].copy()
                    #             for my_data in pass_data_per_adapter[adapter_name][key]:
                    #                 other_data[key].remove(my_data)
                            
                    #         generation_batch_per_adapter[adapter_name] = self._score_completions_diversity(
                    #             generation_batch_per_adapter[adapter_name], 
                    #             pass_data_per_adapter[adapter_name], 
                    #             forward_kwargs_per_adapter[adapter_name],
                    #             other_data, 
                    #             adapter_name,
                    #             current_num_completions
                    #         )
                            
                    #         if self.correctness_gated is True:
                    #             correctness_rewards = generation_batch_per_adapter[adapter_name]["correctness_reward_per_sample"]
                    #             mask = correctness_rewards < self.correctness_threshold
                    #             generation_batch_per_adapter[adapter_name]["diversity_advantages"][mask] = 0.0

                    #         generation_batch_per_adapter[adapter_name]["advantages"] = \
                    #             self.args.correctness_weight_specialist * generation_batch_per_adapter[adapter_name]["correctness_advantages"] + \
                    #             self.args.diversity_weight_specialist * generation_batch_per_adapter[adapter_name]["diversity_advantages"]

                    # if adapter_name == "default":
                    #     generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["correctness_advantages"]
                    # else:
                    #     other_data = {}
                    #     for key in ["completions", "completion_ids_list"]:
                    #         other_data[key] = pass_data_per_adapter["default"][key].copy() # make a copy  # all data
                    #         for my_data in pass_data_per_adapter[adapter_name][key]:
                    #             other_data[key].remove(my_data) # delete my data from all data remove is delete first occurrence only
                    #     # breakpoint()
                        
                    #     generation_batch_per_adapter[adapter_name]= self._score_completions_diversity(generation_batch_per_adapter[adapter_name], 
                    #                                                                                     pass_data_per_adapter[adapter_name], 
                    #                                                                                     forward_kwargs_per_adapter[adapter_name], # [NEED REVIEW]
                    #                                                                                     other_data, 
                    #                                                                                     adapter_name,
                    #                                                                                     current_num_completions)
                        
                    #     if self.correctness_gated is True:
                    #         # If the completion doesn't meet the correctness threshold, set diversity advantage to 0
                    #         correctness_rewards = generation_batch_per_adapter[adapter_name]["correctness_rewards"]
                    #         mask = correctness_rewards < self.correctness_threshold
                    #         # Apply the mask to diversity advantages
                    #         generation_batch_per_adapter[adapter_name]["diversity_advantages"][mask] = 0.0

                    #     generation_batch_per_adapter[adapter_name]["advantages"] = self.args.correctness_weight_specialist * generation_batch_per_adapter[adapter_name]["correctness_advantages"] + \
                    #                                                     self.args.diversity_weight_specialist * generation_batch_per_adapter[adapter_name]["diversity_advantages"]


                import json 
                save_data = []

                for adapter_name in self.all_adapter_names:
                    adapter_data = pass_data_per_adapter[adapter_name]
                    gen_batch = generation_batch_per_adapter.get(adapter_name, {})
                    
                    prompts = adapter_data["prompts"]
                    completions = adapter_data["completions"]
                    
                    # advantages 가져오기 (있으면)
                    advantages = gen_batch.get("advantages", [None] * len(completions))
                    diversity_advantages = gen_batch.get("diversity_advantages", [None] * len(completions))
                    
                    # Get rewards and answer info
                    correctness_rewards = gen_batch.get("correctness_reward_per_sample", [None] * len(completions))
                    answer_info_list = gen_batch.get("answer_info", [None] * len(completions))
                    source_metadata_list = gen_batch.get("source_trace_metadata", [None] * len(completions))

                    # Convert tensors to lists if necessary
                    if advantages is not None:
                        advantages = advantages.tolist() if hasattr(advantages, 'tolist') else list(advantages)
                    else:
                        advantages = [None] * len(completions)
                        
                    if diversity_advantages is not None:
                        diversity_advantages = diversity_advantages.tolist() if hasattr(diversity_advantages, 'tolist') else list(diversity_advantages)
                    else:
                        diversity_advantages = [None] * len(completions)

                    if correctness_rewards is not None:
                        correctness_rewards = correctness_rewards.tolist() if hasattr(correctness_rewards, 'tolist') else list(correctness_rewards)
                    else:
                        correctness_rewards = [None] * len(completions)

                    # Ensure answer_info_list/source_metadata_list are lists
                    if answer_info_list is None:
                        answer_info_list = [None] * len(completions)
                    if source_metadata_list is None:
                        source_metadata_list = [None] * len(completions)

                    # if adapter_name == "default":
                    #     num_to_save = self.args.num_completion_main_adapter
                    #     prompts = prompts[:num_to_save]
                    #     completions = completions[:num_to_save]
                    #     advantages = advantages[:num_to_save] if advantages else [None] * num_to_save
                    #     diversity_advantages = diversity_advantages[:num_to_save] if diversity_advantages else [None] * num_to_save
                    
                    # adapter_entry = {
                    #     "adapter_name": adapter_name,
                    #     "completions": [
                    #         {
                    #             "index": i,
                    #             "prompt": prompt,
                    #             "completion": completion,
                    #             "advantage": adv.item() if hasattr(adv, 'item') else adv,
                    #             "diversity_advantage": div_adv.item() if hasattr(div_adv, 'item') else div_adv,
                    #         }
                    #         for i, (prompt, completion, adv, div_adv) in enumerate(zip(prompts, completions, advantages, diversity_advantages))
                    #     ]
                    # }

                    adapter_entry = {
                        "adapter_name": adapter_name,
                        "completions": []
                    }

                    import numpy as np

                    for i, (prompt, completion, adv, div_adv, corr_reward, answer_info, source_meta) in enumerate(
                        zip(prompts, completions, advantages, diversity_advantages, correctness_rewards, answer_info_list, source_metadata_list)
                    ):
                        # Handle completion format
                        if isinstance(completion, list) and len(completion) > 0:
                            completion_text = completion[0]["content"] if isinstance(completion[0], dict) else str(completion[0])
                        else:
                            completion_text = str(completion)
                        
                        entry = {
                            "index": i,
                            "prompt": prompt,
                            "completion": completion_text,
                            "advantage": float(adv) if adv is not None and not (isinstance(adv, float) and np.isnan(adv)) else None,
                            "diversity_advantage": float(div_adv) if div_adv is not None and not (isinstance(div_adv, float) and np.isnan(div_adv)) else None,
                            "correctness_reward": float(corr_reward) if corr_reward is not None and not (isinstance(corr_reward, float) and np.isnan(corr_reward)) else None,
                        }

                        if isinstance(source_meta, dict):
                            entry["source_trace_metadata"] = source_meta
                            for meta_key in (
                                "sample_id",
                                "source_adapter_name",
                                "requested_source_adapter",
                                "verified_generation_policy",
                                "vllm_lora_local_path",
                                "expected_adapter_manifest",
                                "exported_adapter_manifest",
                                "adapter_hash_match",
                                "planned_update_scope",
                                "planned_advantage_source",
                                "planned_advantage_source_verified",
                                "expected_update_scope",
                                "update_scope",
                                "update_scope_ok",
                                "source_owned_update_required",
                                "source_owned_update_ok",
                                "raw_source_owned_update_ok",
                                "advantage_source",
                                "advantage_source_verified",
                            ):
                                if meta_key in source_meta:
                                    entry[meta_key] = source_meta[meta_key]
                        else:
                            entry["source_trace_metadata"] = None

                        # Add answer info if available
                        if answer_info is not None and isinstance(answer_info, dict):
                            entry["extracted_answer"] = answer_info.get("extracted_answer")
                            entry["gold_answer"] = answer_info.get("gold_answer")
                            entry["is_correct"] = answer_info.get("is_correct")
                            entry["has_think_tag"] = answer_info.get("has_think_tag")
                        else:
                            entry["extracted_answer"] = None
                            entry["gold_answer"] = None
                            entry["is_correct"] = None
                            entry["has_think_tag"] = None
                            
                        adapter_entry["completions"].append(entry)
                        
                    save_data.append(adapter_entry)

                save_path = os.path.join(self.completion_store_path, f"completions_step{self._step}.json")

                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=4, ensure_ascii=False)

                # Keys added by _annotate_and_verify_update_semantics that are plain
                # Python scalars (str / bool). Both shuffle_sequence_dict and
                # split_tensor_dict only understand tensors and lists, so we must
                # pop these keys out before calling either utility and restore them
                # afterwards.
                _SCALAR_META_KEYS = (
                    "expected_update_scope",
                    "update_scope",
                    "advantage_source",
                    "advantage_source_verified",
                )

                for adapter_name in self.all_adapter_names:
                    generation_batch_per_adapter[adapter_name] = split_pixel_values_by_grid(generation_batch_per_adapter[adapter_name])
                    if self.loss_type not in {"dmpo", "pure_dmpo"}:
                        generation_batch_per_adapter[adapter_name] = shuffle_sequence_dict(generation_batch_per_adapter[adapter_name])
                    
                    batch = split_pixel_values_by_grid(generation_batch_per_adapter[adapter_name])
                    saved_meta = {k: batch.pop(k) for k in _SCALAR_META_KEYS if k in batch}
                    if self.loss_type not in {"dmpo", "pure_dmpo"}:
                        batch = shuffle_sequence_dict(batch)
                    batch.update(saved_meta)
                    generation_batch_per_adapter[adapter_name] = batch

                # generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                # self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]

                buffered_per_adapter = {}
                for adapter_name in self.all_adapter_names:
                    adapter_batch = generation_batch_per_adapter[adapter_name]
                    # 각 adapter의 배치를 steps_per_generation 개로 split
                    saved_meta = {k: adapter_batch.pop(k) for k in _SCALAR_META_KEYS if k in adapter_batch}
                    split_batches = split_tensor_dict(adapter_batch, self.args.steps_per_generation)
                    for split_batch in split_batches:
                        split_batch.update(saved_meta)
                    split_batches = [unsplit_pixel_values_by_grid(batch) for batch in split_batches]
                    buffered_per_adapter[adapter_name] = split_batches
                self._buffered_inputs = []
                for step_index in range(self.args.steps_per_generation):
                    step_dict = {
                        adapter_name: buffered_per_adapter[adapter_name][step_index]
                        for adapter_name in self.all_adapter_names
                    }
                    self._buffered_inputs.append(step_dict)
                    
                # generation_batch = split_pixel_values_by_grid(generation_batch)
                # generation_batch = shuffle_sequence_dict(generation_batch)
                # generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                # self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
                
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    # [NEED REVIEW] From Here to ...
    @torch.no_grad()
    def _mean_completion_logp_under_adapter(
        self,
        adapter_name: str,
        prompt_completion_ids: torch.Tensor,   # (N, P+C)
        attention_mask: torch.Tensor,          # (N, P+C)
        completion_mask: torch.Tensor,         # (N, C)
        logits_to_keep: int,                   # C
        forward_kwargs: dict[str, Any] | None = None,
        num_images=None,
    ) -> torch.Tensor:
        """
        Returns mean log-prob of completion tokens under adapter_name.
        Shape: (N,)
        """
        forward_kwargs = forward_kwargs or {}

        # switch adapter
        self._set_adapter_with_logging(self.model, adapter_name, "policy_repulsion_score")
        self.enable_all_lora_grads(self.model)
        
        # breakpoint()

        # compute per-token logps for completion tokens (N, C)
        bs = getattr(self.args, "policy_repulsion_batch_size", 1)
        # breakpoint()
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            self.model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            batch_size=bs,
            compute_entropy=False,
            num_images=num_images,
            **forward_kwargs,
        )
        # breakpoint()

        denom = completion_mask.sum(-1).clamp(min=1.0)
        seq_logp = (per_token_logps * completion_mask).sum(-1) / denom
        return seq_logp


    # @torch.no_grad()
    # def _policy_repulsion_reward(
    #     self,
    #     source_adapter: str,
    #     prompt_completion_ids: torch.Tensor,
    #     attention_mask: torch.Tensor,
    #     completion_mask: torch.Tensor,
    #     logits_to_keep: int,
    #     forward_kwargs: dict[str, Any] | None = None,
    #     num_images=None,
    # ) -> torch.Tensor:
    #     """
    #     Computes policy-space repulsion reward for samples generated by source_adapter.
    #     Returns: rewards shape (N,)
    #     """
    #     reward_type = getattr(self.args, "diversity_reward_type", "external")
    #     target = getattr(self.args, "policy_repulsion_target", "all_other")

    #     # decide which other adapters to compare against
    #     if target == "default_only":
    #         other_adapters = ["default"] if source_adapter != "default" else []
    #     elif target == "all_other":
    #         other_adapters = [a for a in self.all_adapter_names if a != source_adapter]
    #     else:
    #         raise ValueError(f"Unknown policy_repulsion_target: {target}")

    #     if len(other_adapters) == 0:
    #         return torch.zeros(prompt_completion_ids.size(0), device=prompt_completion_ids.device)

    #     # margin variant needs source logp once
    #     if reward_type == "policy_repulsion_margin":
    #         src_logp = self._mean_completion_logp_under_adapter(
    #             source_adapter,
    #             prompt_completion_ids,
    #             attention_mask,
    #             completion_mask,
    #             logits_to_keep,
    #             forward_kwargs=forward_kwargs,
    #             num_images=num_images,
    #         )
    #     else:
    #         src_logp = None

    #     acc = torch.zeros(prompt_completion_ids.size(0), device=prompt_completion_ids.device)
    #     for b in other_adapters:
    #         b_logp = self._mean_completion_logp_under_adapter(
    #             b,
    #             prompt_completion_ids,
    #             attention_mask,
    #             completion_mask,
    #             logits_to_keep,
    #             forward_kwargs=forward_kwargs,
    #             num_images=num_images,
    #         )
    #         if reward_type == "policy_repulsion_nll":
    #             acc += (-b_logp)  # surprisal under other adapter
    #         elif reward_type == "policy_repulsion_margin":
    #             acc += (src_logp - b_logp)
    #         else:
    #             raise ValueError(f"Unknown diversity_reward_type: {reward_type}")

    #     return acc / len(other_adapters)
    # # ... To Here [NEED REVIEW]

    @torch.no_grad()
    def _policy_repulsion_reward(
        self,
        source_adapter: str,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        logits_to_keep: int,
        forward_kwargs: dict[str, Any] | None = None,
        num_images=None,
    ) -> torch.Tensor:
        forward_kwargs = forward_kwargs or {}

        reward_type = getattr(self.args, "diversity_reward_type", "external")
        target = getattr(self.args, "policy_repulsion_target", "all_other")

        # ---- prefix-only mask ----
        prefix_len = int(getattr(self.args, "policy_repulsion_prefix_len", 0) or 0)
        rep_mask = completion_mask
        if prefix_len > 0 and prefix_len < rep_mask.size(1):
            rep_mask = rep_mask.clone()
            rep_mask[:, prefix_len:] = 0

        # other adapters
        if target == "default_only":
            other_adapters = ["default"] if source_adapter != "default" else []
        elif target == "all_other":
            other_adapters = [a for a in self.all_adapter_names if a != source_adapter]
        else:
            raise ValueError(f"Unknown policy_repulsion_target: {target}")

        if len(other_adapters) == 0:
            return torch.zeros(prompt_completion_ids.size(0), device=prompt_completion_ids.device)

        # src logp
        src_logp = self._mean_completion_logp_under_adapter(
            source_adapter,
            prompt_completion_ids,
            attention_mask,
            rep_mask,
            logits_to_keep,
            forward_kwargs=forward_kwargs,
            num_images=num_images,
        )

        # other logps
        other_logps = []
        for b in other_adapters:
            b_logp = self._mean_completion_logp_under_adapter(
                b,
                prompt_completion_ids,
                attention_mask,
                rep_mask,
                logits_to_keep,
                forward_kwargs=forward_kwargs,
                num_images=num_images,
            )
            other_logps.append(b_logp)

        aggregation = getattr(self.args, "policy_repulsion_aggregation", "max")
        stacked = torch.stack(other_logps, dim=0)  # Shape (num_other_adapters, N)
        
        if aggregation == "max":
            b_logp_agg = stacked.max(dim=0).values          # closest (most similar) adapter
        elif aggregation == "mean":
            b_logp_agg = stacked.mean(dim=0)                 # average of all other adapters
        else:
            raise ValueError(
                f"Unknown policy_repulsion_aggregation: '{aggregation}'. "
                "Must be 'max' or 'mean'."
            )

        gap = src_logp - b_logp_agg  # (N,)

        # barrier params (define BEFORE debug)
        m = float(getattr(self.args, "policy_repulsion_barrier_margin", 0.0) or 0.0)
        tau = float(getattr(self.args, "policy_repulsion_barrier_tau", 0.0) or 0.0)

        # debug (now safe)
        self._repulsion_debug = {
            "src_logp": src_logp.detach(),
            "b_logp_agg": b_logp_agg.detach(),
            "gap": gap.detach(),
            "prefix_len": prefix_len,
            "m": m,
            "tau": tau,
        }
        if m > 0.0:
            self._repulsion_debug["barrier_active"] = (gap < m).float().detach()

        # apply reward type
        if reward_type == "policy_repulsion_margin_barrier":
            if m > 0.0:
                if tau and tau > 0.0:
                    reward = -F.softplus((m - gap) / tau)
                else:
                    reward = -torch.relu(m - gap)
            else:
                reward = gap
            return reward

        # plain margin
        return gap

    @profiling_decorator
    def _calculate_diversity_rewards(self, inputs, prompts, completions, completion_ids_list, other_completions, other_completions_ids_list):
        # breakpoint()
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.diversity_reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.diversity_reward_funcs, self.diversity_reward_processing_classes, self.diversity_reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # breakpoint()
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, other_completions=other_completions, other_completions_ids=other_completions_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            # breakpoint()
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # breakpoint()
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func
    
    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        all_answer_info = [None] * len(prompts)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        def _to_float_list(x):
            # x: list/tuple/torch.Tensor -> list of python floats or None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().tolist()
            return list(x)

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)

                    continue

                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )

                # else:
                #     output_reward_func = reward_func(
                #         prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                #     )
                #     # Convert None values to NaN
                #     output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                #     rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            if isinstance(output_reward_func, tuple) and len(output_reward_func) == 2:
                # reward_values, all_answer_info = output_reward_func
                # if all_answer_info[0] is None:
                #     all_answer_info = answer_info_from_func
                # else:
                #     reward_values = output_reward_func
                reward_list, info_list = output_reward_func
                reward_list = _to_float_list(reward_list)

                if info_list is not None:
                    info_list = list(info_list)
                    if len(info_list) != len(prompts):
                        raise ValueError(
                            f"Length of answer info list ({len(info_list)}) does not match number of prompts ({len(prompts)})."
                        )
                    for j, info in enumerate(info_list):
                        if info is not None:
                            all_answer_info[j] = info
                else:
                    reward_list = _to_float_list(output_reward_func)


                if len(reward_list) != len(prompts):
                    raise ValueError(
                        f"Length of reward list ({len(reward_list)}) does not match number of prompts ({len(prompts)})."
                    )

                cleaned = []
                for reward in reward_list:
                    if reward is None:
                        cleaned.append(torch.nan)
                    elif isinstance(reward, torch.Tensor):
                        cleaned.append(float(reward.item()))
                    else:
                        cleaned.append(float(reward))
                
                rewards_per_func[:, i] = torch.tensor(cleaned, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func, all_answer_info

    def _merge_system_prompt(self, messages: list) -> list:
        if not messages or messages[0].get("role") != "system":
            return messages
        system_content = messages[0]["content"]
        merged = list(messages[1:])
        if merged and merged[0].get("role") == "user":
            merged[0] = {**merged[0], "content": system_content + "\n\n" + merged[0]["content"]}
        else:
            merged.insert(0, {"role": "user", "content": system_content})
        return merged

    def check_for_vllm_sleep(self):
        if not (self.use_vllm and self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode):
            return
        if self._vllm_slept:
            return

        torch.cuda.synchronize()
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

        # self.llm.sleep(level=2)
        self.llm.sleep(level=1) #, mode="wait")  # sleep weights but keep kv cache for faster wakeup, since we will wake up in the next iteration anyway
        torch.cuda.synchronize()
        self._vllm_slept = True
            
    def check_for_vllm_wake(self):
        if not (self.use_vllm and self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode):
            return
        if not self._vllm_slept:
            return

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        self.llm.wake_up() 
        # self.llm.wake_up(tags=["weights"])
        # self.llm.collective_rpc("reload_weights")
        # self.llm.wake_up(tags=["kv_cache"])
        self.llm.reset_prefix_cache()

        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

        self._vllm_slept = False
    
    def _generate_single_turn(self, prompts: list, adapter_name: Optional[str] = None):
        device = self.accelerator.device

        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                self.check_for_vllm_wake()
                # self.llm.wake_up(tags=["weights"])

            adapter_key = adapter_name or "__base__"
            if self.state.global_step != self._last_loaded_step_per_adapter.get(adapter_key):
                if adapter_name:
                    self._move_model_to_vllm(adapter_name=adapter_name)
                else:
                    self._move_model_to_vllm()
                self._last_loaded_step_per_adapter[adapter_key] = self.state.global_step

            lora_request = None
            if adapter_name and self.lora_modules:
                adapter_info = next(
                    (m for m in self.lora_modules if m["name"] == adapter_name), None
                )
                if adapter_info:
                    lora_request = LoRARequest(
                        lora_name=adapter_info["name"],
                        lora_int_id=adapter_info["id"],
                        lora_local_path=adapter_info["path"],
                    )

            if is_conversational({"prompt": prompts[0]}):
                prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts = gather_object(prompts)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts[:: self.num_generations]

                    sampling_params = {
                        "n": self.num_generations,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "truncate_prompt_tokens": self.max_prompt_length,
                        "guided_decoding_regex": self.guided_decoding_regex,
                        "generation_kwargs": self.args.generation_kwargs,
                    }

                    with profiling_context(self, "vLLM.generate"):
                        if self.rollout_func is not None:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                ordered_set_of_prompts = [
                                    apply_chat_template(
                                        {"prompt": p}, self.processing_class, **self.chat_template_kwargs
                                    )["prompt"]
                                    for p in ordered_set_of_prompts
                                ]
                            output = self.rollout_func(
                                ordered_set_of_prompts,
                                self.args,
                                self.processing_class,
                                lora_request=all_lora_request,
                            )
                        else:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                # FIXME: this endpoint doesn't exist in vllm_client
                                output = self.vllm_client.chat(
                                    prompts=ordered_set_of_prompts,
                                    **sampling_params,
                                    chat_template_kwargs=self.chat_template_kwargs,
                                    lora_request=lora_request,
                                )
                            else:
                                output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params, lora_request=lora_request)
                        # Extract required fields and collect any extra fields for reward functions
                        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                        extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

                # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
                extra_fields = {}
                for key, values in all_extra_fields.items():
                    if isinstance(values, list):
                        extra_fields[key] = values[process_slice]
                    else:
                        extra_fields[key] = values

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                # breakpoint()
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                lora_request = None
                if adapter_name and self.lora_modules:
                    adapter_info = next(
                        (module for module in self.lora_modules if module["name"] == adapter_name), None
                    )
                    if adapter_info:
                        lora_request = LoRARequest(
                            lora_name=adapter_info["name"],
                            lora_int_id = adapter_info["id"],
                            lora_local_path = adapter_info["path"],
                        )
                        hash_status = self._vllm_lora_hash_results.get(adapter_name, {})
                        print(
                            f"  >> Using LoRA adapter in vLLM x: {adapter_name} "
                            f"(id: {adapter_info['id']}, path: {adapter_info['path']}, "
                            f"hash_status: {hash_status.get('status', 'UNVERIFIED')})"
                        )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                    all_prompts = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts = prompts
                    
                all_prompts = [self._merge_system_prompt(p) for p in all_prompts]

                # if self.args.vllm_enable_sleep_mode:
                #     self.llm.wake_up(tags=["kv_cache"])

                # breakpoint()

                with profiling_context(self, "vLLM.generate"):
                    # breakpoint()
                    if is_conversational({"prompt": prompts[0]}):
                        all_outputs = self.llm.chat(all_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)
                    else:
                        all_outputs = self.llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                # [DEBUG]
                if self.accelerator.is_main_process:
                    print(f"\n[DEBUG _generate_single_turn] adapter={adapter_name}")
                    print(f"  num prompts: {len(all_outputs)}")
                    
                    raw_prompt_ids = all_outputs[0].prompt_token_ids
                    raw_completion_ids = all_outputs[0].outputs[0].token_ids
                    
                    print(f"  [0] prompt_ids[:10]: {raw_prompt_ids[:10]}")
                    print(f"  [0] prompt_ids[-10:]: {raw_prompt_ids[-10:]}")
                    print(f"  [0] prompt decoded: {self.processing_class.decode(raw_prompt_ids, skip_special_tokens=False)[:300]}")
                    print(f"  [0] completion_ids[:20]: {list(raw_completion_ids[:20])}")
                    print(f"  [0] completion decoded (raw): {self.processing_class.decode(raw_completion_ids, skip_special_tokens=False)[:300]}")
                    print(f"  [0] completion decoded (skip_special): {self.processing_class.decode(raw_completion_ids, skip_special_tokens=True)[:300]}")

                    # breakpoint()

                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                extra_fields = {}  # No extra fields for colocate mode

                # if self.args.vllm_enable_sleep_mode:
                #     self.llm.sleep(level=2)
                #     self._vllm_slept = True
                    
                if getattr(self.args, "clear_KV_cache_after_generation", False):
                    try:
                        self.llm.reset_prefix_cache()
                        # torch.cuda.empty_cache()
                        if self.accelerator.is_main_process:
                            print(f"  >> Cleared vLLM cache after generating with adapter: {adapter_name}")
                    except Exception as e:
                        logger.warning(f"Failed to clear vLLM cache: {e}")
                        
        elif self.use_transformers_paged:
            processor_kwargs = {
                "max_length": self.max_prompt_length,
                "truncation": True,
                "add_special_tokens": False,
            }
            if is_conversational({"prompt": prompts[0]}):
                processor_outputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                processor_outputs = self.processing_class(text=prompts, **processor_kwargs)

            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    self._set_adapter_with_logging(unwrapped_model, adapter_name, "transformers_paged_generate")
                    self.enable_all_lora_grads(unwrapped_model)
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"], generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for paged mode

        else:
            # Regular generation path
            processor_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "padding_side": "left",
                "max_length": self.max_prompt_length,
                "truncation": True,
                "add_special_tokens": False,
            }
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(text=prompts, **processor_kwargs)
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                self._set_adapter_with_logging(unwrapped_model, adapter_name, "transformers_generate")
                self.enable_all_lora_grads(unwrapped_model)
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_index = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_index[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_index.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for non-rollout_func paths

        return prompt_ids, completion_ids, logprobs, extra_fields
    
    # def _generate_single_turn(self, prompts: list, adapter_name: str):
    #     device = self.accelerator.device

    #     # Generate completions using either vLLM or regular generation
    #     if self.use_vllm:
    #         if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode: #vllm lorarequest 수정 검토
    #             # wake up colocated vLLM instances if needed
    #             torch.cuda.empty_cache()  # required to avoid OOM in some cases
    #             self.llm.wake_up(tags=["weights"])

    #         # First, update the vLLM weights if needed
    #         if self.state.global_step != self._last_loaded_step:
    #             if adapter_name is None:
    #                 self._move_model_to_vllm()
    #                 self._last_loaded_step = self.state.global_step
    #                 lora_req = None
    #             else:
    #                 self._move_model_to_vllm(adapter_name=adapter_name)
    #                 self._last_loaded_step = self.state.global_step
                    
    #         lora_req = None
    #         if adapter_name and self.lora_modules:
    #             adapter_info = next(
    #                 (module for module in self.lora_modules if module["name"] == adapter_name), None
    #             )
    #             if adapter_info:
    #                 lora_req = LoRARequest(
    #                     lora_name=adapter_info["name"],
    #                     lora_int_id = adapter_info["id"],
    #                     lora_local_path = adapter_info["path"],
    #                 )
    #                 print(f"  >> Using LoRA adapter in vLLM generation: {adapter_name} (id: {adapter_info['id']})")

    #         outputs = self.llm.generate(
    #             prompts=prompts,
    #             sampling_params=self.sampling_params,
    #             lora_request=lora_req
    #         )
    #         if is_conversational({"prompt": prompts[0]}):
    #             prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

    #         # Not gonna use 'server' mode -> Don't need to modify anymore
    #         # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
    #         if self.vllm_mode == "server":
    #             raise NotImplementedError("vLLM server mode is not implemented in this version.")
    #             all_prompts = gather_object(prompts)

    #             if self.accelerator.is_main_process:
    #                 # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
    #                 # num_generations outputs for each one. This is faster than generating outputs for each duplicate
    #                 # prompt individually.
    #                 ordered_set_of_prompts = all_prompts[:: self.num_generations]

    #                 sampling_params = {
    #                     "n": self.num_generations,
    #                     "repetition_penalty": self.repetition_penalty,
    #                     "temperature": self.temperature,
    #                     "top_p": self.top_p,
    #                     "top_k": -1 if self.top_k is None else self.top_k,
    #                     "min_p": 0.0 if self.min_p is None else self.min_p,
    #                     "max_tokens": self.max_completion_length,
    #                     "truncate_prompt_tokens": self.max_prompt_length,
    #                     "guided_decoding_regex": self.guided_decoding_regex,
    #                     "generation_kwargs": self.args.generation_kwargs,
    #                 }
    #                 with profiling_context(self, "vLLM.generate"):
    #                     if self.rollout_func is not None:
    #                         if is_conversational({"prompt": ordered_set_of_prompts[0]}):
    #                             ordered_set_of_prompts = [
    #                                 apply_chat_template(
    #                                     {"prompt": p}, self.processing_class, **self.chat_template_kwargs
    #                                 )["prompt"]
    #                                 for p in ordered_set_of_prompts
    #                             ]
    #                         output = self.rollout_func(
    #                             ordered_set_of_prompts,
    #                             self.args,
    #                             self.processing_class,
    #                         )
    #                     else:
    #                         if is_conversational({"prompt": ordered_set_of_prompts[0]}):
    #                             output = self.vllm_client.chat(
    #                                 messages=ordered_set_of_prompts,
    #                                 **sampling_params,
    #                                 chat_template_kwargs=self.chat_template_kwargs,
    #                             )
    #                         else:
    #                             output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params)
    #                     # Extract required fields and collect any extra fields for reward functions
    #                     required_keys = {"prompt_ids", "completion_ids", "logprobs"}
    #                     extra_fields = {k: v for k, v in output.items() if k not in required_keys}
    #                     payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
    #             else:
    #                 payload = None

    #             # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
    #             obj_list = [payload]
    #             broadcast_object_list(obj_list, from_process=0)
    #             all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

    #             # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
    #             all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

    #             process_slice = slice(
    #                 self.accelerator.process_index * len(prompts),
    #                 (self.accelerator.process_index + 1) * len(prompts),
    #             )
    #             prompt_ids = all_prompt_ids[process_slice]
    #             completion_ids = all_completion_ids[process_slice]
    #             logprobs = all_logprobs[process_slice]

    #             # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
    #             extra_fields = {}
    #             for key, values in all_extra_fields.items():
    #                 if isinstance(values, list):
    #                     extra_fields[key] = values[process_slice]
    #                 else:
    #                     extra_fields[key] = values

    #         # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
    #         elif self.vllm_mode == "colocate":
    #             if self.guided_decoding_regex:
    #                 guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
    #             else:
    #                 guided_decoding = None

    #             generation_kwargs = {
    #                 "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
    #                 "repetition_penalty": self.repetition_penalty,
    #                 "temperature": self.temperature,
    #                 "top_p": self.top_p,
    #                 "top_k": -1 if self.top_k is None else self.top_k,
    #                 "min_p": 0.0 if self.min_p is None else self.min_p,
    #                 "max_tokens": self.max_completion_length,
    #                 "truncate_prompt_tokens": self.max_prompt_length,
    #                 "guided_decoding": guided_decoding,
    #                 "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
    #             }
    #             if self.args.generation_kwargs is not None:
    #                 generation_kwargs.update(self.args.generation_kwargs)
    #             sampling_params = SamplingParams(**generation_kwargs)

    #             if self.vllm_tensor_parallel_size > 1:
    #                 # Gather prompts from all ranks in the TP group and flatten.
    #                 # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
    #                 orig_size = len(prompts)
    #                 gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
    #                 torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
    #                 all_prompts = [p for sublist in gathered_prompts for p in sublist]
    #             else:
    #                 all_prompts = prompts

    #             if self.args.vllm_enable_sleep_mode:
    #                 self.llm.wake_up(tags=["kv_cache"])

    #             with profiling_context(self, "vLLM.generate"):
    #                 if is_conversational({"prompt": prompts[0]}):
    #                     all_outputs = self.llm.chat(all_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_req)
    #                 else:
    #                     all_outputs = self.llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_req)

    #             all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
    #             all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
    #             all_logprobs = [
    #                 [next(iter(lp.values())).logprob for lp in output.logprobs]
    #                 for outputs in all_outputs
    #                 for output in outputs.outputs
    #             ]

    #             if self.vllm_tensor_parallel_size > 1:
    #                 # Slice completions for this rank within its TP group.
    #                 # Each rank generates all outputs — we keep only our share.
    #                 local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
    #                 tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
    #                 prompt_ids = all_prompt_ids[tp_slice]
    #                 completion_ids = all_completion_ids[tp_slice]
    #                 logprobs = all_logprobs[tp_slice]
    #             else:
    #                 prompt_ids = all_prompt_ids
    #                 completion_ids = all_completion_ids
    #                 logprobs = all_logprobs

    #             extra_fields = {}  # No extra fields for colocate mode

    #             if self.args.vllm_enable_sleep_mode:
    #                 self.llm.sleep(level=2)

    #     elif self.use_transformers_paged:
    #         processor_kwargs = {
    #             "max_length": self.max_prompt_length,
    #             "truncation": True,
    #             "add_special_tokens": False,
    #         }
    #         if is_conversational({"prompt": prompts[0]}):
    #             processor_outputs = self.processing_class.apply_chat_template(
    #                 conversation=prompts,
    #                 **processor_kwargs,
    #                 add_generation_prompt=True,
    #                 tokenize=True,
    #                 return_dict=True,
    #                 **self.chat_template_kwargs,
    #             )
    #         else:
    #             processor_outputs = self.processing_class(text=prompts, **processor_kwargs)

    #         with (
    #             profiling_context(self, "transformers.generate_batch"),
    #             unwrap_model_for_generation(
    #                 self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
    #             ) as unwrapped_model,
    #             torch.no_grad(),
    #             FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
    #         ):
    #             # Cast to the appropriate dtype based on training configuration
    #             if self.args.bf16:
    #                 unwrapped_model.to(torch.bfloat16)
    #             elif self.args.fp16:
    #                 unwrapped_model.to(torch.float16)
    #             if self.args.cast_lm_head_to_fp32:
    #                 unwrapped_model.lm_head.to(torch.float32)
    #             with torch.inference_mode():
    #                 # Continuous batching API expects 'inputs' arg only
    #                 all_outputs = unwrapped_model.generate_batch(
    #                     processor_outputs["input_ids"], generation_config=self.generation_config, progress_bar=False
    #                 )
    #                 unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
    #         completion_ids = [output.generated_tokens for output in all_outputs.values()]
    #         prompt_ids = processor_outputs["input_ids"]
    #         logprobs = None  # not used in this case
    #         extra_fields = {}  # No extra fields for paged mode

    #     else:
    #         # Regular generation path
    #         processor_kwargs = {
    #             "return_tensors": "pt",
    #             "padding": True,
    #             "padding_side": "left",
    #             "max_length": self.max_prompt_length,
    #             "truncation": True,
    #             "add_special_tokens": False,
    #         }
    #         if is_conversational({"prompt": prompts[0]}):
    #             generate_inputs = self.processing_class.apply_chat_template(
    #                 conversation=prompts,
    #                 **processor_kwargs,
    #                 add_generation_prompt=True,
    #                 tokenize=True,
    #                 return_dict=True,
    #                 **self.chat_template_kwargs,
    #             )
    #         else:
    #             generate_inputs = self.processing_class(text=prompts, **processor_kwargs)
    #         generate_inputs = super()._prepare_inputs(generate_inputs)

    #         with (
    #             profiling_context(self, "transformers.generate"),
    #             unwrap_model_for_generation(
    #                 self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
    #             ) as unwrapped_model,
    #             torch.no_grad(),
    #             FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
    #         ):
    #             unwrapped_model.set_adapter(adapter_name)  # ensure no adapter is set during generation
    #             prompt_completion_ids = unwrapped_model.generate(
    #                 **generate_inputs, generation_config=self.generation_config, disable_compile=True
    #             )
    #         # Compute prompt length and extract completion ids
    #         prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
    #         prompt_length = prompt_ids.size(1)
    #         completion_ids = prompt_completion_ids[:, prompt_length:]

    #         # Mask everything after the first EOS token
    #         is_eos = completion_ids == self.eos_token_id
    #         eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #         eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #         sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #         completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    #         prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
    #         completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
    #         logprobs = None  # not used in this case
    #         extra_fields = {}  # No extra fields for non-rollout_func paths

    #     return prompt_ids, completion_ids, logprobs, extra_fields

    def _generate(self, prompts: list, adapter_name: str):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # with torch.amp.autocast("cuda"):
        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts, adapter_name)

        if self.accelerator.is_main_process:
            print(f"\n[DEBUG _generate] adapter={adapter_name}")
            print(f"  prompt_ids[0] (list, len={len(prompt_ids[0])}): {prompt_ids[0][:10]}...{prompt_ids[0][-5:]}")
            print(f"  completion_ids[0] (list, len={len(completion_ids[0])}): {completion_ids[0][:20]}")
            print(f"  prompt decoded[0]: {self.processing_class.decode(prompt_ids[0], skip_special_tokens=False)[:200]}")
            print(f"  completion decoded[0] (skip_special=False): {self.processing_class.decode(completion_ids[0], skip_special_tokens=False)[:300]}")
            print(f"  completion decoded[0] (skip_special=True):  {self.processing_class.decode(completion_ids[0], skip_special_tokens=True)[:300]}")
            # breakpoint()

        # breakpoint()

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, extra_fields

    @profiling_decorator
    def _generate_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]], adapter_to_indices: dict[str, list[int]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # breakpoint()
        prompts = [x["prompt"] for x in inputs]
        # breakpoint()

        """
        (Pdb) if self.accelerator.is_main_process: print ("True") 
        True
        (Pdb) print(f"\n[DEBUG prompt input] adapter_to_indices={adapter_to_indices}")

        [DEBUG prompt input] adapter_to_indices={'default': [0, 1, 2, 3], 'diversity_0': [4, 5, 6], 'diversity_1': [7, 8, 9]}
        (Pdb) print(f"  inputs[0]['prompt']: {inputs[0]['prompt']}")
        inputs[0]['prompt']: [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{}. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages.', 'role': 'system'}, {'content': 'A triple of positive integers  $(a,b,c)$  is *brazilian* if   $$ a|bc+1 $$    $$ b|ac+1 $$    $$ c|ab+1 $$  Determine all the brazilian triples.', 'role': 'user'}]
        (Pdb) print(f"  is_conversational: {is_conversational(inputs[0])}")
        is_conversational: True
        (Pdb) print(f"  total inputs: {len(inputs)}")
        total inputs: 10
        """

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        # prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
        #     self._generate(prompts)
        # )
        
        # 1. 전체 데이터 크기 계산 (Batch Size * Sum of generations)
        total_items = len(prompts)  # 혹은 sum(all_adapter_num_completions) * batch_size

        # 2. 결과를 담을 빈 리스트(Placeholder) 생성 (원래 순서 유지를 위함)
        # 크기가 정해져 있으므로 None으로 미리 채워둡니다.
        final_prompt_ids_list = [None] * total_items
        final_completion_ids_list = [None] * total_items
        final_sampling_per_token_logps_list = [None] * total_items
        final_num_items_in_batch = [None] * total_items
        final_source_trace_metadata = [None] * total_items

        if self.use_vllm and self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
            self.check_for_vllm_wake()

        # 3. 어댑터별로 순회하며 생성 및 제자리 채워넣기
        for adapter_name, indices in adapter_to_indices.items():
            # breakpoint()
            # # 해당 어댑터에 필요한 프롬프트만 골라내기
            # for idx in indices:
            #     selected_prompts = [prompts[idx]]
                
            #     # 해당 어댑터로 생성 실행
            #     (
            #         sub_prompt_ids,
            #         sub_completion_ids,
            #         sub_num_items,
            #         sub_logps,
            #         sub_extra
            #     ) = self._generate(selected_prompts, adapter_name)
            #     # breakpoint()
                
            #     if len(sub_extra) > 0:
            #         raise NotImplementedError("Handling extra fields per adapter is not implemented in this version.")
                
            #     # 4. [중요] 생성된 결과를 원래 인덱스 위치(indices)에 매핑하여 저장
            #     # sub_results는 0부터 순서대로 나오지만, 실제 위치는 indices[k] 입니다.
            #     for local_idx, original_idx in enumerate([idx]):
            #         # breakpoint()
            #         final_prompt_ids_list[original_idx] = sub_prompt_ids[local_idx]
            #         final_completion_ids_list[original_idx] = sub_completion_ids[local_idx]
            #         if sub_logps is not None:
            #             final_sampling_per_token_logps_list[original_idx] = sub_logps[local_idx]
            #         final_num_items_in_batch[original_idx] = sub_num_items.item()
            #         # breakpoint()
            #         """
            #         (Pdb) final_num_items_in_batch
            #         [8192, None, None, None, None, None, None, None, None, None]
            #         """
                
            
            selected_prompts = [prompts[i] for i in indices]
            
            # 해당 어댑터로 생성 실행
            (
                sub_prompt_ids,
                sub_completion_ids,
                sub_num_items,
                sub_logps,
                sub_extra
            ) = self._generate(selected_prompts, adapter_name)
            # breakpoint()

            if len(sub_extra) > 0:
                raise NotImplementedError("Handling extra fields per adapter is not implemented in this version.")
            
            # 4. [중요] 생성된 결과를 원래 인덱스 위치(indices)에 매핑하여 저장
            # sub_results는 0부터 순서대로 나오지만, 실제 위치는 indices[k] 입니다.
            source_adapter_idx = self.all_adapter_names.index(adapter_name)
            for local_idx, original_idx in enumerate(indices):
                # breakpoint()
                final_prompt_ids_list[original_idx] = sub_prompt_ids[local_idx]
                final_completion_ids_list[original_idx] = sub_completion_ids[local_idx]
                local_sampling_logps = None
                if sub_logps is not None:
                    local_sampling_logps = sub_logps[local_idx]
                    final_sampling_per_token_logps_list[original_idx] = local_sampling_logps
                final_num_items_in_batch[original_idx] = sub_num_items.item()
                final_source_trace_metadata[original_idx] = self._build_source_trace_metadata(
                    inputs[original_idx],
                    adapter_name,
                    source_adapter_idx,
                    original_idx,
                    local_idx,
                    sub_completion_ids[local_idx],
                    local_sampling_logps,
                )
                # breakpoint()
                """
                (Pdb) final_num_items_in_batch
                [8192, None, None, None, None, None, None, None, None, None]
                """
        
        if self.use_vllm and self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
            self.check_for_vllm_sleep()

        # breakpoint()
        # 5. 최종 결과 변수 할당 (원래 코드와 동일한 변수명으로 복귀)
        prompt_ids_list = final_prompt_ids_list
        completion_ids_list = final_completion_ids_list
        num_items_in_batch = final_num_items_in_batch
        sampling_per_token_logps_list = None
        if sub_logps is not None:
            sampling_per_token_logps_list = final_sampling_per_token_logps_list
        extra_fields = {} # Not implemented for per-adapter extra fields
        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # breakpoint()

        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        if self.args.mini_batch_size is not None:
            batch_size = self.args.mini_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        pass_data = {
            "inputs": inputs,
            "prompts": prompts,
            "completions": completions,
            "completion_ids_list": completion_ids_list,
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
            "batch_size": batch_size,
            "num_images": num_images,
            "sampling_per_token_logps": sampling_per_token_logps,
            "source_trace_metadata": final_source_trace_metadata,
            "prompts_text": prompts_text,
            "completions_text": completions_text,
            "images": images,
        }
        
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
            "source_trace_metadata": final_source_trace_metadata,
        }
        # breakpoint()
        output_per_adapter = {}
        pass_data_per_adapter = {}
        forward_kwargs_per_adapter = {}
        for per_data, all_data in [(pass_data_per_adapter, pass_data), (output_per_adapter, output), (forward_kwargs_per_adapter, forward_kwargs)]:
            for adapter_name, indices in adapter_to_indices.items():
                
                per_data[adapter_name] = {}
                for key in all_data.keys():
                    if adapter_name == "default":
                        per_data[adapter_name][key] = all_data[key]
                        continue
                    
                    sample_val = all_data[key]
                    if sample_val is None:
                        per_data[adapter_name][key] = None
                    
                    elif isinstance(sample_val, list):
                        merged_list = []
                        for idx in indices:
                            merged_list.append(all_data[key][idx])
                        per_data[adapter_name][key] = merged_list

                    # 2) Tensor Type: 차원 변형(View) 후 병합(Cat)이 훨씬 빠름
                    elif isinstance(sample_val, torch.Tensor):
                        per_data[adapter_name][key] = all_data[key][indices]
                    
                    # 3) Int/Float (Scalar): 단순히 합산 (예: Loss 합계 등)
                    elif isinstance(sample_val, (int, float)):
                        per_data[adapter_name][key] = sample_val

                    else:
                        raise NotImplementedError(f"Data type {type(sample_val)} not supported for key {key}")
                    
                    if key == "num_items_in_batch":
                        per_data[adapter_name][key] = torch.Tensor(sum(per_data[adapter_name][key])).long() 
                            
        return output_per_adapter, pass_data_per_adapter, forward_kwargs_per_adapter

    def _log_main_correct_rate_and_group_coverage(
        self,
        full_source_trace_metadata: list[Any] | None,
        main_correct_rate_by_prompt: dict[int, float],
    ) -> None:
        """Logs main_correct_rate(x) distribution and whole-group coverage stats.

        `full_source_trace_metadata` must be the UNSLICED per-step metadata
        list (i.e. `generation_batch_per_adapter["default"]["source_trace_metadata"]`),
        which covers every row generated this step across every adapter --
        default's own correctness pass always scores the full pool, so by the
        time this is called every row already has `answer_correct_float`,
        `prompt_index`, and `source_adapter_name` populated.
        """
        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return

        rates = list(main_correct_rate_by_prompt.values())
        if rates:
            n = len(rates)
            self._metrics[mode]["train/main_correct_rate/mean"].append(sum(rates) / n)
            mean = sum(rates) / n
            variance = sum((r - mean) ** 2 for r in rates) / n
            self._metrics[mode]["train/main_correct_rate/std"].append(variance ** 0.5)

            # N_m (main rollouts per prompt) is fixed for the whole run, so the
            # possible rates are always k / N_m for k in [0, N_m]. We bucket at
            # the canonical (4/3/3)-style quarters for readability, but this is
            # purely descriptive -- rates outside these buckets are ignored by
            # a given frac_* bucket, not dropped from mean/std above.
            def frac_close(target: float) -> float:
                return sum(1 for r in rates if abs(r - target) < 1e-6) / n

            self._metrics[mode]["train/main_correct_rate/frac_0"].append(frac_close(0.0))
            self._metrics[mode]["train/main_correct_rate/frac_025"].append(frac_close(0.25))
            self._metrics[mode]["train/main_correct_rate/frac_05"].append(frac_close(0.5))
            self._metrics[mode]["train/main_correct_rate/frac_075"].append(frac_close(0.75))
            self._metrics[mode]["train/main_correct_rate/frac_1"].append(frac_close(1.0))

        if full_source_trace_metadata is not None:
            coverage = compute_group_coverage_stats(full_source_trace_metadata)
            for key, value in coverage.items():
                self._metrics[mode][f"train/group/{key}"].append(value)

    @profiling_decorator
    def _score_completions_correctness(
        self, generation_batch: dict[str, torch.Tensor | Any], pass_data, pass_forward_kwargs, adapter_name: str, num_completions: int, adapter_index: list[int]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # breakpoint()
        
        inputs = pass_data["inputs"]
        prompts = pass_data["prompts"]
        completions = pass_data["completions"]
        completion_ids_list = pass_data["completion_ids_list"]
        prompt_completion_ids = pass_data["prompt_completion_ids"]
        attention_mask = pass_data["attention_mask"]
        logits_to_keep = pass_data["logits_to_keep"]
        batch_size = pass_data["batch_size"]
        num_images = pass_data["num_images"]
        # sampling_per_token_logps = pass_data["sampling_per_token_logps"]
        sampling_per_token_logps = pass_data.get("sampling_per_token_logps", None)
        source_trace_metadata = pass_data.get("source_trace_metadata", None)
        prompts_text = pass_data["prompts_text"]
        completions_text = pass_data["completions_text"]
        images = pass_data["images"]
        forward_kwargs = pass_forward_kwargs
        
        prompt_ids = generation_batch["prompt_ids"]
        prompt_mask = generation_batch["prompt_mask"]
        completion_ids = generation_batch["completion_ids"]
        completion_mask = generation_batch["completion_mask"]
        num_items_in_batch = generation_batch["num_items_in_batch"]

        # breakpoint()

        with torch.no_grad():
            self._set_adapter_with_logging(self.model, adapter_name, "correctness_score")
            self.enable_all_lora_grads(self.model)
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            # generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            # if self.args.gradient_accumulation_steps % generate_every != 0 or (
            #     self.use_vllm and self.vllm_importance_sampling_correction
            # ):
            #     old_per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            #         self.model,
            #         prompt_completion_ids,
            #         attention_mask,
            #         logits_to_keep,
            #         batch_size=batch_size,
            #         compute_entropy=True,
            #         num_images=num_images,
            #         **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
            #     )
            #     if adapter_name == "default":
            #         completion_token_count = completion_mask[adapter_index].sum().clamp(min=1.0)
            #         def masked_batch_mean(x):
            #             return (x * completion_mask[adapter_index]).sum() / completion_token_count
            #         mean_entropy = masked_batch_mean(entropies[adapter_index])
            #         self._metrics[mode][f"{adapter_name}/gen_entropy"].append(
            #             self.accelerator.gather(mean_entropy).nanmean().item()
            #         )
                    
                    
            #         completion_token_count = completion_mask.sum().clamp(min=1.0)
            #         def masked_batch_mean2(x):
            #             return (x * completion_mask).sum() / completion_token_count
            #         mean_entropy = masked_batch_mean2(entropies)
            #         self._metrics[mode][f"{adapter_name}/all_entropy"].append(
            #             self.accelerator.gather(mean_entropy).nanmean().item()
            #         )
            #     else:
            #         completion_token_count = completion_mask.sum().clamp(min=1.0)
            #         def masked_batch_mean(x):
            #             return (x * completion_mask).sum() / completion_token_count
            #         mean_entropy = masked_batch_mean(entropies)
            #         self._metrics[mode][f"{adapter_name}/gen_entropy"].append(
            #             self.accelerator.gather(mean_entropy).nanmean().item()
            #         )           
            # else:
            #     old_per_token_logps = None
                
            use_importance_weighting = bool(getattr(self.args, "use_importance_weighting", False))
            need_old = (adapter_name == "default") and use_importance_weighting and (sampling_per_token_logps is not None)

            if need_old: #or (self.use_vllm and self.vllm_importance_sampling_correction):
                old_per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    compute_entropy=True,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention
                )
            else:
                old_per_token_logps, entropies = None, None

            importance_sampling_ratio = None

            if use_importance_weighting and adapter_name == "default" and sampling_per_token_logps is not None and old_per_token_logps is not None:
                # old_per_token_logps: log pi_old_default (Logp that evaluated by default adapter)
                # sampling_per_token_logps: log beta (vLLM logp from each adapter generation)
                # => exp(old - behaviour) = pi_old_default / beta
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)

                cap = float(getattr(self.args, "vllm_importance_sampling_cap", 0.0) or 0.0)

                if cap > 0:
                    importance_sampling_ratio = torch.clamp(importance_sampling_ratio, max=cap)
                # Key that was used to calibrate vLLM mismatch, but also can be used for calibrate for off-policy.

            # # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            # if self.use_vllm and self.vllm_importance_sampling_correction:
            #     importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
            #     importance_sampling_ratio = torch.clamp(
            #         importance_sampling_ratio, max=self.vllm_importance_sampling_cap
            #     )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None
        

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func, answer_info = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, num_completions).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_completions, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, num_completions).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(num_completions, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        local_rewards = rewards[process_slice]

        # Slice answer_info for local process
        if answer_info and answer_info[0] is not None:
            # local_answer_info = answer_info[process_slice.start:process_slice.stop]
            local_answer_info = answer_info if answer_info is not None else [None] * len(prompts)
        else:
            local_answer_info = [None] * len(prompts)

        # Pure answer correctness c(x,y) in {0,1}, independent of think/format
        # bonuses -- this is the ONLY field that should ever be used for
        # correctness gating or for verifier-observable correctness rewards
        # such as main_weak_correctness_bonus. Never threshold `local_rewards`
        # (the mixed correctness_reward scalar) for that purpose.
        answer_correct_float = torch.tensor(
            [
                float(info.get("answer_correct_float", 0.0)) if isinstance(info, dict) else 0.0
                for info in local_answer_info
            ],
            dtype=torch.float32,
            device=device,
        )
        if mode == "train":
            own_correct = answer_correct_float[adapter_index] if adapter_name == "default" else answer_correct_float
            self._metrics[mode][f"train/source/{adapter_name}/answer_correct_mean"].append(own_correct.mean().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"{adapter_name}/correctness_rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"{adapter_name}/correctness_rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode][f"{adapter_name}/correctness_reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode][f"{adapter_name}/correctness_reward_std"].append(std_rewards.mean().item())
        self._metrics[mode][f"{adapter_name}/frac_correctness_reward_zero_std"].append(is_std_zero.float().mean().item())
        
        if adapter_name == "default":
            local_rewards_per_func = rewards_per_func[process_slice]
            log_reward_per_func = local_rewards_per_func[adapter_index]
            for i, reward_func_name in enumerate(self.reward_func_names):
                log_mean_rewards = torch.nanmean(log_reward_per_func[:, i]).item()
                self._metrics[mode][f"{adapter_name}/gen_correctness_rewards/{reward_func_name}/mean"].append(log_mean_rewards)
                log_std_func_rewards = nanstd(log_reward_per_func[:, i]).item()
                self._metrics[mode][f"{adapter_name}/gen_correctness_rewards/{reward_func_name}/std"].append(log_std_func_rewards)
            # log_mean_grouped_rewards = mean_grouped_rewards[adapter_index].mean().item()
            # self._metrics[mode][f"{adapter_name}/gen_correctness_reward"].append(log_mean_grouped_rewards)
            # log_std_rewards = std_rewards[adapter_index].mean().item()
            # self._metrics[mode][f"{adapter_name}/gen_correctness_reward_std"].append(log_std_rewards)
            # log_is_std_zero = is_std_zero[adapter_index].float().mean().item()
            # self._metrics[mode][f"{adapter_name}/gen_frac_correctness_reward_zero_std"].append(log_is_std_zero)

        # Log prompt and completion texts
        self._logs[f"{adapter_name}/prompt"].extend(gather_object(prompts_text))
        self._logs[f"{adapter_name}/correctness_completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs[f"{adapter_name}/correctness_rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs[f"{adapter_name}/correctness_advantages"].extend(all_process_advantages.tolist())
        if images is not None:
            self._logs[f"{adapter_name}/images"].extend(gather_object(images))

        # if self.use_vllm and self.vllm_importance_sampling_correction:
        #     delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
        #     delta = delta[completion_mask.bool()]
        #     mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        #     max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        #     self._metrics[mode][f"{adapter_name}/sampling/sampling_logp_difference/mean"].append(
        #         self.accelerator.gather(mean_delta).mean().item()
        #     )
        #     self._metrics[mode][f"{adapter_name}/sampling/sampling_logp_difference/max"].append(
        #         self.accelerator.gather(max_delta).max().item()
        #     )

        #     flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
        #     min_importance_sampling_ratio = (
        #         torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        #     )
        #     mean_importance_sampling_ratio = (
        #         torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        #     )
        #     max_importance_sampling_ratio = (
        #         torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        #     )
        #     self._metrics[mode][f"{adapter_name}/sampling/importance_sampling_ratio/min"].append(
        #         nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
        #     )
        #     self._metrics[mode][f"{adapter_name}/sampling/importance_sampling_ratio/mean"].append(
        #         self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
        #     )
        #     self._metrics[mode][f"{adapter_name}/sampling/importance_sampling_ratio/max"].append(
        #         nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
        #     )

        if source_trace_metadata is not None:
            corr_values = local_rewards.detach().float().cpu().tolist()
            corr_adv_values = advantages.detach().float().cpu().tolist()
            correct_values = answer_correct_float.detach().float().cpu().tolist()
            for idx, record in enumerate(source_trace_metadata):
                if isinstance(record, dict):
                    record["correctness_reward"] = corr_values[idx] if idx < len(corr_values) else None
                    record["correctness_advantage"] = corr_adv_values[idx] if idx < len(corr_adv_values) else None
                    is_correct = bool(correct_values[idx] >= 1.0) if idx < len(correct_values) else None
                    record["is_correct"] = is_correct
                    record["answer_correct"] = is_correct
                    record["answer_correct_float"] = correct_values[idx] if idx < len(correct_values) else None
        if self.use_vllm and use_importance_weighting and importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)

            self._metrics[mode][f"{adapter_name}/sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode][f"{adapter_name}/sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_is = torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            mean_is = torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            max_is = torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)

            self._metrics[mode][f"{adapter_name}/sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_is)).item()
            )
            self._metrics[mode][f"{adapter_name}/sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_is).nanmean().item()
            )
            self._metrics[mode][f"{adapter_name}/sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_is)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "correctness_advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "correctness_reward_per_sample": local_rewards,
            "raw_rewards": local_rewards,
            "answer_info": local_answer_info,
            "answer_correct_float": answer_correct_float,
            "source_trace_metadata": source_trace_metadata,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if use_importance_weighting and importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images

        return output

    @profiling_decorator
    def _score_completions_diversity(
        self,
        generation_batch: dict[str, torch.Tensor | Any],
        pass_data, pass_forward_kwargs, other_data,
        adapter_name: str, num_completions: int,
        main_correct_rate_by_prompt: dict[int, float] | None = None,
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if adapter_name == "default":
            raise ValueError(
                "_score_completions_diversity must never be called for the default/main "
                "adapter -- it remains a collective correctness learner only."
            )

        inputs = pass_data["inputs"]
        prompts = pass_data["prompts"]
        completions = pass_data["completions"]
        completion_ids_list = pass_data["completion_ids_list"]

        reward_type = getattr(self.args, "diversity_reward_type", "external")
        # External reward types use self.diversity_reward_funcs. Trainer-native
        # policy-repulsion and trace-Jaccard rewards are handled below.
        use_policy_repulsion = reward_type in self._POLICY_REPULSION_REWARD_TYPES
        trace_diagnostics = None

        if use_policy_repulsion:
            # 1) local repulsion rewards
            local_rewards = self._policy_repulsion_reward(
                source_adapter=adapter_name,
                prompt_completion_ids=pass_data["prompt_completion_ids"],
                attention_mask=pass_data["attention_mask"],
                completion_mask=generation_batch["completion_mask"],
                logits_to_keep=pass_data["logits_to_keep"],
                forward_kwargs=pass_forward_kwargs,
                num_images=pass_data["num_images"],
            )

            # 2) optional correctness gate -- gate on PURE answer correctness
            # only, never the mixed correctness_reward_per_sample scalar
            # (0.0/0.5/1.0/1.5), which would let a wrong-but-think-tagged
            # sample slip past a threshold like 0.5.
            gate = None
            corr = generation_batch.get("answer_correct_float", None)
            if getattr(self.args, "policy_repulsion_gate_by_correctness", False) and corr is not None:
                thr = float(getattr(self.args, "policy_repulsion_gate_threshold", 1.0))
                gate = (corr >= thr).float()
                local_rewards = local_rewards * gate

            # 3) gather -> global rewards_per_func
            rewards = gather(local_rewards)
            rewards_per_func = rewards.unsqueeze(1)  # (N_global, 1)
            diversity_reward_names = [getattr(self.args, "diversity_reward_type", "policy_repulsion")]
            diversity_weights = torch.ones(1, device=device)

            # 4) debug logging
            dbg = getattr(self, "_repulsion_debug", None)
            if dbg is not None:
                gap_g = gather(dbg["gap"])
                self._metrics[mode][f"{adapter_name}/repulsion/gap_mean"].append(gap_g.mean().item())
                self._metrics[mode][f"{adapter_name}/repulsion/gap_std"].append(gap_g.std().item())

                src_g = gather(dbg["src_logp"])
                oth_g = gather(dbg["b_logp_agg"])
                self._metrics[mode][f"{adapter_name}/repulsion/src_logp_mean"].append(src_g.mean().item())
                self._metrics[mode][f"{adapter_name}/repulsion/other_logp_mean"].append(oth_g.mean().item())

                if "barrier_active" in dbg:
                    act_g = gather(dbg["barrier_active"])
                    self._metrics[mode][f"{adapter_name}/repulsion/barrier_active_frac"].append(act_g.mean().item())

                if gate is not None:
                    gate_g = gather(gate)
                    self._metrics[mode][f"{adapter_name}/repulsion/gate_frac"].append(gate_g.mean().item())

        elif reward_type in self._TRACE_JACCARD_REWARD_TYPES:
            local_reward_values, local_diagnostics = trace_jaccard_diversity_reward(
                prompts=prompts,
                completions=completions,
                other_prompts=other_data["prompts"],
                other_completions=other_data["completions"],
                source_adapter=adapter_name,
                comparison_adapters=other_data.get("adapter_names"),
                candidate_ids=other_data.get("candidate_indices"),
                other_candidate_ids=other_data.get("comparison_indices"),
                exclude_self=bool(other_data.get("exclude_self", False)),
                ngram_size=int(getattr(self.args, "trace_jaccard_ngram_size", 3)),
                return_diagnostics=True,
            )
            local_rewards = torch.tensor(
                local_reward_values, dtype=torch.float32, device=device
            )
            local_rewards = torch.nan_to_num(
                local_rewards, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
            rewards_per_func = gather(local_rewards).unsqueeze(1)
            diversity_reward_names = [reward_type]
            diversity_weights = torch.ones(1, device=device)
            trace_diagnostics = {
                name: gather(
                    torch.tensor(values, dtype=torch.float32, device=device)
                )
                for name, values in local_diagnostics.items()
            }
            trace_diagnostics["reward"] = rewards_per_func[:, 0]

        elif reward_type == "external" or reward_type in self._ONE_MINUS_BLEU_REWARD_TYPES:
            bleu_balance_mode = getattr(self.args, "diversity_bleu_balance_mode", "sample_balanced")
            bleu_main_weight = float(getattr(self.args, "diversity_source_main_weight", 0.5))
            bleu_exclude_self = bool(getattr(self.args, "diversity_bleu_exclude_self", True))
            bleu_text_scope = getattr(self.args, "diversity_bleu_text_scope", "full_completion")

            reference_groups = other_data.get("reference_groups", {})
            source_group_name = other_data.get("source_group_name", None)

            local_reward_values, bleu_diag = compute_one_minus_bleu_rewards_for_adapter(
                source_completions=completions,
                reference_groups=reference_groups,
                balance_mode=bleu_balance_mode,
                main_weight=bleu_main_weight,
                exclude_self=bleu_exclude_self,
                text_scope=bleu_text_scope,
                source_group_name=source_group_name,
            )
            local_rewards = torch.tensor(local_reward_values, dtype=torch.float32, device=device)
            local_rewards = torch.nan_to_num(local_rewards, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            rewards_per_func = gather(local_rewards).unsqueeze(1)
            diversity_reward_names = [reward_type]
            diversity_weights = torch.ones(1, device=device)

            # Debug logging for BLEU reward diagnostics
            if getattr(self.args, "diversity_reward_debug", False):
                debug_interval = int(getattr(self.args, "diversity_reward_debug_steps", 20))
                if self._step % debug_interval == 0:
                    ref_sizes = bleu_diag.get("reference_group_sizes", {})
                    n_ref_default = ref_sizes.get("default", 0)
                    n_ref_other_div = sum(v for k, v in ref_sizes.items() if k != "default")
                    n_ref_total = sum(ref_sizes.values())
                    sim_main = [v for v in bleu_diag["sim_main_mean"] if not math.isnan(v)]
                    sim_other = [v for v in bleu_diag["sim_other_div_mean"] if not math.isnan(v)]
                    gathered_rewards = rewards_per_func[:, 0]
                    logger.info(
                        f"[DiversityBLEU] adapter={adapter_name} step={self._step} "
                        f"balance_mode={bleu_balance_mode} scope={self.diversity_comparison_scope} "
                        f"main_weight={bleu_main_weight} "
                        f"n_src={bleu_diag['num_source']} "
                        f"n_ref_total={n_ref_total} n_ref_default={n_ref_default} "
                        f"n_ref_other_div={n_ref_other_div} "
                        f"sim_main_mean={sum(sim_main)/len(sim_main):.4f}" if sim_main else
                        f"[DiversityBLEU] adapter={adapter_name} step={self._step} "
                        f"balance_mode={bleu_balance_mode} scope={self.diversity_comparison_scope} "
                        f"main_weight={bleu_main_weight} "
                        f"n_src={bleu_diag['num_source']} "
                        f"n_ref_total={n_ref_total} n_ref_default={n_ref_default} "
                        f"n_ref_other_div={n_ref_other_div} "
                        f"reward_mean={gathered_rewards.mean().item():.4f} "
                        f"reward_std={gathered_rewards.std(unbiased=False).item():.4f} "
                        f"reward_min={gathered_rewards.min().item():.4f} "
                        f"reward_max={gathered_rewards.max().item():.4f}"
                    )
                    self._metrics[mode][f"{adapter_name}/bleu_diversity/n_ref_total"].append(float(n_ref_total))
                    self._metrics[mode][f"{adapter_name}/bleu_diversity/n_ref_default"].append(float(n_ref_default))
                    self._metrics[mode][f"{adapter_name}/bleu_diversity/n_ref_other_div"].append(float(n_ref_other_div))
                    if sim_main:
                        self._metrics[mode][f"{adapter_name}/bleu_diversity/sim_main_mean"].append(
                            sum(sim_main) / len(sim_main)
                        )
                    if sim_other:
                        self._metrics[mode][f"{adapter_name}/bleu_diversity/sim_other_div_mean"].append(
                            sum(sim_other) / len(sim_other)
                        )

        elif reward_type in self._MAIN_WEAK_CORRECTNESS_REWARD_TYPES:
            # main_weak_correctness_bonus: reward a diversity sample for being
            # correct on prompts where the main/default adapter itself is
            # weak, instead of rewarding arbitrary textual diversity.
            #
            # --- Old formula (superseded, kept for reference) -----------------
            #   raw_bonus = answer_correct_float * (1.0 - main_correct_rate)
            #   diversity_advantages = normalize(raw_bonus)
            #
            # Problem: (1 - main_correct_rate(x)) is a CONSTANT across every
            # diversity sample from the same prompt x (this branch's group
            # normalization below is scoped to one adapter's own rollouts for
            # a single prompt, i.e. exactly the samples that all share the
            # same main_correct_rate(x)). Group normalization subtracts the
            # group mean and divides by the group std, both of which scale
            # linearly with any shared per-group constant k > 0:
            #   Norm[k * v] == Norm[v]  for a per-group constant k.
            # So "main is weaker on this prompt" barely changed the resulting
            # advantage relative to a prompt where main was already strong --
            # the intended "reward main-weak coverage more" signal was mostly
            # cancelled by normalization.
            #
            # --- New formula -----------------------------------------------
            #   aux_correct_advantages = Norm[c(x, y)]                       (below, shared normalization block)
            #   main_weak_advantages   = (1 - main_correct_rate(x)) * aux_correct_advantages   (post-normalization hook, below)
            #   main_weak_advantages   = main_weak_correctness_bonus_weight * main_weak_advantages
            # The main-weakness factor is applied to the ALREADY-NORMALIZED
            # advantage, so it is not cancelled by the group std division.
            #
            # c(x, y) is PURE answer correctness (answer_correct_float, never
            # the mixed correctness_reward scalar). main_correct_rate(x) was
            # computed once per step from the default adapter's own rollouts,
            # before any diversity adapter was scored, and is passed in here
            # already aligned per prompt_index.
            local_answer_correct = generation_batch.get("answer_correct_float")
            if local_answer_correct is None:
                raise ValueError(
                    "main_weak_correctness_bonus requires 'answer_correct_float' in "
                    "generation_batch -- _score_completions_correctness must run for "
                    "this adapter before _score_completions_diversity."
                )
            local_source_metadata = generation_batch.get("source_trace_metadata") or [None] * len(
                local_answer_correct
            )
            aligned_main_rate = align_main_correct_rate_to_local_rows(
                local_source_metadata,
                main_correct_rate_by_prompt or {},
            )
            # Raw diagnostic value ONLY -- c(x,y)*(1-main_correct_rate(x)).
            # This is never fed into rewards_per_func/normalization anymore;
            # it is recorded purely for logging/metadata (see below).
            raw_bonus_values = compute_main_weak_correctness_bonus(
                local_answer_correct.detach().float().cpu().tolist(),
                aligned_main_rate,
            )
            local_raw_bonus = torch.tensor(raw_bonus_values, dtype=torch.float32, device=device)

            # rewards_per_func carries PURE answer correctness -- the shared
            # normalization block below (identical code path every other
            # diversity reward uses, scoped to this adapter's own rollouts)
            # computes aux_correct_advantages = Norm[c(x, y)] from this.
            local_rewards = local_answer_correct.detach().float()
            rewards_per_func = gather(local_rewards).unsqueeze(1)
            diversity_reward_names = [reward_type]
            diversity_weights = torch.ones(1, device=device)
            gathered_main_rate = gather(
                torch.tensor(aligned_main_rate, dtype=torch.float32, device=device)
            )
            gathered_raw_bonus = gather(local_raw_bonus)

        elif reward_type == "external":
            # Legacy external reward path (e.g. custom reward_funcs_diversity registered at init).
            rewards_per_func = self._calculate_diversity_rewards(
                inputs,
                prompts,
                completions,
                completion_ids_list,
                other_data["completions"],
                other_data["completion_ids_list"],
            )  # already gathered
            diversity_reward_names = self.diversity_reward_func_names
            diversity_weights = self.diversity_reward_weights.to(device)

        else:
            raise ValueError(f"Unknown diversity_reward_type: {reward_type}")

        rewards = (rewards_per_func * diversity_weights.unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, num_completions).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_completions, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, num_completions).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(num_completions, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")

        std_rewards = torch.nan_to_num(
            std_rewards, nan=0.0, posinf=0.0, neginf=0.0
        )
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        advantages = torch.nan_to_num(
            advantages, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        aux_correct_advantages_global = None
        if reward_type in self._MAIN_WEAK_CORRECTNESS_REWARD_TYPES:
            # At this point `advantages` == aux_correct_advantages == Norm[c(x,y)],
            # computed by the shared group-normalization above (rewards_per_func
            # held pure answer correctness for this branch). Snapshot it before
            # applying the main-weakness factor, for diagnostics/metadata below.
            aux_correct_advantages_global = advantages.clone()

            # Apply (1 - main_correct_rate(x)) and main_weak_correctness_bonus_weight
            # AFTER group normalization, not to the raw reward -- see the long
            # comment above this branch's rewards_per_func construction for why
            # a pre-normalization multiply would be cancelled by the group std
            # division. Applying both factors here, to the already-normalized
            # advantage, makes them effective, non-cancelled knobs.
            main_weak_factor_global = 1.0 - gathered_main_rate
            advantages = main_weak_factor_global * advantages
            main_weak_weight = float(getattr(self.args, "main_weak_correctness_bonus_weight", 1.0))
            advantages = main_weak_weight * advantages

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]
        local_rewards = rewards[process_slice]
        local_aux_correct_advantages = (
            aux_correct_advantages_global[process_slice] if aux_correct_advantages_global is not None else None
        )

        # metrics/logs
        for i, name in enumerate(diversity_reward_names):
            if trace_diagnostics is not None:
                values = torch.nan_to_num(
                    rewards_per_func[:, i], nan=0.0, posinf=1.0, neginf=0.0
                )
                reward_mean = values.mean().item()
                reward_std = values.std(unbiased=False).item()
            else:
                reward_mean = torch.nanmean(rewards_per_func[:, i]).item()
                reward_std = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"{adapter_name}/diversity_rewards/{name}/mean"].append(reward_mean)
            self._metrics[mode][f"{adapter_name}/diversity_rewards/{name}/std"].append(reward_std)

        self._metrics[mode][f"{adapter_name}/diversity_reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode][f"{adapter_name}/diversity_reward_std"].append(std_rewards.mean().item())
        self._metrics[mode][f"{adapter_name}/frac_diversity_reward_zero_std"].append(is_std_zero.float().mean().item())

        if trace_diagnostics is not None:
            reward_values = torch.nan_to_num(trace_diagnostics["reward"])
            similarity_values = torch.nan_to_num(
                trace_diagnostics["max_jaccard_similarity"]
            )
            trace_lengths = torch.nan_to_num(trace_diagnostics["trace_length"])
            empty_traces = torch.nan_to_num(trace_diagnostics["empty_trace"])
            comparison_sizes = torch.nan_to_num(trace_diagnostics["comparison_size"])
            prefix = f"{adapter_name}/trace_jaccard"
            self._metrics[mode][f"{prefix}/comparison_scope_intra_adapter"].append(
                float(getattr(self.args, "diversity_comparison_scope", "intra_adapter") == "intra_adapter")
            )
            self._metrics[mode][f"{prefix}/reward_mean"].append(reward_values.mean().item())
            self._metrics[mode][f"{prefix}/reward_std"].append(reward_values.std(unbiased=False).item())
            self._metrics[mode][f"{prefix}/max_jaccard_similarity_mean"].append(similarity_values.mean().item())
            self._metrics[mode][f"{prefix}/max_jaccard_similarity_std"].append(similarity_values.std(unbiased=False).item())
            self._metrics[mode][f"{prefix}/trace_length_mean"].append(trace_lengths.mean().item())
            self._metrics[mode][f"{prefix}/empty_trace_ratio"].append(empty_traces.mean().item())
            self._metrics[mode][f"{prefix}/comparison_size_mean"].append(comparison_sizes.mean().item())

        for i, name in enumerate(diversity_reward_names):
            self._logs[f"{adapter_name}/diversity_rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs[f"{adapter_name}/diversity_advantages"].extend(all_process_advantages.tolist())
        is_main_weak = reward_type in self._MAIN_WEAK_CORRECTNESS_REWARD_TYPES

        source_trace_metadata = generation_batch.get("source_trace_metadata", None)
        if source_trace_metadata is not None:
            div_values = local_rewards.detach().float().cpu().tolist()
            div_adv_values = advantages.detach().float().cpu().tolist()
            if is_main_weak:
                local_raw_bonus_values = local_raw_bonus.detach().float().cpu().tolist()
                local_aux_correct_values = (
                    local_aux_correct_advantages.detach().float().cpu().tolist()
                    if local_aux_correct_advantages is not None
                    else [None] * len(source_trace_metadata)
                )
            for idx, record in enumerate(source_trace_metadata):
                if isinstance(record, dict):
                    record["diversity_reward"] = div_values[idx] if idx < len(div_values) else None
                    record["diversity_advantage"] = div_adv_values[idx] if idx < len(div_adv_values) else None
                    if is_main_weak:
                        main_rate = aligned_main_rate[idx] if idx < len(aligned_main_rate) else None
                        record["main_correct_rate"] = main_rate
                        record["main_weak_factor"] = (1.0 - main_rate) if main_rate is not None else None
                        # Raw diagnostic ONLY: c(x,y)*(1-main_correct_rate(x)).
                        # This is NOT what feeds the loss -- see
                        # main_weak_correctness_advantage below for that.
                        record["main_weak_correctness_bonus"] = (
                            local_raw_bonus_values[idx] if idx < len(local_raw_bonus_values) else None
                        )
                        # aux_correct_advantage = Norm[c(x,y)], BEFORE the
                        # main-weakness factor and weight are applied.
                        record["aux_correct_advantage"] = (
                            local_aux_correct_values[idx] if idx < len(local_aux_correct_values) else None
                        )
                        # The value actually used in the loss -- numerically
                        # identical to diversity_advantage above for this
                        # reward type, re-exposed under a formula-matching
                        # name for easier diagnostics.
                        record["main_weak_correctness_advantage"] = (
                            div_adv_values[idx] if idx < len(div_adv_values) else None
                        )

        if is_main_weak and bool(getattr(self.args, "main_weak_correctness_bonus_log_by_source", True)):
            raw_bonus_g = torch.nan_to_num(gathered_raw_bonus)
            raw_mean = raw_bonus_g.mean().item()
            raw_std = raw_bonus_g.std(unbiased=False).item()
            raw_nonzero_frac = (raw_bonus_g > 0.0).float().mean().item()
            raw_sum = raw_bonus_g.sum().item()

            # Raw diagnostic (c(x,y)*(1-main_correct_rate(x))) -- kept under
            # the original key names for backward compatibility, plus
            # explicit raw_* aliases making clear this is diagnostic only,
            # NOT the training signal (see adv_* below for that).
            self._metrics[mode]["train/main_weak_correctness_bonus/mean"].append(raw_mean)
            self._metrics[mode]["train/main_weak_correctness_bonus/std"].append(raw_std)
            self._metrics[mode]["train/main_weak_correctness_bonus/nonzero_frac"].append(raw_nonzero_frac)
            self._metrics[mode]["train/main_weak_correctness_bonus/raw_sum"].append(raw_sum)
            self._metrics[mode]["train/main_weak_correctness_bonus/raw_mean"].append(raw_mean)
            self._metrics[mode]["train/main_weak_correctness_bonus/raw_std"].append(raw_std)
            self._metrics[mode]["train/main_weak_correctness_bonus/raw_nonzero_frac"].append(raw_nonzero_frac)

            # weighted_sum: kept for backward compatibility, but this is a
            # group-NORMALIZED advantage, so its sum sits near 0 by
            # construction regardless of signal strength -- use adv_abs_mean
            # / adv_l2 below to actually gauge signal strength.
            self._metrics[mode]["train/main_weak_correctness_bonus/weighted_sum"].append(
                all_process_advantages.sum().item()
            )

            adv_g = torch.nan_to_num(all_process_advantages)
            adv_abs_g = adv_g.abs()
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_mean"].append(adv_g.mean().item())
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_std"].append(
                adv_g.std(unbiased=False).item()
            )
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_abs_mean"].append(adv_abs_g.mean().item())
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_abs_sum"].append(adv_abs_g.sum().item())
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_l2"].append(
                torch.sqrt((adv_g ** 2).mean()).item()
            )
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_nonzero_frac"].append(
                (adv_abs_g > 0.0).float().mean().item()
            )
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_positive_frac"].append(
                (adv_g > 0.0).float().mean().item()
            )
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_negative_frac"].append(
                (adv_g < 0.0).float().mean().item()
            )
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_positive_mass"].append(
                adv_g.clamp(min=0.0).sum().item()
            )
            self._metrics[mode]["train/main_weak_correctness_bonus/adv_negative_mass"].append(
                adv_g.clamp(max=0.0).sum().item()
            )

            # Source-level (per diversity adapter). raw_* reuse the
            # process-gathered-but-adapter-scoped raw bonus; adv_* use this
            # adapter's own already-sliced local advantage.
            local_adv = torch.nan_to_num(advantages)
            self._metrics[mode][f"train/source/{adapter_name}/main_weak_bonus_mean"].append(raw_mean)
            self._metrics[mode][f"train/source/{adapter_name}/main_weak_bonus_nonzero_frac"].append(
                raw_nonzero_frac
            )
            self._metrics[mode][f"train/source/{adapter_name}/main_weak_adv_abs_mean"].append(
                local_adv.abs().mean().item()
            )
            self._metrics[mode][f"train/source/{adapter_name}/main_weak_adv_std"].append(
                local_adv.std(unbiased=False).item()
            )

        generation_batch["diversity_reward_per_sample"] = local_rewards
        generation_batch["diversity_advantages"] = advantages
        return generation_batch

    
    
    # @profiling_decorator
    # def _score_completions_diversity(
    #     self, 
    #     generation_batch: dict[str, torch.Tensor | Any], 
    #     pass_data, pass_forward_kwargs, other_data, adapter_name: str, num_completions: int
    # ) -> dict[str, torch.Tensor | Any]:
    #     device = self.accelerator.device
    #     mode = "train" if self.model.training else "eval"
        
    #     inputs = pass_data["inputs"]
    #     prompts = pass_data["prompts"]
    #     completions = pass_data["completions"]
    #     completion_ids_list = pass_data["completion_ids_list"]
        
    #     other_completions = other_data["completions"]
    #     other_completions_ids_list = other_data["completion_ids_list"]

    #     use_policy_repulsion = getattr(self.args, "diversity_reward_type", "external") in {
    #         "policy_repulsion_margin",
    #         "policy_repulsion_margin_barrier",
    #     }

    #     if use_policy_repulsion:
    #         # compute local rewards (N_local,)
    #         local_rewards = self._policy_repulsion_reward(
    #             source_adapter=adapter_name,
    #             prompt_completion_ids=pass_data["prompt_completion_ids"],
    #             attention_mask=pass_data["attention_mask"],
    #             completion_mask=generation_batch["completion_mask"],
    #             logits_to_keep=pass_data["logits_to_keep"],
    #             forward_kwargs=pass_forward_kwargs,
    #             num_images=pass_data["num_images"],
    #         )

    #     dbg = getattr(self, "_repulsion_debug", None)
    #     if dbg is not None:
    #         gap_g = gather(dbg["gap"])
    #         self._metrics[mode][f"{adapter_name}/repulsion/gap_mean"].append(gap_g.mean().item())
    #         self._metrics[mode][f"{adapter_name}/repulsion/gap_std"].append(gap_g.std().item())

    #         src_g = gather(dbg["src_logp"])
    #         oth_g = gather(dbg["b_logp_star"])
    #         self._metrics[mode][f"{adapter_name}/repulsion/src_logp_mean"].append(src_g.mean().item())
    #         self._metrics[mode][f"{adapter_name}/repulsion/other_logp_mean"].append(oth_g.mean().item())

    #         if "barrier_active" in dbg:
    #             act_g = gather(dbg["barrier_active"])
    #             self._metrics[mode][f"{adapter_name}/repulsion/barrier_active_frac"].append(act_g.mean().item())

    #         # optional correctness gate
    #         if getattr(self.args, "policy_repulsion_gate_by_correctness", False) and corr is not None:
    #             gate_frac = gather(gate).float().mean().item()
    #             self._metrics[mode][f"{adapter_name}/repulsion/gate_frac"].append(gate_frac)

    #         # gather across processes for correct group normalization
    #         rewards = gather(local_rewards)
    #         rewards_per_func = rewards.unsqueeze(1)  # shape (N_global, 1)
    #         diversity_reward_names = [getattr(self.args, "diversity_reward_type", "policy_repulsion")]
    #         diversity_weights = torch.ones(1, device=device)

    #     else:
    #         # fallback to your existing external diversity reward funcs (BLEU, etc.)
    #         other_completions = other_data["completions"]
    #         other_completions_ids_list = other_data["completion_ids_list"]

    #         rewards_per_func = self._calculate_diversity_rewards(
    #             inputs,
    #             prompts,
    #             completions,
    #             completion_ids_list,
    #             other_completions,
    #             other_completions_ids_list,
    #         )  # already gathered

    #         diversity_reward_names = self.diversity_reward_func_names
    #         diversity_weights = self.diversity_reward_weights.to(device)  
    #     # ...to here [NEED REVIEW]

    #     # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
    #     # important because rewards will be normalized per group, and completions are distributed. We will later slice
    #     # rewards_per_func to extract each process's subset.
    #     ## Correctness Rewards & Advantages

    #     ## Diversity Rewards & Advantages
    #     # rewards_per_func = self._calculate_diversity_rewards(inputs, prompts, completions, completion_ids_list, other_completions, other_completions_ids_list)
    #     # 요구: prompt에 대해 completions 개수만큼의 reward 요소가 나와야됨 (reward function에서 others와의 점수 자체를 리스트로 받아오는게 아니라 평균내서 총 completions 개수만큼의 요소 가져옴)

    #     # Apply weights to each reward function's output and sum
    #     rewards = (rewards_per_func * diversity_weights.unsqueeze(0)).nansum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, num_completions).mean(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_completions, dim=0)
    #     advantages = rewards - mean_grouped_rewards

    #     if self.scale_rewards in ["group", "none"]:
    #         # If self.scale_rewards = "none", we'll still log group level std
    #         std_rewards = rewards.view(-1, num_completions).std(dim=1)
    #         std_rewards = std_rewards.repeat_interleave(num_completions, dim=0)
    #     elif self.scale_rewards == "batch":
    #         # Compute global std
    #         std_rewards = rewards.std().expand_as(rewards)
    #     else:
    #         raise ValueError(
    #             f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
    #         )

    #     is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
    #     if self.scale_rewards != "none":
    #         advantages = advantages / (std_rewards + 1e-4)

    #     # Slice to keep only the local part of the data
    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
    #     advantages = advantages[process_slice]

    #     # # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
    #     # # for i, diversity_reward_func_name in enumerate(self.diversity_reward_func_names): # [NEED REVIEW] 기존
    #     # for i, diversity_reward_func_name in enumerate(diversity_reward_names): # [NEED REVIEW] 수정본
    #     #     mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
    #     #     self._metrics[mode][f"{adapter_name}/diversity_rewards/{diversity_reward_func_name}/mean"].append(mean_rewards)
    #     #     std_func_rewards = nanstd(rewards_per_func[:, i]).item()
    #     #     self._metrics[mode][f"{adapter_name}/diversity_rewards/{diversity_reward_func_name}/std"].append(std_func_rewards)
    #     # self._metrics[mode][f"{adapter_name}/diversity_reward"].append(mean_grouped_rewards.mean().item())
    #     # self._metrics[mode][f"{adapter_name}/diversity_reward_std"].append(std_rewards.mean().item())
    #     # self._metrics[mode][f"{adapter_name}/frac_diversity_reward_zero_std"].append(is_std_zero.float().mean().item())

    #     # # 로그 부분 수정
    #     # for i, name in enumerate(self.diversity_reward_func_names):
    #     #     self._logs[f"{adapter_name}/diversity_rewards"][name].extend(rewards_per_func[:, i].tolist())
    #     # self._logs[f"{adapter_name}/diversity_advantages"].extend(all_process_advantages.tolist())
            
    #     # generation_batch["diversity_advantages"] = advantages

    #     # return generation_batch

    #     # metrics/logs
    #     for i, name in enumerate(diversity_reward_names):
    #         self._metrics[mode][f"{adapter_name}/diversity_rewards/{name}/mean"].append(
    #             torch.nanmean(rewards_per_func[:, i]).item()
    #         )
    #         self._metrics[mode][f"{adapter_name}/diversity_rewards/{name}/std"].append(
    #             nanstd(rewards_per_func[:, i]).item()
    #         )

    #     self._metrics[mode][f"{adapter_name}/diversity_reward"].append(mean_grouped_rewards.mean().item())
    #     self._metrics[mode][f"{adapter_name}/diversity_reward_std"].append(std_rewards.mean().item())
    #     self._metrics[mode][f"{adapter_name}/frac_diversity_reward_zero_std"].append(is_std_zero.float().mean().item())
        
    #     # if adapter_name == "default":
    #     #     log_reward_per_func = rewards_per_func[process_slice]
    #     #     for i, name in enumerate(diversity_reward_names):
    #     #         self._metrics[mode][f"{adapter_name}/gen_diversity_rewards/{name}/mean"].append(
    #     #             torch.nanmean(log_reward_per_func[:, i]).item()
    #     #         )
    #     #         self._metrics[mode][f"{adapter_name}/gen_diversity_rewards/{name}/std"].append(
    #     #             nanstd(log_reward_per_func[:, i]).item()
    #     #         )
    #     #     log_mean_grouped_rewards = mean_grouped_rewards[process_slice].mean().item()
    #     #     self._metrics[mode][f"{adapter_name}/gen_diversity_reward"].append(log_mean_grouped_rewards)
    #     #     log_std_rewards = std_rewards[process_slice].mean().item()
    #     #     self._metrics[mode][f"{adapter_name}/gen_diversity_reward_std"].append(log_std_rewards)
    #     #     log_is_std_zero = is_std_zero[process_slice].float().mean().item()
    #     #     self._metrics[mode][f"{adapter_name}/gen_frac_diversity_reward_zero_std"].append(log_is_std_zero)

    #     for i, name in enumerate(diversity_reward_names):
    #         self._logs[f"{adapter_name}/diversity_rewards"][name].extend(rewards_per_func[:, i].tolist())
    #     self._logs[f"{adapter_name}/diversity_advantages"].extend(all_process_advantages.tolist())

    #     generation_batch["diversity_advantages"] = advantages
    #     return generation_batch
    
    # def _generate_and_score_completions(
    #     self, inputs: list[dict[str, torch.Tensor | Any]]
    # ) -> dict[str, torch.Tensor | Any]:
    #     device = self.accelerator.device
    #     mode = "train" if self.model.training else "eval"

    #     prompts = [x["prompt"] for x in inputs]

    #     if "images" in inputs[0]:
    #         images = [example.get("images") for example in inputs]
    #     elif "image" in inputs[0]:
    #         images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
    #     else:
    #         images = None
    #     # Transformers requires at least one image in the batch, otherwise it throws an error
    #     if images is not None and all(img_list == [] for img_list in images):
    #         images = None

    #     # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
    #     # [{"role": "user", "content": "What color is the sky?"}] to
    #     # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
    #     if images is not None:
    #         prompts = [
    #             prepare_multimodal_messages(prompt, image_list)
    #             for prompt, image_list in zip(prompts, images, strict=True)
    #         ]

    #     prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
    #         self._generate(prompts)
    #     )

    #     # Convert lists of token IDs to padded tensors
    #     prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
    #     prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
    #     prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
    #     prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
    #     completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
    #     completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
    #     completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
    #     completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
    #     if sampling_per_token_logps_list is not None:
    #         sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
    #         sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
    #     else:
    #         sampling_per_token_logps = None

    #     # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
    #     if self.mask_truncated_completions:
    #         eos_and_pad = [self.eos_token_id, self.pad_token_id]
    #         is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
    #         completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

    #     # Concatenate prompt_mask with completion_mask for logit computation
    #     prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

    #     logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
    #     batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

    #     num_images = [len(img_list) for img_list in images] if images is not None else None

    #     # Get forward_kwargs for models with multimodal inputs
    #     if images is not None:
    #         prompts_text = [
    #             apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
    #             for prompt in prompts
    #         ]
    #         prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
    #         prompt_inputs = super()._prepare_inputs(prompt_inputs)
    #         forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
    #     else:
    #         forward_kwargs = {}

    #     # If token_type_ids are used, extend them with zeros for the completion part
    #     if "token_type_ids" in forward_kwargs:
    #         token_type_ids = forward_kwargs["token_type_ids"]
    #         forward_kwargs["token_type_ids"] = torch.cat(
    #             [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
    #         )

    #     with torch.no_grad():
    #         # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
    #         # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
    #         # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
    #         # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
    #         # old_per_token_logps to None.
    #         # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
    #         # distribution mismatch between vLLM and the training model can be large and harm the training.
    #         generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
    #         if self.args.gradient_accumulation_steps % generate_every != 0 or (
    #             self.use_vllm and self.vllm_importance_sampling_correction
    #         ):
    #             old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
    #                 self.model,
    #                 prompt_completion_ids,
    #                 attention_mask,
    #                 logits_to_keep,
    #                 batch_size,
    #                 num_images=num_images,
    #                 **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
    #             )
    #         else:
    #             old_per_token_logps = None

    #         # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
    #         if self.use_vllm and self.vllm_importance_sampling_correction:
    #             importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
    #             importance_sampling_ratio = torch.clamp(
    #                 importance_sampling_ratio, max=self.vllm_importance_sampling_cap
    #             )

    #         # Compute the per-token log probabilities for the reference model
    #         if self.beta != 0.0:
    #             if self.ref_model is not None:
    #                 ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
    #                     self.ref_model,
    #                     prompt_completion_ids,
    #                     attention_mask,
    #                     logits_to_keep,
    #                     batch_size=batch_size,
    #                     num_images=num_images,
    #                     **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
    #                 )
    #             else:
    #                 with self.accelerator.unwrap_model(self.model).disable_adapter():
    #                     ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
    #                         self.model,
    #                         prompt_completion_ids,
    #                         attention_mask,
    #                         logits_to_keep,
    #                         batch_size=batch_size,
    #                         num_images=num_images,
    #                         **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
    #                     )
    #         else:
    #             ref_per_token_logps = None

    #     # Decode
    #     prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
    #     completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     if is_conversational(inputs[0]):
    #         completions = []
    #         for prompt, completion in zip(prompts, completions_text, strict=True):
    #             bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
    #             completions.append([{"role": "assistant", "content": bootstrap + completion}])
    #     else:
    #         completions = completions_text

    #     # Merge extra_fields from rollout_func into inputs for reward functions
    #     if extra_fields:
    #         for i, inp in enumerate(inputs):
    #             for key, values in extra_fields.items():
    #                 if isinstance(values, list) and i < len(values):
    #                     inp[key] = values[i]
    #                 elif not isinstance(values, list):
    #                     inp[key] = values

    #     # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
    #     # important because rewards will be normalized per group, and completions are distributed. We will later slice
    #     # rewards_per_func to extract each process's subset.
    #     rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    #     # Apply weights to each reward function's output and sum
    #     rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = rewards - mean_grouped_rewards

    #     if self.scale_rewards in ["group", "none"]:
    #         # If self.scale_rewards = "none", we'll still log group level std
    #         std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
    #         std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
    #     elif self.scale_rewards == "batch":
    #         # Compute global std
    #         std_rewards = rewards.std().expand_as(rewards)
    #     else:
    #         raise ValueError(
    #             f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
    #         )

    #     is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
    #     if self.scale_rewards != "none":
    #         advantages = advantages / (std_rewards + 1e-4)

    #     # Slice to keep only the local part of the data
    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
    #     advantages = advantages[process_slice]

    #     # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
    #     for i, reward_func_name in enumerate(self.reward_func_names):
    #         mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
    #         self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
    #         std_func_rewards = nanstd(rewards_per_func[:, i]).item()
    #         self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
    #     self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
    #     self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
    #     self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

    #     # Log prompt and completion texts
    #     self._logs["prompt"].extend(gather_object(prompts_text))
    #     self._logs["completion"].extend(gather_object(completions_text))
    #     for i, name in enumerate(self.reward_func_names):
    #         self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
    #     self._logs["advantages"].extend(all_process_advantages.tolist())

    #     if images is not None:
    #         self._logs["images"].extend(gather_object(images))

    #     if self.use_vllm and self.vllm_importance_sampling_correction:
    #         delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
    #         delta = delta[completion_mask.bool()]
    #         mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
    #         max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
    #         self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
    #             self.accelerator.gather(mean_delta).mean().item()
    #         )
    #         self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
    #             self.accelerator.gather(max_delta).max().item()
    #         )

    #         flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
    #         min_importance_sampling_ratio = (
    #             torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
    #         )
    #         mean_importance_sampling_ratio = (
    #             torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
    #         )
    #         max_importance_sampling_ratio = (
    #             torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
    #         )
    #         self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
    #             nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
    #         )
    #         self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
    #             self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
    #         )
    #         self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
    #             nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
    #         )

    #     output = {
    #         "prompt_ids": prompt_ids,
    #         "prompt_mask": prompt_mask,
    #         "completion_ids": completion_ids,
    #         "completion_mask": completion_mask,
    #         "advantages": advantages,
    #         "num_items_in_batch": num_items_in_batch,
    #     }
    #     if old_per_token_logps is not None:
    #         output["old_per_token_logps"] = old_per_token_logps
    #     if self.use_vllm and self.vllm_importance_sampling_correction:
    #         output["importance_sampling_ratio"] = importance_sampling_ratio
    #     if ref_per_token_logps is not None:
    #         output["ref_per_token_logps"] = ref_per_token_logps
    #     if "pixel_values" in forward_kwargs:
    #         output["pixel_values"] = forward_kwargs["pixel_values"]
    #     if "image_grid_thw" in forward_kwargs:
    #         output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
    #     if "pixel_attention_mask" in forward_kwargs:
    #         output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
    #     if "image_sizes" in forward_kwargs:
    #         output["image_sizes"] = forward_kwargs["image_sizes"]
    #     if "token_type_ids" in forward_kwargs:
    #         output["token_type_ids"] = forward_kwargs["token_type_ids"]
    #     if images is not None:
    #         output["num_images"] = num_images
    #     return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # compute loss and metrics using liger coex loss
        loss, metrics = self.liger_coex_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_coex_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The CoExTrainer does not support returning outputs")
        if self.use_liger_kernel:
            # Compute the loss using the liger coex loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    @staticmethod
    def _dmpo_loss_from_tensors(
        current_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        raw_rewards: torch.Tensor,
        num_generations: int,
        dmpo_temperature: float,
        skip_zero_advantage_groups: bool = False,
        reward_range_tolerance: float = 1e-8,
    ):
        if current_per_token_logps.shape != completion_mask.shape:
            raise ValueError(
                "current_per_token_logps and completion_mask must have identical shapes, got "
                f"{tuple(current_per_token_logps.shape)} and {tuple(completion_mask.shape)}"
            )
        if raw_rewards.ndim != 1:
            raise ValueError(f"raw_rewards must be 1-D, got shape {tuple(raw_rewards.shape)}")
        if raw_rewards.shape[0] != current_per_token_logps.shape[0]:
            raise ValueError(
                "raw_rewards batch dimension must match current_per_token_logps, got "
                f"{raw_rewards.shape[0]} and {current_per_token_logps.shape[0]}"
            )
        if num_generations <= 1:
            raise ValueError(f"num_generations must be > 1 for DMPO, got {num_generations}")
        if dmpo_temperature <= 0:
            raise ValueError(f"dmpo_temperature must be positive, got {dmpo_temperature}")

        batch_size = raw_rewards.shape[0]
        if batch_size % num_generations != 0:
            raise ValueError(
                "DMPO requires complete prompt rollout groups on the local rank: "
                f"batch_size={batch_size}, num_generations={num_generations}. "
                "Use a group-preserving sampler/gather design before enabling distributed cross-rank groups."
            )
        num_prompt_groups = batch_size // num_generations

        mask = completion_mask.to(dtype=current_per_token_logps.dtype)
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_scores = (current_per_token_logps * mask).sum(dim=-1) / seq_lengths

        group_scores = seq_scores.view(num_prompt_groups, num_generations)
        group_rewards = raw_rewards.detach().float().view(num_prompt_groups, num_generations)

        target_dist = torch.softmax(group_rewards / dmpo_temperature, dim=-1).detach()
        policy_dist = torch.softmax(group_scores.float(), dim=-1)

        reward_range = group_rewards.max(dim=-1).values - group_rewards.min(dim=-1).values
        uniform_group_mask = reward_range <= reward_range_tolerance
        if skip_zero_advantage_groups:
            valid_group_mask = ~uniform_group_mask
        else:
            valid_group_mask = torch.ones_like(uniform_group_mask, dtype=torch.bool)

        dm_loss_per_group = ((policy_dist - target_dist) ** 2).mean(dim=-1)
        if valid_group_mask.any():
            dm_loss = dm_loss_per_group[valid_group_mask].mean()
        else:
            dm_loss = group_scores.sum() * 0.0

        with torch.no_grad():
            eps = torch.finfo(torch.float32).eps
            metric_policy = policy_dist.detach()
            metric_target = target_dist.detach()
            if valid_group_mask.any():
                metric_policy_valid = metric_policy[valid_group_mask]
                metric_target_valid = metric_target[valid_group_mask]
                metric_scores_valid = group_scores.detach().float()[valid_group_mask]
            else:
                metric_policy_valid = metric_policy.new_zeros((1, num_generations))
                metric_target_valid = metric_target.new_zeros((1, num_generations))
                metric_scores_valid = group_scores.detach().float().new_zeros((1, num_generations))

            target_entropy = -(
                metric_target_valid * torch.log(metric_target_valid.clamp_min(eps))
            ).sum(dim=-1).mean()
            policy_entropy = -(
                metric_policy_valid * torch.log(metric_policy_valid.clamp_min(eps))
            ).sum(dim=-1).mean()
            target_policy_mse = ((metric_policy_valid - metric_target_valid) ** 2).mean(dim=-1).mean()
            target_policy_l1 = (metric_policy_valid - metric_target_valid).abs().mean(dim=-1).mean()
            target_to_policy_kl = (
                metric_target_valid
                * (
                    torch.log(metric_target_valid.clamp_min(eps))
                    - torch.log(metric_policy_valid.clamp_min(eps))
                )
            ).sum(dim=-1).mean()

            metrics = {
                "target_entropy": target_entropy,
                "policy_entropy": policy_entropy,
                "target_policy_mse": target_policy_mse,
                "target_policy_l1": target_policy_l1,
                "target_to_policy_kl": target_to_policy_kl,
                "target_max": metric_target_valid.max(),
                "target_min": metric_target_valid.min(),
                "policy_max": metric_policy_valid.max(),
                "policy_min": metric_policy_valid.min(),
                "uniform_reward_group_ratio": uniform_group_mask.float().mean(),
                "valid_group_ratio": valid_group_mask.float().mean(),
                "num_groups": torch.tensor(float(num_prompt_groups), device=seq_scores.device),
                "mean_completion_score": metric_scores_valid.mean(),
                "std_completion_score": metric_scores_valid.std(unbiased=False),
            }

        debug = {
            "num_groups": num_prompt_groups,
            "group_size": num_generations,
            "raw_reward_shape": tuple(raw_rewards.shape),
            "seq_score_shape": tuple(group_scores.shape),
            "target_dist_shape": tuple(target_dist.shape),
            "policy_dist_shape": tuple(policy_dist.shape),
            "target_dist": target_dist,
            "policy_dist": policy_dist,
            "seq_scores": group_scores,
            "target_row_sums": target_dist.detach().sum(dim=-1),
            "policy_row_sums": policy_dist.detach().sum(dim=-1),
            "target_requires_grad": target_dist.requires_grad,
            "policy_requires_grad": policy_dist.requires_grad,
        }
        return dm_loss, metrics, debug

    def _reduce_base_policy_loss(self, per_token_loss, completion_mask, inputs, base_loss_type):
        if base_loss_type == "grpo":
            return (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        if base_loss_type == "bnpo":
            return (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        if base_loss_type == "dr_grpo":
            return (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        if base_loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            return (per_token_loss * completion_mask).sum() / normalizer
        raise ValueError(f"Unknown base policy loss type: {base_loss_type}")

    def _compute_dmpo_loss(self, current_per_token_logps, completion_mask, raw_rewards):
        return self._dmpo_loss_from_tensors(
            current_per_token_logps=current_per_token_logps,
            completion_mask=completion_mask,
            raw_rewards=raw_rewards,
            num_generations=self.num_generations,
            dmpo_temperature=self.dmpo_temperature,
            skip_zero_advantage_groups=self.dmpo_skip_zero_advantage_groups,
        )

    @staticmethod
    def _metric_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().item()
        return float(value)

    def _record_dmpo_metrics(self, mode, base_loss, dm_loss, total_loss, dm_metrics):
        if not self.dmpo_log_metrics:
            return
        prefix = "train/dmpo" if mode == "train" else "eval/dmpo"
        weighted_dm_loss = dm_loss.detach() * float(self.dmpo_beta)
        values = {
            "base_loss": base_loss.detach(),
            "dm_loss": dm_loss.detach(),
            "weighted_dm_loss": weighted_dm_loss,
            "total_loss": total_loss.detach(),
            **dm_metrics,
        }
        for name, value in values.items():
            self._metrics[mode][f"{prefix}/{name}"].append(self._metric_scalar(value))

    def _maybe_print_dmpo_sanity(self, base_loss, dm_loss, total_loss, debug):
        if not self.dmpo_sanity_check or self._dmpo_sanity_printed:
            return
        if not self.accelerator.is_main_process:
            return

        def _short_list(tensor):
            return tensor.detach().float().cpu()[: min(3, tensor.numel())].tolist()

        print("[DMPO_SANITY]")
        print(f"loss_type={self.loss_type}")
        print(f"base_loss_type={self.dmpo_base_loss_type}")
        print(f"num_groups={debug['num_groups']}")
        print(f"group_size={debug['group_size']}")
        print(f"raw_reward_shape={debug['raw_reward_shape']}")
        print(f"seq_score_shape={debug['seq_score_shape']}")
        print(f"target_dist_shape={debug['target_dist_shape']}")
        print(f"policy_dist_shape={debug['policy_dist_shape']}")
        print(f"target_row_sums={_short_list(debug['target_row_sums'])}")
        print(f"policy_row_sums={_short_list(debug['policy_row_sums'])}")
        print(f"target_requires_grad={debug['target_requires_grad']}")
        print(f"policy_requires_grad={debug['policy_requires_grad']}")
        print(f"base_loss={self._metric_scalar(base_loss)}")
        print(f"dm_loss={self._metric_scalar(dm_loss)}")
        print(f"total_loss={self._metric_scalar(total_loss)}")
        self._dmpo_sanity_printed = True

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        adapter_name = inputs.get("_adapter_name", "default")
        mode = "train" if self.model.training else "eval"
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        print(f"[DEBUG _compute_loss] input_ids.shape: {input_ids.shape}")
        print(f"[DEBUG _compute_loss] completion_ids.shape: {completion_ids.shape}")
        print(f"[DEBUG _compute_loss] adapter_name: {adapter_name}")

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_was_provided = old_per_token_logps is not None
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # if self.use_vllm and self.vllm_importance_sampling_correction:
        #     per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        # Originally based on Config flag (use_vllm), but applied to work dependent on 'importance_sampling_ratio' presence
        # if adapter_name == "default" and "importance_sampling_ratio" in inputs and inputs["importance_sampling_ratio"] is not None:
        #     per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        use_importance_weighting = bool(getattr(self.args, "use_importance_weighting", False))
        if use_importance_weighting and adapter_name == "default" and inputs.get("importance_sampling_ratio") is not None:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        dm_metrics = None
        dm_debug = None
        if self.loss_type in {"grpo", "bnpo", "dr_grpo", "dapo"}:
            base_loss_unscaled = self._reduce_base_policy_loss(
                per_token_loss,
                completion_mask,
                inputs,
                base_loss_type=self.loss_type,
            )
            total_loss_unscaled = base_loss_unscaled
        elif self.loss_type == "dmpo":
            raw_rewards = inputs.get("raw_rewards", inputs.get("correctness_reward_per_sample"))
            if raw_rewards is None:
                raise KeyError("DMPO requires raw_rewards or correctness_reward_per_sample in inputs.")
            base_loss_unscaled = self._reduce_base_policy_loss(
                per_token_loss,
                completion_mask,
                inputs,
                base_loss_type=self.dmpo_base_loss_type,
            )
            should_apply_dm = adapter_name == "default"
            if self.dmpo_candidate_scope == "main_only" and adapter_name != "default":
                raise NotImplementedError("DMPO main_only currently supports only the default adapter update path.")
            if should_apply_dm:
                dm_loss_unscaled, dm_metrics, dm_debug = self._compute_dmpo_loss(
                    current_per_token_logps=per_token_logps,
                    completion_mask=completion_mask,
                    raw_rewards=raw_rewards,
                )
                total_loss_unscaled = base_loss_unscaled + self.dmpo_beta * dm_loss_unscaled
            else:
                total_loss_unscaled = base_loss_unscaled
        elif self.loss_type == "pure_dmpo":
            raw_rewards = inputs.get("raw_rewards", inputs.get("correctness_reward_per_sample"))
            if raw_rewards is None:
                raise KeyError("pure_dmpo requires raw_rewards or correctness_reward_per_sample in inputs.")
            if adapter_name == "default":
                dm_loss_unscaled, dm_metrics, dm_debug = self._compute_dmpo_loss(
                    current_per_token_logps=per_token_logps,
                    completion_mask=completion_mask,
                    raw_rewards=raw_rewards,
                )
                base_loss_unscaled = dm_loss_unscaled.new_zeros(())
                total_loss_unscaled = self.dmpo_beta * dm_loss_unscaled
            elif self.dmpo_candidate_scope == "collective":
                base_loss_unscaled = per_token_loss.sum() * 0.0
                total_loss_unscaled = base_loss_unscaled
            else:
                raise NotImplementedError("pure_dmpo main_only currently supports only the default adapter update path.")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = total_loss_unscaled / self.current_gradient_accumulation_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"
        if dm_metrics is not None:
            self._record_dmpo_metrics(
                mode,
                base_loss=base_loss_unscaled,
                dm_loss=dm_loss_unscaled,
                total_loss=total_loss_unscaled,
                dm_metrics=dm_metrics,
            )
            self._maybe_print_dmpo_sanity(
                base_loss=base_loss_unscaled,
                dm_loss=dm_loss_unscaled,
                total_loss=total_loss_unscaled,
                debug=dm_debug,
            )

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # if self.beta != 0.0:
        #     mean_kl = masked_batch_mean(per_token_kl)
        #     self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # mean_entropy = masked_batch_mean(entropies)
        # self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # # Compute the clipped probability ratios
        # is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        # is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        # is_region_clipped = is_low_clipped | is_high_clipped

        # low_clip = masked_batch_mean(is_low_clipped.float())
        # high_clip = masked_batch_mean(is_high_clipped.float())
        # clip_ratio = masked_batch_mean(is_region_clipped.float())

        # gathered_low_clip = self.accelerator.gather(low_clip)
        # self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        # self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        # gathered_high_clip = self.accelerator.gather(high_clip)
        # self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        # self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        # gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        # self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode][f"{adapter_name}/kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode][f"{adapter_name}/train_entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode][f"{adapter_name}/clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode][f"{adapter_name}/clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())

        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode][f"{adapter_name}/clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode][f"{adapter_name}/clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())

        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode][f"{adapter_name}/clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        self._emit_ratio_trace_and_metadata(
            inputs=inputs,
            adapter_name=adapter_name,
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            old_was_provided=old_was_provided,
            log_ratio=log_ratio,
            ratio=coef_1,
            clip_mask=is_region_clipped,
            advantages=advantages,
            completion_mask=completion_mask,
            loss=loss,
        )

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        self._log_lora_fingerprints(mode)

        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            for adapter_name in self.all_adapter_names:
                prompt_key = f"{adapter_name}/prompt"
                
                if prompt_key not in self._logs or len(self._logs[prompt_key]) == 0:
                    continue
                
                if is_rich_available():
                    print_prompt_completions_sample(
                        self._logs[f"{adapter_name}/prompt"],
                        self._logs[f"{adapter_name}/correctness_completion"],
                        self._logs[f"{adapter_name}/correctness_rewards"],
                        self._logs[f"{adapter_name}/correctness_advantages"],
                        self.state.global_step,
                        self.num_completions_to_print,
                    )

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs[f"{adapter_name}/prompt"]),
                    "adapter": [adapter_name] * len(self._logs[f"{adapter_name}/prompt"]),
                    "prompt": list(self._logs[f"{adapter_name}/prompt"]),
                    "completion": list(self._logs[f"{adapter_name}/correctness_completion"]),
                    **{k: list(v) for k, v in self._logs[f"{adapter_name}/correctness_rewards"].items()},
                    "advantage": list(self._logs[f"{adapter_name}/correctness_advantages"]),
                }

                # Diversity rewards 추가 (default 제외)
                # if adapter_name != "default" and f"{adapter_name}/diversity_rewards" in self._logs:
                #     for k, v in self._logs[f"{adapter_name}/diversity_rewards"].items():
                #         table[f"diversity_{k}"] = list(v)
                #     table["diversity_advantage"] = list(self._logs[f"{adapter_name}/diversity_advantages"])
                # Diversity rewards 추가 (default 제외, no_div 모드에서는 스킵)
                if (
                    adapter_name != "default" 
                    and not self.args.no_div
                    and f"{adapter_name}/diversity_rewards" in self._logs
                    and len(self._logs[f"{adapter_name}/diversity_rewards"]) > 0
                ):
                    for k, v in self._logs[f"{adapter_name}/diversity_rewards"].items():
                        if len(v) > 0:  # 빈 리스트 체크
                            table[f"diversity_{k}"] = list(v)
                    if f"{adapter_name}/diversity_advantages" in self._logs and len(self._logs[f"{adapter_name}/diversity_advantages"]) > 0:
                        table["diversity_advantage"] = list(self._logs[f"{adapter_name}/diversity_advantages"])

                df_base = pd.DataFrame(table)
                images_raw = self._logs.get(f"{adapter_name}_images", [])

                for logging_backend in self.args.report_to:
                    if logging_backend == "wandb":
                        if images_raw:
                            images = []
                            for image_list in images_raw:
                                images.append([wandb.Image(image) for image in image_list])
                            df = pd.concat([df_base, pd.Series(images, name="image")], axis=1, copy=False)
                        else:
                            df = df_base

                        if self.wandb_log_unique_prompts:
                            df = df.drop_duplicates(subset=["prompt"])

                        wandb.log({f"{adapter_name}/completions": wandb.Table(dataframe=df)})

                    if logging_backend == "trackio":
                        df = df_base
                        trackio.log({f"{adapter_name}/completions": trackio.Table(dataframe=df)})
                        
    # def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
    #     mode = "train" if self.model.training else "eval"
    #     metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

    #     # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
    #     # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
    #     if mode == "eval":
    #         metrics = {f"eval_{key}": val for key, val in metrics.items()}

    #     logs = {**logs, **metrics}
    #     super().log(logs, start_time)
    #     self._metrics[mode].clear()

    #     if self.accelerator.is_main_process and self.log_completions:
    #         if is_rich_available():
    #             print_prompt_completions_sample(
    #                 self._logs["prompt"],
    #                 self._logs["completion"],
    #                 self._logs["rewards"],
    #                 self._logs["advantages"],
    #                 self.state.global_step,
    #                 self.num_completions_to_print,
    #             )

    #         table = {
    #             "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
    #             "prompt": self._logs["prompt"],
    #             "completion": self._logs["completion"],
    #             **self._logs["rewards"],
    #             "advantage": self._logs["advantages"],
    #         }

    #         df_base = pd.DataFrame(table)
    #         images_raw = self._logs["images"] or []

    #         for logging_backend in self.args.report_to:
    #             if logging_backend == "wandb":
    #                 if images_raw:
    #                     images = []
    #                     for image_list in self._logs["images"]:
    #                         images.append([wandb.Image(image) for image in image_list])
    #                     df = pd.concat([df_base, pd.Series(images, name="image")], axis=1, copy=False)
    #                 else:
    #                     df = df_base

    #                 if self.wandb_log_unique_prompts:
    #                     df = df.drop_duplicates(subset=["prompt"])

    #                 wandb.log({"completions": wandb.Table(dataframe=df)})

    #             if logging_backend == "trackio":
    #                 if images_raw:
    #                     # TODO: Implement once supported upstream https://github.com/gradio-app/trackio/issues/334
    #                     logger.info("Skipping image logging for Trackio")
    #                     df = df_base
    #                     # images = []
    #                     # for image_list in self._logs["images"]:
    #                     #     images.append([trackio.Image(image) for image in image_list])
    #                     # df = pd.concat([df_base, pd.Series(images, name="image")], axis=1, copy=False)
    #                 else:
    #                     df = df_base

    #                 trackio.log({"completions": trackio.Table(dataframe=df)})


    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        for i,j in model.named_parameters(): 
            print(f"{i}: {j.numel()} - requires_grad: {j.requires_grad}")
        print(f"saving checkpoint at step {self.state.global_step}")
            
        if self.args.hub_model_id is None:
            model_name = Path(self.trainer_output).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
