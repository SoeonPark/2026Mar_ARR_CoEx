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

import inspect
import os
import textwrap
import time
import warnings
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
    from peft import PeftConfig, PeftModel

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
                        adapter_path = os.path.join(self.lora_temp_dir, f"{adapter_name}_adapter")
                        os.makedirs(adapter_path, exist_ok=True)

                        # Save each adapter to a separate directory
                        model.set_adapter(adapter_name)
                        # Save only PEFT adapters to the adapter_path
                        model.save_pretrained(adapter_path, save_adapter = True, save_config = True)

                        self.lora_modules.append(
                            {
                                "name": adapter_name,
                                "path": adapter_path,
                                "id": adapter_index + 1, 
                            }
                        )
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

    import math
    
    def _match_adapter_param(self, name: str, adapter_name: str) -> bool:
        if adapter_name == "default":
            return (".default." in name) or ("_default_" in name) or (".default_" in name) or ("_default." in name)

        tag1 = f".{adapter_name}."
        tag2 = f"_{adapter_name}_"
        tag3 = f".{adapter_name}_"
        tag4 = f"_{adapter_name}."
        return (tag1 in name) or (tag2 in name) or (tag3 in name) or (tag4 in name)

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
                self.model.set_adapter(adapter_name)
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
        # adapter_info: {"name": str, "path": str, "id": int}
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
            ok = eng.add_lora(
                LoRARequest(
                    lora_name=adapter_info["name"],
                    lora_int_id=adapter_info["id"],
                    lora_local_path=adapter_info["path"],
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

                self.model.set_adapter(adapter)
                self.enable_all_lora_grads(self.model)
                adapter_path = adapter_info["path"]
                
                # DeepSpeed ZeRO-3의 경우 전체 파라미터를 gather
                with gather_if_zero3(list(self.model.parameters())):
                    if self.accelerator.is_main_process:
                        print(f"  >> [vLLM Sync] Merging Adapter '{adapter}'...")
                        self.model.save_pretrained(
                            adapter_path, 
                            save_adapter=True, 
                            save_config=True, 
                            safe_serialization=True
                        )
                    print(f"  >> [vLLM Sync] Adapter '{adapter}' saved to '{adapter_path}'.")
                    
                    # Synchronize with barrier for DeepSpeed
                    if zero_stage_3:
                        torch.distributed.barrier()
                
                self._reload_lora_in_vllm_colocate(adapter_info)

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
                            elif self.diversity_comparison_scope == "all_other":
                                comparison_indices = [
                                    index
                                    for index in range(len(all_pass_data["completions"]))
                                    if index not in source_index_set
                                ]
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
                            
                            generation_batch_per_adapter[adapter_name] = self._score_completions_diversity(
                                generation_batch_per_adapter[adapter_name], 
                                pass_data_per_adapter[adapter_name], 
                                forward_kwargs_per_adapter[adapter_name],
                                other_data, 
                                adapter_name,
                                current_num_completions
                            )
                            
                            if self.correctness_gated is True:
                                correctness_rewards = generation_batch_per_adapter[adapter_name]["correctness_reward_per_sample"]
                                mask = correctness_rewards < self.correctness_threshold
                                generation_batch_per_adapter[adapter_name]["diversity_advantages"][mask] = 0.0

                            # no_correctness 모드: diversity만으로 학습
                            if self.args.no_correctness:
                                generation_batch_per_adapter[adapter_name]["advantages"] = generation_batch_per_adapter[adapter_name]["diversity_advantages"]
                            else:
                                generation_batch_per_adapter[adapter_name]["advantages"] = \
                                    self.args.correctness_weight_specialist * generation_batch_per_adapter[adapter_name]["correctness_advantages"] + \
                                    self.args.diversity_weight_specialist * generation_batch_per_adapter[adapter_name]["diversity_advantages"]

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

                    # Ensure answer_info_list is a list
                    if answer_info_list is None:
                        answer_info_list = [None] * len(completions)

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

                    for i, (prompt, completion, adv, div_adv, corr_reward, answer_info) in enumerate(
                        zip(prompts, completions, advantages, diversity_advantages, correctness_rewards, answer_info_list)
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

                for adapter_name in self.all_adapter_names:
                    generation_batch_per_adapter[adapter_name] = split_pixel_values_by_grid(generation_batch_per_adapter[adapter_name])
                    if self.loss_type not in {"dmpo", "pure_dmpo"}:
                        generation_batch_per_adapter[adapter_name] = shuffle_sequence_dict(generation_batch_per_adapter[adapter_name])
                    
                # generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                # self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]

                buffered_per_adapter = {}
                for adapter_name in self.all_adapter_names:
                    adapter_batch = generation_batch_per_adapter[adapter_name]
                    # 각 adapter의 배치를 steps_per_generation 개로 split
                    split_batches = split_tensor_dict(adapter_batch, self.args.steps_per_generation)
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
        self.model.set_adapter(adapter_name)
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
                        print(f"  >> Using LoRA adapter in vLLM x: {adapter_name} (id: {adapter_info['id']})")

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
                    unwrapped_model.set_adapter(adapter_name)
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
                unwrapped_model.set_adapter(adapter_name)
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
            for local_idx, original_idx in enumerate(indices):
                # breakpoint()
                final_prompt_ids_list[original_idx] = sub_prompt_ids[local_idx]
                final_completion_ids_list[original_idx] = sub_completion_ids[local_idx]
                if sub_logps is not None:
                    final_sampling_per_token_logps_list[original_idx] = sub_logps[local_idx]
                final_num_items_in_batch[original_idx] = sub_num_items.item()
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
            self.model.set_adapter(adapter_name)
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
        adapter_name: str, num_completions: int
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

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

            # 2) optional correctness gate (FIX: corr 정의)
            gate = None
            corr = generation_batch.get("correctness_reward_per_sample", None)
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

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

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
