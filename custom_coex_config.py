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

import warnings
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class CoExConfig(TrainingArguments):
    r"""
    Configuration class for the [`CoExTrainer`].

    This class includes only the parameters that are specific to CoEx training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`str`, `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`CoExTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents
            the model from generating different logprobs for the same input.
        cast_lm_head_to_fp32 (`bool`, *optional*, defaults to `False`):
            Whether to cast the language modeling head of the policy and reference models to float32. As recommended by
            the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only supported when the model
            has untied word embedding and language modeling head layers i.e. `tie_word_embeddings` in the model config
            is False.

        > Parameters that control the data preprocessing
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes * per_device_batch_size
            * gradient_accumulation_steps) must be evenly divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        generation_batch_size: (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
            generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
        steps_per_generation: (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
            with `generation_batch_size`.
        temperature (`float`, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        use_transformers_paged (`bool`, *optional*, defaults to `False`):
            Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
            paged implementation will be used for generation instead of the default padded implementation. This
            parameter is only effective when `use_vllm` is set to `False`.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        vllm_guided_decoding_regex (`str`, *optional*):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)

        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step and woken
            for weight sync and generation.

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
            training speed.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        delta (`float`, *optional*):
            Enables the upper clipping bound in two-sided CoEx loss when set to a float. If `None` (default), standard
            CoEx clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in
            the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        importance_sampling_level (`str`, *optional*, defaults to `"token"`):
            Controls whether importance sampling ratios are computed at the `"token"` or `"sequence"` level. `"token"`
            keeps the raw per-token log-probability ratios (one weight per token). `"sequence"` averages the
            log-probability ratios across valid tokens to produce a single ratio per sequence. The [GSPO
            paper](https://huggingface.co/papers/2507.18071) shows that sequence-level sampling often yields more
            stable training and better alignment with sequence-level rewards.
        reward_weights (`list[float]`, *optional*):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Specifies the scaling strategy for rewards. Supported values are:

            - `True` or `"group"` (default): rewards are scaled by the standard deviation within each group, ensuring
              unit variance within a group.
            - `"batch"`: rewards are scaled by the standard deviation across the entire batch, as recommended in the
              [PPO Lite paper](https://huggingface.co/papers/2508.08221).
            - `False` or `"none"`: no scaling is applied. The [Dr. CoEx
              paper](https://huggingface.co/papers/2503.20783) recommends not scaling rewards, as scaling by the
              standard deviation introduces a question-level difficulty bias.
        loss_type (`str`, *optional*, defaults to `"dapo"`):
            Specifies the loss formulation to use. Supported values are:

            - `"coex"`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to
              length bias—this approach tends to prefer shorter completions with positive advantages and longer ones
              with negative advantages.
            - `"dr_coex"`: Aggregates token-level losses by normalizing with a global constant. This method was
              introduced in the [Dr. CoEx paper](https://huggingface.co/papers/2503.20783) to eliminate length bias.
              The value of the constant corresponds to `max_completion_length`.
            - `"dapo"` (default): Aggregates token-level losses by normalizing with the number of active token in the
              global accumulated batch. This method was introduced in the [DAPO
              paper](https://huggingface.co/papers/2503.14476) to eliminate length bias.
            - `"bnpo"`: Aggregates token-level losses by normalizing with the number of active token in the local
              batch. Note that normalization is performed over the local batch only, so results may slightly vary
              depending on the local batch size, despite a constant effective batch size. When using
              `per_device_train_batch_size==1`, the loss is equivalent to the CoEx loss.
        mask_truncated_completions (`bool`, *optional*, defaults to `False`):
            When enabled, truncated completions are excluded from the loss calculation, preventing them from being
            incorrectly penalized and introducing noise during training. According to the
            [DAPO](https://huggingface.co/papers/2503.14476) paper, this is a good practice for training stability.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.
        top_entropy_quantile (`float`, *optional*, defaults to `1.0`):
            ρ parameter from [Beyond the 80/20 Rule](https://huggingface.co/papers/2506.01939). Keeps in the policy
            loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence
            position, improving results. Range: `[0.0-1.0]`. A value of `0.0` masks all but the highest entropy token;
            `1.0` keeps all tokens. The paper recommends a value of `0.2`. If used with
            `mask_truncated_completions=True`, only tokens from non-truncated completions are considered.
        use_liger_loss (`bool`, *optional*):
            Whether to use Liger loss.

            <Deprecated version="0.25.0">

            Parameter `use_liger_loss` is deprecated and will be removed in version 0.28.0. Use `use_liger_kernel`
            instead.

            </Deprecated>

        vllm_importance_sampling_correction (`bool`, *optional*, defaults to `True`):
            Whether to apply Truncated Importance Sampling (TIS) between vLLM completion logprobs and recomputed
            logprobs. [Your Efficient RL Framework Secretly Brings You Off-Policy RL
            Training](https://fengyao.notion.site/off-policy-rl) highlights that using a separate generation framework
            (such as vLLM) can introduce off-policy effects due to subtle implementation differences between generation
            and training backends. TIS is proposed as a remedy for this issue.
        vllm_importance_sampling_cap (`float`, *optional*, defaults to `2.0`):
            Truncation parameter C for Truncated Importance Sampling (TIS). This sets an upper bound on the importance
            sampling ratio, improving training stability.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or
            `trackio`.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        wandb_log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, all prompts
            are logged.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]
    
    num_diversity_adapters: int = field(
        default=2,
        metadata={"help": "Number of diversity adapters to use."}
    )
    num_completion_per_diversity_adapter: int = field(
        default=3,
        metadata={"help": "Number of completions to generate per diversity adapter."}
    )
    num_completion_main_adapter: int = field(
        default=4,
        metadata={"help": "Number of completions to generate for the main adapter."}
    )
    diversity_reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each diversity reward function. Must match the number of diversity reward functions. If `None`, all "
            "diversity rewards are weighted equally with weight `1.0`."
        },
    )
    diversity_comparison_scope: str = field(
        default="intra_adapter",
        metadata={
            "help": "Comparison pool for diversity rewards: 'intra_adapter' compares only within the same diversity "
            "adapter rollout group, excluding the current rollout; 'all_other' compares against all candidates except "
            "the source adapter's own rollouts.",
        },
    )
    diversity_bleu_balance_mode: str = field(
        default="sample_balanced",
        metadata={
            "help": "Averaging mode for 1-BLEU diversity reward when diversity_comparison_scope='all_other'. "
            "'sample_balanced': every reference completion receives equal weight (simple average over all N_m + (K-1)*N_d "
            "references). 'source_balanced': compute per-source average similarity first, then combine with "
            "diversity_source_main_weight, making the reward independent of rollout allocation size."
        },
    )
    diversity_source_main_weight: float = field(
        default=0.5,
        metadata={
            "help": "alpha_m in [0, 1] for source-balanced 1-BLEU diversity reward. Controls how strongly the "
            "diversity policy is encouraged to differ from the main/default policy. 0.5 weights main and other "
            "diversity sources equally; 0.6 is recommended when inference uses only the main policy."
            "Only used when diversity_bleu_balance_mode='source_balanced'."
        },
    )
    diversity_bleu_exclude_self: bool = field(
        default=True,
        metadata={
            "help": "Whether to exclude a sample from comparing against itself in 1-BLEU diversity reward. "
            "Applies to both BLEU and trace-Jaccard rewards. In all_other scope this is a no-op since the "
            "source adapter's completions are never in the reference pool. In intra_adapter scope this "
            "prevents inflated self-similarity pulling down the reward."
        },
    )
    diversity_reward_debug: bool = field(
        default=False,
        metadata={
            "help": "If True, log detailed per-adapter diversity reward diagnostics (reference pool sizes, "
            "balance mode, reward statistics, source component similarities) every "
            "diversity_reward_debug_steps optimization steps."
        },
    )
    diversity_reward_debug_steps: int = field(
        default=20,
        metadata={
            "help": "Emit detailed diversity reward debug logs every N optimization steps when "
            "diversity_reward_debug=True."
        },
    )
    diversity_bleu_text_scope: str = field(
        default="full_completion",
        metadata={
            "help": "Text extraction scope for 1-BLEU diversity reward. "
            "'full_completion': BLEU over the entire completion text (default). "
            "'reasoning_only': BLEU over <think>...</think> blocks only. "
            "'answer_only': BLEU over the final answer portion only."
        },
    )
    mini_batch_size: int = field(
        default=None,
        metadata={"help": "Mini-batch size for forward."}
    )
    logprob_token_chunk_size: int = field(
        default=64,
        metadata={
            "help": "Number of completion positions projected through the LM head at once when computing logprobs. "
            "Smaller values reduce peak vocabulary-logit memory."
        },
    )
    logprob_sanity_check: bool = field(
        default=False,
        metadata={
            "help": "Compare PEFT-wrapper and chunked selected logprobs once per active adapter."
        },
    )
    memory_profiling: bool = field(
        default=True,
        metadata={"help": "Log CUDA memory at generation, scoring, backward, and optimizer boundaries."},
    )
    memory_profile_interval: int = field(
        default=1,
        metadata={"help": "Profile CUDA memory every N global steps."},
    )
    adapter_sanity_check_steps: int = field(
        default=10,
        metadata={
            "help": "Log detailed per-adapter LoRA norms, gradients, optimizer membership, and optimizer-step deltas "
            "every N optimizer steps (the first step is always logged). Set to 0 to disable detailed checks."
        },
    )
    log_adapter_switches: bool = field(
        default=True,
        metadata={"help": "Log every explicit PEFT adapter switch without changing adapter behavior."},
    )
    vllm_lora_hash_check: bool = field(
        default=False,
        metadata={"help": "After each vLLM LoRA export, compare training-side adapter tensors with the exported safetensors by SHA-256."},
    )
    vllm_lora_hash_check_interval: int = field(
        default=1,
        metadata={"help": "Run vLLM LoRA tensor hash checks every N optimizer steps when vllm_lora_hash_check is enabled."},
    )
    vllm_lora_hash_check_strict: bool = field(
        default=True,
        metadata={"help": "Raise immediately if a vLLM LoRA export hash check fails."},
    )
    source_owned_trace: bool = field(
        default=False,
        metadata={"help": "Log ratio/source-policy invariants and write response-level source-owned update metadata."},
    )
    source_owned_trace_log_steps: int = field(
        default=1,
        metadata={"help": "Emit source-owned ratio trace logs every N optimizer steps when source_owned_trace is enabled."},
    )

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model and reference model
    model_init_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `CoExTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )
    cast_lm_head_to_fp32: bool = field(
        default=False,
        metadata={
            "help": "Whether to cast the language modeling head of the policy and reference, models to float32."
            "As recommended by the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only supported when the model"
            " has untied word embedding and language modeling head layers i.e. `tie_word_embeddings` in the model config is False."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in CoEx we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: int | None = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        }, 
    )
    max_completion_length: int | None = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: bool | None = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: int | None = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: float | None = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation. This parameter is only effective when `use_vllm` is set to `False`."
        },
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step "
            "and woken for weight sync and generation."
        },
    )
    vllm_guided_decoding_regex: str | None = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Pop out KV Cache after generation to free up GPU memory for training when `vllm_mode` is set to `'colocate'`.
    clear_KV_cache_after_generation: bool = field(
        default=True,
        metadata={
            "help": "Whether to clear the vLLM KV cache after generation to free up GPU memory for training. This setting "
            "only applies when `vllm_mode` is set to `'colocate'`."
        },
    )

    # Parameters that control the training
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: float | None = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided CoEx loss when set to a float. If `None` "
            "(default), standard CoEx clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. CoEx paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', "
            "'dr_grpo', 'dmpo', and 'pure_dmpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GrPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the grpo loss."
        },
    )
    dmpo_base_loss_type: str = field(
        default="grpo",
        metadata={"help": "Base policy loss for DMPO: grpo, bnpo, dr_grpo, or dapo."},
    )
    dmpo_beta: float = field(
        default=1.0,
        metadata={"help": "Weight for the DMPO distribution-matching regularizer."},
    )
    dmpo_temperature: float = field(
        default=1.0 / 15.0,
        metadata={"help": "Temperature for the reward-induced target distribution in DMPO."},
    )
    dmpo_skip_zero_advantage_groups: bool = field(
        default=False,
        metadata={"help": "If True, omit uniform-reward groups from the DMPO regularizer."},
    )
    dmpo_candidate_scope: str = field(
        default="main_only",
        metadata={"help": "Candidate scope for DMPO. main_only is implemented; collective is a placeholder."},
    )
    dmpo_log_metrics: bool = field(
        default=True,
        metadata={"help": "Log DMPO scalar diagnostics."},
    )
    dmpo_sanity_check: bool = field(
        default=False,
        metadata={"help": "Print one detailed DMPO sanity-check block on the first valid batch."},
    )

    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `0.0` masks all but the highest entropy token; `1.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )
    use_liger_loss: bool = field(
        default=None,
        metadata={"help": "Whether to use the Liger CoEx loss."},
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply Truncated Importance Sampling (TIS) between vLLM completion logprobs and "
            "recomputed logprobs. Your Efficient RL Framework Secretly Brings You Off-Policy RL "
            "Training highlights that using a separate generation framework (such as vLLM) can introduce off-policy "
            "effects due to subtle implementation differences between generation and training backends. TIS is "
            "proposed as a remedy for this issue."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=2.0,
        metadata={
            "help": "Truncation parameter C for Truncated Importance Sampling (TIS). This sets an upper bound on the "
            "importance sampling ratio, improving training stability."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    wandb_log_unique_prompts: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )

    # Settings for Specialist Adapter
    correctness_weight_specialist: float = field(
        default = 0.7,
        metadata={"help": "Weight for correctness reward for the specialist adapter."}
    )
    diversity_weight_specialist: float = field(
        default = 0.3,
        metadata={"help": "Weight for diversity reward for the specialist adapter."}
    )
    correctness_gated: bool = field(
        default = False,
        metadata={"help": "Whether to mask diversity reward depends on correctness score for diversity adapters."}
    )
    correctness_threshold: float = field(
        default = 0.5,
        metadata={"help": "Threshold for correctness score to gate diversity reward."}
    )
    use_importance_weighting: bool = field(
        default=False,
        metadata={"help": "Whether to apply importance weighting to main-adapter samples."},
    )

    # Settings for Distribution Repulsion
    diversity_reward_type: str = field(
        default="external",
        metadata={"help": "external|one_minus_bleu|one_minus_bleu_score|1-bleu|trace_jaccard|trace_jaccard3|policy_repulsion_margin|policy_repulsion_margin_barrier|main_weak_correctness_bonus"},
    )
    trace_jaccard_ngram_size: int = field(
        default=3,
        metadata={"help": "N-gram size for trace_jaccard rewards. The proposed reward uses 3."},
    )
    trace_jaccard_aggregation: str = field(
        default="max",
        metadata={"help": "Comparison aggregation for trace_jaccard. Currently only max is supported."},
    )
    policy_repulsion_target: str = field(
        default="all_other",
        metadata={"help": "all_other|default_only"},
    )
    policy_repulsion_batch_size: int = field(
        default=1,
        metadata={"help": "micro-batch size when evaluating other adapters' logprobs"},
    )
    policy_repulsion_aggregation: str = field(
        default="max",
        metadata={
            "help": "How to aggregate other adapters' logprobs for margin gap. "
            "'max': gap = src_logp - max(other_logps). Repulsion from the closest (most similar) adapter. "
            "'mean': gap = src_logp - mean(other_logps). Repulsion from the average of all other adapters."
        },
    )
    policy_repulsion_gate_by_correctness: bool = field(
        default=False,
        metadata={"help": "gate diversity reward by correctness reward threshold"},
    )
    policy_repulsion_gate_threshold: float = field(
        default=0.0,
        metadata={"help": "threshold on correctness_reward_per_sample to enable diversity reward"},
    )
    no_div: bool = field(
        default=False,
        metadata={"help": "If True, diversity adapters exist but train with correctness reward only (no diversity loss). For ablation study."}
    )
    no_correctness: bool = field(
        default=False,
        metadata={"help": "If True, diversity adapters train with diversity reward only (no correctness loss). For ablation study."}
    )
    policy_repulsion_prefix_len: int = field(
        default=0,
        metadata={"help": "If >0, compute repulsion on first L completion tokens (prefix-only)."},
    )
    policy_repulsion_barrier_margin: float = field(
        default=0.0,
        metadata={"help": "Minimum required margin m for barrier: gap = logp_src - logp_other. Active if >0."},
    )
    policy_repulsion_barrier_tau: float = field(
        default=0.0,
        metadata={"help": "If >0, use soft barrier with softplus temperature tau; else hard ReLU barrier."},
    )

    # Settings for main_weak_correctness_bonus
    main_weak_correctness_bonus_weight: float = field(
        default=1.0,
        metadata={
            "help": "Multiplier applied to the raw main_weak_correctness_bonus reward "
            "(c(x,y) * (1 - main_correct_rate(x))) before it is group-normalized into "
            "diversity_advantages. Only used when diversity_reward_type='main_weak_correctness_bonus'. "
            "Note: the *effective* coefficient on the final advantage is "
            "diversity_weight_specialist * main_weak_correctness_bonus_weight, since "
            "diversity_weight_specialist is applied again after normalization."
        },
    )
    main_weak_correctness_bonus_log_by_source: bool = field(
        default=True,
        metadata={"help": "If True, emit per-diversity-source main_weak_correctness_bonus metrics."},
    )
    main_weak_correctness_bonus_use_answer_correct_only: bool = field(
        default=True,
        metadata={
            "help": "If True (recommended, only supported value for now), main_weak_correctness_bonus uses the "
            "pure answer_correct_float field rather than the mixed correctness_reward scalar for both c(x,y) and "
            "main_correct_rate(x)."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

        if self.trace_jaccard_ngram_size <= 0:
            raise ValueError("trace_jaccard_ngram_size must be positive")
        if self.trace_jaccard_aggregation != "max":
            raise ValueError("trace_jaccard_aggregation currently supports only max")
        if self.logprob_token_chunk_size <= 0:
            raise ValueError("logprob_token_chunk_size must be positive")
        if self.memory_profile_interval <= 0:
            raise ValueError("memory_profile_interval must be positive")
        if self.adapter_sanity_check_steps < 0:
            raise ValueError("adapter_sanity_check_steps must be non-negative")
        if self.vllm_lora_hash_check_interval <= 0:
            raise ValueError("vllm_lora_hash_check_interval must be positive")
        if self.source_owned_trace_log_steps <= 0:
            raise ValueError("source_owned_trace_log_steps must be positive")
        if self.diversity_comparison_scope not in {"intra_adapter", "all_other"}:
            raise ValueError("diversity_comparison_scope must be one of {'intra_adapter', 'all_other'}")
        if self.diversity_bleu_balance_mode not in {"sample_balanced", "source_balanced"}:
            raise ValueError(
                "diversity_bleu_balance_mode must be one of {'sample_balanced', 'source_balanced'}, "
                f"got {self.diversity_bleu_balance_mode!r}"
            )
        if not (0.0 <= self.diversity_source_main_weight <= 1.0):
            raise ValueError(
                f"diversity_source_main_weight must be in [0.0, 1.0], got {self.diversity_source_main_weight}"
            )
        if self.diversity_reward_debug_steps <= 0:
            raise ValueError("diversity_reward_debug_steps must be positive")
        if self.diversity_bleu_text_scope not in {"full_completion", "reasoning_only", "answer_only"}:
            raise ValueError(
                "diversity_bleu_text_scope must be one of {'full_completion', 'reasoning_only', 'answer_only'}, "
                f"got {self.diversity_bleu_text_scope!r}"
            )
        valid_diversity_reward_types = {
            "external",
            "one_minus_bleu",
            "one_minus_bleu_score",
            "1-bleu",
            "trace_jaccard",
            "trace_jaccard3",
            "policy_repulsion_margin",
            "policy_repulsion_margin_barrier",
            "main_weak_correctness_bonus",
        }
        if self.diversity_reward_type not in valid_diversity_reward_types:
            raise ValueError(
                "diversity_reward_type must be one of "
                f"{sorted(valid_diversity_reward_types)}, got {self.diversity_reward_type!r}"
            )
        valid_loss_types = {"grpo", "bnpo", "dr_grpo", "dapo", "dmpo", "pure_dmpo"}
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {sorted(valid_loss_types)}, got {self.loss_type!r}")
        valid_dmpo_base_loss_types = {"grpo", "bnpo", "dr_grpo", "dapo"}
        if self.dmpo_base_loss_type not in valid_dmpo_base_loss_types:
            raise ValueError(
                "dmpo_base_loss_type must be one of "
                f"{sorted(valid_dmpo_base_loss_types)}, got {self.dmpo_base_loss_type!r}"
            )
        if self.dmpo_temperature <= 0:
            raise ValueError("dmpo_temperature must be positive")
        if self.dmpo_beta < 0:
            raise ValueError("dmpo_beta must be non-negative")
        if self.dmpo_candidate_scope not in {"main_only", "collective"}:
            raise ValueError("dmpo_candidate_scope must be one of {'main_only', 'collective'}")

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "CoEx requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.use_liger_loss is not None:
            warnings.warn(
                "The `use_liger_loss` argument is deprecated and will be removed in version 0.28.0. Please use "
                "`use_liger_kernel` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_liger_kernel = self.use_liger_loss

        if self.loss_type in {"dmpo", "pure_dmpo"} and self.use_liger_kernel:
            raise NotImplementedError("DMPO is not implemented for the Liger fused loss path.")

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided CoEx loss yet.")
