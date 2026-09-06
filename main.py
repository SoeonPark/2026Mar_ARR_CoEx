# ./train/train.py
import gc
import os
import sys
sys.set_int_max_str_digits(0)
import importlib.util

import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import trl
from trl import TrlParser, ModelConfig, GRPOConfig, GRPOTrainer
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
import time

# Custom imports
from rewards import (
    extract_hash_answer,
    correctness_reward_func,
    levenstein_distance,
    bert_score,
    bleu_score,
    one_minus_bleu_score,
    trace_jaccard_diversity_reward,
    correctness_reward_func_rule
)
from data_utils import (
    get_gsm8k_dataset, 
    get_hf_math_dataset, 
    get_anker_math_dataset, 
    set_random_seed,
    get_open_rs_dataset
)

from transformers import TrainerCallback
import json
from pathlib import Path
from collections import defaultdict

from custom_coex_trainer import CoExTrainer
from custom_coex_config import CoExConfig

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

def build_experiment_name(args, coex_config) -> str:

    timestamp = time.strftime('%m%d_%H%M%S')
    
    # Model short name
    model_short = args.model_path.split('/')[-1].lower()
    model_short = model_short.replace('deepseek-r1-distill-', '').replace('meta-llama-', '').replace('llama-', '')
    
    # Method tag
    if coex_config.loss_type in {"dmpo", "pure_dmpo"}:
        method_tag = (
            f"{coex_config.loss_type}_singleSource_G{coex_config.num_generations}"
            f"_beta{coex_config.dmpo_beta:g}_temp{coex_config.dmpo_temperature:.4g}"
            f"_base{coex_config.dmpo_base_loss_type}"
        )
    elif coex_config.num_diversity_adapters > 0:
            ablation_tag = ""
            if coex_config.no_div:
                ablation_tag = "_noDiv"
            elif coex_config.no_correctness:
                ablation_tag = "_noCorr"
            method_tag = f"coex_{coex_config.num_diversity_adapters}{ablation_tag}-lr_{coex_config.learning_rate}-base_{coex_config.num_completion_main_adapter}-diversity_{coex_config.num_completion_per_diversity_adapter}-correctnessGated_{coex_config.correctness_gated}-lossType_{coex_config.loss_type}"
    else:
        method_tag = "grpo"
        
    # Reward functions tag
    corr_tag = "+".join(args.correctness_reward_funcs)
    if coex_config.num_diversity_adapters > 0 and args.diversity_reward_funcs:
        div_tag = "+".join(args.diversity_reward_funcs)
        reward_tag = f"{corr_tag}_{div_tag}"
    else:
        reward_tag = corr_tag
    
    # Total generations
    total_gens = coex_config.num_generations
    
    if coex_config.diversity_reward_type == 'external':
        return f"{method_tag}_{model_short}-num_gens{total_gens}-reward_funcs{reward_tag}-lossType{coex_config.loss_type}-r_{LORA_R}-alpha_{LORA_ALPHA}-dropout_{LORA_DROPOUT}-{timestamp}"
    else:
        return f"{method_tag}_{model_short}-num_gens{total_gens}-diversityRewardType{coex_config.diversity_reward_type}-r_{LORA_R}-alpha_{LORA_ALPHA}-dropout_{LORA_DROPOUT}-{timestamp}"

from torch.distributed.elastic.multiprocessing.errors import record


def cleanup_colocated_vllm(trainer) -> None:
    """Release a colocated vLLM engine while CUDA is still fully alive."""
    if not (
        getattr(trainer, "use_vllm", False)
        and getattr(trainer, "vllm_mode", None) == "colocate"
        and getattr(trainer, "llm", None) is not None
    ):
        return

    print("[vLLM Cleanup] Starting colocated engine cleanup")
    if getattr(trainer, "_vllm_slept", False):
        trainer.check_for_vllm_wake()

    llm = trainer.llm
    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)
    driver_wrapper = getattr(executor, "driver_worker", None)
    worker = getattr(driver_wrapper, "worker", None)

    if driver_wrapper is not None:
        driver_wrapper.worker = None
    if executor is not None:
        executor.driver_worker = None
    if engine is not None:
        engine.model_executor = None
    trainer.llm = None

    del worker
    del driver_wrapper
    del executor
    del engine
    del llm
    gc.collect()

    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.instance
    if allocator is not None:
        live_allocations = len(allocator.pointer_to_data)
        pool_entries = list(allocator.allocator_and_pools.values())
        allocator.allocator_and_pools.clear()
        print(
            "[vLLM Cleanup] "
            f"live_allocations={live_allocations} "
            f"memory_pools={len(pool_entries)}"
        )
        while pool_entries:
            mem_pool, pluggable_allocator = pool_entries.pop()
            del mem_pool
            del pluggable_allocator
        CuMemAllocator.instance = None
        del allocator
        gc.collect()

    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[vLLM Cleanup] Completed colocated engine cleanup")


@record
def main(coex_config, model_config, args, use_vllm: Optional[bool] = True) -> None:
    set_random_seed(coex_config.seed)

    # Load Dataset
    if args.dataset_name == "gsm8k":
        dataset = get_gsm8k_dataset(split=args.dataset_split)
    elif args.dataset_name == "hf_math":
        dataset = get_hf_math_dataset(split=args.dataset_split)
    elif args.dataset_name == "anker_math":
        dataset = get_anker_math_dataset(split=args.dataset_split)
    elif args.dataset_name == "open_rs":
        dataset = get_open_rs_dataset(split=args.dataset_split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    print(f"  >> Loaded {args.dataset_name} dataset with {len(dataset)} samples from split '{args.dataset_split}'")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_enable_fp32_cpu_offload=False
    # )

    compute_dtype = (
        torch.bfloat16
        if coex_config.bf16 and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=model_config.trust_remote_code)
    flash_attn_available = importlib.util.find_spec("flash_attn") is not None
    attn_impl = "flash_attention_2" if flash_attn_available else "sdpa"
    cfg.attn_implementation = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=cfg,
        quantization_config=bnb_config,
        trust_remote_code=model_config.trust_remote_code, # True, False
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
        # device_map="auto", 
        # device_map={"": torch.cuda.current_device()},
        device_map={"": local_rank}, 
    )

    print("param dtype:", next(model.parameters()).dtype)
    print("cuda bf16 supported:", torch.cuda.is_bf16_supported())

    print("embed dtype:", model.get_input_embeddings().weight.dtype)
    print("lm_head dtype:", model.get_output_embeddings().weight.dtype)
    print("first layer dtype:", next(model.model.parameters()).dtype)
    print("attn impl:", getattr(model.config, "attn_implementation", None))
    print("flash_attn available:", flash_attn_available)

    peft_config = LoraConfig(
        # r=model_config.lora_r,
        # lora_alpha=model_config.lora_alpha,
        # lora_dropout=model_config.lora_dropout,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # task_type="CAUSAL_LM"
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=coex_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, peft_config)
    
    for i in range(coex_config.num_diversity_adapters):
        model.add_adapter(f"diversity_{i}", peft_config=peft_config)
        print(f"  >> Added adapter diversity__{i}")
    print(model)
    
    model.set_adapter("default")
    
    print("  >> Active adapter: default")
    print(f"  >> Active-adapter trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=model_config.trust_remote_code, 
        use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("has chat_template:", getattr(tokenizer, "chat_template", None) is not None)

    diversity_reward_type = coex_config.diversity_reward_type
    if diversity_reward_type in {"trace_jaccard", "trace_jaccard3"}:
        diversity_reward_funcs = [trace_jaccard_diversity_reward]
    elif diversity_reward_type in {"one_minus_bleu", "one_minus_bleu_score", "1-bleu", "external"}:
        diversity_reward_funcs = [one_minus_bleu_score]
    elif diversity_reward_type in {"policy_repulsion_margin", "policy_repulsion_margin_barrier"}:
        # Policy-repulsion modes compute their reward inside CoExTrainer.
        diversity_reward_funcs = []
    elif diversity_reward_type == "main_weak_correctness_bonus":
        # main_weak_correctness_bonus is computed inside CoExTrainer using the
        # already-scored default-adapter correctness, not an external reward_func.
        diversity_reward_funcs = []
    else:
        raise ValueError(f"Unknown diversity_reward_type: {diversity_reward_type}")

    trainer = CoExTrainer(
        args=coex_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs_correctness=[correctness_reward_func],
        reward_funcs_diversity=diversity_reward_funcs,
        completion_store_path=f"completion_storage/{args.experiment_name}",
        trainer_output=f"trainer_output/{args.experiment_name}",
    )

    m = trainer.accelerator.unwrap_model(trainer.model)
    print("is_gradient_checkpointing:", getattr(m, "is_gradient_checkpointing", None))
    print("_gradient_checkpointing_kwargs:", getattr(m, "_gradient_checkpointing_kwargs", None))

    try:
        trainer.train()

        final_checkpoint = os.path.join(
            f"trainer_output/{args.experiment_name}",
            "final_checkpoint_default",
        )
        model.set_adapter("default")
        model.save_pretrained(final_checkpoint)
        print(f"[DONE] Final model saved to: {final_checkpoint}")

        for i in range(coex_config.num_diversity_adapters):
            adapter_checkpoint = os.path.join(
                f"trainer_output/{args.experiment_name}",
                f"final_checkpoint_diversity_{i}",
            )
            model.set_adapter(f"diversity_{i}")
            model.save_pretrained(adapter_checkpoint)
            print(
                f"[DONE] Diversity adapter {i} saved to: "
                f"{adapter_checkpoint}"
            )
    finally:
        cleanup_colocated_vllm(trainer)

@dataclass
class CustomArgument:
    model_path: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        metadata={"help": "Path to the model"}
    )
    dataset_name: str = field(
        default="open_rs",
        metadata={"help": "Dataset name: gsm8k, hf_math, anker_math, open_rs"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    experiment_name: str = field(
        default="auto",
        metadata={"help": "Experiment name. Use 'auto' for automatic naming."}
    )
    
    # Reward function selection (comma-separated)
    correctness_reward_funcs: List[str] = field(
        default_factory=lambda: ["correctness"],
        metadata={"help": "Correctness reward functions (comma-separated): correctness, correctness_rule"}
    )
    diversity_reward_funcs: List[str] = field(
        default_factory=lambda: ["bleu"],
        metadata={"help": "External diversity reward labels; diversity_reward_type selects trainer-native rewards."}
    )
    wandb_project: str = field(
        default="auto",
        metadata={"help": "WandB project name. Use 'auto' for automatic naming."}
    )
    wandb_entity: str = field(
        default="none",
        metadata={"help": "WandB entity/team name. Use none to omit it."}
    )
    wandb_mode: str = field(
        default="offline",
        metadata={"help": "WandB mode: online, offline, or disabled."}
    )

    
if __name__ == "__main__":       
    parser = TrlParser((CoExConfig, ModelConfig, CustomArgument))
    coex_config, model_config, args = parser.parse_args_and_config()

    coex_config.model_init_kwargs = coex_config.model_init_kwargs or {}
    coex_config.model_init_kwargs["trust_remote_code"] = model_config.trust_remote_code

    args.experiment_name = build_experiment_name(args, coex_config) if args.experiment_name == "auto" else args.experiment_name
    
    if args.wandb_project == "auto":
        wandb_project = f"CoEx-{args.model_path.split('/')[-1]}"
    else:
        wandb_project = args.wandb_project
    
    if args.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be one of: online, offline, disabled")

    wandb_entity = None if args.wandb_entity.lower() == "none" else args.wandb_entity
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity is None:
        os.environ.pop("WANDB_ENTITY", None)
    else:
        os.environ["WANDB_ENTITY"] = wandb_entity

    coex_config.run_name = args.experiment_name
    coex_config.report_to = [] if args.wandb_mode == "disabled" else ["wandb"]

    coex_config.num_generations = (
        coex_config.num_completion_main_adapter + 
        coex_config.num_completion_per_diversity_adapter * coex_config.num_diversity_adapters
    )
    coex_config.generation_batch_size = coex_config.num_generations
    coex_config.per_device_train_batch_size = coex_config.num_generations

    print(f"  >> num_generations: {coex_config.num_generations}")
    print(f"  >> per_device_train_batch_size: {coex_config.per_device_train_batch_size}")
    print(f"  >> generation_batch_size: {coex_config.generation_batch_size}")
    print(f"  >> loss_type: {coex_config.loss_type}")
    print(f"  >> dmpo_base_loss_type: {coex_config.dmpo_base_loss_type}")
    print(f"  >> dmpo_beta: {coex_config.dmpo_beta}")
    print(f"  >> dmpo_temperature: {coex_config.dmpo_temperature}")
    print(f"  >> dmpo_skip_zero_advantage_groups: {coex_config.dmpo_skip_zero_advantage_groups}")
    print(f"  >> dmpo_candidate_scope: {coex_config.dmpo_candidate_scope}")

    if coex_config.loss_type in {"dmpo", "pure_dmpo"}:
        source_mixed_rollouts = (
            coex_config.num_diversity_adapters
            * coex_config.num_completion_per_diversity_adapter
            != 0
        )
        if coex_config.dmpo_candidate_scope == "collective":
            print(
                "  >> DMPO candidate scope: collective CoEx-DM mode "
                "(source-mixed rollout group)."
            )
        elif coex_config.dmpo_candidate_scope == "main_only" and source_mixed_rollouts:
            raise ValueError(
                "Faithful DMPO main_only baseline must not use source-mixed rollouts: "
                f"num_diversity_adapters={coex_config.num_diversity_adapters}, "
                f"num_completion_per_diversity_adapter={coex_config.num_completion_per_diversity_adapter}. "
                "Use dmpo_candidate_scope=collective for the explicit CoEx-DM 4/3/3 mode."
            )

    main(coex_config, model_config, args, use_vllm=coex_config.use_vllm)
