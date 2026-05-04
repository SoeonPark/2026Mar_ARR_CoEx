# ./train/train.py
import os
import subprocess
import wandb
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
    if coex_config.num_diversity_adapters > 0:
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

from functools import wraps

def attach_set_adapter_callback(model, callback):
    """
    Wrap model.set_adapter(adapter_names) so that callback(model, adapter_names)
    is called after every set_adapter().
    """
    orig = model.set_adapter  # bound method

    @wraps(orig)
    def wrapped(adapter_names, *args, **kwargs):
        out = orig(adapter_names, *args, **kwargs)
        callback(model, adapter_names)
        return out

    model.set_adapter = wrapped
    return model

from torch.distributed.elastic.multiprocessing.errors import record

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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
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
        torch_dtype=torch.float16,
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    for i in range(coex_config.num_diversity_adapters):
        model.add_adapter(f"diversity_{i}", peft_config=peft_config)
        print(f"  >> Added adapter diversity__{i}")
    print(model)
    
    def force_enable_all_lora_grads(model, adapter_names=None):
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    attach_set_adapter_callback(model, force_enable_all_lora_grads)
    
    model.set_adapter("default")
    
    print(f"  >> Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for i,j in model.named_parameters(): 
        print(f"{i}: {j.numel()} - requires_grad: {j.requires_grad}")

    # WandB setup
    wandb_project = f"CoEx-{args.model_path.split('/')[-1]}"
    wandb.init(project=wandb_project, name=args.experiment_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=model_config.trust_remote_code, 
        use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("has chat_template:", getattr(tokenizer, "chat_template", None) is not None)

    trainer = CoExTrainer(
        args=coex_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs_correctness=[correctness_reward_func],
        reward_funcs_diversity=[one_minus_bleu_score],
        completion_store_path=f"completion_storage/{args.experiment_name}",
        trainer_output=f"trainer_output/{args.experiment_name}",
    )

    m = trainer.accelerator.unwrap_model(trainer.model)
    print("is_gradient_checkpointing:", getattr(m, "is_gradient_checkpointing", None))
    print("_gradient_checkpointing_kwargs:", getattr(m, "_gradient_checkpointing_kwargs", None))

    trainer.train()

    final_checkpoint = os.path.join(f"trainer_output/{args.experiment_name}", f"final_checkpoint_default")
    model.set_adapter("default")
    model.save_pretrained(final_checkpoint)
    print(f"[DONE] Final model saved to: {final_checkpoint}")
    
    for i in range(coex_config.num_diversity_adapters):
        adapter_checkpoint = os.path.join(f"trainer_output/{args.experiment_name}", f"final_checkpoint_diversity_{i}")
        model.set_adapter(f"diversity_{i}")
        model.save_pretrained(adapter_checkpoint)
        print(f"[DONE] Diversity adapter {i} saved to: {adapter_checkpoint}")

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
        metadata={"help": "Diversity reward functions (comma-separated): bleu, bert, levenshtein"}
    )
    wandb_project: str = field(
        default="auto",
        metadata={"help": "WandB project name. Use 'auto' for automatic naming."}
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
    
    wandb.init(project=wandb_project, name=args.experiment_name)

    coex_config.num_generations = (
        coex_config.num_completion_main_adapter + 
        coex_config.num_completion_per_diversity_adapter * coex_config.num_diversity_adapters
    )
    coex_config.generation_batch_size = coex_config.num_generations
    coex_config.per_device_train_batch_size = coex_config.num_generations

    print(f"  >> num_generations: {coex_config.num_generations}")
    print(f"  >> per_device_train_batch_size: {coex_config.per_device_train_batch_size}")
    print(f"  >> generation_batch_size: {coex_config.generation_batch_size}")
        
    main(coex_config, model_config, args, use_vllm=coex_config.use_vllm)
