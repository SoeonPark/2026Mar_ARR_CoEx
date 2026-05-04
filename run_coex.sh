#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="1"

CHECK_ONLY="${CHECK_ONLY:-0}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb online
    wandb login "$WANDB_API_KEY"
else
    export WANDB_MODE=offline
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_FLASHINFER=1
export TORCH_CHECKPOINT_USE_REENTRANT=0
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_V1=0

MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-auto}"

if [[ $MODEL_PATH == *"DeepSeek-R1-Distill-Qwen-1.5B"* ]]; then
    MODEL_ABB="DeepSeek-R1-Distill-Qwen-1.5B"
    MAX_EVAL_GTOK=4096
else
    echo "Error: Model path '$MODEL_PATH' does not match any known models."
    exit 1
fi

echo "Selected Model: $MODEL_ABB (Util: $GPU_MEM_UTIL, MaxEvalGTok: $MAX_EVAL_GTOK)"

loss_type="grpo"
diversity_reward_type="policy_repulsion_margin"
num_completion_per_diversity_adapter=2
num_completion_main_adapter=4
num_diversity_adapters=3
num_generations=$(( num_completion_per_diversity_adapter * num_diversity_adapters + num_completion_main_adapter ))
no_correctness=False
no_div=False
learning_rate=1e-4
max_steps=1000
use_importance_weighting=False
policy_repulsion_target="all_other"
vllm_enable_sleep_mode=True
trust_remote_code=False
clear_KV_cache_after_generation=True
gradient_checkpointing=False
policy_repulsion_aggregation="max"

num_processes=1

generation_batch_size=$(( num_generations * num_processes ))

STEPS=(1000 900 800 700)

wandb_project="CoEx-${MODEL_ABB}"

experiment_name="coex_lr_${learning_rate}-base_${num_completion_main_adapter}-${num_diversity_adapters}_div_${num_completion_per_diversity_adapter}_lT_${loss_type}-ndiv_${no_div}-ncorr_${no_correctness}-divType_${diversity_reward_type}-correctnessGated_False-useIW_${use_importance_weighting}-repulsionTarget_${policy_repulsion_target}-aggregation_${policy_repulsion_aggregation}-$(date +%m%d_%H%M%S)"

echo "num_generations=$num_generations"
echo "per_device_train_batch_size passed as: $num_generations"

if [[ "$CHECK_ONLY" == "1" ]]; then
    python -m py_compile main.py custom_coex_trainer.py custom_coex_config.py data_utils.py prepare_model_inf.py utils.py reward_func.py rewards/__init__.py rewards/parsing.py rewards/correctness.py rewards/diversity.py rewards/format.py
    python -c "import main; import prepare_model_inf; print('CHECK_ONLY_OK')"
    exit 0
fi

python -m accelerate.commands.launch --num_processes $num_processes main.py \
    --model_path $MODEL_PATH \
    --use_vllm True \
    --vllm_mode "colocate" \
    --num_generations $num_generations \
    --generation_batch_size $generation_batch_size \
    --per_device_train_batch_size $num_generations \
    --gradient_accumulation_steps 2 \
    --max_completion_length 3584 \
    --max_prompt_length 512 \
    --learning_rate $learning_rate \
    --max_steps $max_steps \
    --vllm_tensor_parallel_size 1 \
    --mini_batch_size 1 \
    --save_steps 100 \
    --logging_steps 1 \
    --epsilon 0.2 \
    --beta 0.04 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --temperature 0.7 \
    --log_completions True \
    --correctness_gated False \
    --num_completion_per_diversity_adapter ${num_completion_per_diversity_adapter} \
    --num_completion_main_adapter ${num_completion_main_adapter} \
    --num_diversity_adapters ${num_diversity_adapters} \
    --no_correctness ${no_correctness} \
    --no_div ${no_div} \
    --wandb_project $wandb_project \
    --experiment_name $experiment_name \
    --diversity_reward_type ${diversity_reward_type} \
    --policy_repulsion_target ${policy_repulsion_target} \
    --vllm_enable_sleep_mode ${vllm_enable_sleep_mode} \
    --trust_remote_code ${trust_remote_code} \
    --clear_KV_cache_after_generation ${clear_KV_cache_after_generation} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --policy_repulsion_aggregation ${policy_repulsion_aggregation} \


echo "========== Starting model merge for all checkpoints =========="
for STEP in "${STEPS[@]}"; do
    echo ">>> Merging checkpoint-${STEP}..."
    python prepare_model_inf.py \
        --base_model_path $MODEL_PATH \
        --base_dir . \
        --experiment_name $experiment_name \
        --step $STEP
done
echo "========== Model merge complete =========="
