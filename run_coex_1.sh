#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COEX_ENV_DIR="/home/nlplab/anaconda3/envs/coex"
PYTHON="${COEX_ENV_DIR}/bin/python"
export PATH="${COEX_ENV_DIR}/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CHECK_ONLY=0
export MASTER_PORT=29511

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_FLASHINFER=1
export TORCH_CHECKPOINT_USE_REENTRANT=0
# expandable_segments:True는 vLLM CuMemAllocator(sleep mode)와 충돌 → 제거.
# https://github.com/pytorch/pytorch/issues/147851
export VLLM_USE_V1=0

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B-Instruct}"

if [[ $MODEL_PATH == *"Qwen2.5-Math-1.5B-Instruct"* ]]; then
    MODEL_ABB="Qwen2.5-Math-1.5B-Instruct"
    GPU_MEM_UTIL=0.7
    MAX_EVAL_GTOK=4096
elif [[ $MODEL_PATH == *"Phi-4-mini-instruct"* ]]; then
    MODEL_ABB="Phi-4-mini-instruct"
    GPU_MEM_UTIL=0.6
    MAX_EVAL_GTOK=4096
elif [[ $MODEL_PATH == *"DeepSeek-R1-Distill-Qwen-1.5B"* ]]; then
    MODEL_ABB="DeepSeek-R1-Distill-Qwen-1.5B"
    GPU_MEM_UTIL=0.7
    MAX_EVAL_GTOK=4096
elif [[ $MODEL_PATH == *"Llama-3.2-3B-Instruct"* ]]; then
    MODEL_ABB="Llama-3.2-3B-Instruct"
    GPU_MEM_UTIL=0.5
    MAX_EVAL_GTOK=4096
elif [[ $MODEL_PATH == *"Qwen/Qwen3-0.6B"* ]]; then
    MODEL_ABB="Qwen3-0.6B"
    GPU_MEM_UTIL=0.8
    MAX_EVAL_GTOK=2048
elif [[ $MODEL_PATH == *"Qwen/Qwen3-1.7B"* ]]; then
    MODEL_ABB="Qwen3-1.7B"
    GPU_MEM_UTIL=0.8
    MAX_EVAL_GTOK=2048
else
    echo "Error: Model path '$MODEL_PATH' does not match any known models."
    exit 1
fi


echo "Selected Model: $MODEL_ABB (MaxEvalGTok: $MAX_EVAL_GTOK, VllmGpuMemUtil: $GPU_MEM_UTIL)"

loss_type="grpo"
diversity_reward_type="main_weak_correctness_bonus" # "one_minus_bleu_score" "trace_jaccard" "policy_repulsion_margin" "main_weak_correctness_bonus"

# CoEx 4/3/3 rollout layout:
# - default/main adapter sees all 10 candidates and updates with correctness GRPO.
# - each diversity adapter generates 3 rollouts and uses one_minus_bleu_score for diversity reward.
num_completion_per_diversity_adapter=3
num_completion_main_adapter=4
num_diversity_adapters=2
num_generations=$(( num_completion_per_diversity_adapter * num_diversity_adapters + num_completion_main_adapter ))

# Keep original CoEx default: diversity adapters use correctness/diversity weighted sum.
# Set True only for a diversity-only ablation.
no_correctness=False
no_div=False
learning_rate=1e-6
beta=0.04
max_steps=1000 # 1000
use_importance_weighting=False
policy_repulsion_target="all_other" # "all_other" "default_only"
vllm_enable_sleep_mode=True
trust_remote_code=False
clear_KV_cache_after_generation=True
gradient_checkpointing=True
policy_repulsion_aggregation="mean" # "mean" "max"
logprob_token_chunk_size=64
memory_profiling=True
memory_profile_interval=1
adapter_sanity_check_steps=10
# Leave empty for a fresh run. Setting a checkpoint path automatically adds
# --resume_from_checkpoint and marks the W&B run/experiment name with [RESUME].
resume_from_checkpoint=""

# Used only when diversity_reward_type="trace_jaccard".
trace_jaccard_ngram_size=3
trace_jaccard_aggregation="max"
# intra_adapter: compare each diversity rollout against the other rollouts from the same diversity adapter.
# all_other: compare against all candidates except this adapter own rollouts.
diversity_comparison_scope="all_other"
diversity_bleu_balance_mode="source_balanced"   # "sample_balanced" "source_balanced"
diversity_source_main_weight=0.5

# Correctness/diversity combination.
correctness_gated=False
correctness_threshold=0.5
correctness_weight_specialist=0.7
diversity_weight_specialist=0.3

# WandB settings.
wandb_mode="online" # "online" "offline" "disabled"
wandb_project="CoEx-${MODEL_ABB}"
wandb_entity="none" # WandB entity/team name, or "none".
wandb_log_unique_prompts=False
log_completions=True

case "$diversity_reward_type" in
    trace_jaccard|trace_jaccard3|policy_repulsion_margin|policy_repulsion_margin_barrier|external|one_minus_bleu|one_minus_bleu_score|1-bleu|main_weak_correctness_bonus)
        ;;
    *)
        echo "Error: unsupported diversity_reward_type '$diversity_reward_type'."
        echo "Supported values: trace_jaccard, trace_jaccard3, policy_repulsion_margin, policy_repulsion_margin_barrier, external, one_minus_bleu, one_minus_bleu_score, 1-bleu, main_weak_correctness_bonus"
        exit 1
        ;;
esac

if [[ "$diversity_reward_type" == "trace_jaccard" || "$diversity_reward_type" == "trace_jaccard3" ]]; then
    diversity_aggregation="$trace_jaccard_aggregation"
elif [[ "$diversity_reward_type" == "one_minus_bleu" || "$diversity_reward_type" == "one_minus_bleu_score" || "$diversity_reward_type" == "1-bleu" ]]; then
    diversity_aggregation="mean_bleu"
else
    diversity_aggregation="$policy_repulsion_aggregation"
fi

num_processes=1

generation_batch_size=$(( num_generations * num_processes ))

STEPS=("${max_steps}")

# Compact abbreviations
_scope=$(echo "$diversity_comparison_scope" | sed 's/intra_adapter/intra/;s/all_other/all/')
_bal=""
if [[ "$diversity_reward_type" == *"bleu"* || "$diversity_reward_type" == "1-bleu" ]]; then
    _bal="-bal$(echo "$diversity_bleu_balance_mode" | sed 's/sample_balanced/samp/;s/source_balanced/src/')"
fi
_repulse=""
if [[ "$diversity_reward_type" == *"repulsion"* ]]; then
    _repulse="-tgt${policy_repulsion_target}-agg${policy_repulsion_aggregation}"
fi

resume_args=()
resume_tag=""
if [[ -n "$resume_from_checkpoint" ]]; then
    resume_args=(--resume_from_checkpoint "$resume_from_checkpoint")
    resume_tag="[RESUME]"
fi

experiment_name="${resume_tag}CoEx_${diversity_bleu_balance_mode}_lt-${loss_type}_dt-${diversity_reward_type}_G${num_generations}-m${num_completion_main_adapter}-d${num_diversity_adapters}x${num_completion_per_diversity_adapter}_lr${learning_rate}_beta${beta}_scope-${_scope}${_bal}_cw${correctness_weight_specialist}-dw${diversity_weight_specialist}_gate-${correctness_gated}_ndiv-${no_div}_ncorr-${no_correctness}_iw-${use_importance_weighting}${_repulse}-$(date +%m%d_%H%M%S)"

echo "num_generations=$num_generations"
echo "per_device_train_batch_size passed as: $num_generations"
echo "rollout_layout=main${num_completion_main_adapter}+div${num_diversity_adapters}x${num_completion_per_diversity_adapter}"
echo "loss_type=$loss_type"
echo "diversity_reward_type=$diversity_reward_type"
echo "diversity_comparison_scope=$diversity_comparison_scope"
echo "no_correctness=$no_correctness (False => diversity adapters use correctness/diversity weighted sum)"
echo "specialist_weights=correctness:${correctness_weight_specialist},diversity:${diversity_weight_specialist}"

if [[ "$CHECK_ONLY" == "1" ]]; then
    "$PYTHON" -m py_compile main.py custom_coex_trainer.py custom_coex_config.py data_utils.py prepare_model_inf.py utils.py reward_func.py rewards/__init__.py rewards/parsing.py rewards/correctness.py rewards/diversity.py rewards/format.py
    "$PYTHON" -c "import main; import prepare_model_inf; print('CHECK_ONLY_OK')"
    exit 0
fi

"$PYTHON" -m accelerate.commands.launch --num_processes $num_processes \
    --main_process_port "$MASTER_PORT" \
    main.py \
    --model_path $MODEL_PATH \
    --use_vllm True \
    --vllm_mode "colocate" \
    --vllm_gpu_memory_utilization ${GPU_MEM_UTIL} \
    --num_generations $num_generations \
    --generation_batch_size $generation_batch_size \
    --per_device_train_batch_size $num_generations \
    --gradient_accumulation_steps 2 \
    --loss_type ${loss_type} \
    --max_completion_length 3584 \
    --max_prompt_length 512 \
    --learning_rate $learning_rate \
    --max_steps $max_steps \
    --vllm_tensor_parallel_size 1 \
    --mini_batch_size 10 \
    --logprob_token_chunk_size ${logprob_token_chunk_size} \
    --memory_profiling ${memory_profiling} \
    --memory_profile_interval ${memory_profile_interval} \
    --adapter_sanity_check_steps ${adapter_sanity_check_steps} \
    --bf16 True \
    --fp16 False \
    --save_steps 100 \
    --logging_steps 1 \
    --epsilon 0.2 \
    --beta 0.04 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --temperature 0.7 \
    --log_completions ${log_completions} \
    --wandb_log_unique_prompts ${wandb_log_unique_prompts} \
    --correctness_gated ${correctness_gated} \
    --correctness_threshold ${correctness_threshold} \
    --num_completion_per_diversity_adapter ${num_completion_per_diversity_adapter} \
    --num_completion_main_adapter ${num_completion_main_adapter} \
    --num_diversity_adapters ${num_diversity_adapters} \
    --no_correctness ${no_correctness} \
    --no_div ${no_div} \
    --use_importance_weighting ${use_importance_weighting} \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --wandb_mode ${wandb_mode} \
    --experiment_name "${experiment_name}" \
    --run_name "${experiment_name}" \
    --report_to wandb \
    --trace_jaccard_ngram_size ${trace_jaccard_ngram_size} \
    --trace_jaccard_aggregation ${trace_jaccard_aggregation} \
    --diversity_comparison_scope ${diversity_comparison_scope} \
    --correctness_weight_specialist ${correctness_weight_specialist} \
    --diversity_weight_specialist ${diversity_weight_specialist} \
    --diversity_reward_type ${diversity_reward_type} \
    --policy_repulsion_target ${policy_repulsion_target} \
    --vllm_enable_sleep_mode ${vllm_enable_sleep_mode} \
    --trust_remote_code ${trust_remote_code} \
    --clear_KV_cache_after_generation ${clear_KV_cache_after_generation} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --policy_repulsion_aggregation ${policy_repulsion_aggregation} \
    --diversity_bleu_balance_mode ${diversity_bleu_balance_mode} \
    --diversity_source_main_weight ${diversity_source_main_weight} \
    "${resume_args[@]}"

echo "========== Starting model merge for all checkpoints =========="
for STEP in "${STEPS[@]}"; do
    echo ">>> Merging checkpoint-${STEP}..."
    "$PYTHON" prepare_model_inf.py \
        --base_model_path $MODEL_PATH \
        --base_dir . \
        --experiment_name "$experiment_name" \
        --step $STEP
done
echo "========== Model merge complete =========="
