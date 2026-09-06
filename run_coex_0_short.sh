#!/usr/bin/env bash
# Short validation run (max_steps=20) derived from run_coex_0.sh.
# Purpose: verify OOM safety, backward/optimizer stability, vLLM sleep/wake,
#          reward/loss/grad_norm NaN, checkpoint save, wandb logging
#          before committing to the full 1000-step run.
#
# Differences from run_coex_0.sh:
#   max_steps           : 1000 → 20
#   save_steps          : 50   → 10   (checkpoint at step 10 & 20)
#   adapter_sanity_check_steps: 10 → 5
#   COEX_LOGPROB_SANITY_CHECK  : explicitly 0 (already verified)
#   LOG_FILE / GPU_CSV  : /tmp/coex-short0-TIMESTAMP.{log,csv}
#   Checkpoint merge    : runs at step 20 instead of 1000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COEX_ENV_DIR="/home/nlplab/anaconda3/envs/coex"
PYTHON="${COEX_ENV_DIR}/bin/python"
export PATH="${COEX_ENV_DIR}/bin:$PATH"
export CUDA_VISIBLE_DEVICES="0"

CHECK_ONLY=0
export MASTER_PORT=29510

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_FLASHINFER=1
export TORCH_CHECKPOINT_USE_REENTRANT=0
# expandable_segments:True는 vLLM CuMemAllocator(sleep mode)와 충돌 → 제거.
# https://github.com/pytorch/pytorch/issues/147851
export VLLM_USE_V1=0

# Sanity check already verified in smoke test — disable for production-like run.
export COEX_LOGPROB_SANITY_CHECK=0

MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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
    GPU_MEM_UTIL=0.8
    MAX_EVAL_GTOK=4096
elif [[ $MODEL_PATH == *"Llama-3.2-3B-Instruct"* ]]; then
    MODEL_ABB="Llama-3.2-3B-Instruct"
    GPU_MEM_UTIL=0.6
    MAX_EVAL_GTOK=4096
elif [[ $MODEL_PATH == *"Qwen/Qwen3-0.6B"* ]]; then
    MODEL_ABB="Qwen3-0.6B"
    GPU_MEM_UTIL=0.7
    MAX_EVAL_GTOK=2048
else
    echo "Error: Model path '$MODEL_PATH' does not match any known models."
    exit 1
fi

echo "Selected Model: $MODEL_ABB (MaxEvalGTok: $MAX_EVAL_GTOK, VllmGpuMemUtil: $GPU_MEM_UTIL)"

loss_type="grpo"
diversity_reward_type="trace_jaccard"
num_completion_per_diversity_adapter=2
num_completion_main_adapter=4
num_diversity_adapters=3
num_generations=$(( num_completion_per_diversity_adapter * num_diversity_adapters + num_completion_main_adapter ))
no_correctness=False
no_div=False
learning_rate=1e-6
max_steps=20              # SHORT RUN: was 1000
use_importance_weighting=False
policy_repulsion_target="all_other"
vllm_enable_sleep_mode=True
trust_remote_code=False
clear_KV_cache_after_generation=True
gradient_checkpointing=True
policy_repulsion_aggregation="mean"
logprob_token_chunk_size=64
memory_profiling=True
memory_profile_interval=1
adapter_sanity_check_steps=5  # SHORT RUN: was 10 (check at step 5, 10, 15, 20)

trace_jaccard_ngram_size=3
trace_jaccard_aggregation="max"

correctness_gated=False
correctness_threshold=0.5
correctness_weight_specialist=0.7
diversity_weight_specialist=0.3

wandb_mode="online"
wandb_project="CoEx-${MODEL_ABB}"
wandb_entity="none"
wandb_log_unique_prompts=False
log_completions=True

case "$diversity_reward_type" in
    trace_jaccard|trace_jaccard3|policy_repulsion_margin|policy_repulsion_margin_barrier|external|one_minus_bleu|one_minus_bleu_score|1-bleu)
        ;;
    *)
        echo "Error: unsupported diversity_reward_type '$diversity_reward_type'."
        exit 1
        ;;
esac

if [[ "$diversity_reward_type" == "trace_jaccard" || "$diversity_reward_type" == "trace_jaccard3" ]]; then
    diversity_aggregation="$trace_jaccard_aggregation"
else
    diversity_aggregation="$policy_repulsion_aggregation"
fi

num_processes=1
generation_batch_size=$(( num_generations * num_processes ))

STEPS=("${max_steps}")

TS="$(date +%m%d_%H%M%S)"
LOG_FILE="/tmp/coex-short0-${TS}.log"
GPU_CSV="/tmp/coex-short0-gpu-${TS}.csv"

experiment_name="coex_SHORT20_lr_${learning_rate}-base_${num_completion_main_adapter}-${num_diversity_adapters}_div_${num_completion_per_diversity_adapter}_lT_${loss_type}-${TS}"

echo "========================================================"
echo "  SHORT RUN (max_steps=${max_steps})"
echo "  Log  : $LOG_FILE"
echo "  GPU  : $GPU_CSV"
echo "  Exp  : $experiment_name"
echo "========================================================"
echo "num_generations=$num_generations"

# Background GPU memory recorder (samples every 5 s).
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 5 > "$GPU_CSV" 2>&1 &
GPU_MON_PID=$!
echo "GPU monitor PID: $GPU_MON_PID  →  $GPU_CSV"

cleanup() {
    kill "$GPU_MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

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
    --mini_batch_size 1 \
    --logprob_token_chunk_size ${logprob_token_chunk_size} \
    --memory_profiling ${memory_profiling} \
    --memory_profile_interval ${memory_profile_interval} \
    --adapter_sanity_check_steps ${adapter_sanity_check_steps} \
    --bf16 True \
    --fp16 False \
    --save_steps 10 \
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
    --experiment_name ${experiment_name} \
    --run_name ${experiment_name} \
    --report_to wandb \
    --trace_jaccard_ngram_size ${trace_jaccard_ngram_size} \
    --trace_jaccard_aggregation ${trace_jaccard_aggregation} \
    --correctness_weight_specialist ${correctness_weight_specialist} \
    --diversity_weight_specialist ${diversity_weight_specialist} \
    --diversity_reward_type ${diversity_reward_type} \
    --policy_repulsion_target ${policy_repulsion_target} \
    --vllm_enable_sleep_mode ${vllm_enable_sleep_mode} \
    --trust_remote_code ${trust_remote_code} \
    --clear_KV_cache_after_generation ${clear_KV_cache_after_generation} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --policy_repulsion_aggregation ${policy_repulsion_aggregation} \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
echo "  Training exit code: $EXIT_CODE"
echo "  Log saved to      : $LOG_FILE"
echo ""
echo "  Quick NaN check:"
grep -iE "nan|inf" "$LOG_FILE" | grep -iE "grad_norm|loss|reward" | tail -20 || echo "  (no NaN/Inf lines found in grad_norm/loss/reward)"
echo ""
echo "  GPU peak (MiB):"
awk -F',' 'NR>1 {gsub(/ MiB/,"",$3); if($3+0 > max) max=$3+0} END {print "  peak used =", max, "MiB"}' "$GPU_CSV" || true
echo "========================================================"

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Training FAILED with exit code $EXIT_CODE. Skipping model merge."
    exit $EXIT_CODE
fi

echo "========== Merging checkpoint-${max_steps} =========="
"$PYTHON" prepare_model_inf.py \
    --base_model_path $MODEL_PATH \
    --base_dir . \
    --experiment_name $experiment_name \
    --step $max_steps
echo "========== Model merge complete =========="
