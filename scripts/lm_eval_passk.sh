#!/bin/bash
# =============================================================================
# lm_eval_passk.sh
# - sampling only: AIME24 + AIME25 pass@16
# =============================================================================

# nohup bash lm_eval_passk.sh > eval_log/eval_passk_gpu1.log 2>&1 &

export CUDA_VISIBLE_DEVICES="1"
export HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0

DATE=$(date +%m%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLING_CONFIG="${SCRIPT_DIR}/eval/configs/eval_sampling.yaml"
CUSTOM_TASKS="${SCRIPT_DIR}/eval/custom_tasks"

# =============================================================================
# 헬퍼 함수
# =============================================================================

get_category() {
    local dir_name=$1
    if [[ "$dir_name" =~ base_10-0_div_0 ]]; then
        echo "GRPO"
    elif [[ "$dir_name" =~ base_[0-9]+-[0-9]+_div_[0-9]+ ]]; then
        echo "Proposed"
    else
        echo "Pretrained"
    fi
}

get_base_model() {
    local path=$1
    if [[ "$path" =~ Qwen3-1\.7B ]]; then
        echo "Qwen3-1.7B"
    elif [[ "$path" =~ Qwen2\.5-Math-1\.5B ]]; then
        echo "Qwen2.5-Math-1.5B"
    elif [[ "$path" =~ DeepSeek-R1-Distill-Qwen-1\.5B ]]; then
        echo "DeepSeek-R1-Distill-Qwen-1.5B"
    else
        echo "DeepSeek-R1-Distill-Qwen-1.5B"
    fi
}

# -----------------------------------------------------------------------------
# run_sampling: AIME24 + AIME25 pass@16, temperature=0.6
# -----------------------------------------------------------------------------
run_sampling() {
    local model_path=$1
    local model_name=$2
    local category=$3
    local max_len=${4:-32768}
    local dtype=${5:-float16}

    local output_path="./eval_results/${DATE}/${category}/sampling/${model_name}"

    echo ""
    echo "=========================================="
    echo "[SAMPLING] AIME24 + AIME25  (pass@16)"
    echo "  path : $model_path"
    echo "  out  : $output_path"
    echo "=========================================="

    lm-eval run \
        --config "$SAMPLING_CONFIG" \
        --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
        --apply_chat_template true \
        --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=${max_len}" \
        --include_path "$CUSTOM_TASKS" \
        --output_path "$output_path"
}

# -----------------------------------------------------------------------------
# eval_checkpoints: checkpoint-XXX 범위 반복
# -----------------------------------------------------------------------------
eval_checkpoints() {
    local run_dir=$1
    local step_start=$2
    local step_end=$3
    local step_interval=$4
    local max_len=${5:-32768}
    local dtype=${6:-float16}

    local dir_name
    dir_name=$(basename "$run_dir")
    local category
    category=$(get_category "$dir_name")
    local base_model
    base_model=$(get_base_model "$run_dir")

    for step in $(seq "$step_start" "$step_interval" "$step_end"); do
        local ckpt_path="${run_dir}/checkpoint-${step}"
        if [[ ! -d "$ckpt_path" ]]; then
            echo "[SKIP] Not found: $ckpt_path"
            continue
        fi
        local model_name="${base_model}/${dir_name}/step${step}"
        run_sampling "$ckpt_path" "$model_name" "$category" "$max_len" "$dtype"
    done
}


# =============================================================================
# 실행 블록
# =============================================================================

# ── 0. Pretrained ─────────────────────────────────────────────────────────────
run_sampling \
    "/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" \
    "DeepSeek-R1-Distill-Qwen-1.5B/Pretrained" \
    "Pretrained" \
    32768

# ── 1. GRPO (10-0_div_0)  step 700~1000 ──────────────────────────────────────
eval_checkpoints \
    "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_external-correctnessGated_False-useIW_False-0205_161145" \
    700 700 100 \
    32768

# # ── 2. Proposed (4-3_div_2)  step 700~1000 ───────────────────────────────────
# eval_checkpoints \
#     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0216_154758" \
#     700 800 100 \
#     32768

# ── 3. DRA-GRPO (나중에) ──────────────────────────────────────────────────────
# run_sampling \
#     "/home/nlplab/.cache/huggingface/hub/models--SpiceRL--DRA-DR.GRPO/snapshots/2e2054a53fab825a5916aeb731c6ae8aa6596e53" \
#     "DeepSeek-R1-Distill-Qwen-1.5B/DRA-GRPO" \
#     "DRA-GRPO" \
#     32768