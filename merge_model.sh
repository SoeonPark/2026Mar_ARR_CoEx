#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/home/nlplab/ssd2/soeon/2026Mar_ARR_CoEx"
TRAINER_OUTPUT="${BASE_DIR}/trainer_output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP_START=1000 # 700
STEP_END=1000
STEP_INTERVAL=100

declare -A GROUP_BASE_MODEL
GROUP_BASE_MODEL["DeepSeek"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GROUP_BASE_MODEL["Phi"]="microsoft/Phi-4-mini-instruct"
GROUP_BASE_MODEL["Qwen2.5"]="Qwen/Qwen2.5-Math-1.5B-Instruct"
GROUP_BASE_MODEL["Qwen0.6"]="Qwen/Qwen3-0.6B"
GROUP_BASE_MODEL["Llama"]="meta-llama/Llama-3.2-3B-Instruct"
GROUP_BASE_MODEL["Qwen3.1.7"]="Qwen/Qwen3-1.7B"

DEEPSEEK_MODELS=(
    # "coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0609_173218"
    # "DMPO_singleSource_G10_beta1_temp0.0667_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0616_134603"
    # "coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0617_202854"
    # "DMPO_collective_G10_main4_div3x2_beta1_temp0.0667_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0617_103737"
    # "CoEx_GRPO_oneMinusBLEU_G10_main4_div3x2_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0621_010126"
    "CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0707_121837"
)

PHI_MODELS=(
    "coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0612_033207"
)

QWEN_SMALL_MODELS=(
    "coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0607_133430"
)

QWEN_MATH_MODELS=(
    # "CoEx_source_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m10-d0x0_lr1e-6_beta0.04_scope-intra-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0711_182450"
    # "CoEx_source_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m4-d2x3_lr1e-6_beta0.04_scope-intra-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0712_144632"
    "CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0713_181313"
)

LLAMA_MODELS=(
    "coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0614_032025"
)

QWEN3_MODELS=(
    # "CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m10-d0x0_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0710_141525"
    "CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0711_164611"
)

merge_group() {
    local group_name=$1
    local base_model=${GROUP_BASE_MODEL[$group_name]}
    shift
    local model_list=("$@")

    echo ""
    echo "=========================================="
    echo "GROUP: ${group_name}  |  base: ${base_model}"
    echo "=========================================="

    for model_item in "${model_list[@]}"; do
        local full_path
        [[ "$model_item" == /* ]] && full_path="$model_item" || full_path="${TRAINER_OUTPUT}/${model_item}"

        local exp_name
        exp_name="$(basename "$full_path")"

        local ckpt_dirs
        mapfile -t ckpt_dirs < <(
            ls -d "${full_path}/checkpoint-"* 2>/dev/null \
            | awk -F'checkpoint-' '{print $NF, $0}' \
            | sort -n \
            | awk -v start="$STEP_START" -v end="$STEP_END" -v interval="$STEP_INTERVAL" \
                '$1 >= start && $1 <= end && $1 % interval == 0 {print $2}'
        )

        if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
            echo "[SKIP] No checkpoints found: $full_path"
            continue
        fi

        echo "  Found checkpoints: ${ckpt_dirs[*]}"

        for ckpt_path in "${ckpt_dirs[@]}"; do
            local step
            step="$(basename "$ckpt_path" | sed 's/checkpoint-//')"

            local out_dir="${BASE_DIR}/merged_output/${exp_name}/checkpoint-${step}"
            if [[ -d "$out_dir" ]]; then
                echo "[SKIP] Already merged: $out_dir"
                continue
            fi

            echo ""
            echo "  [MERGE] ${exp_name} / step${step}"
            echo "    adapter : $ckpt_path"
            echo "    out     : $out_dir"

            python "${SCRIPT_DIR}/prepare_model_inf.py" \
                --base_model_path "$base_model" \
                --adapter_path    "$ckpt_path" \
                --out_dir         "$out_dir" \
                --qproj_path      auto \
                --dtype           bfloat16 \
                --device          cuda
        done
    done
}
# merge_group "DeepSeek" "${DEEPSEEK_MODELS[@]}"
# merge_group "Phi"      "${PHI_MODELS[@]}"
# merge_group "Qwen0.6"     "${QWEN_SMALL_MODELS[@]}"
merge_group "Qwen2.5"     "${QWEN_MATH_MODELS[@]}"
# merge_group "Llama"    "${LLAMA_MODELS[@]}"
# merge_group "Qwen3.1.7"     "${QWEN3_MODELS[@]}"

echo ""
echo "All merge tasks finished."