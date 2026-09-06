#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SERVER="${SERVER:-10}"

notify() {
  local message="$1"
  curl -s -d "${message}" "ntfy.sh/soeon_server${SERVER}" >/dev/null 2>&1 || true
}

DATE=$(date +%m%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CONFIG="${SCRIPT_DIR}/configs/basic_eval_pretrained.yaml"
SAMPLING_CONFIG="${SCRIPT_DIR}/eval/configs/eval_sampling.yaml"
CUSTOM_TASKS="${SCRIPT_DIR}/eval/custom_tasks"

# Task control — all overridable via env vars at call time:
#   BASE_TASKS="aime24,aime25" ./eval.sh                   # run subset of default tasks
#   BASE_TASKS="aime" EXTRA_INCLUDE_PATH="..." ./eval.sh   # custom task from yaml
#   EXTRA_TASKS="aime_1983_2024" ./eval.sh                 # default tasks + extra
_DEFAULT_TASKS="aime24,aime25,aime,minerva_math500,gsm8k,mmlu_pro_math"
BASE_TASKS="${BASE_TASKS:-${_DEFAULT_TASKS}}"
EXTRA_TASKS="${EXTRA_TASKS:-}"
EXTRA_INCLUDE_PATH="${EXTRA_INCLUDE_PATH:-}"

# Always passes --tasks so the script (not the config YAML) controls task selection.
build_task_args() {
    local tasks="$BASE_TASKS"
    [[ -n "$EXTRA_TASKS" ]] && tasks="${tasks},${EXTRA_TASKS}"
    printf -- '--tasks %s' "$tasks"
    [[ -n "$EXTRA_INCLUDE_PATH" ]] && printf ' --include_path %s' "$EXTRA_INCLUDE_PATH"
}

# Suffix for output/log paths when tasks differ from the default set, to avoid
# clashing with existing results from a different task configuration.
build_step_suffix() {
    local suffix=""
    [[ "$BASE_TASKS" != "$_DEFAULT_TASKS" ]] && suffix="${BASE_TASKS//,/_}"
    if [[ -n "$EXTRA_TASKS" ]]; then
        [[ -n "$suffix" ]] && suffix+="+"
        suffix+="${EXTRA_TASKS//,/_}"
    fi
    [[ -n "$suffix" ]] && printf '+%s' "$suffix"
}

BASE_PATH="/home/nlplab/ssd2/soeon/2026Mar_ARR_CoEx/merged_output"

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
    if [[ "$path" =~ DeepSeek-R1-Distill-Qwen-1\.5B ]]; then
        echo "DeepSeek-R1-Distill-Qwen-1.5B"
    elif [[ "$path" =~ Phi-4-mini ]]; then
        echo "Phi-4-mini-instruct"
    elif [[ "$path" =~ Qwen0\.6 ]]; then
        echo "Qwen0.6"
    elif [[ "$path" =~ Qwen2\.5-Math-1\.5B ]]; then
        echo "Qwen2.5-Math-1.5B"
    elif [[ "$path" =~ Llama-3\.2-3B ]]; then
        echo "Llama-3.2-3B-Instruct"
    else
        echo "Model-Unknown"
    fi
}

find_existing_eval_output() {
    local group_name=$1
    local category=$2
    local dir_name=$3
    local step=$4
    local suffix="${5:-}"   # optional: e.g. "+aime_1983_2024"

    local matches=()
    local candidate

    shopt -s nullglob
    matches=(./eval_results/*/"${group_name}"/"${category}"/"${dir_name}"/"step${step}${suffix}")
    shopt -u nullglob

    for candidate in "${matches[@]}"; do
        if [[ -d "$candidate" ]] && [[ -n "$(find "$candidate" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

run_eval() {
    local model_path=$1
    local model_name=$2
    local output_path=$3
    local max_len=$4
    local max_gen=$5
    local dtype=${6:-float16}
    local config=${7:-$CONFIG}
    local fewshot_as_multiturn=${8:-true}

    echo ""
    echo "=========================================="
    echo "[EVAL] ${model_name}"
    echo "  Out : $output_path"
    echo "=========================================="

    notify "[START] Pretrained | ${model_name}"

    local fewshot_flag=""
    [[ "$fewshot_as_multiturn" == "true" ]] && fewshot_flag="--fewshot_as_multiturn"

    local extra_task_args
    extra_task_args="$(build_task_args)"

    if lm-eval run \
        --config "$config" \
        --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
        --apply_chat_template true \
        --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
        $fewshot_flag \
        $extra_task_args \
        --output_path "$output_path"; then
        notify "[DONE] Pretrained | ${model_name}"
    else
        notify "[ERROR] Pretrained | ${model_name} — check logs"
    fi
}

run_eval_pretrained() {
    local model_path=$1
    local model_name=$2
    local max_len=$3
    local max_gen=$4
    local dtype=${5:-float16}
    local config=${6:-$CONFIG}
    local fewshot_as_multiturn=${7:-true}

    local base_output="./eval_results/${DATE}/Pretrained/${model_name}"
    run_eval "$model_path" "$model_name" "$base_output" "$max_len" "$max_gen" "$dtype" "$config" "$fewshot_as_multiturn"
}

run_sampling() {
    local model_path=$1
    local model_name=$2
    local category=$3
    local group_name=$4
    local max_len=${5:-32768}
    local dtype=${6:-bfloat16}
    local config=${7:-$SAMPLING_CONFIG}

    local output_path="./eval_results/${DATE}/${category}/sampling/${group_name}/${model_name}"

    echo ""
    echo "=========================================="
    echo "[SAMPLING] Group: $group_name"
    echo "  Model : $model_name"
    echo "  Out   : $output_path"
    echo "=========================================="

    notify "[START] ${group_name} | ${model_name}"

    if lm-eval run \
        --config "$SAMPLING_CONFIG" \
        --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
        --apply_chat_template true \
        --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=${max_len}" \
        --include_path "$CUSTOM_TASKS" \
        --output_path "$output_path"; then
        notify "[DONE] ${group_name} | ${model_name}"
    else
        notify "[ERROR] ${group_name} | ${model_name} — check logs"
    fi
}


eval_checkpoints() {
    local run_dir=$1
    local group_name=$2
    local step_start=$3
    local step_end=$4
    local step_interval=$5
    local max_len=$6
    local max_gen=$7
    local dtype=${8:-float16}
    local config=${9:-$CONFIG}

    local dir_name
    dir_name=$(basename "$run_dir")

    local category
    category=$(get_category "$dir_name")

    local base_model
    base_model=$(get_base_model "$run_dir")

    local step_suffix
    step_suffix="$(build_step_suffix)"

    for step in $(seq "$step_start" "$step_interval" "$step_end"); do
        local ckpt_path="${run_dir}/checkpoint-${step}"

        if [[ ! -d "$ckpt_path" ]]; then
            echo "[SKIP] Not found: $ckpt_path"
            continue
        fi

        local output_path="./eval_results/${DATE}/${group_name}/${category}/${dir_name}/step${step}${step_suffix}"
        local log_file="eval_log/${group_name}_${category}_${dir_name}_step${step}${step_suffix}.log"
        mkdir -p "$(dirname "$log_file")"

        local existing_output=""
        if existing_output="$(find_existing_eval_output "$group_name" "$category" "$dir_name" "$step" "$step_suffix")"; then
            echo "[SKIP] Already evaluated: ${group_name} | ${category} | ${dir_name} | step${step}${step_suffix}"
            echo "       found: $(realpath -m "$existing_output")"
            echo "       new would be: $(realpath -m "$output_path")"
            notify "[SKIP] Already evaluated: ${group_name} | ${category} | ${dir_name} | step${step}${step_suffix}"
            continue
        fi

        echo "=========================================="
        echo "[EVAL] ${group_name} | ${category}"
        echo "Run  : ${dir_name}"
        echo "Step : ${step}"
        echo "Path : ${ckpt_path}"
        echo "Out  : $(realpath -m "$output_path")"
        echo "Log  : $(realpath -m "$log_file")"
        echo "=========================================="

        notify "[START] ${group_name} | ${category} | ${dir_name} | step${step}"

        local extra_task_args
        extra_task_args="$(build_task_args)"

        if lm-eval run \
            --config "$config" \
            --model_args "pretrained=${ckpt_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
            --apply_chat_template true \
            --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
            $extra_task_args \
            --output_path "$output_path" \
            > "$log_file" 2>&1; then

            notify "[DONE] ${group_name} | ${category} | ${dir_name} | step${step}"
        else
            notify "[ERROR] ${group_name} | ${category} | ${dir_name} | step${step}"
        fi
    done
}

eval_specific_steps() {
    local run_dir=$1
    local group_name=$2
    local max_len=$3
    shift 3
    local steps=("$@")

    for step in "${steps[@]}"; do
        eval_checkpoints "$run_dir" "$group_name" "$step" "$step" 1 "$max_len" "$max_len" bfloat16
    done
}

GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
TIME_INTERVAL="${TIME_INTERVAL:-3m}"

echo ">>> Waiting for GPU ${GPU_ID} to be free..."
while true; do
    PYTHON_PROCESS_COUNT="$(
        nvidia-smi -i "${GPU_ID}" \
            --query-compute-apps=process_name \
            --format=csv,noheader 2>/dev/null \
        | awk 'BEGIN{IGNORECASE=1} /python/{c++} END{print c+0}'
    )"

    if [[ "${PYTHON_PROCESS_COUNT}" -eq 0 ]]; then
        echo ">>> GPU ${GPU_ID} is free. Starting eval..."
        notify "GPU ${GPU_ID} is free. Starting eval."
        break
    fi

    echo ">>> GPU ${GPU_ID} busy (python=${PYTHON_PROCESS_COUNT}), waiting ${TIME_INTERVAL}..."
    sleep "${TIME_INTERVAL}"
done

notify "Temporary DeepSeek eval started"

# # 2) 433 Only for 700 yet 
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0617_202854"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000
# 700 1000 100 32768 32768 bfloat16

# # GRPO
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0617_202854"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# GRPO (433 -- No Diversity)
# RUN_NAME="${BASE_PATH}/CoEx_GRPO_oneMinusBLEU_G10_main4_div3x2_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0624_225048"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# # GRPO (433)
# RUN_NAME="${BASE_PATH}/CoEx_GRPO_oneMinusBLEU_G10_main4_div3x2_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0621_010126"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 300

# # GRPO (433 -- New regime of BLEU comparison)
# RUN_NAME="${BASE_PATH}/CoEx_lt-grpo_dt-one_minus_bleu_score_G10-m4-d2x3_lr1e-6_beta0.04_scope-all-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0628_151401"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# # GRPO (433 -- New regime of BLEU comparison 2)
# RUN_NAME="${BASE_PATH}/CoEx_sample_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m4-d2x3_lr1e-6_beta0.04_scope-all-balsamp_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0701_103150"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# # GRPO (4222 -- BLEU comparison 1)
# RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m4-d3x2_lr1e-6_beta0.04_scope-intra-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0701_151421"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# GRPO (433 covers main weakness)
# RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0707_121837"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# # Dr.GRPO
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0618_203254"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# # DMPO
# RUN_NAME="${BASE_PATH}/DMPO_singleSource_G10_beta1_temp0.0667_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0616_134603"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

# # DMPO (433)
# RUN_NAME="${BASE_PATH}/CoEx_GRPO_oneMinusBLEU_G10_main4_div3x2_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0621_010126"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 300

# 3) ours old: 200 only
# OURS_700_RUN="${BASE_PATH}/coex_lr_1e-6-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0510_155659"
# eval_specific_steps "$OURS_700_RUN" "DeepSeek" 32768 200

# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_4-3_div_2_lT_grpo-ndiv_True-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0511_231511"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 900

# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_sapo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0524_125146"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 50

# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_4-2_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0520_091807"
# eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 700

# # Sampling — checkpoint runs (pass@64)
# DS_GRPO_RUN="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0509_055052"
# run_sampling "${DS_GRPO_RUN}/checkpoint-100"    "coex_10-0_grpo-0509_step100"    "GRPO"     "DeepSeek" 32768 bfloat16

# DS_PROP_RUN="${BASE_PATH}/RESUE_coex_lr_1e-6-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0511_171729"
# run_sampling "${DS_PROP_RUN}/checkpoint-300"    "coex_4-3-2_grpo-0511_step300"   "Proposed" "DeepSeek" 32768 bfloat16

# DS_DRGRPO_RUN="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0510_160444"
# run_sampling "${DS_DRGRPO_RUN}/checkpoint-1000" "coex_10-0_drgrpo-0510_step1000" "GRPO"     "DeepSeek" 32768 bfloat16

# Pretrained — greedy eval
# run_eval_pretrained "Qwen/Qwen2.5-Math-1.5B-Instruct"           "Qwen2.5-Math-1.5B-Instruct"    4096  4096  float16
# run_eval_pretrained "meta-llama/Llama-3.2-3B-Instruct"          "Llama-3.2-3B-Instruct"         32768 32768 float16
# run_eval_pretrained "microsoft/Phi-4-mini-instruct"              "Phi-4-mini-instruct"            32768 32768 float16
# run_eval_pretrained "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "DeepSeek-R1-Distill-Qwen-1.5B" 32768 32768 float16

# LLaMA
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0614_032025"
# eval_checkpoints "$RUN_NAME" "Llama" 700 1000 100 32768 32768 bfloat16

# Phi
## GRPO
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0517_141047"
# eval_specific_steps "$RUN_NAME" "Phi" 32768 1000
## DR.GRPO
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0519_052354"
# eval_specific_steps "$RUN_NAME" "Phi" 32768 1000
# ## 4222
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0612_033207"
# eval_specific_steps "$RUN_NAME" "Phi" 32768 700 1000 100

# Qwen 0.6
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0607_133430"
# eval_specific_steps "$RUN_NAME" "Qwen" 32768 100 800 900
# 4222 
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0515_180204"
# eval_specific_steps "$RUN_NAME" "Qwen" 32768 1000

# Qwen 2.5 Math
## GRPO
# RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m10-d0x0_lr1e-6_beta0.04_scope-intra-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0711_182450"
# eval_specific_steps "$RUN_NAME" "Qwen2.5-Math" 4096 1000
## DR.GRPO
# RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0521_025354"
# eval_specific_steps "$RUN_NAME" "Qwen2.5-Math" 4096 1000
## Proposed
# RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m4-d2x3_lr1e-6_beta0.04_scope-intra-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0712_144632"
# eval_specific_steps "$RUN_NAME" "Qwen2.5-Math" 4096 1000
RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0713_181313"
eval_specific_steps "$RUN_NAME" "Qwen2.5-Math" 4096 1000

# Qwen3 1.7B
# RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m10-d0x0_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0710_141525"
# eval_specific_steps "$RUN_NAME" "Qwen3.1.7" 32768 1000
# RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0711_164611"
# eval_specific_steps "$RUN_NAME" "Qwen3.1.7" 32768 1000

notify "Temporary eval finished"
echo "All temporary eval tasks finished."