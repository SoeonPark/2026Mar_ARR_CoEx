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
LMEVAL_BIN="${LMEVAL_BIN:-${SCRIPT_DIR}/lm-evaluation-harness/coex_eval/bin/lm-eval}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/lm-evaluation-harness/coex_eval/bin/python}"
SAMPLING_EXPORTER="${SCRIPT_DIR}/eval/export_sampling_records.py"
SAMPLING_K_VALUES=(${SAMPLING_K_VALUES:-${SAMPLING_K:-32 16 8}})
SAMPLING_K="${SAMPLING_K_VALUES[0]}"
SAMPLING_TASK_INCLUDE_PATH="$CUSTOM_TASKS"
SAMPLING_TEMP_TASK_DIRS=()
SAMPLING_SEED="${SAMPLING_SEED:-1234}"
SAMPLING_SOURCE_ID="${SAMPLING_SOURCE_ID:-main}"
SAMPLING_ADAPTER_NAME="${SAMPLING_ADAPTER_NAME:-default}"

BASE_PATH="/home/nlplab/ssd2/soeon/2026Mar_ARR_CoEx/merged_output"

cleanup_sampling_task_dirs() {
    local task_dir
    for task_dir in "${SAMPLING_TEMP_TASK_DIRS[@]}"; do
        if [[ -n "$task_dir" && -d "$task_dir" && "$task_dir" == "${TMPDIR:-/tmp}"/coex_sampling_k* ]]; then
            rm -rf "$task_dir"
        fi
    done
}
trap cleanup_sampling_task_dirs EXIT

prepare_sampling_task_dir() {
    local sampling_k=$1
    local task_dir
    local task_yaml
    local reported_pass_k="["
    local pass_k
    local separator=""

    if [[ ! "$sampling_k" =~ ^[0-9]+$ ]] || (( sampling_k <= 0 )); then
        echo "[ERROR] Invalid SAMPLING_K: $sampling_k" >&2
        return 1
    fi

    for pass_k in 1 2 4 8 16 32; do
        if (( pass_k <= sampling_k )); then
            reported_pass_k+="${separator}${pass_k}"
            separator=", "
        fi
    done
    reported_pass_k+="]"

    task_dir="$(mktemp -d "${TMPDIR:-/tmp}/coex_sampling_k${sampling_k}.XXXXXX")" || return 1
    SAMPLING_TEMP_TASK_DIRS+=("$task_dir")

    cp "${CUSTOM_TASKS}/utils.py" "${task_dir}/utils.py" || return 1
    for task_yaml in aime24_passk.yaml aime25_passk.yaml; do
        cp "${CUSTOM_TASKS}/${task_yaml}" "${task_dir}/${task_yaml}" || return 1
        perl -0pi -e "s/repeats: \d+/repeats: ${sampling_k}/g; s/(function: take_first_k\n\s+k:) \d+/\$1 ${sampling_k}/g; s/name: pass\d+/name: pass${sampling_k}/g; s/sampling_k: \d+/sampling_k: ${sampling_k}/g; s/generations_per_prompt: \d+/generations_per_prompt: ${sampling_k}/g; s/reported_pass_k: \[[^\]]+\]/reported_pass_k: ${reported_pass_k}/g" "${task_dir}/${task_yaml}" || return 1
    done

    SAMPLING_TASK_INCLUDE_PATH="$task_dir"
}

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

    local matches=()
    local candidate

    shopt -s nullglob
    matches=(./eval_results/*/"${group_name}"/"${category}"/"${dir_name}"/"step${step}")
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

    if lm-eval run \
        --config "$config" \
        --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
        --apply_chat_template true \
        --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
        $fewshot_flag \
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
    # Supported call forms:
    #   run_sampling "DeepSeek-R1-Distill-Qwen-1.5B"
    #   run_sampling "$RUN_NAME" "DeepSeek" 32768 1000
    local model_or_run=$1
    local group_name=${2:-DeepSeek}
    local max_len=${3:-32768}
    local checkpoint_step=${4:-}
    local dtype=${5:-bfloat16}
    local config=${6:-$SAMPLING_CONFIG}
    local max_gen_toks=${MAX_GEN_TOKS:-$((max_len - 2048))}
    local task_include_path=${SAMPLING_TASK_INCLUDE_PATH:-$CUSTOM_TASKS}

    local model_path
    local model_name
    local category

    if [[ -n "$checkpoint_step" ]]; then
        model_path="${model_or_run}/checkpoint-${checkpoint_step}"
        model_name="$(basename "$model_or_run")_step${checkpoint_step}"

        case "$model_name" in
            CoEx_GRPO_oneMinusBLEU_*) category="1-BLEU_GRPO" ;;
            CoEx_DMPO_collective_oneMinusBLEU_*) category="1-BLEU_DMPO" ;;
            DMPO_singleSource_*) category="DMPO" ;;
            *base_10-0_div_0*) category="GRPO" ;;
            *) category="FineTuned" ;;
        esac

        # A checkpoint that is still being trained is an expected skip.
        if [[ ! -d "$model_path" ]]; then
            echo "[SKIP] Checkpoint not found yet: $model_path"
            notify "[SKIP] Checkpoint not found yet | ${model_name}"
            return 0
        fi
        if [[ ! -f "${model_path}/config.json" ]]; then
            echo "[ERROR] Not a merged model checkpoint: $model_path"
            SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
            return 1
        fi
    else
        category="Pretrained"
        case "$model_or_run" in
            DeepSeek-R1-Distill-Qwen-1.5B)
                model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                model_name="DeepSeek-R1-Distill-Qwen-1.5B"
                ;;
            *)
                model_path="$model_or_run"
                model_name="$(basename "$model_or_run")"
                ;;
        esac
    fi

    if [[ ! -x "$LMEVAL_BIN" ]]; then
        echo "[ERROR] lm-eval executable not found: $LMEVAL_BIN"
        echo "        Override it with LMEVAL_BIN=/path/to/lm-eval"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
    fi
    if [[ ! -f "$config" ]]; then
        echo "[ERROR] Sampling config not found: $config"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
    fi
    if [[ ! -d "$task_include_path" ]]; then
        echo "[ERROR] Custom task directory not found: $task_include_path"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
    fi
    if [[ ! -x "$PYTHON_BIN" || ! -f "$SAMPLING_EXPORTER" ]]; then
        echo "[ERROR] Sampling record exporter is unavailable."
        echo "        Python: $PYTHON_BIN"
        echo "        Exporter: $SAMPLING_EXPORTER"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
    fi
    if (( max_gen_toks <= 0 || max_gen_toks >= max_len )); then
        echo "[ERROR] MAX_GEN_TOKS must be positive and smaller than max_len."
        echo "        MAX_GEN_TOKS=${max_gen_toks}, max_len=${max_len}"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
    fi

    local output_path="./eval_results/${DATE}/${category}/sampling/pass@${SAMPLING_K}/${group_name}/${model_name}"

    echo ""
    echo "=========================================="
    echo "[SAMPLING] Group: $group_name"
    echo "  Model : $model_name"
    echo "  Path  : $model_path"
    echo "  K     : $SAMPLING_K"
    echo "  Tasks : $task_include_path"
    echo "  Limit : context=${max_len}, generation=${max_gen_toks}"
    echo "  Out   : $output_path"
    echo "=========================================="

    notify "[START] ${group_name} | ${model_name}"

    if "$LMEVAL_BIN" run \
        --config "$config" \
        --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true,seed=${SAMPLING_SEED}" \
        --apply_chat_template \
        --gen_kwargs "do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=${max_gen_toks}" \
        --seed "0,${SAMPLING_SEED},${SAMPLING_SEED},${SAMPLING_SEED}" \
        --include_path "$task_include_path" \
        --output_path "$output_path"; then
        if "$PYTHON_BIN" "$SAMPLING_EXPORTER" \
            --output-root "$output_path" \
            --source-id "$SAMPLING_SOURCE_ID" \
            --adapter-name "$SAMPLING_ADAPTER_NAME" \
            --seed "$SAMPLING_SEED" \
            --sampling-k "$SAMPLING_K"; then
            notify "[DONE] ${group_name} | ${model_name}"
            return 0
        fi
        echo "[ERROR] Failed to export individual sampling records: $output_path"
        notify "[ERROR] Sampling record export failed | ${model_name}"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
    else
        notify "[ERROR] ${group_name} | ${model_name} — check logs"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        return 1
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

    for step in $(seq "$step_start" "$step_interval" "$step_end"); do
        local ckpt_path="${run_dir}/checkpoint-${step}"

        if [[ ! -d "$ckpt_path" ]]; then
            echo "[SKIP] Not found: $ckpt_path"
            continue
        fi

        local output_path="./eval_results/${DATE}/${group_name}/${category}/${dir_name}/step${step}"
        local log_file="eval_log/${group_name}_${category}_${dir_name}_step${step}.log"
        mkdir -p "$(dirname "$log_file")"

        local existing_output=""
        if existing_output="$(find_existing_eval_output "$group_name" "$category" "$dir_name" "$step")"; then
            echo "[SKIP] Already evaluated: ${group_name} | ${category} | ${dir_name} | step${step}"
            echo "       found: $(realpath -m "$existing_output")"
            echo "       new would be: $(realpath -m "$output_path")"
            notify "[SKIP] Already evaluated: ${group_name} | ${category} | ${dir_name} | step${step}"
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

        if lm-eval run \
            --config "$config" \
            --model_args "pretrained=${ckpt_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
            --apply_chat_template true \
            --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
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

notify "Temporary DeepSeek eval started for K=${SAMPLING_K_VALUES[*]}"
SAMPLING_FAILURES=0

run_sampling_suite() {
    # # Pretrained DeepSeek-R1-Distill-Qwen-1.5B
    # run_sampling "DeepSeek-R1-Distill-Qwen-1.5B"

    # # GRPO (10)
    # RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0617_202854"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 1000

    # # 1-BLEU/GRPO (433)
    # RUN_NAME="${BASE_PATH}/CoEx_GRPO_oneMinusBLEU_G10_main4_div3x2_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0621_010126"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 300

    # # GRPO (433 - No Diversity)
    # RUN_NAME="${BASE_PATH}/CoEx_GRPO_oneMinusBLEU_G10_main4_div3x2_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0624_225048"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 1000

    # # GRPO (433 - New 1-BLEU/GRPO)
    # RUN_NAME="${BASE_PATH}/CoEx_lt-grpo_dt-one_minus_bleu_score_G10-m4-d2x3_lr1e-6_beta0.04_scope-all-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0628_151401"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 1000

    # # # GRPO (433 - New regime of BLEU comparison 2)
    # RUN_NAME="${BASE_PATH}/CoEx_sample_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m4-d2x3_lr1e-6_beta0.04_scope-all-balsamp_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0701_103150"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 1000

    # # GRPO (4222 -- BLEU comparison 1)
    # RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-one_minus_bleu_score_G10-m4-d3x2_lr1e-6_beta0.04_scope-intra-balsrc_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0701_151421"
    # eval_specific_steps "$RUN_NAME" "DeepSeek" 32768 1000

    # GRPO (433 main weakness)
    RUN_NAME="${BASE_PATH}/CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0707_121837"
    run_sampling "$RUN_NAME" "DeepSeek" 32768 1000
    
    # # DMPO (10)
    # RUN_NAME="${BASE_PATH}/DMPO_singleSource_G10_beta1_temp0.0667_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0616_134603"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 1000

    # # 1-BLEU/DMPO (433)
    # RUN_NAME="${BASE_PATH}/CoEx_DMPO_collective_oneMinusBLEU_G10_main4_div3x2_beta1.0_temp0.06666666666666667_DeepSeek-R1-Distill-Qwen-1.5B_lr_1e-6-0621_010227"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 550

    # # Dr.GRPO (10)
    # RUN_NAME="${BASE_PATH}/coex_lr_1e-6-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0618_203254"
    # run_sampling "$RUN_NAME" "DeepSeek" 32768 1000
}

for SAMPLING_K in "${SAMPLING_K_VALUES[@]}"; do
    echo ""
    echo ">>> Starting sampling eval with SAMPLING_K=${SAMPLING_K}"
    notify "Temporary DeepSeek eval pass@${SAMPLING_K} started"

    if ! prepare_sampling_task_dir "$SAMPLING_K"; then
        notify "[ERROR] Failed to prepare sampling tasks for pass@${SAMPLING_K}"
        SAMPLING_FAILURES=$((SAMPLING_FAILURES + 1))
        continue
    fi

    run_sampling_suite
    SAMPLING_TASK_INCLUDE_PATH="$CUSTOM_TASKS"
done

if (( SAMPLING_FAILURES > 0 )); then
    notify "Temporary eval finished with ${SAMPLING_FAILURES} failure(s)"
    echo "Sampling eval finished with ${SAMPLING_FAILURES} failure(s)."
    exit 1
fi

notify "Temporary eval finished"
echo "All temporary eval tasks finished successfully."
