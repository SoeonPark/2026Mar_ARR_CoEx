# # nohup bash lm_eval.sh > eval_log/0224_eval_pretraineds.log 2>&1 &
# # nohup bash lm_eval.sh > eval_log/0227_eval_pretraineds_gpu2.log 2>&1 &

# # nohup bash lm_eval_3.sh > eval_log/0302_eval_qwen2_5_math.log 2>&1 &
# # nohup bash lm_eval_3.sh > eval_log/0309_eval_phi4_pr_gpu3.log 2>&1 &
# # nohup bash lm_eval_3.sh > eval_log/0311_eval_qwen_mthInst_pretrained_grpo_gpu3.log 2>&1 &
# # nohup bash lm_eval_3.sh >> eval_log/DeepSeek_dapo_ablation_gpu3.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# export HF_DATASETS_OFFLINE=1
# export VLLM_USE_V1=0

# DATE=$(date +%m%d)

# get_category() {
#     local dir_name=$1
#     if [[ "$dir_name" =~ base_10-0_div_0 ]]; then
#         echo "GRPO"
#     elif [[ "$dir_name" =~ base_[0-9]+-[0-9]+_div_[0-9]+ ]]; then
#         echo "Proposed"
#     else
#         echo "Pretrained"
#     fi
# }

# get_base_model() {
#     local path=$1
#     if [[ "$path" =~ Qwen3-1\.7B ]]; then
#         echo "Qwen3-1.7B"
#     elif [[ "$path" =~ Qwen2\.5-Math-1\.5B ]]; then
#         echo "Qwen2.5-Math-1.5B"
#     elif [[ "$path" =~ DeepSeek-R1-Distill-Qwen-1\.5B ]]; then
#         echo "DeepSeek-R1-Distill-Qwen-1.5B"
#     else
#         echo "DeepSeek-R1-Distill-Qwen-1.5B"
#     fi
# }

# run_eval() {
#     local model_path=$1
#     local model_name=$2
#     local base_output=$3
#     local max_len=$4
#     local max_gen=$5
#     local dtype=${6:-float16}
#     local config=${7:-basic_eval_pretrained.yaml}
#     local fewshot_as_multiturn=${8:-true}

#     echo "=========================================="
#     echo "Evaluating: $model_name  [dtype=${dtype}, fewshot_multiturn=${fewshot_as_multiturn}]"
#     echo "  config: $config"
#     echo "  path:   $model_path"
#     echo "=========================================="

#     # fewshot_as_multiturn=false일 때만 플래그 추가
#     local multiturn_flag=""
#     if [[ "$fewshot_as_multiturn" == "false" ]]; then
#         multiturn_flag="--fewshot_as_multiturn false"
#     fi

#     lm-eval run --config "$config" \
#         --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
#         --apply_chat_template true \
#         $multiturn_flag \
#         --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
#         --output_path "${base_output}/${model_name}"
# }

# run_eval_pretrained() {
#     local model_path=$1
#     local model_name=$2
#     local max_len=$3
#     local max_gen=$4
#     local dtype=${5:-float16}
#     local config=${6:-basic_eval_pretrained.yaml}
#     local fewshot_as_multiturn=${7:-true}

#     local base_output="./eval_results/${DATE}/Pretrained/ApplyChatTemplate_True"
#     run_eval "$model_path" "$model_name" "$base_output" "$max_len" "$max_gen" "$dtype" "$config" "$fewshot_as_multiturn"
# }

# eval_checkpoints() {
#     local run_dir=$1
#     local step_start=$2
#     local step_end=$3
#     local step_interval=$4
#     local max_len=$5
#     local max_gen=$6
#     local dtype=${7:-float16}

#     local dir_name
#     dir_name=$(basename "$run_dir")
#     local category
#     category=$(get_category "$dir_name")
#     local base_model
#     base_model=$(get_base_model "$run_dir")
#     local base_output="./eval_results/${DATE}/${category}/ApplyChatTemplate_True"

#     for step in $(seq "$step_start" "$step_interval" "$step_end"); do
#         local ckpt_path="${run_dir}/checkpoint-${step}"
#         if [[ ! -d "$ckpt_path" ]]; then
#             echo "[SKIP] Not found: $ckpt_path"
#             continue
#         fi
#         local model_name="${base_model}/${dir_name}/step${step}"
#         run_eval "$ckpt_path" "$model_name" "$base_output" "$max_len" "$max_gen" "$dtype"
#     done
# }


# # # ════════════════════════════════════════════════════════════════════
# # # Qwen3-1.7B  (step 700 ~ 1000, interval 100)
# # # ════════════════════════════════════════════════════════════════════

# # ## Proposed (4-2 / 4-3)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_coex_lr_1e-4-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0203_045614" \
# #     700 1000 100 40960 32768

# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/Qwen/Qwen3-1.7B_coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0122_164804" \
# #     700 1000 100 40960 32768

# # ## GRPO (10-0)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/Qwen/Qwen3-1.7B_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False_ncorr_False_bleu_correctnessGated_False_0108_171813" \
# #     700 1000 100 40960 32768

# # # ════════════════════════════════════════════════════════════════════
# # # DeepSeek-R1-Distill-Qwen-1.5B  (step 700 ~ 1000, interval 100)
# # # ════════════════════════════════════════════════════════════════════
# ## Pretrained
# # run_eval_pretrained \
# #     "/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" \
# #     "DeepSeek-R1-Distill-Qwen-1.5B/Pretrained" \
# #     40960 32768

# ## GRPO (10-0)
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_external-correctnessGated_False-useIW_False-0205_161145" \
# #     700 1000 100 40960 32768

# # ## Proposed (4-3 / 4-2)
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0216_154758" \
# #     700 1000 100 40960 32768

# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0219_180233" \
# #     700 1000 100 40960 32768

# # # DAPO
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_10-0_div_0_lT_dapo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0313_120647" \
# #     700 1000 100 40960 32768
# # echo "===== DAPO done! ====="

# # echo "===== Starting Ablation Eval: 4-2-2 => 4-3 ====="
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-2_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0311_113635" \
# #     700 1000 100 40960 32768

# echo "===== Starting Ablation Eval: 4-2 ====="
# eval_checkpoints \
#     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-1_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0311_023051" \
#     700 1000 100 40960 32768



# # ════════════════════════════════════════════════════════════════════
# # Qwen2.5-Math-1.5B  (step 400 ~ 500 only, interval 100)
# # max_len=4096, max_gen=2048
# # ════════════════════════════════════════════════════════════════════

# ## GRPO (10-0)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_external-correctnessGated_False-useIW_False-0207_155729" \
# #     400 500 100 4096 2048

# # ## Proposed (4-3 / 4-2)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2_coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0209_124422" \
# #     400 500 100 4096 2048

# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2_coex_lr_1e-4-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0216_155828" \
# #     400 500 100 4096 2048

# # eval_checkpoints \
# #     "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0303_163013" \
# #     500 500 100 40960 32768

# # ==════════════════════════════════════════════════════════════════════
# # # Qwen2.5-Math-1.5B-Instruct Pretrained
# # run_eval_pretrained \
# #     "/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35" \
# #     "Qwen2.5-Math-1.5B-Instruct/Pretrained" \
# #     4096 2048

# # # Qwen2.5-Math-1.5B-Instruct (step 700~1000, interval 100)
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0310_102709" \
#     # 700 1000 100 4096 2048

# echo "===== All done! ====="


#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V1=0

SERVER="${SERVER:-7}"

notify() {
  local message="$1"
  curl -s -d "${message}" "ntfy.sh/soeon_server${SERVER}" >/dev/null 2>&1 || true
}

DATE=$(date +%m%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/basic_eval_pretrained.yaml"
SAMPLING_CONFIG="${SCRIPT_DIR}/basic_eval_pretrained.yaml"
# # nohup bash lm_eval.sh > eval_log/0224_eval_pretraineds.log 2>&1 &
# # nohup bash lm_eval.sh > eval_log/0227_eval_pretraineds_gpu2.log 2>&1 &

# # nohup bash lm_eval_3.sh > eval_log/0302_eval_qwen2_5_math.log 2>&1 &
# # nohup bash lm_eval_3.sh > eval_log/0309_eval_phi4_pr_gpu3.log 2>&1 &
# # nohup bash lm_eval_3.sh > eval_log/0311_eval_qwen_mthInst_pretrained_grpo_gpu3.log 2>&1 &
# # nohup bash lm_eval_3.sh >> eval_log/DeepSeek_dapo_ablation_gpu3.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# export HF_DATASETS_OFFLINE=1
# export VLLM_USE_V1=0

# DATE=$(date +%m%d)

# get_category() {
#     local dir_name=$1
#     if [[ "$dir_name" =~ base_10-0_div_0 ]]; then
#         echo "GRPO"
#     elif [[ "$dir_name" =~ base_[0-9]+-[0-9]+_div_[0-9]+ ]]; then
#         echo "Proposed"
#     else
#         echo "Pretrained"
#     fi
# }

# get_base_model() {
#     local path=$1
#     if [[ "$path" =~ Qwen3-1\.7B ]]; then
#         echo "Qwen3-1.7B"
#     elif [[ "$path" =~ Qwen2\.5-Math-1\.5B ]]; then
#         echo "Qwen2.5-Math-1.5B"
#     elif [[ "$path" =~ DeepSeek-R1-Distill-Qwen-1\.5B ]]; then
#         echo "DeepSeek-R1-Distill-Qwen-1.5B"
#     else
#         echo "DeepSeek-R1-Distill-Qwen-1.5B"
#     fi
# }

# run_eval() {
#     local model_path=$1
#     local model_name=$2
#     local base_output=$3
#     local max_len=$4
#     local max_gen=$5
#     local dtype=${6:-float16}
#     local config=${7:-basic_eval_pretrained.yaml}
#     local fewshot_as_multiturn=${8:-true}

#     echo "=========================================="
#     echo "Evaluating: $model_name  [dtype=${dtype}, fewshot_multiturn=${fewshot_as_multiturn}]"
#     echo "  config: $config"
#     echo "  path:   $model_path"
#     echo "=========================================="

#     # fewshot_as_multiturn=false일 때만 플래그 추가
#     local multiturn_flag=""
#     if [[ "$fewshot_as_multiturn" == "false" ]]; then
#         multiturn_flag="--fewshot_as_multiturn false"
#     fi

#     lm-eval run --config "$config" \
#         --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
#         --apply_chat_template true \
#         $multiturn_flag \
#         --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
#         --output_path "${base_output}/${model_name}"
# }

# run_eval_pretrained() {
#     local model_path=$1
#     local model_name=$2
#     local max_len=$3
#     local max_gen=$4
#     local dtype=${5:-float16}
#     local config=${6:-basic_eval_pretrained.yaml}
#     local fewshot_as_multiturn=${7:-true}

#     local base_output="./eval_results/${DATE}/Pretrained/ApplyChatTemplate_True"
#     run_eval "$model_path" "$model_name" "$base_output" "$max_len" "$max_gen" "$dtype" "$config" "$fewshot_as_multiturn"
# }

# eval_checkpoints() {
#     local run_dir=$1
#     local step_start=$2
#     local step_end=$3
#     local step_interval=$4
#     local max_len=$5
#     local max_gen=$6
#     local dtype=${7:-float16}

#     local dir_name
#     dir_name=$(basename "$run_dir")
#     local category
#     category=$(get_category "$dir_name")
#     local base_model
#     base_model=$(get_base_model "$run_dir")
#     local base_output="./eval_results/${DATE}/${category}/ApplyChatTemplate_True"

#     for step in $(seq "$step_start" "$step_interval" "$step_end"); do
#         local ckpt_path="${run_dir}/checkpoint-${step}"
#         if [[ ! -d "$ckpt_path" ]]; then
#             echo "[SKIP] Not found: $ckpt_path"
#             continue
#         fi
#         local model_name="${base_model}/${dir_name}/step${step}"
#         run_eval "$ckpt_path" "$model_name" "$base_output" "$max_len" "$max_gen" "$dtype"
#     done
# }


# # # ════════════════════════════════════════════════════════════════════
# # # Qwen3-1.7B  (step 700 ~ 1000, interval 100)
# # # ════════════════════════════════════════════════════════════════════

# # ## Proposed (4-2 / 4-3)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_coex_lr_1e-4-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0203_045614" \
# #     700 1000 100 40960 32768

# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/Qwen/Qwen3-1.7B_coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0122_164804" \
# #     700 1000 100 40960 32768

# # ## GRPO (10-0)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/Qwen/Qwen3-1.7B_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False_ncorr_False_bleu_correctnessGated_False_0108_171813" \
# #     700 1000 100 40960 32768

# # # ════════════════════════════════════════════════════════════════════
# # # DeepSeek-R1-Distill-Qwen-1.5B  (step 700 ~ 1000, interval 100)
# # # ════════════════════════════════════════════════════════════════════
# ## Pretrained
# # run_eval_pretrained \
# #     "/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" \
# #     "DeepSeek-R1-Distill-Qwen-1.5B/Pretrained" \
# #     40960 32768

# ## GRPO (10-0)
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_external-correctnessGated_False-useIW_False-0205_161145" \
# #     700 1000 100 40960 32768

# # ## Proposed (4-3 / 4-2)
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0216_154758" \
# #     700 1000 100 40960 32768

# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0219_180233" \
# #     700 1000 100 40960 32768

# # # DAPO
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_10-0_div_0_lT_dapo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0313_120647" \
# #     700 1000 100 40960 32768
# # echo "===== DAPO done! ====="

# # echo "===== Starting Ablation Eval: 4-2-2 => 4-3 ====="
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-2_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0311_113635" \
# #     700 1000 100 40960 32768

# echo "===== Starting Ablation Eval: 4-2 ====="
# eval_checkpoints \
#     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_4-1_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0311_023051" \
#     700 1000 100 40960 32768



# # ════════════════════════════════════════════════════════════════════
# # Qwen2.5-Math-1.5B  (step 400 ~ 500 only, interval 100)
# # max_len=4096, max_gen=2048
# # ════════════════════════════════════════════════════════════════════

# ## GRPO (10-0)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_external-correctnessGated_False-useIW_False-0207_155729" \
# #     400 500 100 4096 2048

# # ## Proposed (4-3 / 4-2)
# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2_coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0209_124422" \
# #     400 500 100 4096 2048

# # eval_checkpoints \
# #     "/home/nlplab/ssd1/gyop/research/2601_arr/merged_output/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2_coex_lr_1e-4-base_4-2_div_3_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0216_155828" \
# #     400 500 100 4096 2048

# # eval_checkpoints \
# #     "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0303_163013" \
# #     500 500 100 40960 32768

# # ==════════════════════════════════════════════════════════════════════
# # # Qwen2.5-Math-1.5B-Instruct Pretrained
# # run_eval_pretrained \
# #     "/home/nlplab/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35" \
# #     "Qwen2.5-Math-1.5B-Instruct/Pretrained" \
# #     4096 2048

# # # Qwen2.5-Math-1.5B-Instruct (step 700~1000, interval 100)
# # eval_checkpoints \
# #     "/home/nlplab/hdd1/gyop/research/2601_arr/merged_output/coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0310_102709" \
#     # 700 1000 100 4096 2048

# echo "===== All done! ====="


#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0

SERVER="${SERVER:-7}"

notify() {
  local message="$1"
  curl -s -d "${message}" "ntfy.sh/soeon_server${SERVER}" >/dev/null 2>&1 || true
}

DATE=$(date +%m%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
CONFIG="${SCRIPT_DIR}/basic_eval_pretrained.yaml"
SAMPLING_CONFIG="${SCRIPT_DIR}/basic_eval_pretrained.yaml"

BASE_PATH="/home/nlplab/hdd1/gyop/research/2601_arr/merged_output"

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
    elif [[ "$path" =~ Qwen2\.5-Math-1\.5B ]]; then
        echo "Qwen2.5-Math-1.5B"
    elif [[ "$path" =~ Llama-3\.2-3B ]]; then
        echo "Llama-3.2-3B-Instruct"
    else
        echo "Model-Unknown"
    fi
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

run_eval() {
    local model_path=$1
    local model_name=$2
    local base_output=$3
    local max_len=$4
    local max_gen=$5
    local dtype=${6:-float16}
    local config=${7:-$CONFIG}
    local fewshot_as_multiturn=${8:-true}

    local log_name
    log_name="$(echo "${model_name}" | tr '/' '__')"
    local log_file="./eval_log/${log_name}.out"
    mkdir -p ./eval_log

    echo "=========================================="
    echo "Evaluating: $model_name  [dtype=${dtype}, fewshot_multiturn=${fewshot_as_multiturn}]"
    echo "  config: $config"
    echo "  path:   $model_path"
    echo "=========================================="

    local multiturn_flag=""
    if [[ "$fewshot_as_multiturn" == "false" ]]; then
        multiturn_flag="--fewshot_as_multiturn false"
    fi

    notify "[START] ${model_name}"

    if lm-eval run --config "$config" \
        --model_args "pretrained=${model_path},dtype=${dtype},max_model_len=${max_len},max_num_batched_tokens=${max_len},gpu_memory_utilization=0.9,trust_remote_code=true" \
        --apply_chat_template true \
        $multiturn_flag \
        --gen_kwargs "temperature=0,top_p=1,max_gen_toks=${max_gen}" \
        --output_path "${base_output}/${model_name}"; then
        notify "[DONE] ${model_name}"
    else
        notify "[ERROR] ${model_name} — check logs"
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

        local model_name="${base_model}/${dir_name}/step${step}"

        local output_path="./eval_results/${DATE}/${group_name}/${category}/${dir_name}/step${step}"

        local log_file="eval_log/${group_name}_${category}_${dir_name}_step${step}.log"
        mkdir -p "$(dirname "$log_file")"

        echo "=========================================="
        echo "[EVAL] ${group_name} | ${category}"
        echo "Run  : ${dir_name}"
        echo "Step : ${step}"
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
# 1. DeepSeek-R1-Distill-Qwen-1.5B
DEEPSEEK_MODELS=(
    "/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_external-correctnessGated_False-useIW_False-0205_161145"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0216_154758"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0317_131117"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0329_012113"
    "/home/nlplab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_coex_lr_1e-4-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0228_150903"
    "coex_lr_1e-4-base_4-3_div_2_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0310_201954"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_True-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0317_122742"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0317_131126"
)

# 2. Phi-4-mini-instruct
PHI_MODELS=(
    "coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0308_104155"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0303_163013"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0326_224256"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0328_125756"
    "coex_lr_1e-4-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0311_152340"
    "coex_lr_1e-4-base_4-3_div_2_lT_dr_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0312_150441"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_True-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0319_172357"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0324_193001"
)

# 3. Qwen2.5-Math-1.5B-Instruct # "max_position_embeddings": 4096,
QWEN_MATH_MODELS=(
    "coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0310_102709"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0318_154318"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_default_only-0320_124410"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0328_125741"
    "coex_lr_1e-4-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0321_114206"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_True-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0321_120208"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0330_173325"
)

# 4. Llama-3.1B-Instruct
LLAMA_MODELS=(
    "coex_lr_1e-4-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0322_094039"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-0324_192958"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_default_only-0326_225417"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_mean-0328_125756"
    "coex_lr_1e-4-base_10-0_div_0_lT_dr_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-0324_193831"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_True-ncorr_False-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0330_211537"
    "coex_lr_1e-4-base_4-3_div_2_lT_grpo-ndiv_False-ncorr_True-divType_policy_repulsion_margin-correctnessGated_False-useIW_False-repulsionTarget_all_other-0326_224736"
)

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

run_group() {
    local group_name=$1
    local group_max_len=$2
    shift 2
    local group_list=("$@")

    for model_item in "${group_list[@]}"; do
        local full_path
        [[ "$model_item" == /* ]] && full_path="$model_item" || full_path="${BASE_PATH}/${model_item}"

        eval_checkpoints "$full_path" "$group_name" 100 1000 100 "$group_max_len" "$group_max_len" bfloat16
    done
}
notify "Eval started: Llama / DeepSeek / Phi / Qwen"

run_group "Llama"    32768 "${LLAMA_MODELS[@]}"
run_group "DeepSeek" 32768 "${DEEPSEEK_MODELS[@]}"
run_group "Phi"      32768 "${PHI_MODELS[@]}"
run_group "Qwen"     4096  "${QWEN_MATH_MODELS[@]}"

notify "All eval tasks finished."
echo "All tasks finished."
