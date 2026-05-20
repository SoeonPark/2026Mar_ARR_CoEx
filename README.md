# 2026Mar_ARR_CoEx

Official implementation of **"CoEx-GRPO: Collective-Experience Rollouts for Group-Relative Policy Optimization under Verifiable Rewards"** (submitted to ACL ARR March 2026).

### Overview

Standard GRPO suffers from homogenized exploration: as the policy converges, sampled responses for each prompt become increasingly similar, reducing reward variance and weakening the group-relative learning signal. **CoEx-GRPO** addresses this through a multi-policy reinforcement learning framework:

- **Multi-Policy Training** — Jointly trains a Main Policy (the final model) and multiple Diversity Policies implemented as independent LoRA adapters sharing the same frozen base model.
- **Collective Experience** — Aggregates rollouts from all policies into a single Global Candidate Set per prompt, ensuring the Main Policy always sees diverse comparisons.
- **Asymmetric Optimization** — The Main Policy learns from the full collective experience (correctness reward only), while Diversity Policies are trained with a weighted combination of correctness and Policy-Repulsion rewards to maintain exploration in distinct regions of the response space.

### Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA 12.x
- Flash Attention 2

`pip install -r requirements.txt`

Key dependencies (see `requirements.txt` for pinned versions):

| Package                    | Version                                     | Purpose                      |
| -------------------------- | ------------------------------------------- | ---------------------------- |
| `trl`                      | 0.25.0                                      | Base GRPO trainer            |
| `peft`                     | 0.18.0                                      | LoRA adapter management      |
| `vllm`                     | 0.10.2                                      | Efficient rollout generation |
| `math-verify`              | 0.8.0                                       | Symbolic answer verification |
| `flash_attn`               | pip install flash-attn --no-build-isolation |
| Memory-efficient attention |

### Project Structure

```text
.
├── main.py
├── custom_coex_trainer.py
├── custom_coex_config.py
├── reward_func.py
├── rewards/
├── data_utils.py
├── prepare_model_inf.py
├── baseline.sh
├── run_coex.sh
├── configs/
├── eval/
├── scripts/
├── requirements.txt
└── README.md
```

### Data Preparation

The framework primarily uses the **OPEN-RS** dataset, a curated mixture of 7,000 mathematical reasoning problems from sources like OPEN-S1 and DEEPSCALER.

- Training Set: Includes problems with step-by-step solutions and verifiable gold answers.
- Loading: Use `data_utils.py` to automatically fetch and format datasets.

```python
from data_utils import get_open_rs_dataset
dataset = get_open_rs_dataset(split="train")
```

### Training

Training uses DeepSeek-R1-Distill-Qwen-1.5B as the base model with QLoRA (4-bit NF4 quantization). The Main Policy and each Diversity Policy are independent LoRA adapters (`r=16`, `alpha=32`) sharing the frozen base weights. Rollout generation uses vLLM in colocate mode.

Ready-to-use launch scripts are provided:

```bash
bash run_coex.sh
```

```bash
bash baseline.sh
```

`run_coex.sh` configures the following default setup:

- Main Policy adapter (4 completions) + 3 Diversity Policy adapters (2 completions each) = 10 generations per prompt
- Diversity reward: `policy_repulsion_margin`
- Loss: GRPO with PPO-style clipping (`epsilon=0.2`) and KL penalty (`beta=0.04`)
- Learning rate: `1e-4`, cosine schedule with min LR ratio `0.1`
- Max completion length: `3584` tokens
- Gradient accumulation: `2` steps
- Logging and checkpointing via W&B

`baseline.sh` uses the same training stack with only the Main Policy active.

Estimated training time: ~87.3 hours on a single **RTX3090** (24GB).

### Evaluation

After training, merge the LoRA adapter into the base model:

```bash
python prepare_model_inf.py \
    --base_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --experiment_name <EXPERIMENT_NAME> \
    --step <CHECKPOINT_STEP>
```

Then evaluate on downstream benchmarks using the evaluation scripts under `scripts/` and `eval/`.

- **Benchmarks:** AIME 2024, AIME 2025, MATH-500, GSM8K, MMLU-MATH-PRO
- **Metrics:** Accuracy is calculated using symbolic equivalence checking via the `math_verify` library.

### Main Results

<img width="1016" height="416" alt="image" src="https://github.com/user-attachments/assets/8f85474f-f21b-482d-bd48-5837a3b7e76f" />

### Citation

```bibtex
@article{anonymous2026coex,
    title  = {CoEx-GRPO: Collective-Experience Rollouts for Group-Relative Policy Optimization under Verifiable Rewards},
    author = {Anonymous},
    year   = {2026},
    note   = {Under review at ACL 2026 via ARR}
}
```

### License
