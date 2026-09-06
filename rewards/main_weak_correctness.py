"""Helpers for the `main_weak_correctness_bonus` diversity reward.

This reward does not measure textual diversity. For a diversity-adapter
sample y that answers prompt x correctly, it rewards the sample more when the
main/default adapter's own rollouts on x were mostly wrong:

    main_correct_rate(x) = mean_i c(x, y_main_i)          over the main adapter's own rollouts
    bonus(x, y)          = c(x, y) * (1 - main_correct_rate(x))

where c(x, y) in {0, 1} is *pure* answer correctness (never a scalar that
mixes in think-tag/format bonuses).

These helpers are intentionally framework-agnostic (plain floats/dicts/lists,
no torch) so they can be unit-tested without any model or trainer state.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def compute_main_correct_rate_by_prompt(
    source_trace_metadata: Sequence[Any],
    main_source_name: str = "default",
    correct_field: str = "answer_correct_float",
    prompt_index_field: str = "prompt_index",
    source_field: str = "source_adapter_name",
) -> dict[int, float]:
    """Per-prompt main/default answer-correctness rate, c-bar_m(x).

    Args:
        source_trace_metadata: sequence of per-row metadata dicts (or None),
            each expected to carry `prompt_index`, `source_adapter_name`, and
            a pure-correctness field (`answer_correct_float` by default).
            Only rows whose `source_adapter_name == main_source_name` are
            averaged; rows from any other adapter are ignored, so this is
            robust to how many diversity adapters exist and to any ordering
            (it does not assume contiguous index ranges).
        main_source_name: adapter name identifying "main" rows (default:
            "default").
        correct_field: key holding pure answer correctness as a 0.0/1.0 float.
        prompt_index_field: key holding the integer prompt id.
        source_field: key holding the adapter name that generated the row.

    Returns:
        dict mapping prompt_index -> mean main answer correctness in [0, 1].
        A prompt with zero main rows is simply absent from the returned dict
        (callers should decide their own fallback, see
        `align_main_correct_rate_to_local_rows`).
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for record in source_trace_metadata:
        if not isinstance(record, dict):
            continue
        if record.get(source_field) != main_source_name:
            continue
        prompt_index = record.get(prompt_index_field)
        if prompt_index is None:
            continue
        correct = record.get(correct_field)
        if correct is None:
            continue
        sums[prompt_index] = sums.get(prompt_index, 0.0) + float(correct)
        counts[prompt_index] = counts.get(prompt_index, 0) + 1
    return {p: sums[p] / counts[p] for p in sums}


def align_main_correct_rate_to_local_rows(
    local_source_metadata: Sequence[Any],
    main_correct_rate_by_prompt: Mapping[int, float],
    default_rate: float = 0.0,
    prompt_index_field: str = "prompt_index",
) -> list[float]:
    """Broadcasts per-prompt main_correct_rate onto a diversity adapter's own local rows.

    `local_source_metadata` must be the SAME adapter's own source-trace
    metadata list, in the same row order as the tensor it will be zipped
    with (e.g. `generation_batch["source_trace_metadata"]` inside
    `_score_completions_diversity`) -- i.e. it must already be sliced to that
    adapter's own rollout rows, not the full/other-adapter comparison pool.

    Returns:
        list[float] of length len(local_source_metadata); `default_rate` is
        used for any row missing a resolvable prompt_index (should not happen
        in practice, but keeps this helper total).
    """
    result: list[float] = []
    for record in local_source_metadata:
        rate = default_rate
        if isinstance(record, dict):
            prompt_index = record.get(prompt_index_field)
            if prompt_index in main_correct_rate_by_prompt:
                rate = main_correct_rate_by_prompt[prompt_index]
        result.append(rate)
    return result


def compute_main_weak_correctness_bonus(
    answer_correct_float: Sequence[float],
    main_correct_rate: Sequence[float],
) -> list[float]:
    """Raw DIAGNOSTIC value R_diag(x, y) = c(x, y) * (1 - c-bar_m(x)).

    NOTE: as of the post-normalization formula revision, this value is no
    longer fed into group normalization / diversity_advantages. It is kept
    purely for logging and per-sample metadata (`record["main_weak_correctness_bonus"]`)
    -- see `apply_main_weak_factor` for the value that actually drives
    training. The old (superseded) approach normalized THIS value directly;
    that cancelled the (1 - main_correct_rate) scale factor almost entirely,
    since it is constant within each prompt's diversity-adapter group and
    group normalization divides out per-group constants.

    Both inputs must be the same length and row-aligned (one entry per
    diversity-adapter sample).
    """
    if len(answer_correct_float) != len(main_correct_rate):
        raise ValueError(
            "answer_correct_float and main_correct_rate must be the same length, "
            f"got {len(answer_correct_float)} vs {len(main_correct_rate)}"
        )
    return [
        float(c) * (1.0 - float(rate))
        for c, rate in zip(answer_correct_float, main_correct_rate)
    ]


def apply_main_weak_factor(
    aux_correct_advantages: Sequence[float],
    main_correct_rate: Sequence[float],
    weight: float = 1.0,
) -> list[float]:
    """A_main_weak(x, y) = weight * (1 - main_correct_rate(x)) * aux_correct_advantages(x, y).

    `aux_correct_advantages` must already be the group-NORMALIZED pure
    correctness advantage (Norm[c(x,y)], scoped to one diversity adapter's
    own rollouts for one prompt) -- i.e. the main-weakness factor is applied
    AFTER normalization, not before, so it is not cancelled by the group
    std division the way the old raw-reward-then-normalize formula was.
    """
    if len(aux_correct_advantages) != len(main_correct_rate):
        raise ValueError(
            "aux_correct_advantages and main_correct_rate must be the same length, "
            f"got {len(aux_correct_advantages)} vs {len(main_correct_rate)}"
        )
    return [
        float(weight) * (1.0 - float(rate)) * float(a)
        for a, rate in zip(aux_correct_advantages, main_correct_rate)
    ]


def normalize_within_group(
    values: Sequence[float],
    num_completions: int,
    scale_rewards: str = "group",
    std_epsilon: float = 1e-4,
) -> list[float]:
    """Reference implementation of the trainer's shared group-normalization step.

    Mirrors -- using the exact same torch ops and defaults -- the generic
    normalization block in `_score_completions_diversity` (mean-center per
    contiguous group of `num_completions` rows, then divide by the group's
    std + `std_epsilon` when `scale_rewards != "none"`). This lets unit
    tests validate main_weak_correctness_bonus's advantage formula
    end-to-end without instantiating a full model/trainer.

    The production code path does NOT call this function -- it reuses the
    trainer's own inline tensor ops directly (so BLEU/trace-Jaccard/
    policy-repulsion/main_weak all share one implementation). This is a
    behavior-matching test double, kept here so the formula it mirrors is
    documented in one place.
    """
    import torch

    rewards = torch.as_tensor(list(values), dtype=torch.float32)
    mean_grouped = rewards.view(-1, num_completions).mean(dim=1).repeat_interleave(num_completions, dim=0)
    advantages = rewards - mean_grouped

    if scale_rewards in ("group", "none"):
        std_rewards = rewards.view(-1, num_completions).std(dim=1).repeat_interleave(num_completions, dim=0)
    elif scale_rewards == "batch":
        std_rewards = rewards.std().expand_as(rewards)
    else:
        raise ValueError(f"Invalid scale_rewards: {scale_rewards!r}")

    std_rewards = torch.nan_to_num(std_rewards, nan=0.0, posinf=0.0, neginf=0.0)
    if scale_rewards != "none":
        advantages = advantages / (std_rewards + std_epsilon)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    return advantages.tolist()


def compute_group_coverage_stats(
    source_trace_metadata: Sequence[Any],
    main_source_name: str = "default",
    correct_field: str = "answer_correct_float",
    prompt_index_field: str = "prompt_index",
    source_field: str = "source_adapter_name",
) -> dict[str, float]:
    """Per-prompt coverage statistics over the WHOLE rollout group (main + all diversity sources).

    Every row in `source_trace_metadata` must already carry a pure
    correctness field (`answer_correct_float`), a `prompt_index`, and a
    `source_adapter_name` -- i.e. this should be called after every adapter
    in the step has had its correctness scored (or, since "default" holds
    the full unsliced pool, immediately after default's own scoring pass is
    enough as long as diversity rows are also included in that same list).

    Returns a dict with:
        frac_main_all_wrong:   frac of prompts with main_correct_rate(x) == 0
        frac_main_all_correct: frac of prompts with main_correct_rate(x) == 1
        frac_main_mixed:       frac of prompts with 0 < main_correct_rate(x) < 1
        main_all_wrong_diversity_any_correct_rate:
            among main-all-wrong prompts, frac where >=1 diversity sample is correct
        any_correct_rate_total_group:
            frac of prompts where >=1 sample in the WHOLE group is correct
        mixed_correctness_group_rate_total:
            frac of prompts where the whole group is neither all-wrong nor all-correct
    """
    prompts: dict[int, dict[str, list[float]]] = {}
    for record in source_trace_metadata:
        if not isinstance(record, dict):
            continue
        prompt_index = record.get(prompt_index_field)
        correct = record.get(correct_field)
        if prompt_index is None or correct is None:
            continue
        bucket = prompts.setdefault(prompt_index, {"main": [], "div": [], "all": []})
        correct = float(correct)
        bucket["all"].append(correct)
        if record.get(source_field) == main_source_name:
            bucket["main"].append(correct)
        else:
            bucket["div"].append(correct)

    zero = {
        "frac_main_all_wrong": 0.0,
        "frac_main_all_correct": 0.0,
        "frac_main_mixed": 0.0,
        "main_all_wrong_diversity_any_correct_rate": 0.0,
        "any_correct_rate_total_group": 0.0,
        "mixed_correctness_group_rate_total": 0.0,
    }
    n_prompts = len(prompts)
    if n_prompts == 0:
        return zero

    n_main_all_wrong = 0
    n_main_all_correct = 0
    n_main_mixed = 0
    n_main_all_wrong_div_any_correct = 0
    n_any_correct_total = 0
    n_mixed_total = 0

    for bucket in prompts.values():
        main_vals = bucket["main"]
        all_vals = bucket["all"]
        div_vals = bucket["div"]

        if main_vals:
            main_rate = sum(main_vals) / len(main_vals)
            if main_rate <= 0.0:
                n_main_all_wrong += 1
                if any(v >= 1.0 for v in div_vals):
                    n_main_all_wrong_div_any_correct += 1
            elif main_rate >= 1.0:
                n_main_all_correct += 1
            else:
                n_main_mixed += 1

        total_correct = sum(all_vals)
        if total_correct > 0:
            n_any_correct_total += 1
        if 0 < total_correct < len(all_vals):
            n_mixed_total += 1

    n_main_known = n_main_all_wrong + n_main_all_correct + n_main_mixed
    main_denom = n_main_known if n_main_known > 0 else 1

    return {
        "frac_main_all_wrong": n_main_all_wrong / main_denom,
        "frac_main_all_correct": n_main_all_correct / main_denom,
        "frac_main_mixed": n_main_mixed / main_denom,
        "main_all_wrong_diversity_any_correct_rate": (
            n_main_all_wrong_div_any_correct / n_main_all_wrong if n_main_all_wrong > 0 else 0.0
        ),
        "any_correct_rate_total_group": n_any_correct_total / n_prompts,
        "mixed_correctness_group_rate_total": n_mixed_total / n_prompts,
    }
