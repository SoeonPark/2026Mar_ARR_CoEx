from __future__ import annotations

import json
import math
import re
import warnings
from collections import defaultdict
from typing import Any, List

import numpy as np
from evaluate import load
from sentence_transformers import SentenceTransformer, util

from .parsing import extract_solutions


def _get_answer(answers: Any, index: int) -> str:
    if isinstance(answers, (list, tuple)):
        return answers[index] if index < len(answers) else ""
    return answers if answers is not None else ""


_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"<answer\b[^>]*>|<final_?answer\b[^>]*>|\bfinal\s+answer\s*:|(?:^|\n)\s*answer\s*:",
    re.IGNORECASE,
)


def response_to_text(response: Any) -> str:
    """Convert standard or conversational completion objects to plain text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("content", response))
    if isinstance(response, (list, tuple)):
        parts = []
        for item in response:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(response)


def extract_reasoning_trace(response: Any) -> str:
    """Extract reasoning while excluding an explicitly marked final answer."""
    text = response_to_text(response).strip()
    if not text:
        return ""

    think_matches = _THINK_RE.findall(text)
    if think_matches:
        return "\n".join(block.strip() for block in think_matches if block.strip())

    open_think = _OPEN_THINK_RE.search(text)
    if open_think:
        text = text[open_think.end() :]

    answer_match = _FINAL_ANSWER_RE.search(text)
    if answer_match:
        return text[: answer_match.start()].strip()

    if "####" in text:
        return text.split("####", 1)[0].strip()

    return text


def tokenize_reasoning_trace(trace: str) -> list[str]:
    """Whitespace-tokenize after lightweight lowercase/punctuation normalization."""
    normalized = re.sub(r"[^\w]+", " ", trace.lower(), flags=re.UNICODE)
    return normalized.split()


def make_ngram_set(tokens: list[str], ngram_size: int = 3) -> set[tuple[str, ...]]:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if len(tokens) < ngram_size:
        return set()
    return {
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    }


def jaccard_similarity(
    left: set[tuple[str, ...]], right: set[tuple[str, ...]]
) -> float:
    """Return a finite Jaccard similarity with explicit empty-set behavior."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 1.0
    similarity = len(left & right) / len(union)
    return float(min(1.0, max(0.0, similarity)))


def trace_jaccard_similarity(
    left_response: Any, right_response: Any, ngram_size: int = 3
) -> float:
    left_tokens = tokenize_reasoning_trace(extract_reasoning_trace(left_response))
    right_tokens = tokenize_reasoning_trace(extract_reasoning_trace(right_response))
    return jaccard_similarity(
        make_ngram_set(left_tokens, ngram_size),
        make_ngram_set(right_tokens, ngram_size),
    )


def _stable_prompt_key(prompt: Any) -> str:
    try:
        return json.dumps(prompt, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        return repr(prompt)


def trace_jaccard_diversity_reward(
    prompts: list[Any],
    completions: list[Any],
    other_completions: list[Any],
    other_prompts: list[Any] | None = None,
    candidate_ids: list[Any] | None = None,
    other_candidate_ids: list[Any] | None = None,
    exclude_self: bool = False,
    ngram_size: int = 3,
    return_diagnostics: bool = False,
    **_: Any,
) -> list[float] | tuple[list[float], dict[str, list[float]]]:
    """Compute 1 - max trace Jaccard similarity against a configurable comparison pool."""
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have equal lengths")
    if other_prompts is not None and len(other_prompts) != len(other_completions):
        raise ValueError("other_prompts and other_completions must have equal lengths")
    if candidate_ids is not None and len(candidate_ids) != len(completions):
        raise ValueError("candidate_ids and completions must have equal lengths")
    if other_candidate_ids is not None and len(other_candidate_ids) != len(other_completions):
        raise ValueError("other_candidate_ids and other_completions must have equal lengths")

    comparison_by_prompt: dict[str, list[tuple[Any, set[tuple[str, ...]]]]] = defaultdict(list)
    all_comparison_ngrams: list[tuple[Any, set[tuple[str, ...]]]] = []
    for index, other_completion in enumerate(other_completions):
        other_trace = extract_reasoning_trace(other_completion)
        other_ngrams = make_ngram_set(
            tokenize_reasoning_trace(other_trace), ngram_size
        )
        other_id = other_candidate_ids[index] if other_candidate_ids is not None else None
        comparison_item = (other_id, other_ngrams)
        all_comparison_ngrams.append(comparison_item)
        if other_prompts is not None:
            comparison_by_prompt[_stable_prompt_key(other_prompts[index])].append(
                comparison_item
            )

    rewards = []
    max_similarities = []
    trace_lengths = []
    empty_traces = []
    comparison_sizes = []

    for index, completion in enumerate(completions):
        trace = extract_reasoning_trace(completion)
        tokens = tokenize_reasoning_trace(trace)
        ngrams = make_ngram_set(tokens, ngram_size)
        if other_prompts is None:
            comparison_items = all_comparison_ngrams
        else:
            comparison_items = comparison_by_prompt.get(
                _stable_prompt_key(prompts[index]), []
            )
        if exclude_self and candidate_ids is not None and other_candidate_ids is not None:
            current_id = candidate_ids[index]
            comparisons = [other for other_id, other in comparison_items if other_id != current_id]
        else:
            comparisons = [other for _, other in comparison_items]

        similarities = [jaccard_similarity(ngrams, other) for other in comparisons]
        # A missing comparison set indicates malformed rollout metadata. Avoid
        # granting a maximal diversity reward in that case.
        max_similarity = max(similarities, default=1.0)
        if not math.isfinite(max_similarity):
            max_similarity = 0.0
        max_similarity = min(1.0, max(0.0, max_similarity))
        reward = min(1.0, max(0.0, 1.0 - max_similarity))

        rewards.append(reward)
        max_similarities.append(max_similarity)
        trace_lengths.append(float(len(tokens)))
        empty_traces.append(float(len(tokens) == 0))
        comparison_sizes.append(float(len(comparisons)))

    if not return_diagnostics:
        return rewards
    return rewards, {
        "max_jaccard_similarity": max_similarities,
        "trace_length": trace_lengths,
        "empty_trace": empty_traces,
        "comparison_size": comparison_sizes,
    }


trace_jaccard3_similarity = trace_jaccard_similarity
trace_jaccard3_reward = trace_jaccard_diversity_reward


def levenstein_distance(prompts: List[str], completions: List[str], completion_ids: List, **kwargs) -> List[float]:
    if len(completions) < 2:
        return [0.0]

    current_prompt = prompts[0]
    current_completion = completions[0]
    answers = kwargs.get("answer", [""] * len(prompts))
    current_answer = answers[0]

    similarity_scores = []

    for i in range(1, len(completions)):
        compare_completion = completions[i]
        compare_answer = answers[i] if i < len(answers) else ""

        sol1 = extract_solutions([current_prompt], [current_completion], [current_answer])
        sol2 = extract_solutions([prompts[i]], [compare_completion], [compare_answer])

        if not sol1 and not sol2:
            similarity = 1.0
        elif not sol1 or not sol2:
            similarity = 0.0
        else:
            leven = load("character")
            max_len = max(len(sol1), len(sol2))
            try:
                distance = leven.compute(predictions=[sol1], references=[sol2])["character_error_rate"] * max_len
                similarity = 1.0 - (distance / max_len)
            except Exception:
                similarity = 1.0

        similarity_scores.append(similarity)

    return similarity_scores


def get_embedding(model_name: str) -> np.ndarray:
    if not hasattr(get_embedding, "model"):
        get_embedding.model = SentenceTransformer(model_name)
    return get_embedding.model


def bert_score(prompts: List[str], completions: List[str], completion_ids: List[str], **kwargs) -> float:
    if len(completions) < 2:
        return [0.0]

    current_prompt = prompts[0]
    current_completion = completions[0]
    answers = kwargs.get("answer", [""] * len(prompts))
    current_answer = answers[0]

    similarity_scores = []

    model = get_embedding()
    print(f"  >> Loaded embedding model: {get_embedding.model.__class__.__name__.upper()}")

    for i in range(1, len(completions)):
        compare_completion = completions[i]
        compare_answer = answers[i] if i < len(answers) else ""

        sol1 = extract_solutions([current_prompt], [current_completion], [current_answer])
        sol2 = extract_solutions([prompts[i]], [compare_completion], [compare_answer])

        try:
            embeddings = model.encode([sol1, sol2], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        except Exception:
            similarity = 1.0

        similarity_scores.append(similarity)

    return similarity_scores


bleu = load("bleu")


def bleu_score(prompts: List[str], completions: List[str], other_completions: List[str], **kwargs) -> List[float]:
    all_other_completions = []
    answers = kwargs.get("answer", [""] * len(prompts))
    for other_comp_i in other_completions:
        sol2 = extract_solutions([prompts[0]], [other_comp_i], [_get_answer(answers, 0)])
        if not sol2:
            sol2 = ""
        all_other_completions.append(sol2)

    similarity_scores = []
    for i in range(len(completions)):
        sol = extract_solutions([prompts[i]], [completions[i]], [_get_answer(answers, i)])

        if not sol:
            similarity_scores.append(1.0)
        else:
            scores = bleu.compute(predictions=[sol], references=[all_other_completions])
            similarity = scores["bleu"]
            similarity_scores.append(1 - similarity)

    return similarity_scores


def one_minus_bleu_score(prompts: List[str], completions: List[str], other_completions: List[str], **kwargs) -> List[float]:
    all_other_solutions = []
    answers = kwargs.get("answer", [""] * len(prompts))
    for other_comp_i in other_completions:
        sol2 = extract_solutions([prompts[0]], [other_comp_i], [_get_answer(answers, 0)])
        all_other_solutions.append(sol2 if sol2 else "")

    similarity_scores = []
    for i in range(len(completions)):
        sol = extract_solutions([prompts[i]], [completions[i]], [_get_answer(answers, i)])

        if not sol or not all_other_solutions:
            similarity_scores.append(1.0)
        else:
            individual_bleus = []
            for other_sol in all_other_solutions:
                score = bleu.compute(predictions=[sol], references=[[other_sol]])
                individual_bleus.append(score["bleu"])

            avg_similarity = sum(individual_bleus) / len(individual_bleus)
            similarity_scores.append(1 - avg_similarity)

    return similarity_scores


# ---------------------------------------------------------------------------
# Principled all-other 1-BLEU diversity reward
# ---------------------------------------------------------------------------

def _extract_completion_text(completion: Any, text_scope: str = "full_completion") -> str:
    """Centralized text extractor for BLEU-based diversity rewards.

    text_scope values
    -----------------
    "full_completion"  : full completion string (default)
    "reasoning_only"   : <think>...</think> block only
    "answer_only"      : first assistant message content (legacy extract_solutions behavior)
    """
    if text_scope == "full_completion":
        return response_to_text(completion)
    elif text_scope == "reasoning_only":
        return extract_reasoning_trace(completion)
    elif text_scope == "answer_only":
        sol = extract_solutions([""], [completion], [""])
        return sol if sol else ""
    else:
        raise ValueError(
            f"Unknown diversity_bleu_text_scope: {text_scope!r}. "
            "Allowed values: 'full_completion', 'reasoning_only', 'answer_only'."
        )


def compute_one_minus_bleu_rewards_for_adapter(
    source_completions: list[Any],
    reference_groups: dict[str, list[Any]],
    balance_mode: str = "sample_balanced",
    main_weight: float = 0.5,
    exclude_self: bool = True,
    text_scope: str = "full_completion",
    source_group_name: str | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Per-sample 1-BLEU diversity reward with sample- or source-balanced averaging.

    Parameters
    ----------
    source_completions
        Completions produced by this diversity adapter (length N_d).
    reference_groups
        Mapping from source name to list of reference completions.
        E.g. ``{"default": [...], "diversity_1": [...]}``.
        For ``intra_adapter`` mode pass ``{adapter_name: source_completions}``.
    balance_mode
        ``"sample_balanced"`` : every reference completion gets equal weight.
        ``"source_balanced"``  : main and other-diversity sources are averaged at
        the source level first, then combined with ``main_weight``.
    main_weight
        α_m ∈ [0, 1]. Weight for the "default" source in source-balanced mode.
        Ignored in sample-balanced mode.
    exclude_self
        Whether to skip comparing a sample against itself.  Applies only when
        ``source_group_name`` matches a key in ``reference_groups`` (i.e.
        intra-adapter comparisons).  In all-other mode this is a no-op.
    text_scope
        Text extraction scope passed to ``_extract_completion_text``.
    source_group_name
        Name of the current adapter (used solely for self-exclusion logic).
        Set to ``None`` for all-other mode where the source adapter is absent
        from ``reference_groups``.

    Returns
    -------
    rewards : list[float] of length len(source_completions)
    diagnostics : dict with diagnostic lists/values
    """
    if not reference_groups:
        raise ValueError(
            "reference_groups is empty. At least one reference source must be "
            "provided to compute a diversity reward."
        )
    if balance_mode not in {"sample_balanced", "source_balanced"}:
        raise ValueError(
            f"Unknown balance_mode: {balance_mode!r}. "
            "Allowed: 'sample_balanced', 'source_balanced'."
        )
    if not (0.0 <= main_weight <= 1.0):
        raise ValueError(f"main_weight must be in [0, 1], got {main_weight}")

    # Warn about sample-count imbalance when using sample_balanced
    if balance_mode == "sample_balanced":
        group_sizes = {k: len(v) for k, v in reference_groups.items()}
        unique_sizes = set(group_sizes.values())
        if len(unique_sizes) > 1:
            warnings.warn(
                "diversity_bleu_balance_mode='sample_balanced' with unequal source "
                f"sample counts ({group_sizes}). Source weights will be proportional "
                "to sample counts and will change if rollout allocation changes. "
                "Consider 'source_balanced' mode for allocation-independent rewards.",
                UserWarning,
                stacklevel=2,
            )

    n_src = len(source_completions)

    # Pre-extract all texts once
    src_texts: list[str] = [_extract_completion_text(c, text_scope) for c in source_completions]
    ref_texts: dict[str, list[str]] = {
        grp: [_extract_completion_text(c, text_scope) for c in comps]
        for grp, comps in reference_groups.items()
    }

    rewards: list[float] = []
    diag_sim_main: list[float] = []
    diag_sim_other_div: list[float] = []

    for src_local_idx in range(n_src):
        src_text = src_texts[src_local_idx]

        if balance_mode == "sample_balanced":
            # Flatten all references with per-position self-exclusion
            flat_refs: list[str] = []
            for grp_name, grp_ref_list in ref_texts.items():
                for ref_local_idx, ref_text in enumerate(grp_ref_list):
                    if exclude_self and grp_name == source_group_name and ref_local_idx == src_local_idx:
                        continue
                    flat_refs.append(ref_text)

            if not flat_refs:
                rewards.append(1.0)
                diag_sim_main.append(float("nan"))
                diag_sim_other_div.append(float("nan"))
                continue

            bleu_vals: list[float] = []
            for ref_text in flat_refs:
                if not src_text or not ref_text:
                    bleu_vals.append(0.0)
                else:
                    s = bleu.compute(predictions=[src_text], references=[[ref_text]])
                    bleu_vals.append(s["bleu"])

            avg_sim = sum(bleu_vals) / len(bleu_vals)
            rewards.append(max(0.0, min(1.0, 1.0 - avg_sim)))
            diag_sim_main.append(float("nan"))
            diag_sim_other_div.append(float("nan"))

        else:  # source_balanced
            source_sims: dict[str, float] = {}

            for grp_name, grp_ref_list in ref_texts.items():
                valid_refs = [
                    rt for ri, rt in enumerate(grp_ref_list)
                    if not (exclude_self and grp_name == source_group_name and ri == src_local_idx)
                ]
                if not valid_refs:
                    continue

                grp_bleus: list[float] = []
                for ref_text in valid_refs:
                    if not src_text or not ref_text:
                        grp_bleus.append(0.0)
                    else:
                        s = bleu.compute(predictions=[src_text], references=[[ref_text]])
                        grp_bleus.append(s["bleu"])

                source_sims[grp_name] = sum(grp_bleus) / len(grp_bleus)

            if not source_sims:
                rewards.append(1.0)
                diag_sim_main.append(float("nan"))
                diag_sim_other_div.append(float("nan"))
                continue

            S_main = source_sims.get("default", None)
            other_vals = [v for k, v in source_sims.items() if k != "default"]

            if S_main is not None and other_vals:
                # General case: main + other diversity sources
                S_other = sum(other_vals) / len(other_vals)
                sim = main_weight * S_main + (1.0 - main_weight) * S_other
                diag_sim_main.append(S_main)
                diag_sim_other_div.append(S_other)
            elif S_main is not None:
                # K=1 (no other diversity policies): reduce to pure main comparison
                sim = S_main
                diag_sim_main.append(S_main)
                diag_sim_other_div.append(float("nan"))
            else:
                # Main references absent: use only other diversity sources
                sim = sum(other_vals) / len(other_vals)
                diag_sim_main.append(float("nan"))
                diag_sim_other_div.append(sim)

            rewards.append(max(0.0, min(1.0, 1.0 - sim)))

    diagnostics: dict[str, Any] = {
        "sim_main_mean": diag_sim_main,
        "sim_other_div_mean": diag_sim_other_div,
        "num_source": n_src,
        "reference_group_sizes": {k: len(v) for k, v in reference_groups.items()},
    }
    return rewards, diagnostics
