"""Dependency-light reward helpers shared by CoEx reward implementations."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any


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
