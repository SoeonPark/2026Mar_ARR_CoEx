"""Shared scoring helpers for repeated-sampling AIME tasks."""

from math import comb

from lm_eval.tasks.aime.utils import process_results as process_single_result


PASS_KS = (1, 2, 4, 8, 16, 32)


def _pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Return the standard unbiased pass@k estimate."""
    if num_samples < k:
        raise ValueError(f"pass@{k} requires at least {k} samples")
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def process_results_pass_at_k(doc: dict, results: list) -> dict[str, float]:
    """Score all generated responses for one AIME problem.

    The ``take_first_k`` filter preserves the repeats as a nested list, so the
    evaluator supplies ``[[response_1, ..., response_k]]`` here.
    """
    responses = (
        results[0]
        if len(results) == 1 and isinstance(results[0], list)
        else results
    )

    num_samples = len(responses)
    if num_samples <= 0:
        raise ValueError(
            "Expected at least one generation per AIME problem, "
            f"got {num_samples}"
        )

    correct = sum(
        process_single_result(doc, [response])["exact_match"]
        for response in responses
    )

    return {
        f"pass@{k}": _pass_at_k(num_samples, correct, k)
        for k in PASS_KS
        if k <= num_samples
    }
