#!/usr/bin/env python3
"""Analyze repeated-sampling AIME results produced by lm-evaluation-harness.

The script treats each ``samples_aime{24,25}_passk_*.jsonl`` row as one
benchmark problem, recursively flattens its generations, keeps main-policy
samples for final metrics, and writes response-, problem-, and summary-level
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence


try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAVE_SKLEARN = True
except ImportError:
    np = None
    TfidfVectorizer = None
    HAVE_SKLEARN = False


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|\\[A-Za-z]+|[^\s]")
INTEGER_RE = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?![\w.])")
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.I | re.S)
ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.I | re.S)
FINAL_ANSWER_RE = re.compile(r"final\s+answer", re.I)
MATH_PATTERNS = (
    re.compile(r"\\\[(.*?)\\\]", re.S),
    re.compile(r"\\\((.*?)\\\)", re.S),
    re.compile(r"(?<!\\)\$(?!\$)(.*?)(?<!\\)\$", re.S),
)
PROVENANCE_KEYS = ("source_id", "source", "adapter_name", "adapter", "policy")
MAIN_SOURCE_VALUES = {"", "main", "main_policy", "main-policy", "default", "none", "null"}
EXPECTED_PROBLEMS = 30
PRIMARY_PATH_THRESHOLD = 0.80


PARSED_SAMPLE_FIELDS = [
    "method",
    "benchmark",
    "file_path",
    "problem_id",
    "sample_idx",
    "K_observed",
    "target_raw",
    "target_norm",
    "pred_raw",
    "pred_norm",
    "is_correct",
    "parse_success",
    "boxed_answer_found",
    "answer_source",
    "source_id",
    "adapter_name",
    "policy",
    "included_final_metric",
    "raw_response",
    "think_text",
    "text_for_metrics",
    "length_chars",
    "length_tokens_approx",
]


@dataclass
class ResponseLeaf:
    response: str
    metadata: dict[str, Any]


@dataclass
class ParsedResponse:
    sample_idx: int
    response: str
    think_text: str
    text_for_metrics: str
    tokens: list[str]
    pred_raw: str | None
    pred_norm: str | None
    answer_source: str | None
    boxed_answer_found: bool
    parse_success: bool
    is_correct: bool
    provenance: dict[str, str]
    included: bool


def warn(message: str, events: list[dict[str, Any]], **context: Any) -> None:
    print(f"[WARNING] {message}", file=sys.stderr)
    events.append({"type": "warning", "message": message, **context})


def discover_files(root: Path) -> list[Path]:
    files = list(root.rglob("samples_aime24_passk_*.jsonl"))
    files += list(root.rglob("samples_aime25_passk_*.jsonl"))
    return sorted(set(files))


def infer_benchmark(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("samples_aime24_passk_"):
        return "aime24"
    if name.startswith("samples_aime25_passk_"):
        return "aime25"
    return "unknown"


def infer_method(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    category = relative.parts[0] if relative.parts else ""
    if category == "Pretrained":
        return "Pretrained"
    if category.startswith("1-BLEU"):
        return "CoEx"
    if category == "DMPO":
        return "DMPO"
    if category == "GRPO":
        return "GRPO"
    # Fallback for layouts without a method category directly below root.
    meaningful_parts = [part for part in relative.parts if not part.startswith("__home__")]
    joined = "/".join(meaningful_parts)
    if "Pretrained" in meaningful_parts:
        return "Pretrained"
    if "1-BLEU_GRPO" in joined or any(part.startswith("CoEx_GRPO") for part in meaningful_parts):
        return "CoEx"
    if any(part == "DMPO" or part.startswith("DMPO_") for part in meaningful_parts):
        return "DMPO"
    if "GRPO" in meaningful_parts and "sampling" in meaningful_parts:
        return "GRPO"
    return "Unknown"


def expected_k_from_path(path: Path, root: Path) -> int:
    rel_parts = path.relative_to(root).parts
    return 32 if "pass@32" in rel_parts else 16


def first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def recursive_flatten_responses(value: Any, inherited: dict[str, Any] | None = None) -> list[ResponseLeaf]:
    inherited = dict(inherited or {})
    leaves: list[ResponseLeaf] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            leaves.extend(recursive_flatten_responses(item, inherited))
        return leaves
    if isinstance(value, dict):
        metadata = dict(inherited)
        for key in PROVENANCE_KEYS:
            if key in value:
                metadata[key] = value[key]
        response = first_present(value, ("response", "text", "output", "generated_text"))
        if response is not None:
            return [ResponseLeaf(str(response), metadata)]
        for key in ("resps", "responses", "samples", "outputs", "generations"):
            if key in value:
                leaves.extend(recursive_flatten_responses(value[key], metadata))
        return leaves
    if value is not None:
        leaves.append(ResponseLeaf(str(value), inherited))
    return leaves


def row_provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in PROVENANCE_KEYS if key in row}


def normalize_provenance(metadata: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in PROVENANCE_KEYS:
        value = metadata.get(key)
        if value is not None:
            result[key] = str(value)
    return result


def is_main_policy(provenance: dict[str, str]) -> bool:
    for key, value in provenance.items():
        normalized = value.strip().lower()
        if key in {"source_id", "source", "policy"} and normalized not in MAIN_SOURCE_VALUES:
            return False
        if key in {"adapter_name", "adapter"} and normalized not in MAIN_SOURCE_VALUES:
            return False
    return True


def extract_problem_id(row: dict[str, Any], row_index: int) -> str:
    doc = row.get("doc")
    if isinstance(doc, dict) and doc.get("ID") is not None:
        return str(doc["ID"])
    if row.get("ID") is not None:
        return str(row["ID"])
    if row.get("doc_id") is not None:
        return str(row["doc_id"])
    return str(row_index)


def extract_target(row: dict[str, Any]) -> Any:
    if row.get("target") is not None:
        return row["target"]
    doc = row.get("doc")
    if isinstance(doc, dict) and doc.get("Answer") is not None:
        return doc["Answer"]
    return row.get("Answer")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def extract_think_text(response: str) -> str:
    matches = THINK_RE.findall(response)
    if matches:
        return "\n".join(match.strip() for match in matches if match.strip())
    open_match = re.search(r"<think\b[^>]*>", response, re.I)
    if open_match:
        tail = response[open_match.end() :]
        answer_start = re.search(r"<answer\b", tail, re.I)
        return tail[: answer_start.start() if answer_start else None].strip()
    return response.strip()


def boxed_contents(text: str) -> list[str]:
    results: list[str] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[start : index - 1])
    return results


def normalize_integer(value: Any, allow_search: bool = False) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if 0 <= value <= 999 else None
    if isinstance(value, float) and value.is_integer():
        integer = int(value)
        return str(integer) if 0 <= integer <= 999 else None
    text = str(value).strip()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"\\(?:boxed|fbox)\s*\{(.*)\}", r"\1", text, flags=re.S)
    text = re.sub(r"\\text\s*\{(.*?)\}", r"\1", text, flags=re.S)
    text = text.strip().rstrip(".").strip()
    if re.search(r"\\(?:d?frac|tfrac)|\d\s*/\s*\d", text):
        return None
    direct = re.fullmatch(r"[+]?\d+", text)
    if direct:
        digits = direct.group().lstrip("+")
        significant = digits.lstrip("0") or "0"
        # Reject pathological generations without imposing the AIME 0..999
        # range here; normalization itself should still map "1,234" to 1234.
        if len(significant) > 32:
            return None
        integer = int(significant)
        return str(integer) if integer >= 0 else None
    if allow_search:
        matches = INTEGER_RE.findall(text)
        for match in reversed(matches):
            normalized = normalize_integer(match, allow_search=False)
            if normalized is not None:
                return normalized
    return None


def last_integer(text: str) -> tuple[str | None, str | None]:
    matches = INTEGER_RE.findall(text)
    for raw in reversed(matches):
        normalized = normalize_integer(raw)
        if normalized is not None:
            return raw, normalized
    return None, None


def extract_final_answer(response: str) -> tuple[str | None, str | None, str | None, bool]:
    boxes = boxed_contents(response)
    for content in reversed(boxes):
        normalized = normalize_integer(content, allow_search=False)
        if normalized is not None:
            return content, normalized, "boxed", True

    answer_sections = ANSWER_RE.findall(response)
    for section in reversed(answer_sections):
        raw, normalized = last_integer(section)
        if normalized is not None:
            return raw, normalized, "answer_tag", bool(boxes)

    final_markers = list(FINAL_ANSWER_RE.finditer(response))
    for marker in reversed(final_markers):
        raw, normalized = last_integer(response[marker.end() :])
        if normalized is not None:
            return raw, normalized, "final_answer", bool(boxes)

    raw, normalized = last_integer(response)
    return raw, normalized, "last_integer" if normalized is not None else None, bool(boxes)


def text_for_reasoning_metrics(think_text: str) -> str:
    text = ANSWER_RE.sub(" ", think_text)
    marker = FINAL_ANSWER_RE.search(text)
    if marker:
        text = text[: marker.start()]
    text = re.sub(r"\\(?:boxed|fbox)\s*\{[^{}]*\}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n or n <= 0:
        return math.nan
    if c <= 0:
        return 0.0
    if k == 1:
        return c / n
    if k == n:
        return 1.0
    if n - c < k:
        return 1.0
    product = 1.0
    for value in range(n - c + 1, n + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    return (
        zip(*(islice(tokens, offset, None) for offset in range(n)))
        if len(tokens) >= n
        else ()
    )


def _reference_length(lengths: list[int], candidate_index: int) -> int:
    candidate_length = lengths[candidate_index]
    alternatives = [length for index, length in enumerate(lengths) if index != candidate_index]
    return min(alternatives, key=lambda length: (abs(length - candidate_length), length))


def text_group_metrics(
    token_lists: list[list[str]], distinct_ns: list[int]
) -> tuple[float, dict[int, float]]:
    """Compute optimized self-BLEU and corpus-style distinct-n for one problem.

    For BLEU, each n-gram stores the two largest counts across references. This
    gives the same leave-one-out clipped precision as repeatedly constructing
    K-1 reference lists, without the prohibitive O(K^2 * text_length) work.
    """
    size = len(token_lists)
    lengths = [len(tokens) for tokens in token_lists]
    log_precision = [0.0] * size
    distinct: dict[int, float] = {}
    orders = sorted(set(distinct_ns) | {1, 2, 3, 4})

    for order in orders:
        counters: list[Counter[tuple[str, ...]]] = []
        top_counts: dict[tuple[str, ...], list[int]] = {}
        total_ngrams = 0
        for sample_index, tokens in enumerate(token_lists):
            counter = Counter(iter_ngrams(tokens, order))
            counters.append(counter)
            total_ngrams += sum(counter.values())
            for ngram, count in counter.items():
                stats = top_counts.get(ngram)
                if stats is None:
                    top_counts[ngram] = [count, sample_index, 1, 0]
                elif count > stats[0]:
                    stats[3] = stats[0]
                    stats[0] = count
                    stats[1] = sample_index
                    stats[2] = 1
                elif count == stats[0]:
                    stats[2] += 1
                elif count > stats[3]:
                    stats[3] = count

        if order in distinct_ns:
            distinct[order] = len(top_counts) / (total_ngrams + 1e-12)

        if order <= 4 and size >= 2:
            for sample_index, counter in enumerate(counters):
                denominator = sum(counter.values())
                if denominator == 0:
                    precision = 1e-12
                else:
                    clipped = 0
                    for ngram, count in counter.items():
                        maximum, owner, ties, second = top_counts[ngram]
                        reference_max = second if owner == sample_index and ties == 1 else maximum
                        clipped += min(count, reference_max)
                    # NLTK method1-style smoothing for zero modified precision.
                    precision = clipped / denominator if clipped else 0.1 / denominator
                log_precision[sample_index] += 0.25 * math.log(max(precision, 1e-300))

        del counters, top_counts

    if size < 2:
        return math.nan, distinct

    scores: list[float] = []
    for index, candidate_length in enumerate(lengths):
        if candidate_length == 0:
            scores.append(0.0)
            continue
        reference_length = _reference_length(lengths, index)
        brevity_penalty = (
            1.0
            if candidate_length > reference_length
            else math.exp(1.0 - reference_length / max(candidate_length, 1))
        )
        scores.append(brevity_penalty * math.exp(log_precision[index]))
    return mean_finite(scores), distinct


def canonicalize_equation(equation: str) -> str:
    text = equation.lower().replace("\\left", "").replace("\\right", "")
    text = re.sub(
        r"\\(?:d?frac|tfrac)\{([^{}]+)\}\{([^{}]+)\}",
        r"(\1)/(\2)",
        text,
    )
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"([,;:.])\1+", r"\1", text)
    return text.strip(" \t\r\n,;:.$")


def extract_equations(trace: str) -> set[str]:
    candidates: list[str] = []
    for pattern in MATH_PATTERNS:
        candidates.extend(pattern.findall(trace))
    for line in trace.splitlines():
        if re.search(r"=|=>|\\implies|\blog\b|\\log|\bsqrt\b|\\sqrt|\\frac|\^|_", line, re.I):
            candidates.append(line)
    equations = {canonicalize_equation(candidate) for candidate in candidates}
    return {equation for equation in equations if equation}


def normalize_trace(trace: str) -> str:
    text = trace.lower()
    text = re.sub(r"\\(?:boxed|fbox)\s*\{[^{}]*\}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity_matrix(texts: list[str]) -> list[list[float]]:
    size = len(texts)
    if size <= 1:
        return [[1.0]] if size == 1 else []
    if HAVE_SKLEARN:
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                lowercase=False,
                sublinear_tf=True,
                max_features=50_000,
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform(texts)
            dense = (matrix @ matrix.T).toarray()
            return dense.tolist()
        except ValueError:
            pass
    similarities = [[0.0] * size for _ in range(size)]
    for left in range(size):
        similarities[left][left] = 1.0
        for right in range(left + 1, size):
            value = difflib.SequenceMatcher(None, texts[left], texts[right], autojunk=False).ratio()
            similarities[left][right] = similarities[right][left] = value
    return similarities


def combined_similarity_matrix(traces: list[str], equations: list[set[str]]) -> list[list[float]]:
    normalized = [normalize_trace(trace) for trace in traces]
    text_matrix = text_similarity_matrix(normalized)
    size = len(traces)
    combined = [[0.0] * size for _ in range(size)]
    for left in range(size):
        combined[left][left] = 1.0
        for right in range(left + 1, size):
            union = equations[left] | equations[right]
            equation_similarity = len(equations[left] & equations[right]) / len(union) if union else 0.0
            value = 0.6 * text_matrix[left][right] + 0.4 * equation_similarity
            combined[left][right] = combined[right][left] = value
    return combined


def connected_components(similarity: list[list[float]], threshold: float) -> list[list[int]]:
    size = len(similarity)
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(size):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in range(size):
                if neighbor not in visited and similarity[current][neighbor] >= threshold:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def cluster_representative(component: list[int], similarity: list[list[float]]) -> int:
    return max(
        component,
        key=lambda index: sum(similarity[index][other] for other in component) / len(component),
    )


def analyze_correct_paths(
    correct_samples: list[ParsedResponse],
    thresholds: list[float],
    method: str,
    benchmark: str,
    file_path: str,
    problem_id: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    count = len(correct_samples)
    result: dict[str, float] = {}
    exports: list[dict[str, Any]] = []
    if count == 0:
        for threshold in thresholds:
            suffix = threshold_suffix(threshold)
            result[f"unique_correct_path_{suffix}"] = 0
            result[f"unique_correct_path_norm_{suffix}"] = 0.0
            exports.append(
                {
                    "method": method,
                    "benchmark": benchmark,
                    "file_path": file_path,
                    "problem_id": problem_id,
                    "threshold": threshold,
                    "correct_count": 0,
                    "unique_correct_path": 0,
                    "clusters": [],
                }
            )
        return result, exports

    traces = [sample.think_text for sample in correct_samples]
    equation_sets = [extract_equations(trace) for trace in traces]
    similarity = combined_similarity_matrix(traces, equation_sets)
    for threshold in thresholds:
        components = connected_components(similarity, threshold)
        suffix = threshold_suffix(threshold)
        result[f"unique_correct_path_{suffix}"] = len(components)
        result[f"unique_correct_path_norm_{suffix}"] = len(components) / max(count, 1)
        clusters: list[dict[str, Any]] = []
        for cluster_id, component in enumerate(components):
            representative = cluster_representative(component, similarity)
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "sample_indices": [correct_samples[index].sample_idx for index in component],
                    "representative_text": traces[representative],
                    "equations": sorted(equation_sets[representative]),
                }
            )
        exports.append(
            {
                "method": method,
                "benchmark": benchmark,
                "file_path": file_path,
                "problem_id": problem_id,
                "threshold": threshold,
                "correct_count": count,
                "unique_correct_path": len(components),
                "clusters": clusters,
            }
        )
    return result, exports


def threshold_suffix(threshold: float) -> str:
    return f"t{round(threshold * 100):03d}"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean_finite(values: Iterable[Any]) -> float:
    usable = [float(value) for value in values if finite(value)]
    return sum(usable) / len(usable) if usable else math.nan


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ci(values: list[Any], iterations: int, rng: random.Random) -> tuple[float, float]:
    if not values or not any(finite(value) for value in values):
        return math.nan, math.nan
    bootstraps: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimate = mean_finite(sample)
        if finite(estimate):
            bootstraps.append(estimate)
    bootstraps.sort()
    return percentile(bootstraps, 0.025), percentile(bootstraps, 0.975)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else "NaN"


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_problem_delta_rows(
    problem_rows: list[dict[str, Any]],
    delta_metrics: list[str],
    baseline_methods: Sequence[str] = ("GRPO", "DMPO"),
) -> list[dict[str, Any]]:
    """Pair CoEx and baseline problems and return CoEx-minus-baseline deltas."""
    index = {
        (
            str(row["method"]),
            str(row["benchmark"]),
            int(row["K_observed"]),
            str(row["problem_id"]),
        ): row
        for row in problem_rows
    }
    delta_rows: list[dict[str, Any]] = []
    coex_rows = sorted(
        (row for row in problem_rows if row["method"] == "CoEx"),
        key=lambda row: (str(row["benchmark"]), int(row["K_observed"]), str(row["problem_id"])),
    )
    for coex in coex_rows:
        benchmark = str(coex["benchmark"])
        observed_k = int(coex["K_observed"])
        problem_id = str(coex["problem_id"])
        for baseline_method in baseline_methods:
            baseline = index.get((baseline_method, benchmark, observed_k, problem_id))
            if baseline is None:
                continue
            output: dict[str, Any] = {
                "comparison": f"CoEx_vs_{baseline_method}",
                "coex_method": "CoEx",
                "baseline_method": baseline_method,
                "benchmark": benchmark,
                "K_observed": observed_k,
                "problem_id": problem_id,
                "delta_definition": "CoEx - baseline",
                "coex_file_path": coex["file_path"],
                "baseline_file_path": baseline["file_path"],
            }
            for metric in delta_metrics:
                coex_value = as_float(coex.get(metric))
                baseline_value = as_float(baseline.get(metric))
                output[f"coex_{metric}"] = coex_value
                output[f"baseline_{metric}"] = baseline_value
                output[f"delta_{metric}"] = (
                    coex_value - baseline_value
                    if finite(coex_value) and finite(baseline_value)
                    else math.nan
                )
            delta_rows.append(output)
    return delta_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 16, 32])
    parser.add_argument("--distinct_ns", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--path_thresholds", nargs="+", type=float, default=[0.75, 0.80, 0.85])
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(root)
    if not files:
        raise SystemExit(f"No AIME pass@k sample JSONL files found under {root}")

    parse_events: list[dict[str, Any]] = []
    file_mappings: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    path_cluster_exports: list[dict[str, Any]] = []
    parsing_groups: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    k_distributions: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
    coex_unknown_provenance_files: set[str] = set()

    parsed_path = out_dir / "parsed_samples.csv"
    with parsed_path.open("w", newline="", encoding="utf-8") as parsed_handle:
        parsed_writer = csv.DictWriter(parsed_handle, fieldnames=PARSED_SAMPLE_FIELDS)
        parsed_writer.writeheader()

        for file_path in files:
            relative_path = str(file_path.relative_to(root))
            method = infer_method(file_path, root)
            benchmark = infer_benchmark(file_path)
            expected_k = expected_k_from_path(file_path, root)
            if method == "Unknown":
                warn("Could not infer method; using Unknown", parse_events, file_path=relative_path)

            rows: list[dict[str, Any]] = []
            with file_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise TypeError("row is not a JSON object")
                        rows.append(value)
                    except Exception as exc:
                        parse_events.append(
                            {
                                "type": "json_parse_error",
                                "file_path": relative_path,
                                "line_number": line_number,
                                "error": repr(exc),
                            }
                        )

            if len(rows) != EXPECTED_PROBLEMS:
                warn(
                    f"Expected {EXPECTED_PROBLEMS} problems, found {len(rows)}",
                    parse_events,
                    method=method,
                    benchmark=benchmark,
                    file_path=relative_path,
                )

            observed_values: list[int] = []
            for row_index, row in enumerate(rows):
                problem_id = extract_problem_id(row, row_index)
                target_raw = extract_target(row)
                target_norm = normalize_integer(target_raw, allow_search=True)
                inherited_provenance = row_provenance(row)
                leaves = recursive_flatten_responses(row.get("resps", row.get("responses", [])), inherited_provenance)
                k_observed = len(leaves)
                observed_values.append(k_observed)
                if k_observed != expected_k:
                    warn(
                        f"Observed K={k_observed}, expected K={expected_k} from path",
                        parse_events,
                        method=method,
                        benchmark=benchmark,
                        file_path=relative_path,
                        problem_id=problem_id,
                    )

                parsed_responses: list[ParsedResponse] = []
                provenance_present = bool(inherited_provenance) or any(leaf.metadata for leaf in leaves)
                if method == "CoEx" and not provenance_present and relative_path not in coex_unknown_provenance_files:
                    coex_unknown_provenance_files.add(relative_path)
                    warn(
                        "CoEx samples have no source/policy field; assuming main policy, but source mixing cannot be verified",
                        parse_events,
                        method=method,
                        benchmark=benchmark,
                        file_path=relative_path,
                    )

                for sample_index, leaf in enumerate(leaves):
                    provenance = normalize_provenance(leaf.metadata)
                    included = is_main_policy(provenance)
                    think_text = extract_think_text(leaf.response)
                    metric_text = text_for_reasoning_metrics(think_text)
                    pred_raw, pred_norm, answer_source, boxed_found = extract_final_answer(leaf.response)
                    parse_success = pred_norm is not None
                    is_correct = bool(parse_success and target_norm is not None and pred_norm == target_norm)
                    tokens = tokenize(metric_text)
                    parsed = ParsedResponse(
                        sample_idx=sample_index,
                        response=leaf.response,
                        think_text=think_text,
                        text_for_metrics=metric_text,
                        tokens=tokens,
                        pred_raw=pred_raw,
                        pred_norm=pred_norm,
                        answer_source=answer_source,
                        boxed_answer_found=boxed_found,
                        parse_success=parse_success,
                        is_correct=is_correct,
                        provenance=provenance,
                        included=included,
                    )
                    parsed_responses.append(parsed)
                    parsed_writer.writerow(
                        {
                            "method": method,
                            "benchmark": benchmark,
                            "file_path": relative_path,
                            "problem_id": problem_id,
                            "sample_idx": sample_index,
                            "K_observed": k_observed,
                            "target_raw": target_raw,
                            "target_norm": target_norm,
                            "pred_raw": pred_raw,
                            "pred_norm": pred_norm,
                            "is_correct": int(is_correct),
                            "parse_success": int(parse_success),
                            "boxed_answer_found": int(boxed_found),
                            "answer_source": answer_source,
                            "source_id": provenance.get("source_id", provenance.get("source", "")),
                            "adapter_name": provenance.get("adapter_name", provenance.get("adapter", "")),
                            "policy": provenance.get("policy", ""),
                            "included_final_metric": int(included),
                            "raw_response": leaf.response,
                            "think_text": think_text,
                            "text_for_metrics": metric_text,
                            "length_chars": len(metric_text),
                            "length_tokens_approx": len(tokens),
                        }
                    )
                    group_key = (method, benchmark, k_observed)
                    parsing_groups[group_key]["responses"] += 1
                    parsing_groups[group_key]["included"] += int(included)
                    parsing_groups[group_key]["parsed"] += int(parse_success and included)
                    parsing_groups[group_key]["boxed"] += int(boxed_found and included)
                    if not parse_success:
                        parse_events.append(
                            {
                                "type": "answer_parse_failure",
                                "method": method,
                                "benchmark": benchmark,
                                "file_path": relative_path,
                                "problem_id": problem_id,
                                "sample_idx": sample_index,
                                "response_tail": leaf.response[-500:],
                            }
                        )
                    if not included:
                        parse_events.append(
                            {
                                "type": "non_main_sample_excluded",
                                "method": method,
                                "benchmark": benchmark,
                                "file_path": relative_path,
                                "problem_id": problem_id,
                                "sample_idx": sample_index,
                                "provenance": provenance,
                            }
                        )

                main_samples = [sample for sample in parsed_responses if sample.included]
                correct_samples = [sample for sample in main_samples if sample.is_correct]
                n_main = len(main_samples)
                correct_count = len(correct_samples)
                all_self_bleu, all_distinct = text_group_metrics(
                    [sample.tokens for sample in main_samples], args.distinct_ns
                )
                if correct_count >= 2:
                    correct_self_bleu, correct_distinct = text_group_metrics(
                        [sample.tokens for sample in correct_samples], args.distinct_ns
                    )
                else:
                    correct_self_bleu = math.nan
                    correct_distinct = {order: math.nan for order in args.distinct_ns}

                path_metrics, cluster_exports = analyze_correct_paths(
                    correct_samples=correct_samples,
                    thresholds=args.path_thresholds,
                    method=method,
                    benchmark=benchmark,
                    file_path=relative_path,
                    problem_id=problem_id,
                )
                path_cluster_exports.extend(cluster_exports)
                problem_metric: dict[str, Any] = {
                    "method": method,
                    "benchmark": benchmark,
                    "file_path": relative_path,
                    "problem_id": problem_id,
                    "K_observed": k_observed,
                    "K_main": n_main,
                    "correct_count": correct_count,
                    "correct_rate": correct_count / n_main if n_main else math.nan,
                    "parse_success_rate": mean_finite(
                        int(sample.parse_success) for sample in main_samples
                    ),
                    "boxed_answer_rate": mean_finite(
                        int(sample.boxed_answer_found) for sample in main_samples
                    ),
                    "self_bleu_all": all_self_bleu,
                    "self_bleu_correct": correct_self_bleu,
                    "diversity_bleu_all": 1 - all_self_bleu if finite(all_self_bleu) else math.nan,
                    "diversity_bleu_correct": 1 - correct_self_bleu if finite(correct_self_bleu) else math.nan,
                    "avg_length_tokens": mean_finite(len(sample.tokens) for sample in main_samples),
                }
                for k in args.ks:
                    problem_metric[f"pass@{k}"] = pass_at_k(n_main, correct_count, k)
                for order in args.distinct_ns:
                    problem_metric[f"distinct_{order}"] = all_distinct.get(order, math.nan)
                    problem_metric[f"distinct_{order}_correct"] = correct_distinct.get(order, math.nan)
                problem_metric.update(path_metrics)
                problem_rows.append(problem_metric)
                k_distributions[(method, benchmark, expected_k)][k_observed] += 1

                # Release very large response/token objects before reading the next problem.
                del parsed_responses, main_samples, correct_samples

            unique_observed = sorted(set(observed_values))
            file_mappings.append(
                {
                    "method": method,
                    "benchmark": benchmark,
                    "file_path": relative_path,
                    "observed_K": ",".join(map(str, unique_observed)),
                    "num_problems": len(rows),
                    "expected_K": expected_k,
                }
            )

    problem_fields = [
        "method",
        "benchmark",
        "file_path",
        "problem_id",
        "K_observed",
        "K_main",
        *[f"pass@{k}" for k in args.ks],
        "correct_count",
        "correct_rate",
        "parse_success_rate",
        "boxed_answer_rate",
        "self_bleu_all",
        "self_bleu_correct",
        "diversity_bleu_all",
        "diversity_bleu_correct",
        *[f"distinct_{order}" for order in args.distinct_ns],
        *[f"distinct_{order}_correct" for order in args.distinct_ns],
        *[f"unique_correct_path_{threshold_suffix(value)}" for value in args.path_thresholds],
        *[f"unique_correct_path_norm_{threshold_suffix(value)}" for value in args.path_thresholds],
        "avg_length_tokens",
    ]
    write_csv(out_dir / "problem_metrics.csv", problem_rows, problem_fields)

    delta_metrics = [
        *[f"pass@{k}" for k in args.ks],
        "correct_count",
        "correct_rate",
        "parse_success_rate",
        "boxed_answer_rate",
        "self_bleu_all",
        "self_bleu_correct",
        "diversity_bleu_all",
        "diversity_bleu_correct",
        *[f"distinct_{order}" for order in args.distinct_ns],
        *[f"distinct_{order}_correct" for order in args.distinct_ns],
        *[f"unique_correct_path_{threshold_suffix(value)}" for value in args.path_thresholds],
        *[f"unique_correct_path_norm_{threshold_suffix(value)}" for value in args.path_thresholds],
        "avg_length_tokens",
    ]
    delta_rows = build_problem_delta_rows(problem_rows, delta_metrics)
    delta_fields = [
        "comparison",
        "coex_method",
        "baseline_method",
        "benchmark",
        "K_observed",
        "problem_id",
        "delta_definition",
        "coex_file_path",
        "baseline_file_path",
    ]
    for metric in delta_metrics:
        delta_fields.extend(
            [f"coex_{metric}", f"baseline_{metric}", f"delta_{metric}"]
        )
    write_csv(out_dir / "problem_level_deltas.csv", delta_rows, delta_fields)

    summary_metric_map: list[tuple[str, str]] = [
        *[(f"pass@{k}", f"pass@{k}") for k in args.ks],
        ("correct_rate", "correct_rate"),
        ("parse_success_rate", "parse_success_rate"),
        ("self_bleu_all", "self_bleu_all"),
        ("self_bleu_correct", "self_bleu_correct"),
        ("diversity_bleu_all", "diversity_bleu_all"),
        ("diversity_bleu_correct", "diversity_bleu_correct"),
        *[(f"distinct_{order}", f"distinct_{order}") for order in args.distinct_ns],
        (
            f"unique_correct_path_{threshold_suffix(PRIMARY_PATH_THRESHOLD)}",
            f"unique_correct_path_{threshold_suffix(PRIMARY_PATH_THRESHOLD)}",
        ),
        (
            f"unique_correct_path_norm_{threshold_suffix(PRIMARY_PATH_THRESHOLD)}",
            f"unique_correct_path_norm_{threshold_suffix(PRIMARY_PATH_THRESHOLD)}",
        ),
        ("avg_length_tokens", "avg_length_tokens"),
    ]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in problem_rows:
        grouped[(row["method"], row["benchmark"], int(row["K_observed"]))].append(row)

    rng = random.Random(args.bootstrap_seed)
    summary_rows: list[dict[str, Any]] = []
    for (method, benchmark, observed_k), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "method": method,
            "benchmark": benchmark,
            "K_observed": observed_k,
            "num_problems": len(rows),
            "file_paths": " | ".join(sorted({str(row["file_path"]) for row in rows})),
        }
        for problem_key, output_key in summary_metric_map:
            values = [row.get(problem_key, math.nan) for row in rows]
            summary[f"mean_{output_key}"] = mean_finite(values)
            low, high = bootstrap_ci(values, args.bootstrap_iters, rng)
            summary[f"ci_low_{output_key}"] = low
            summary[f"ci_high_{output_key}"] = high
        summary_rows.append(summary)

    summary_fields = ["method", "benchmark", "K_observed", "num_problems", "file_paths"]
    for _, output_key in summary_metric_map:
        summary_fields.extend(
            [f"mean_{output_key}", f"ci_low_{output_key}", f"ci_high_{output_key}"]
        )
    write_csv(out_dir / "summary_metrics.csv", summary_rows, summary_fields)

    with (out_dir / "path_clusters.json").open("w", encoding="utf-8") as handle:
        json.dump(path_cluster_exports, handle, ensure_ascii=False, indent=2)
    with (out_dir / "parse_errors.json").open("w", encoding="utf-8") as handle:
        json.dump(parse_events, handle, ensure_ascii=False, indent=2)

    # Sanity warnings based on the completed analysis.
    for key, counts in parsing_groups.items():
        method, benchmark, observed_k = key
        included = counts["included"]
        parse_rate = counts["parsed"] / included if included else 0.0
        if parse_rate < 0.9:
            warn(
                f"parse_success_rate={parse_rate:.3f} is below 0.9",
                parse_events,
                method=method,
                benchmark=benchmark,
                K_observed=observed_k,
            )
    for summary in summary_rows:
        value = summary.get("mean_self_bleu_correct", math.nan)
        matching_rows = grouped[(summary["method"], summary["benchmark"], summary["K_observed"])]
        finite_fraction = sum(finite(row.get("self_bleu_correct")) for row in matching_rows) / len(matching_rows)
        if finite_fraction < 0.5:
            warn(
                f"self-BLEU-correct is NaN for {1-finite_fraction:.1%} of problems",
                parse_events,
                method=summary["method"],
                benchmark=summary["benchmark"],
                K_observed=summary["K_observed"],
                mean_self_bleu_correct=value,
            )
    # Re-write after adding aggregate sanity warnings.
    with (out_dir / "parse_errors.json").open("w", encoding="utf-8") as handle:
        json.dump(parse_events, handle, ensure_ascii=False, indent=2)

    markdown_path = out_dir / "summary_metrics.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Sampling Metric Summary\n\n")
        handle.write("Final metrics include main-policy samples only. Missing provenance is assumed main and warned.\n\n")
        handle.write("## File mapping\n\n")
        handle.write("| Method | Benchmark | K | Problems | File |\n|---|---:|---:|---:|---|\n")
        for mapping in file_mappings:
            handle.write(
                f"| {mapping['method']} | {mapping['benchmark']} | {mapping['observed_K']} | "
                f"{mapping['num_problems']} | `{mapping['file_path']}` |\n"
            )
        handle.write("\n## Metrics\n\n")
        handle.write(
            "| Method | Benchmark | K | pass@1 | pass@16 | pass@32 | correct rate | "
            "self-BLEU | self-BLEU-correct | distinct-2 | UniqueCorrectPath@0.80 | avg tokens |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} | {row['benchmark']} | {row['K_observed']} | "
                f"{format_metric(row.get('mean_pass@1'))} | {format_metric(row.get('mean_pass@16'))} | "
                f"{format_metric(row.get('mean_pass@32'))} | {format_metric(row.get('mean_correct_rate'))} | "
                f"{format_metric(row.get('mean_self_bleu_all'))} | "
                f"{format_metric(row.get('mean_self_bleu_correct'))} | "
                f"{format_metric(row.get('mean_distinct_2'))} | "
                f"{format_metric(row.get('mean_unique_correct_path_t080'))} | "
                f"{format_metric(row.get('mean_avg_length_tokens'), 1)} |\n"
            )
        handle.write("\n## Metric definitions\n\n")
        handle.write("- **pass@k:** HumanEval unbiased estimator, computed from main-policy samples per problem.\n")
        handle.write("- **self-BLEU:** leave-one-out BLEU-4 over all reasoning texts; lower means more diverse.\n")
        handle.write("- **self-BLEU-correct:** the same metric restricted to correct samples; NaN with fewer than two.\n")
        handle.write("- **distinct-n:** unique n-grams divided by all n-grams, with all-sample values primary.\n")
        handle.write("- **UniqueCorrectPath:** connected components among correct traces using 0.6 TF-IDF char similarity + 0.4 equation Jaccard.\n")
        handle.write("\n## Interpretation notes\n\n")
        handle.write("- Lower self-BLEU means greater diversity, but should be read with correct rate.\n")
        handle.write("- self-BLEU-correct and UniqueCorrectPath use correct samples only.\n")
        handle.write("- UniqueCorrectPath is an automatic proxy; inspect all threshold columns and path_clusters.json.\n")
        handle.write("- Confidence intervals are 95% problem-level bootstrap intervals.\n")

    print("\n[FILES FOUND]")
    for mapping in file_mappings:
        print(
            f"{mapping['method']} / {mapping['benchmark']} / K={mapping['observed_K']} / "
            f"problems={mapping['num_problems']} / {mapping['file_path']}"
        )
    print("\n[PARSING]")
    for key in sorted(parsing_groups):
        method, benchmark, observed_k = key
        counts = parsing_groups[key]
        included = counts["included"]
        print(
            f"{method:10s} {benchmark:7s} K={observed_k:2d} "
            f"parse={counts['parsed']/included if included else math.nan:.4f} "
            f"boxed={counts['boxed']/included if included else math.nan:.4f} "
            f"responses={counts['responses']} main={included}"
        )
    print("num responses per problem distribution:")
    for key in sorted(k_distributions):
        print(f"  {key}: {dict(sorted(k_distributions[key].items()))}")

    print("\n[METRICS]")
    header = (
        f"{'method':10s} {'bench':7s} {'K':>3s} {'pass@1':>8s} {'pass@16':>8s} "
        f"{'pass@32':>8s} {'correct':>8s} {'sBLEU':>8s} {'sBLEU-c':>8s} "
        f"{'dist-2':>8s} {'UCP@80':>8s} {'tokens':>9s}"
    )
    print(header)
    for row in summary_rows:
        print(
            f"{row['method']:10s} {row['benchmark']:7s} {row['K_observed']:3d} "
            f"{format_metric(row.get('mean_pass@1')):>8s} "
            f"{format_metric(row.get('mean_pass@16')):>8s} "
            f"{format_metric(row.get('mean_pass@32')):>8s} "
            f"{format_metric(row.get('mean_correct_rate')):>8s} "
            f"{format_metric(row.get('mean_self_bleu_all')):>8s} "
            f"{format_metric(row.get('mean_self_bleu_correct')):>8s} "
            f"{format_metric(row.get('mean_distinct_2')):>8s} "
            f"{format_metric(row.get('mean_unique_correct_path_t080')):>8s} "
            f"{format_metric(row.get('mean_avg_length_tokens'), 1):>9s}"
        )

    delta_coverage = Counter(
        (row["comparison"], row["benchmark"], row["K_observed"])
        for row in delta_rows
    )
    print("\n[PROBLEM-LEVEL DELTAS] (CoEx - baseline)")
    for key, count in sorted(delta_coverage.items()):
        print(f"{key[0]} / {key[1]} / K={key[2]}: {count} matched problems")
    for baseline_method in ("GRPO", "DMPO"):
        for benchmark in ("aime24", "aime25"):
            for observed_k in sorted({int(row['K_observed']) for row in problem_rows}):
                if not any(
                    row["comparison"] == f"CoEx_vs_{baseline_method}"
                    and row["benchmark"] == benchmark
                    and int(row["K_observed"]) == observed_k
                    for row in delta_rows
                ):
                    print(
                        f"[DELTA WARNING] no matched CoEx_vs_{baseline_method} rows for "
                        f"{benchmark}, K={observed_k}"
                    )

    print(f"\nWrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
