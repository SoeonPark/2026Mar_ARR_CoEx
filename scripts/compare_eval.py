#!/usr/bin/env python3
"""Scan eval_results/ and print a performance comparison table in the terminal.

Usage:
  python scripts/compare_eval.py [PATTERN ...]                 # normal accuracy metrics
  python scripts/compare_eval.py [PATTERN ...] --passk          # pass@k breakdown
  python scripts/compare_eval.py --list                         # list all discovered runs

PATTERN is matched (case-insensitive substring, or glob if it contains * or ?)
against the run label, which is the results_*.json's directory path relative
to eval_results/ (minus the auto-generated lm-eval sanitized leaf folder).
This is exactly the RUN_NAME / model_name string you wrote in eval.sh /
eval_sampling.sh, so you can copy-paste it straight from there.

No PATTERN given -> every run found is included.
Multiple PATTERNs -> OR'd together (any match keeps the run), so you can
compare several runs side by side in one table.

--runs-file FILE reads more patterns from a file (one per line, blank lines
and '#' comments ignored) -- handy since wandb tracks runs by the base name
without the step/pass@k suffix, e.g. a file containing:

  CoEx_source_balanced_lt-grpo_dt-main_weak_correctness_bonus_G10-m4-d2x3_lr1e-6_beta0.04_scope-all_cw0.7-dw0.3_gate-False_ndiv-False_ncorr-False_iw-False-0707_121837
  coex_lr_1e-6-base_10-0_div_0_lT_grpo-ndiv_False-ncorr_False-divType_trace_jaccard-correctnessGated_False-useIW_False-repulsionTarget_all_other-aggregation_max-0617_202854

pulls in every step/checkpoint/pass@k variant found for each of those runs,
grouped together in the output (a "group" column shows which listed name
matched, and rows are sorted/clustered by that instead of by raw path).
"""
import argparse
import fnmatch
import json
import re
import signal
import sys
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_ROOT = SCRIPT_DIR.parent / "eval_results"

PASSK_RE = re.compile(r"^pass@(\d+),pass(\d+)$")
STDERR_RE = re.compile(r"_stderr(,|$)")
NON_METRIC_KEYS = {"name", "alias", "sample_len"}

# Preferred metric/filter when a task reports more than one (e.g. gsm8k has
# both flexible-extract and strict-match; math tasks have both exact_match
# and the more lenient math_verify checker).
METRIC_PRIORITY = ["math_verify", "exact_match", "acc_norm", "acc", "pass@1"]
FILTER_PRIORITY = ["none", "custom-extract", "flexible-extract", "strict-match"]


def find_results(eval_root):
    """Yield (label, json_path, data) for every results_*.json under eval_root."""
    for path in sorted(eval_root.rglob("results*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip unreadable {path}: {e}", file=sys.stderr)
            continue
        # path = .../<label>/<lm-eval-sanitized-model-dir>/results_*.json
        label_dir = path.parent.parent
        try:
            label = str(label_dir.relative_to(eval_root))
        except ValueError:
            label = str(label_dir)
        yield label, path, data


def load_patterns_file(path):
    patterns = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def match_pattern(label, model_name, patterns):
    """Return the first pattern (preserving list order) that matches, or None
    if patterns is non-empty and none matched. Returns '' if patterns is empty
    (matches everything, ungrouped)."""
    if not patterns:
        return ""
    hay = [label.lower(), (model_name or "").lower()]
    for pat in patterns:
        pat_l = pat.lower()
        if any(ch in pat for ch in "*?"):
            if any(fnmatch.fnmatch(h, pat_l) for h in hay):
                return pat
        else:
            if any(pat_l in h for h in hay):
                return pat
    return None


def short_label(label, group):
    """Strip the matched group substring out of label, leaving just the
    distinguishing bits (date/category/step/pass@k folder)."""
    if not group:
        return label
    idx = label.lower().find(group.lower())
    if idx == -1:
        return label
    before, after = label[:idx], label[idx + len(group):]
    combined = (before.rstrip("/") + "/" + after.lstrip("/")).strip("/")
    return combined or label


def collect_latest(entries, keep_all=False):
    """entries: list of (label, path, data, group). Collapse to newest per label unless keep_all."""
    if keep_all:
        return entries
    best = {}
    for label, path, data, group in entries:
        ts = data.get("date", path.stat().st_mtime)
        if label not in best or ts > best[label][0]:
            best[label] = (ts, path, data, group)
    return [(label, v[1], v[2], v[3]) for label, v in sorted(best.items())]


def pick_primary_metric(metric_dict):
    """metric_dict: {'exact_match,none': 0.5, 'exact_match_stderr,none': 0.1, ...}
    Returns (key, value) for the best metric to display, or (None, None)."""
    candidates = []
    for key, val in metric_dict.items():
        if key in NON_METRIC_KEYS or STDERR_RE.search(key):
            continue
        if "," not in key:
            continue
        metric, filt = key.split(",", 1)
        if PASSK_RE.match(key):
            metric = "pass@1" if metric == "pass@1" else metric
        m_rank = METRIC_PRIORITY.index(metric) if metric in METRIC_PRIORITY else 99
        f_rank = FILTER_PRIORITY.index(filt) if filt in FILTER_PRIORITY else 99
        candidates.append(((m_rank, f_rank), key, val))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], candidates[0][2]


def fmt_pct(val):
    if val is None:
        return "-"
    return f"{val * 100:.2f}"


def sort_rows(rows, patterns):
    """rows: list of (label, path, data, group). Cluster by requested pattern order,
    then by the run's distinguishing suffix within a group."""
    pattern_order = {p: i for i, p in enumerate(patterns)}
    return sorted(rows, key=lambda r: (pattern_order.get(r[3], len(patterns)), short_label(r[0], r[3])))


def build_normal_table(rows, task_filter, forced_metric, show_group):
    all_tasks = []
    for _, _, data, _g in rows:
        for t in data.get("results", {}):
            if t not in all_tasks:
                all_tasks.append(t)
    if task_filter:
        all_tasks = [t for t in all_tasks if any(f.lower() in t.lower() for f in task_filter)]

    table = []
    task_metric_used = {}
    for label, _path, data, group in rows:
        results = data.get("results", {})
        row = {"run": short_label(label, group)}
        if show_group:
            row["group"] = group
        vals = []
        for task in all_tasks:
            metrics = results.get(task)
            if not metrics:
                row[task] = "-"
                continue
            if forced_metric:
                key = next((k for k in metrics if k.split(",")[0] == forced_metric), None)
                val = metrics.get(key) if key else None
            else:
                key, val = pick_primary_metric(metrics)
            if key and task not in task_metric_used:
                task_metric_used[task] = key.split(",", 1)[1] if "," in key else key
            row[task] = fmt_pct(val)
            if val is not None:
                vals.append(val)
        row["avg"] = fmt_pct(sum(vals) / len(vals)) if vals else "-"
        table.append(row)

    headers = (["group"] if show_group else []) + ["run"] + all_tasks + ["avg"]
    return table, headers, task_metric_used


def build_passk_tables(rows, task_filter, show_group):
    """Returns {task: (table_rows, headers)}"""
    all_tasks = []
    for _, _, data, _g in rows:
        for t, metrics in data.get("results", {}).items():
            if any(PASSK_RE.match(k) for k in metrics) and t not in all_tasks:
                all_tasks.append(t)
    if task_filter:
        all_tasks = [t for t in all_tasks if any(f.lower() in t.lower() for f in task_filter)]

    tables = {}
    for task in all_tasks:
        k_cols = set()
        per_row = []
        for label, _path, data, group in rows:
            metrics = data.get("results", {}).get(task, {})
            passk_vals = {}
            cfg_k = None
            for key, val in metrics.items():
                m = PASSK_RE.match(key)
                if not m:
                    continue
                k, n = int(m.group(1)), int(m.group(2))
                passk_vals[k] = val
                cfg_k = max(cfg_k or 0, n)
            if not passk_vals:
                continue
            k_cols.update(passk_vals.keys())
            per_row.append((label, group, cfg_k, passk_vals))

        if not per_row:
            continue
        k_cols = sorted(k_cols)
        table = []
        for label, group, cfg_k, passk_vals in per_row:
            row = {"run": short_label(label, group), "cfg_K": cfg_k}
            if show_group:
                row["group"] = group
            for k in k_cols:
                row[f"pass@{k}"] = fmt_pct(passk_vals.get(k))
            table.append(row)
        headers = (["group"] if show_group else []) + ["run", "cfg_K"] + [f"pass@{k}" for k in k_cols]
        tables[task] = (table, headers)
    return tables


def print_table(table, headers):
    rows = [[r.get(h, "-") for h in headers] for r in table]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
                  for i, h in enumerate(headers)]
        print(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
        print("-+-".join("-" * w for w in widths))
        for r in rows:
            print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*", help="run-name substrings (or glob) to match, OR'd together")
    ap.add_argument("--runs-file", default="", help="file with one run-name pattern per line (# comments ok)")
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT), help="root of eval_results/ (default: %(default)s)")
    ap.add_argument("--passk", action="store_true", help="show pass@k breakdown tables instead of accuracy table")
    ap.add_argument("--tasks", default="", help="comma-separated substrings to filter tasks (e.g. aime24,gsm8k)")
    ap.add_argument("--metric", default="", help="force a specific metric name (e.g. math_verify, exact_match)")
    ap.add_argument("--all-timestamps", action="store_true", help="don't collapse reruns to the latest one")
    ap.add_argument("--list", action="store_true", help="just list discovered run labels and exit")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.is_dir():
        print(f"[ERROR] eval_root not found: {eval_root}", file=sys.stderr)
        sys.exit(1)

    task_filter = [t.strip() for t in args.tasks.split(",") if t.strip()]
    raw_entries = list(find_results(eval_root))

    if args.list:
        labels = sorted({label for label, _p, _d in raw_entries})
        for l in labels:
            print(l)
        print(f"\n{len(labels)} run(s) found under {eval_root}", file=sys.stderr)
        return

    patterns = list(dict.fromkeys(
        (load_patterns_file(args.runs_file) if args.runs_file else []) + list(args.patterns)
    ))

    entries = []
    for l, p, d in raw_entries:
        group = match_pattern(l, d.get("model_name"), patterns)
        if group is None:
            continue
        entries.append((l, p, d, group))
    if not entries:
        print("[INFO] no matching results found. Try --list to see available run labels.", file=sys.stderr)
        return

    rows = collect_latest(entries, keep_all=args.all_timestamps)
    rows = sort_rows(rows, patterns)
    show_group = bool(patterns) and len(patterns) > 1

    if args.passk:
        tables = build_passk_tables(rows, task_filter, show_group)
        if not tables:
            print("[INFO] no pass@k results found among the matched runs.", file=sys.stderr)
            return
        for task, (table, headers) in tables.items():
            print(f"\n=== {task} ===")
            print_table(table, headers)
    else:
        table, headers, task_metric_used = build_normal_table(rows, task_filter, args.metric or None, show_group)
        print_table(table, headers)
        if task_metric_used:
            print("\nmetric used per task:", file=sys.stderr)
            for t, m in task_metric_used.items():
                print(f"  {t}: {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
