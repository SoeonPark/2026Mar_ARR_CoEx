"""Export lm-eval's grouped responses as one JSON object per generation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TASK_FROM_FILENAME = re.compile(r"^samples_(.+?)_\d{4}-\d{2}-\d{2}T")


def _responses(sample: dict[str, Any]) -> list[str]:
    responses: list[str] = []
    for value in sample.get("resps", []):
        if isinstance(value, list):
            responses.extend(str(response) for response in value)
        else:
            responses.append(str(value))
    return responses


def _generation_args(sample: dict[str, Any]) -> dict[str, Any]:
    arguments = sample.get("arguments", {})
    if not isinstance(arguments, dict):
        return {}
    first_request = arguments.get("gen_args_0", {})
    if not isinstance(first_request, dict):
        return {}
    gen_kwargs = first_request.get("arg_1", {})
    return gen_kwargs if isinstance(gen_kwargs, dict) else {}


def _task_name(sample_path: Path) -> str:
    match = TASK_FROM_FILENAME.match(sample_path.name)
    return match.group(1) if match else sample_path.stem


def export_records(
    output_root: Path,
    source_id: str,
    adapter_name: str,
    seed: int,
    sampling_k: int,
) -> tuple[Path, int]:
    sample_paths = sorted(output_root.rglob("samples_*_*.jsonl"))
    sample_paths = [
        path
        for path in sample_paths
        if _task_name(path) in {"aime24_passk", "aime25_passk"}
    ]
    if not sample_paths:
        raise FileNotFoundError(f"No pass@k sample JSONL found under {output_root}")

    output_file = output_root / "sampling_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sample_id = 0
    with output_file.open("w", encoding="utf-8") as writer:
        for sample_path in sample_paths:
            task = _task_name(sample_path)
            with sample_path.open(encoding="utf-8") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    gen_kwargs = _generation_args(sample)
                    doc = sample.get("doc", {})
                    problem_id = None
                    if isinstance(doc, dict):
                        problem_id = doc.get("ID", doc.get("id"))

                    for response_id, response in enumerate(_responses(sample)):
                        record = {
                            "sample_id": sample_id,
                            "source_id": source_id,
                            "adapter_name": adapter_name,
                            "task": task,
                            "doc_id": sample.get("doc_id"),
                            "problem_id": problem_id,
                            "response_id": response_id,
                            "response": response,
                            "seed": seed,
                            "sampling_k": sampling_k,
                            "temperature": gen_kwargs.get("temperature"),
                            "top_p": gen_kwargs.get("top_p"),
                            "max_gen_toks": gen_kwargs.get("max_gen_toks"),
                        }
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sample_id += 1

    return output_file, sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-id", default="main")
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--sampling-k", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file, count = export_records(
        output_root=args.output_root,
        source_id=args.source_id,
        adapter_name=args.adapter_name,
        seed=args.seed,
        sampling_k=args.sampling_k,
    )
    print(f"Exported {count} individual sampling records to {output_file}")


if __name__ == "__main__":
    main()
