from __future__ import annotations

import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def load_jsonl(path: str | Path) -> list[JsonDict]:
    """Load a JSONL file and report line-level parse errors clearly."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    items: list[JsonDict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected a JSON object in {jsonl_path} at line {line_number}, "
                    f"got {type(item).__name__}."
                )
            items.append(item)
    return items


def write_jsonl(items: list[JsonDict], path: str | Path) -> None:
    """Write one JSON object per line with stable UTF-8 encoding."""
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_sample(sample: JsonDict, task_name: str, index: int | None = None) -> None:
    """Validate the required fields for one QA or summarization sample."""
    prefix = f"sample {index}" if index is not None else "sample"

    if task_name == "qa":
        required = ("id", "task", "context", "question", "answers")
        missing = [field for field in required if field not in sample]
        if missing:
            raise ValueError(f"QA {prefix} is missing required fields: {missing}")
        if sample["task"] != "qa":
            raise ValueError(f"QA {prefix} has task={sample['task']!r}, expected 'qa'.")
        if not isinstance(sample["answers"], list) or not sample["answers"]:
            raise ValueError(f"QA {prefix} must contain a non-empty answers list.")
        if not all(str(answer).strip() for answer in sample["answers"]):
            raise ValueError(f"QA {prefix} contains an empty answer string.")

    elif task_name == "summarization":
        required = ("id", "task", "context", "reference")
        missing = [field for field in required if field not in sample]
        if missing:
            raise ValueError(
                f"Summarization {prefix} is missing required fields: {missing}"
            )
        if sample["task"] != "summarization":
            raise ValueError(
                f"Summarization {prefix} has task={sample['task']!r}, "
                "expected 'summarization'."
            )
        if not str(sample["reference"]).strip():
            raise ValueError(f"Summarization {prefix} has an empty reference.")
    else:
        raise ValueError(f"Unsupported task_name: {task_name}")

    if not str(sample["id"]).strip():
        raise ValueError(f"{prefix} has an empty id.")
    if not str(sample["context"]).strip():
        raise ValueError(f"{prefix} has an empty context.")


def validate_dataset(items: list[JsonDict], task_name: str) -> None:
    if not items:
        raise ValueError(f"No samples found for task {task_name!r}.")
    for index, sample in enumerate(items, start=1):
        validate_sample(sample, task_name=task_name, index=index)
