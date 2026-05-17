from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.loaders import validate_dataset, write_jsonl


DEFAULT_PATHS = (
    Path("data/processed/pilot_qa.jsonl"),
    Path("data/processed/pilot_sum.jsonl"),
)


def parse_json_stream(text: str, path: Path) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    items: list[dict[str, Any]] = []

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            item, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse {path} near character {index}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a JSON value that is not an object.")
        items.append(item)
    return items


def parse_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} line {line_number} is not exactly one valid JSON object: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            items.append(item)
    return items


def infer_task(path: Path, items: list[dict[str, Any]]) -> str:
    if items and items[0].get("task") in {"qa", "summarization"}:
        return str(items[0]["task"])
    if "qa" in path.name:
        return "qa"
    if "sum" in path.name:
        return "summarization"
    raise ValueError(f"Could not infer task for {path}; include a task field.")


def validate_one_file(path: Path, fix: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    needs_rewrite = False
    try:
        items = parse_strict_jsonl(path)
    except ValueError:
        items = parse_json_stream(path.read_text(encoding="utf-8"), path)
        needs_rewrite = True

    task_name = infer_task(path, items)
    validate_dataset(items, task_name=task_name)

    if needs_rewrite:
        if not fix:
            raise ValueError(f"{path} is parseable but not valid JSONL. Re-run without --no-fix.")
        write_jsonl(items, path)
        print(f"Rewrote {path} as valid JSONL with {len(items)} records.")
    else:
        print(f"OK: {path} ({len(items)} {task_name} records)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project JSONL data files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_PATHS),
        help="JSONL files to validate. Defaults to the pilot QA and summarization files.",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Validate only; do not rewrite parseable files that are not strict JSONL.",
    )
    args = parser.parse_args()

    for path in args.paths:
        validate_one_file(path, fix=not args.no_fix)


if __name__ == "__main__":
    main()
