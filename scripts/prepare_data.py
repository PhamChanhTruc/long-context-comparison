from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.loaders import write_jsonl


def require_datasets():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The datasets package is required for data preparation. Install with "
            "`pip install datasets`."
        ) from exc
    return load_dataset


def flatten_qasper_full_text(full_text: Any) -> str:
    parts: list[str] = []
    if isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, dict):
                section_name = section.get("section_name")
                if section_name:
                    parts.append(str(section_name))
                paragraphs = section.get("paragraphs", [])
                if isinstance(paragraphs, list):
                    parts.extend(str(paragraph) for paragraph in paragraphs)
            else:
                parts.append(str(section))
    elif isinstance(full_text, dict):
        for value in full_text.values():
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            else:
                parts.append(str(value))
    elif full_text:
        parts.append(str(full_text))
    return "\n".join(part for part in parts if part.strip())


def qasper_answer_strings(answer_record: Any) -> list[str]:
    if isinstance(answer_record, dict) and "answer" in answer_record:
        answer_record = answer_record["answer"]
    if not isinstance(answer_record, dict):
        return [str(answer_record)] if str(answer_record).strip() else []
    if answer_record.get("unanswerable"):
        return []

    answers: list[str] = []
    for span in answer_record.get("extractive_spans") or []:
        if str(span).strip():
            answers.append(str(span))
    free_form = answer_record.get("free_form_answer")
    if free_form and str(free_form).strip():
        answers.append(str(free_form))
    yes_no = answer_record.get("yes_no")
    if isinstance(yes_no, bool):
        answers.append("yes" if yes_no else "no")
    return list(dict.fromkeys(answers))


def prepare_qasper(output_path: Path, max_samples: int) -> int:
    load_dataset = require_datasets()
    try:
        dataset = load_dataset("allenai/qasper", split="validation")
    except Exception as exc:
        raise RuntimeError(
            "Could not load allenai/qasper validation data. Check internet access, "
            "Hugging Face availability, or cached datasets."
        ) from exc

    rows: list[dict[str, Any]] = []
    for paper in tqdm(dataset, desc="Qasper"):
        context = flatten_qasper_full_text(paper.get("full_text") or paper.get("abstract"))
        if not context:
            continue

        qas = paper.get("qas")
        if isinstance(qas, list):
            for question_record in qas:
                question = str(question_record.get("question", "")).strip()
                answers: list[str] = []
                for answer_record in question_record.get("answers") or []:
                    answers.extend(qasper_answer_strings(answer_record))
                answers = list(dict.fromkeys(answer for answer in answers if answer.strip()))
                if not question or not answers:
                    continue
                rows.append(
                    {
                        "id": str(question_record.get("question_id") or f"qasper_{len(rows):05d}"),
                        "task": "qa",
                        "context": context,
                        "question": question,
                        "answers": answers,
                    }
                )
                if len(rows) >= max_samples:
                    write_jsonl(rows, output_path)
                    return len(rows)
        elif paper.get("question"):
            answers = qasper_answer_strings(paper.get("answers") or paper.get("answer"))
            if answers:
                rows.append(
                    {
                        "id": str(paper.get("question_id") or f"qasper_{len(rows):05d}"),
                        "task": "qa",
                        "context": context,
                        "question": str(paper["question"]),
                        "answers": answers,
                    }
                )
                if len(rows) >= max_samples:
                    write_jsonl(rows, output_path)
                    return len(rows)

    write_jsonl(rows, output_path)
    return len(rows)


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value and str(value).strip():
            return str(value)
    return ""


def prepare_govreport(output_path: Path, max_samples: int) -> int:
    load_dataset = require_datasets()
    try:
        dataset = load_dataset("ccdv/govreport-summarization", split="validation")
    except Exception as exc:
        raise RuntimeError(
            "Could not load ccdv/govreport-summarization validation data. Check "
            "internet access, Hugging Face availability, or cached datasets."
        ) from exc

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(dataset, desc="GovReport")):
        context = first_present(record, ("report", "document", "article", "text"))
        reference = first_present(record, ("summary", "highlights", "abstract"))
        if not context or not reference:
            continue
        rows.append(
            {
                "id": str(record.get("id") or f"govreport_{index:05d}"),
                "task": "summarization",
                "context": context,
                "reference": reference,
            }
        )
        if len(rows) >= max_samples:
            break

    write_jsonl(rows, output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qasper QA and GovReport data.")
    parser.add_argument("--qa_samples", type=int, default=50)
    parser.add_argument("--sum_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if args.qa_samples <= 0 or args.sum_samples <= 0:
        raise ValueError("--qa_samples and --sum_samples must be positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = args.output_dir / "main_qa.jsonl"
    sum_path = args.output_dir / "main_sum.jsonl"

    qa_count = prepare_qasper(qa_path, args.qa_samples)
    sum_count = prepare_govreport(sum_path, args.sum_samples)

    print(f"Wrote {qa_count} QA samples to {qa_path}")
    print(f"Wrote {sum_count} summarization samples to {sum_path}")


if __name__ == "__main__":
    main()
