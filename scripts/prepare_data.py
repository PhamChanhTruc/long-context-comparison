from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.loaders import JsonDict, write_jsonl


QASPER_DATASET = "allenai/qasper"
GOVREPORT_DATASET = "ccdv/govreport-summarization"


def require_datasets():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The datasets package is required for data preparation. Install it with "
            "`pip install datasets`."
        ) from exc
    return load_dataset


def clean_text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def flatten_qasper_full_text(full_text: Any) -> str:
    """Build Qasper context from section names and section paragraphs."""
    parts: list[str] = []

    if isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, dict):
                section_name = clean_text(section.get("section_name"))
                if section_name:
                    parts.append(section_name)
                paragraphs = section.get("paragraphs") or []
                if isinstance(paragraphs, list):
                    parts.extend(clean_text(paragraph) for paragraph in paragraphs)
                else:
                    paragraph_text = clean_text(paragraphs)
                    if paragraph_text:
                        parts.append(paragraph_text)
            else:
                section_text = clean_text(section)
                if section_text:
                    parts.append(section_text)
    elif isinstance(full_text, dict):
        for section_name, paragraphs in full_text.items():
            section_name_text = clean_text(section_name)
            if section_name_text:
                parts.append(section_name_text)
            if isinstance(paragraphs, list):
                parts.extend(clean_text(paragraph) for paragraph in paragraphs)
            else:
                paragraph_text = clean_text(paragraphs)
                if paragraph_text:
                    parts.append(paragraph_text)
    else:
        text = clean_text(full_text)
        if text:
            parts.append(text)

    return "\n".join(part for part in parts if part)


def _answer_payload(answer_record: Any) -> dict[str, Any] | None:
    if isinstance(answer_record, dict) and isinstance(answer_record.get("answer"), dict):
        return answer_record["answer"]
    return answer_record if isinstance(answer_record, dict) else None


def _yes_no_answer(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    text = clean_text(value).lower()
    if text in {"yes", "true"}:
        return "yes"
    if text in {"no", "false"}:
        return "no"
    return ""


def qasper_answer_strings(answer_record: Any) -> list[str]:
    """Extract free-form, extractive-span, and yes/no answers from Qasper."""
    answer = _answer_payload(answer_record)
    if not answer or answer.get("unanswerable"):
        return []

    answers: list[str] = []

    free_form = clean_text(answer.get("free_form_answer"))
    if free_form:
        answers.append(free_form)

    extractive_spans = answer.get("extractive_spans") or []
    if isinstance(extractive_spans, str):
        extractive_spans = [extractive_spans]
    for span in extractive_spans:
        span_text = clean_text(span)
        if span_text:
            answers.append(span_text)

    yes_no = _yes_no_answer(answer.get("yes_no"))
    if yes_no:
        answers.append(yes_no)

    return list(dict.fromkeys(answers))


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def _qasper_context(record: dict[str, Any]) -> str:
    return flatten_qasper_full_text(record.get("full_text")) or clean_text(
        record.get("abstract")
    )


def _qasper_question_rows(record: dict[str, Any], context: str) -> Iterable[JsonDict]:
    qas = record.get("qas")
    if isinstance(qas, list):
        for index, question_record in enumerate(qas):
            if not isinstance(question_record, dict):
                continue
            question = clean_text(question_record.get("question"))
            answers: list[str] = []
            for answer_record in question_record.get("answers") or []:
                answers.extend(qasper_answer_strings(answer_record))
            answers = list(dict.fromkeys(answer for answer in answers if answer))
            if not question or not answers:
                continue

            question_id = clean_text(question_record.get("question_id"))
            paper_id = clean_text(record.get("id"))
            row_id = question_id or (
                f"{paper_id}_q{index:04d}" if paper_id else f"qasper_{index:04d}"
            )
            yield {
                "id": row_id,
                "task": "qa",
                "context": context,
                "question": question,
                "answers": answers,
            }
        return

    question = clean_text(record.get("question"))
    answers: list[str] = []
    raw_answers = record.get("answers") or record.get("answer") or []
    if isinstance(raw_answers, list):
        for answer_record in raw_answers:
            answers.extend(qasper_answer_strings(answer_record))
    else:
        answers.extend(qasper_answer_strings(raw_answers))
    answers = list(dict.fromkeys(answer for answer in answers if answer))
    if question and answers:
        yield {
            "id": clean_text(record.get("question_id"))
            or clean_text(record.get("id"))
            or "qasper_0000",
            "task": "qa",
            "context": context,
            "question": question,
            "answers": answers,
        }


def collect_qasper_rows(dataset: Iterable[dict[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for record in tqdm(dataset, desc="Qasper"):
        context = _qasper_context(record)
        if not context:
            continue
        rows.extend(_qasper_question_rows(record, context))
    return rows


def collect_govreport_rows(dataset: Iterable[dict[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, record in enumerate(tqdm(dataset, desc="GovReport")):
        context = first_present(record, ("report", "document", "article", "text"))
        reference = first_present(record, ("summary", "highlights", "abstract"))
        if not context or not reference:
            continue
        rows.append(
            {
                "id": clean_text(record.get("id")) or f"govreport_{index:05d}",
                "task": "summarization",
                "context": context,
                "reference": reference,
            }
        )
    return rows


def select_deterministic_rows(
    rows: list[JsonDict],
    max_samples: int,
    seed: int,
) -> list[JsonDict]:
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if max_samples <= 0:
        return ordered
    return ordered[:max_samples]


def prepare_qasper(output_path: Path, max_samples: int, seed: int) -> int:
    load_dataset = require_datasets()
    try:
        dataset = load_dataset(QASPER_DATASET, split="validation")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {QASPER_DATASET} validation data. Check internet "
            "access, Hugging Face availability, authentication, or cached datasets."
        ) from exc

    rows = select_deterministic_rows(
        collect_qasper_rows(dataset),
        max_samples=max_samples,
        seed=seed,
    )
    write_jsonl(rows, output_path)
    return len(rows)


def prepare_govreport(output_path: Path, max_samples: int, seed: int) -> int:
    load_dataset = require_datasets()
    try:
        dataset = load_dataset(GOVREPORT_DATASET, split="validation")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {GOVREPORT_DATASET} validation data. Check internet "
            "access, Hugging Face availability, authentication, or cached datasets."
        ) from exc

    rows = select_deterministic_rows(
        collect_govreport_rows(dataset),
        max_samples=max_samples,
        seed=seed,
    )
    write_jsonl(rows, output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qasper QA and GovReport data.")
    parser.add_argument(
        "--qa_samples",
        type=int,
        default=50,
        help="Number of Qasper QA samples to write. Use <=0 for all valid samples.",
    )
    parser.add_argument(
        "--sum_samples",
        type=int,
        default=50,
        help="Number of GovReport summarization samples to write. Use <=0 for all valid samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = args.output_dir / "main_qa.jsonl"
    sum_path = args.output_dir / "main_sum.jsonl"

    qa_count = prepare_qasper(qa_path, args.qa_samples, args.seed)
    sum_count = prepare_govreport(sum_path, args.sum_samples, args.seed)

    print(f"Wrote {qa_count} QA samples to {qa_path}")
    print(f"Wrote {sum_count} summarization samples to {sum_path}")


if __name__ == "__main__":
    main()
