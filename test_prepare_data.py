from scripts.prepare_data import (
    collect_govreport_rows,
    collect_qasper_rows,
    flatten_qasper_full_text,
    qasper_answer_strings,
    select_deterministic_rows,
)
from src.loaders import validate_dataset


def test_qasper_context_flattening() -> None:
    full_text = [
        {
            "section_name": "Introduction",
            "paragraphs": [" First paragraph. ", "Second paragraph."],
        },
        {"section_name": "Methods", "paragraphs": ["Method details."]},
    ]

    context = flatten_qasper_full_text(full_text)

    assert context == "\n".join(
        [
            "Introduction",
            "First paragraph.",
            "Second paragraph.",
            "Methods",
            "Method details.",
        ]
    )


def test_qasper_answer_extraction() -> None:
    answer_record = {
        "answer": {
            "unanswerable": False,
            "free_form_answer": "A generated answer",
            "extractive_spans": ["span one", "span two"],
            "yes_no": True,
        }
    }

    assert qasper_answer_strings(answer_record) == [
        "A generated answer",
        "span one",
        "span two",
        "yes",
    ]
    assert qasper_answer_strings({"answer": {"unanswerable": True}}) == []


def test_prepare_data_jsonl_fields() -> None:
    qasper_rows = collect_qasper_rows(
        [
            {
                "id": "paper_a",
                "full_text": [
                    {"section_name": "Intro", "paragraphs": ["Qasper context."]}
                ],
                "qas": [
                    {
                        "question_id": "paper_a_q1",
                        "question": "What is tested?",
                        "answers": [
                            {
                                "answer": {
                                    "unanswerable": False,
                                    "free_form_answer": "",
                                    "extractive_spans": ["Qasper context"],
                                    "yes_no": None,
                                }
                            }
                        ],
                    }
                ],
            }
        ]
    )
    govreport_rows = collect_govreport_rows(
        [
            {
                "id": "report_a",
                "report": "A long government report.",
                "summary": "A short summary.",
            }
        ]
    )

    validate_dataset(qasper_rows, task_name="qa")
    validate_dataset(govreport_rows, task_name="summarization")
    assert set(qasper_rows[0]) == {"id", "task", "context", "question", "answers"}
    assert set(govreport_rows[0]) == {"id", "task", "context", "reference"}


def test_deterministic_selection_keeps_all_when_nonpositive() -> None:
    rows = [
        {"id": "b", "task": "qa", "context": "c", "question": "q", "answers": ["a"]},
        {"id": "a", "task": "qa", "context": "c", "question": "q", "answers": ["a"]},
    ]

    first = select_deterministic_rows(rows, max_samples=0, seed=42)
    second = select_deterministic_rows(rows, max_samples=0, seed=42)
    limited = select_deterministic_rows(rows, max_samples=1, seed=42)

    assert first == second
    assert len(first) == 2
    assert len(limited) == 1
