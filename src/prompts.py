from __future__ import annotations

from typing import Any


def build_qa_prompt(context: str, question: str) -> str:
    return (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        'Return only the shortest answer span from the context. If the answer is not '
        'in the context, return "unknown".\n'
        "Answer:"
    )


def build_summarization_prompt(context: str) -> str:
    return (
        "Summarize the following document in 2-4 concise, factual sentences.\n\n"
        "Document:\n"
        f"{context}\n\n"
        "Summary:"
    )


def build_prompt(
    sample: dict[str, Any],
    task_name: str,
    model_name: str | None = None,
) -> str:
    del model_name
    if task_name == "qa":
        return build_qa_prompt(
            context=str(sample["context"]),
            question=str(sample["question"]),
        )
    if task_name == "summarization":
        return build_summarization_prompt(context=str(sample["context"]))
    raise ValueError(f"Unsupported task_name: {task_name}")
