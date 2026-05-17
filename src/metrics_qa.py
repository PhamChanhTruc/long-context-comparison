from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


INSTRUCTION_ARTIFACTS = (
    "do not explain",
    "do not summarize the document",
    "do not repeat the question",
    "give a short answer only",
    "return only the shortest answer",
    "return only the answer span",
    "use only information from the context",
)


def normalize_text(text: str) -> str:
    """Normalize text for extractive QA EM/F1 comparisons."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _ensure_answer_list(gold_answers: str | Iterable[str] | None) -> list[str]:
    if gold_answers is None:
        return []
    if isinstance(gold_answers, str):
        return [gold_answers]
    return [str(answer) for answer in gold_answers if str(answer).strip()]


def _strip_instruction_artifacts(text: str) -> str:
    cleaned = text
    for artifact in INSTRUCTION_ARTIFACTS:
        cleaned = re.sub(
            rf"\b{re.escape(artifact)}\b\.?",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned


def clean_qa_prediction(prediction: str) -> str:
    """Remove prompt echoes and instruction artifacts from a QA generation."""
    text = str(prediction or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    text = _strip_instruction_artifacts(text)

    answer_matches = list(re.finditer(r"\banswer\s*[:\-]\s*", text, flags=re.IGNORECASE))
    if answer_matches:
        text = text[answer_matches[-1].end() :]

    lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(context|question|instructions?|document|summary)\s*:", line, re.I):
            continue
        lines.append(line)

    text = lines[0] if lines else text.strip()
    text = re.sub(
        r"^(final\s+answer|response|answer)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(the\s+answer\s+is|answer\s+is)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().strip("\"'` ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text.strip()


def extract_short_answer(prediction: str) -> str:
    """Heuristically turn a full-sentence QA output into a short answer."""
    text = clean_qa_prediction(prediction)
    if not text:
        return ""

    if normalize_text(text) in {"unknown", "unsupported", "not supported"}:
        return "unknown"

    lowered = text.lower()
    sentence_patterns = [
        " is used for ",
        " are used for ",
        " was used for ",
        " were used for ",
        " is located in ",
        " are located in ",
        " was located in ",
        " were located in ",
        " orbits ",
        " is ",
        " are ",
        " was ",
        " were ",
    ]

    if len(text.split()) <= 24:
        for pattern in sentence_patterns:
            if pattern in lowered:
                index = lowered.rfind(pattern)
                candidate = text[index + len(pattern) :].strip()
                candidate = re.sub(r"^(the answer is|answer is)\s+", "", candidate, flags=re.I)
                candidate = re.sub(r"[.,;:!?]+$", "", candidate).strip()
                if candidate:
                    return candidate

    return text


def exact_match(prediction: str, gold_answers: str | Iterable[str] | None) -> float:
    """Return 1.0 when the prediction exactly matches any gold answer."""
    answers = _ensure_answer_list(gold_answers)
    if not answers:
        return 0.0
    normalized_prediction = normalize_text(prediction)
    return max(float(normalized_prediction == normalize_text(answer)) for answer in answers)


def f1_score_single(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold_answer).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, gold_answers: str | Iterable[str] | None) -> float:
    """Compute token-level F1 against the best matching gold answer."""
    answers = _ensure_answer_list(gold_answers)
    if not answers:
        return 0.0
    return max(f1_score_single(prediction, answer) for answer in answers)


def max_f1_score(prediction: str, gold_answers: str | Iterable[str] | None) -> float:
    return f1_score(prediction, gold_answers)
