import re


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_qa_prediction(prediction: str) -> str:
    text = str(prediction).strip()

    # Giữ dòng đầu tiên nếu model sinh nhiều dòng
    text = text.split("\n")[0].strip()

    # Bỏ các tiền tố kiểu "Answer: ..."
    text = re.sub(r"^(answer|response)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    # Bỏ khoảng trắng và dấu câu cuối câu
    text = text.strip().strip("\"' ")
    text = re.sub(r"[ \t\r\n]+", " ", text)
    text = re.sub(r"[.,;:!?]+$", "", text)

    return text.strip()


def extract_short_answer(prediction: str) -> str:
    """
    Heuristic cho QA ngắn:
    - 'The capital of France is Paris.' -> 'Paris'
    - 'The moon orbits the Earth.' -> 'the Earth'
    - 'Python is used for web development...' -> 'web development...'
    """
    text = clean_qa_prediction(prediction)
    lowered = text.lower()

    patterns = [
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

    # Chỉ áp dụng heuristic cho câu ngắn-vừa
    if len(text.split()) <= 20:
        for pattern in patterns:
            if pattern in lowered:
                idx = lowered.rfind(pattern)
                candidate = text[idx + len(pattern):].strip()
                candidate = re.sub(r"[.,;:!?]+$", "", candidate).strip()
                if candidate:
                    return candidate

    return text


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    pred = normalize_text(prediction)
    return max(float(pred == normalize_text(g)) for g in gold_answers)


def f1_score_single(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold_answer).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = {}
    for token in pred_tokens:
        if token in gold_tokens:
            common[token] = min(pred_tokens.count(token), gold_tokens.count(token))

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1_score(prediction: str, gold_answers: list[str]) -> float:
    return max(f1_score_single(prediction, answer) for answer in gold_answers)