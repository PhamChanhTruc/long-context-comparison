import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random
from ast import literal_eval

import pandas as pd
import yaml

from src.inference import run_one_sample
from src.loaders import load_jsonl
from src.metrics_qa import (
    clean_qa_prediction,
    extract_short_answer,
    exact_match,
    max_f1_score,
)
from src.metrics_sum import compute_rouge
from src.model_utils import load_model_and_tokenizer, resolve_device
from src.prompts import build_prompt


def parse_answers(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [text]
    return [str(value)]


def sample_diverse_qa(data, max_samples, seed=42):
    """
    Ưu tiên lấy tối đa 1 câu hỏi cho mỗi paper trước,
    sau đó mới lấy thêm từ phần còn lại nếu chưa đủ max_samples.
    """
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    selected = []
    leftovers = []
    seen_papers = set()

    for item in shuffled:
        sample_id = item.get("id", "")
        paper_id = sample_id.rsplit("_q", 1)[0] if "_q" in sample_id else sample_id

        if paper_id not in seen_papers:
            selected.append(item)
            seen_papers.add(paper_id)
        else:
            leftovers.append(item)

        if len(selected) >= max_samples:
            return selected[:max_samples]

    rng.shuffle(leftovers)
    selected.extend(leftovers)
    return selected[:max_samples]


def select_samples(data, task_name, max_samples, seed=42):
    if max_samples is None or max_samples >= len(data):
        max_samples = len(data)

    if task_name == "qa":
        return sample_diverse_qa(data, max_samples=max_samples, seed=seed)

    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    return shuffled[:max_samples]


def sanitize_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(".", "_")


def compute_qa_metrics_from_df(pred_df: pd.DataFrame):
    raw_em_scores = []
    raw_f1_scores = []
    post_em_scores = []
    post_f1_scores = []

    for _, row in pred_df.iterrows():
        answers = parse_answers(row.get("reference", []))

        raw_pred = str(row.get("raw_prediction", row.get("prediction", "")) or "")
        post_pred = str(row.get("prediction", "") or "")

        if not post_pred:
            post_pred = extract_short_answer(clean_qa_prediction(raw_pred))

        raw_em = max(exact_match(raw_pred, ans) for ans in answers) if answers else 0.0
        raw_f1 = max_f1_score(raw_pred, answers) if answers else 0.0
        post_em = max(exact_match(post_pred, ans) for ans in answers) if answers else 0.0
        post_f1 = max_f1_score(post_pred, answers) if answers else 0.0

        raw_em_scores.append(raw_em)
        raw_f1_scores.append(raw_f1)
        post_em_scores.append(post_em)
        post_f1_scores.append(post_f1)

    return {
        "raw_exact_match": sum(raw_em_scores) / len(raw_em_scores) if raw_em_scores else 0.0,
        "raw_f1": sum(raw_f1_scores) / len(raw_f1_scores) if raw_f1_scores else 0.0,
        "post_exact_match": sum(post_em_scores) / len(post_em_scores) if post_em_scores else 0.0,
        "post_f1": sum(post_f1_scores) / len(post_f1_scores) if post_f1_scores else 0.0,
    }


def compute_sum_metrics_from_df(pred_df: pd.DataFrame):
    predictions = pred_df["prediction"].fillna("").astype(str).tolist()
    references = pred_df["reference"].fillna("").astype(str).tolist()
    return compute_rouge(predictions, references)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flan_qa.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task_name = cfg["task"]["name"]
    data_path = cfg["task"]["data_path"]
    max_samples = cfg["task"].get("max_samples", None)

    model_name = cfg["model"]["name"]
    architecture = cfg["model"]["architecture"]
    use_4bit = cfg["model"].get("use_4bit", False)

    max_input_tokens = cfg["generation"]["max_input_tokens"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    do_sample = cfg["generation"]["do_sample"]
    temperature = cfg["generation"]["temperature"]

    runtime_device = cfg["runtime"].get("device", "auto")

    predictions_dir = Path(cfg["output"]["predictions_dir"])
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    logs_dir = Path(cfg["output"].get("logs_dir", "outputs/logs"))

    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(runtime_device)
    print("Device:", device)

    data = load_jsonl(data_path)
    data = select_samples(data, task_name=task_name, max_samples=max_samples, seed=42)
    print("Loaded samples:", len(data))

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        architecture=architecture,
        device=device,
        use_4bit=use_4bit,
    )

    rows = []
    for i, sample in enumerate(data, start=1):
        out = run_one_sample(
            sample=sample,
            task_name=task_name,
            model_name=model_name,
            architecture=architecture,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            build_prompt_fn=build_prompt,
        )
        rows.append(out)

        if task_name == "qa":
            raw_pred = out.get("raw_prediction", out.get("prediction", ""))
            post_pred = out.get("prediction", "")
            print(f"[{i}/{len(data)}] done | raw={raw_pred} | post={post_pred}")
        else:
            print(f"[{i}/{len(data)}] done | prediction={out.get('prediction', '')}")

    pred_df = pd.DataFrame(rows)

    model_tag = sanitize_model_name(model_name)
    pred_path = predictions_dir / f"{task_name}_{model_tag}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print("Saved predictions to:", pred_path)

    if task_name == "qa":
        metrics = compute_qa_metrics_from_df(pred_df)
    elif task_name == "summarization":
        metrics = compute_sum_metrics_from_df(pred_df)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    metrics.update({
        "task": task_name,
        "model": model_name,
        "architecture": architecture,
        "num_samples": len(pred_df),
        "avg_latency_sec": pred_df["latency_sec"].mean() if "latency_sec" in pred_df.columns else None,
        "avg_input_tokens": pred_df["input_tokens"].mean() if "input_tokens" in pred_df.columns else None,
        "avg_output_tokens": pred_df["output_tokens"].mean() if "output_tokens" in pred_df.columns else None,
    })

    metrics_df = pd.DataFrame([metrics])
    metrics_path = metrics_dir / f"{task_name}_{model_tag}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print("Saved metrics to:", metrics_path)
    print(metrics_df)


if __name__ == "__main__":
    main()