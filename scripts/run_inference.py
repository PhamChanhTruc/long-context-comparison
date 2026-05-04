import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import random
import time

import pandas as pd
import yaml

from src.inference import run_one_sample
from src.loaders import load_jsonl
from src.metrics_qa import compute_qa_metrics
from src.metrics_sum import compute_summarization_metrics
from src.model_utils import load_model_and_tokenizer, resolve_device
from src.prompts import build_prompt


def sample_diverse_qa(data, max_samples, seed=42):
    """
    Ưu tiên lấy tối đa 1 câu hỏi cho mỗi paper trước,
    sau đó mới lấy thêm từ phần còn lại nếu chưa đủ max_samples.
    Điều này giúp giảm hiện tượng nhiều mẫu liên tiếp đến từ cùng một paper.
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
    """
    Chọn mẫu theo cách ổn định, có seed cố định.
    - QA: lấy đa dạng theo paper
    - Summarization: shuffle rồi cắt
    """
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
        metrics = compute_qa_metrics(pred_df)
    elif task_name == "summarization":
        metrics = compute_summarization_metrics(pred_df)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    metrics_df = pd.DataFrame([metrics])
    metrics_path = metrics_dir / f"{task_name}_{model_tag}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print("Saved metrics to:", metrics_path)
    print(metrics_df)


if __name__ == "__main__":
    main()