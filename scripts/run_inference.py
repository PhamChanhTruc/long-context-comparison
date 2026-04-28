import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from src.loaders import load_jsonl
from src.prompts import build_prompt
from src.model_utils import load_model_and_tokenizer, resolve_device
from src.inference import run_one_sample
from src.metrics_qa import exact_match, max_f1_score, clean_qa_prediction, extract_short_answer
from src.metrics_sum import compute_rouge


def main():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task_name = cfg["task"]["name"]
    data_path = cfg["task"]["data_path"]
    max_samples = cfg["task"]["max_samples"]

    model_name = cfg["model"]["name"]
    architecture = cfg["model"]["architecture"]

    max_input_tokens = cfg["generation"]["max_input_tokens"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    do_sample = cfg["generation"]["do_sample"]
    temperature = cfg["generation"]["temperature"]

    device = resolve_device(cfg["runtime"]["device"])
    print("Device:", device)

    data = load_jsonl(data_path)[:max_samples]
    print("Loaded samples:", len(data))

    model, tokenizer = load_model_and_tokenizer(model_name, architecture, device)

    results = []
    for idx, sample in enumerate(data):
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

        if task_name == "qa":
            out["prediction_clean"] = clean_qa_prediction(out["prediction"])
            out["prediction_postprocessed"] = extract_short_answer(out["prediction"])
            print(
                f"[{idx + 1}/{len(data)}] done | "
                f"raw={out['prediction']} | "
                f"post={out['prediction_postprocessed']}"
            )
        else:
            out["prediction_clean"] = out["prediction"]
            out["prediction_postprocessed"] = out["prediction"]
            print(f"[{idx + 1}/{len(data)}] done | prediction={out['prediction']}")

        results.append(out)

    df = pd.DataFrame(results)

    out_dir = Path(cfg["output"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model_name.split("/")[-1].replace(".", "_")

    pred_path = out_dir / f"{task_name}_{model_slug}.csv"
    df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print("Saved predictions to:", pred_path)

    if task_name == "qa":
        raw_ems, raw_f1s = [], []
        post_ems, post_f1s = [], []

        for row, raw in zip(results, data):
            golds = raw["answers"]

            raw_ems.append(exact_match(row["prediction"], golds))
            raw_f1s.append(max_f1_score(row["prediction"], golds))

            post_ems.append(exact_match(row["prediction_postprocessed"], golds))
            post_f1s.append(max_f1_score(row["prediction_postprocessed"], golds))

        metrics = pd.DataFrame([{
            "task": task_name,
            "model": model_name,
            "architecture": architecture,
            "num_samples": len(results),
            "raw_exact_match": sum(raw_ems) / len(raw_ems),
            "raw_f1": sum(raw_f1s) / len(raw_f1s),
            "post_exact_match": sum(post_ems) / len(post_ems),
            "post_f1": sum(post_f1s) / len(post_f1s),
            "avg_latency_sec": float(df["latency_sec"].mean()),
            "avg_input_tokens": float(df["input_tokens"].mean()),
            "avg_output_tokens": float(df["output_tokens"].mean()),
        }])

    elif task_name == "summarization":
        preds = [r["prediction"] for r in results]
        refs = [r["reference"] for r in results]
        rouge_scores = compute_rouge(preds, refs)

        metrics = pd.DataFrame([{
            "task": task_name,
            "model": model_name,
            "architecture": architecture,
            "num_samples": len(results),
            "rouge1": rouge_scores.get("rouge1", 0.0),
            "rouge2": rouge_scores.get("rouge2", 0.0),
            "rougeL": rouge_scores.get("rougeL", 0.0),
            "rougeLsum": rouge_scores.get("rougeLsum", 0.0),
            "avg_latency_sec": float(df["latency_sec"].mean()),
            "avg_input_tokens": float(df["input_tokens"].mean()),
            "avg_output_tokens": float(df["output_tokens"].mean()),
        }])

    metric_path = metrics_dir / f"{task_name}_{model_slug}_metrics.csv"
    metrics.to_csv(metric_path, index=False, encoding="utf-8-sig")
    print("Saved metrics to:", metric_path)
    print(metrics)

    print(df.head())


if __name__ == "__main__":
    main()