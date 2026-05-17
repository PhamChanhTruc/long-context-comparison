from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import resolve_effective_max_input_tokens, run_one_sample
from src.loaders import load_jsonl, validate_dataset
from src.metrics_qa import exact_match, f1_score
from src.metrics_sum import compute_rouge
from src.model_utils import load_model_and_tokenizer, resolve_device
from src.prompts import build_prompt


LOGGER = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return config


def _require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Config is missing required section: {section}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    task = _require_section(config, "task")
    model = _require_section(config, "model")
    generation = _require_section(config, "generation")
    _require_section(config, "runtime")
    _require_section(config, "output")

    task_name = task.get("name")
    if task_name not in {"qa", "summarization"}:
        raise ValueError("task.name must be either 'qa' or 'summarization'.")
    data_path = task.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError(f"task.data_path does not exist: {data_path}")
    max_samples = task.get("max_samples")
    if max_samples is not None and (not isinstance(max_samples, int) or max_samples <= 0):
        raise ValueError("task.max_samples must be a positive integer or null.")

    if not model.get("name"):
        raise ValueError("model.name is required.")
    if model.get("architecture") not in {"encoder-decoder", "decoder-only"}:
        raise ValueError("model.architecture must be 'encoder-decoder' or 'decoder-only'.")

    for field in ("max_input_tokens", "max_new_tokens"):
        value = generation.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"generation.{field} must be a positive integer.")
    if not isinstance(generation.get("do_sample"), bool):
        raise ValueError("generation.do_sample must be true or false.")


def select_samples(
    data: list[dict[str, Any]],
    max_samples: int | None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    selected = data[:]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def sanitize_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(".", "_").replace("-", "-")


def parse_reference_answers(row: pd.Series) -> list[str]:
    references_json = row.get("references_json")
    if isinstance(references_json, str) and references_json.strip():
        try:
            parsed = json.loads(references_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid references_json for sample {row.get('id', '<unknown>')}: {exc}"
            ) from exc
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(
                f"references_json for sample {row.get('id', '<unknown>')} must be "
                "a non-empty JSON list."
            )
        return [str(answer) for answer in parsed if str(answer).strip()]

    reference = row.get("reference", "")
    if pd.isna(reference) or not str(reference).strip():
        return []
    return [str(reference)]


def compute_qa_metrics_from_df(pred_df: pd.DataFrame) -> dict[str, float]:
    raw_em_scores: list[float] = []
    raw_f1_scores: list[float] = []
    post_em_scores: list[float] = []
    post_f1_scores: list[float] = []

    for _, row in pred_df.iterrows():
        answers = parse_reference_answers(row)
        raw_prediction = str(row.get("raw_prediction", "") or "")
        post_prediction = str(
            row.get("post_prediction", row.get("prediction", "")) or ""
        )

        raw_em_scores.append(exact_match(raw_prediction, answers))
        raw_f1_scores.append(f1_score(raw_prediction, answers))
        post_em_scores.append(exact_match(post_prediction, answers))
        post_f1_scores.append(f1_score(post_prediction, answers))

    count = len(pred_df)
    return {
        "raw_exact_match": sum(raw_em_scores) / count if count else 0.0,
        "raw_f1": sum(raw_f1_scores) / count if count else 0.0,
        "post_exact_match": sum(post_em_scores) / count if count else 0.0,
        "post_f1": sum(post_f1_scores) / count if count else 0.0,
    }


def compute_sum_metrics_from_df(pred_df: pd.DataFrame) -> dict[str, float]:
    prediction_column = "post_prediction" if "post_prediction" in pred_df.columns else "prediction"
    predictions = pred_df[prediction_column].fillna("").astype(str).tolist()
    references = pred_df["reference"].fillna("").astype(str).tolist()
    return compute_rouge(predictions, references)


def warn_before_overwrite(path: Path) -> None:
    if path.exists():
        LOGGER.warning("Overwriting existing file: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot inference for one YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flan_qa.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    validate_config(config)

    task_cfg = config["task"]
    model_cfg = config["model"]
    generation_cfg = config["generation"]
    runtime_cfg = config["runtime"]
    output_cfg = config["output"]

    task_name = task_cfg["name"]
    data_path = Path(task_cfg["data_path"])
    max_samples = task_cfg.get("max_samples")
    shuffle = bool(task_cfg.get("shuffle", False))
    seed = int(task_cfg.get("seed", 42))

    model_name = model_cfg["name"]
    architecture = model_cfg["architecture"]
    use_4bit = bool(model_cfg.get("use_4bit", False))

    configured_max_input_tokens = int(generation_cfg["max_input_tokens"])
    max_new_tokens = int(generation_cfg["max_new_tokens"])
    do_sample = bool(generation_cfg["do_sample"])
    temperature = float(generation_cfg.get("temperature", 0.0))

    predictions_dir = Path(output_cfg["predictions_dir"])
    metrics_dir = Path(output_cfg["metrics_dir"])
    logs_dir = Path(output_cfg.get("logs_dir", "outputs/logs"))
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(runtime_cfg.get("device", "auto"))
    LOGGER.info("Device: %s", device)

    data = load_jsonl(data_path)
    validate_dataset(data, task_name=task_name)
    data = select_samples(data, max_samples=max_samples, shuffle=shuffle, seed=seed)
    LOGGER.info("Loaded %s samples from %s.", len(data), data_path)

    try:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            architecture=architecture,
            device=device,
            use_4bit=use_4bit,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model {model_name!r}. Large models may require a GPU, "
            "more RAM/VRAM, Hugging Face access, or a prior model download."
        ) from exc

    max_input_tokens = resolve_effective_max_input_tokens(
        model=model,
        tokenizer=tokenizer,
        requested_max_input_tokens=configured_max_input_tokens,
    )

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(data, start=1):
        output = run_one_sample(
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
        rows.append(output)
        LOGGER.info(
            "[%s/%s] %s done in %.3fs",
            index,
            len(data),
            sample["id"],
            output["latency_sec"],
        )

    pred_df = pd.DataFrame(rows)
    model_tag = sanitize_model_name(model_name)
    run_name = config.get("run_name") or f"{task_name}_{model_tag}"
    pred_path = predictions_dir / f"{run_name}.csv"
    warn_before_overwrite(pred_path)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved predictions to %s", pred_path)

    if task_name == "qa":
        task_metrics = compute_qa_metrics_from_df(pred_df)
    else:
        task_metrics = compute_sum_metrics_from_df(pred_df)

    metrics = {
        "task": task_name,
        "model_name": model_name,
        "model": model_name,
        "model_type": architecture,
        "architecture": architecture,
        "config_path": str(args.config),
        "num_samples": len(pred_df),
        "max_samples": max_samples,
        "max_input_tokens": max_input_tokens,
        "configured_max_input_tokens": configured_max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "total_latency_sec": pred_df["latency_sec"].sum(),
        "avg_latency_sec": pred_df["latency_sec"].mean(),
        "avg_input_tokens": pred_df["input_tokens"].mean(),
        "avg_output_tokens": pred_df["output_tokens"].mean(),
        "num_truncated": int(pred_df["truncated"].sum()),
    }
    metrics.update(task_metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = metrics_dir / f"{run_name}_metrics.csv"
    warn_before_overwrite(metrics_path)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved metrics to %s", metrics_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
