from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))


LOGGER = logging.getLogger(__name__)


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(digits)
    return df


def warn_before_overwrite(path: Path) -> None:
    if path.exists():
        LOGGER.warning("Overwriting existing comparison output: %s", path)


def print_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("No rows." if df.empty else df.to_string(index=False))


def read_metric_files(metrics_dir: Path) -> pd.DataFrame:
    files = sorted(metrics_dir.glob("*_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No per-run metric files found in {metrics_dir}")

    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file)
        frame["source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_main_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    main_scores: list[float | None] = []
    main_score_names: list[str | None] = []

    for _, row in df.iterrows():
        task = row.get("task")
        if task == "qa":
            for column in ("post_f1", "f1", "raw_f1"):
                if column in row and pd.notna(row[column]):
                    main_scores.append(float(row[column]))
                    main_score_names.append(column)
                    break
            else:
                main_scores.append(None)
                main_score_names.append(None)
        elif task == "summarization":
            score = row.get("rougeL")
            main_scores.append(float(score) if pd.notna(score) else None)
            main_score_names.append("rougeL" if pd.notna(score) else None)
        else:
            main_scores.append(None)
            main_score_names.append(None)

    df["main_score_name"] = main_score_names
    df["main_score"] = main_scores
    return df


def ordered_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "task",
        "model_name",
        "model",
        "model_type",
        "architecture",
        "num_samples",
        "max_samples",
        "max_input_tokens",
        "max_new_tokens",
        "raw_exact_match",
        "raw_f1",
        "post_exact_match",
        "post_f1",
        "rouge1",
        "rouge2",
        "rougeL",
        "main_score_name",
        "main_score",
        "total_latency_sec",
        "avg_latency_sec",
        "avg_input_tokens",
        "avg_output_tokens",
        "num_truncated",
        "source_file",
    ]
    return [column for column in preferred if column in df.columns]


def save_optional_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.info("Skipping figures because matplotlib is unavailable: %s", exc)
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    for task_name, task_df in df.groupby("task"):
        task_df = task_df.dropna(subset=["main_score"]).copy()
        if task_df.empty:
            continue
        labels = task_df.get("model_name", task_df.get("model", task_df["source_file"]))
        labels = labels.astype(str).str.split("/").str[-1]

        fig_width = max(6, min(12, 1.5 * len(task_df)))
        plt.figure(figsize=(fig_width, 4))
        plt.bar(labels, task_df["main_score"])
        plt.ylabel(task_df["main_score_name"].iloc[0] or "main score")
        plt.title(f"{task_name} model comparison")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = figures_dir / f"{task_name}_main_score.png"
        warn_before_overwrite(out_path)
        plt.savefig(out_path, dpi=150)
        plt.close()
        LOGGER.info("Saved figure to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics_dir = Path("outputs/metrics")
    figures_dir = Path("outputs/figures")

    all_metrics = read_metric_files(metrics_dir)
    all_metrics = add_main_score(all_metrics)
    all_metrics = all_metrics[ordered_columns(all_metrics)]
    sort_columns = ["task", "main_score"]
    sort_ascending = [True, False]
    if "avg_latency_sec" in all_metrics.columns:
        sort_columns.append("avg_latency_sec")
        sort_ascending.append(True)
    all_metrics = all_metrics.sort_values(
        by=sort_columns,
        ascending=sort_ascending,
        na_position="last",
    )
    all_metrics = round_numeric(all_metrics, digits=4)

    out_path = metrics_dir / "all_metrics_comparison.csv"
    warn_before_overwrite(out_path)
    all_metrics.to_csv(out_path, index=False, encoding="utf-8-sig")

    qa_df = all_metrics[all_metrics["task"] == "qa"].copy()
    qa_out = metrics_dir / "qa_metrics_comparison.csv"
    warn_before_overwrite(qa_out)
    qa_df.to_csv(qa_out, index=False, encoding="utf-8-sig")

    sum_df = all_metrics[all_metrics["task"] == "summarization"].copy()
    sum_out = metrics_dir / "summarization_metrics_comparison.csv"
    warn_before_overwrite(sum_out)
    sum_df.to_csv(sum_out, index=False, encoding="utf-8-sig")

    leaderboard_cols = [
        column
        for column in (
            "task",
            "model_name",
            "model",
            "model_type",
            "architecture",
            "main_score_name",
            "main_score",
            "avg_latency_sec",
            "avg_output_tokens",
            "source_file",
        )
        if column in all_metrics.columns
    ]
    leaderboard = all_metrics[leaderboard_cols].copy()
    leaderboard_out = metrics_dir / "leaderboard.csv"
    warn_before_overwrite(leaderboard_out)
    leaderboard.to_csv(leaderboard_out, index=False, encoding="utf-8-sig")

    save_optional_figures(all_metrics, figures_dir)

    LOGGER.info("Saved full comparison to %s", out_path)
    LOGGER.info("Saved QA comparison to %s", qa_out)
    LOGGER.info("Saved summarization comparison to %s", sum_out)
    LOGGER.info("Saved leaderboard to %s", leaderboard_out)

    print_section("FULL METRICS TABLE", all_metrics)
    print_section("QA COMPARISON", qa_df)
    print_section("SUMMARIZATION COMPARISON", sum_df)
    print_section("LEADERBOARD", leaderboard)


if __name__ == "__main__":
    main()
