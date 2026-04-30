import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(digits)
    return df


def print_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if df.empty:
        print("No rows.")
    else:
        print(df.to_string(index=False))


def main():
    metrics_dir = Path("outputs/metrics")
    files = sorted(
        [f for f in metrics_dir.glob("*_metrics.csv") if f.name != "all_metrics_comparison.csv"]
    )

    if not files:
        print("No metric files found in outputs/metrics")
        return

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        dfs.append(df)

    all_metrics = pd.concat(dfs, ignore_index=True)

    # Chuẩn hóa cột để dễ nhìn
    desired_cols = [
        "task",
        "model",
        "architecture",
        "num_samples",
        # QA metrics
        "raw_exact_match",
        "raw_f1",
        "post_exact_match",
        "post_f1",
        # old QA metrics fallback
        "exact_match",
        "f1",
        # summarization metrics
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        # efficiency
        "avg_latency_sec",
        "avg_input_tokens",
        "avg_output_tokens",
        "source_file",
    ]

    existing_cols = [c for c in desired_cols if c in all_metrics.columns]
    all_metrics = all_metrics[existing_cols]
    all_metrics = round_numeric(all_metrics, digits=4)

    # Lưu full table
    out_path = metrics_dir / "all_metrics_comparison.csv"
    all_metrics.to_csv(out_path, index=False, encoding="utf-8-sig")

    # -----------------------------
    # QA view
    # -----------------------------
    qa_df = all_metrics[all_metrics["task"] == "qa"].copy()

    qa_cols_priority = [
        "task",
        "model",
        "architecture",
        "num_samples",
        "raw_exact_match",
        "raw_f1",
        "post_exact_match",
        "post_f1",
        "exact_match",
        "f1",
        "avg_latency_sec",
        "avg_input_tokens",
        "avg_output_tokens",
        "source_file",
    ]
    qa_cols = [c for c in qa_cols_priority if c in qa_df.columns]
    qa_df = qa_df[qa_cols]

    # Sắp xếp QA: ưu tiên post_f1, nếu không có thì dùng f1/raw_f1
    if "post_f1" in qa_df.columns:
        qa_df = qa_df.sort_values(by=["post_f1", "avg_latency_sec"], ascending=[False, True])
    elif "f1" in qa_df.columns:
        qa_df = qa_df.sort_values(by=["f1", "avg_latency_sec"], ascending=[False, True])
    elif "raw_f1" in qa_df.columns:
        qa_df = qa_df.sort_values(by=["raw_f1", "avg_latency_sec"], ascending=[False, True])

    qa_out = metrics_dir / "qa_metrics_comparison.csv"
    qa_df.to_csv(qa_out, index=False, encoding="utf-8-sig")

    # -----------------------------
    # Summarization view
    # -----------------------------
    sum_df = all_metrics[all_metrics["task"] == "summarization"].copy()

    sum_cols_priority = [
        "task",
        "model",
        "architecture",
        "num_samples",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "avg_latency_sec",
        "avg_input_tokens",
        "avg_output_tokens",
        "source_file",
    ]
    sum_cols = [c for c in sum_cols_priority if c in sum_df.columns]
    sum_df = sum_df[sum_cols]

    # Sắp xếp summarization: ưu tiên rougeL
    if "rougeL" in sum_df.columns:
        sum_df = sum_df.sort_values(by=["rougeL", "avg_latency_sec"], ascending=[False, True])

    sum_out = metrics_dir / "summarization_metrics_comparison.csv"
    sum_df.to_csv(sum_out, index=False, encoding="utf-8-sig")

    # -----------------------------
    # Lightweight leaderboard
    # -----------------------------
    leaderboard_rows = []

    for _, row in qa_df.iterrows():
        score = None
        score_name = None

        if "post_f1" in row and pd.notna(row.get("post_f1", None)):
            score = row["post_f1"]
            score_name = "post_f1"
        elif "f1" in row and pd.notna(row.get("f1", None)):
            score = row["f1"]
            score_name = "f1"
        elif "raw_f1" in row and pd.notna(row.get("raw_f1", None)):
            score = row["raw_f1"]
            score_name = "raw_f1"

        leaderboard_rows.append({
            "task": "qa",
            "model": row["model"],
            "architecture": row["architecture"],
            "main_score_name": score_name,
            "main_score": score,
            "avg_latency_sec": row.get("avg_latency_sec"),
            "avg_output_tokens": row.get("avg_output_tokens"),
        })

    for _, row in sum_df.iterrows():
        leaderboard_rows.append({
            "task": "summarization",
            "model": row["model"],
            "architecture": row["architecture"],
            "main_score_name": "rougeL" if "rougeL" in row else None,
            "main_score": row.get("rougeL"),
            "avg_latency_sec": row.get("avg_latency_sec"),
            "avg_output_tokens": row.get("avg_output_tokens"),
        })

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_df = round_numeric(leaderboard_df, digits=4)

    leaderboard_out = metrics_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_out, index=False, encoding="utf-8-sig")

    # -----------------------------
    # Print to terminal
    # -----------------------------
    print(f"Saved full comparison file to: {out_path}")
    print(f"Saved QA comparison file to: {qa_out}")
    print(f"Saved summarization comparison file to: {sum_out}")
    print(f"Saved leaderboard file to: {leaderboard_out}")

    print_section("FULL METRICS TABLE", all_metrics)
    print_section("QA COMPARISON", qa_df)
    print_section("SUMMARIZATION COMPARISON", sum_df)
    print_section("LEADERBOARD", leaderboard_df)


if __name__ == "__main__":
    main()