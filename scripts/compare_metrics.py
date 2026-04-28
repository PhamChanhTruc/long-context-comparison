import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd


def main():
    metrics_dir = Path("outputs/metrics")
    files = list(metrics_dir.glob("*_metrics.csv"))

    if not files:
        print("No metric files found in outputs/metrics")
        return

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        dfs.append(df)

    all_metrics = pd.concat(dfs, ignore_index=True)

    preferred_cols = [
        "task",
        "model",
        "architecture",
        "num_samples",
        "exact_match",
        "f1",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "avg_latency_sec",
        "avg_input_tokens",
        "avg_output_tokens",
        "source_file",
    ]

    existing_cols = [c for c in preferred_cols if c in all_metrics.columns]
    all_metrics = all_metrics[existing_cols]

    out_path = metrics_dir / "all_metrics_comparison.csv"
    all_metrics.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("Saved comparison file to:", out_path)
    print()
    print(all_metrics)


if __name__ == "__main__":
    main()