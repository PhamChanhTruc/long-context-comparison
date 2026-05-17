from __future__ import annotations

from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute mean ROUGE F1 scores without requiring metric downloads."""
    if len(predictions) != len(references):
        raise ValueError(
            f"ROUGE needs the same number of predictions and references; got "
            f"{len(predictions)} predictions and {len(references)} references."
        )
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(str(reference), str(prediction))
        for metric_name in totals:
            totals[metric_name] += scores[metric_name].fmeasure

    return {metric_name: value / len(predictions) for metric_name, value in totals.items()}
