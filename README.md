# Long Context Comparison

Final NLP project for comparing encoder-decoder and decoder-only language models on
long-context tasks.

## Project Goal

The project runs zero-shot inference on two tasks and compares quality and latency:

- Question Answering
- Summarization

Model groups:

- Encoder-decoder: FLAN-T5, LongT5
- Decoder-only: TinyLlama, Mistral

## Repository Structure

```text
configs/                 YAML experiment configs
data/processed/           Pilot JSONL data and optional prepared main data
scripts/                  Validation, data prep, inference, comparison scripts
src/                      Data loading, prompts, inference, metrics, model utilities
outputs/                  Local predictions, metrics, logs, and figures (gitignored)
test_load_data.py         Lightweight data/config tests
test_model_qa.py          Lightweight QA metric and inference-output tests
```

## Installation

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux or Colab:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` is optional and only needed for CUDA 4-bit configs such as Mistral.
Install it separately when your CUDA environment supports it:

```bash
pip install bitsandbytes
```

## Data

Pilot files are included:

- `data/processed/pilot_qa.jsonl`
- `data/processed/pilot_sum.jsonl`

These are small local files for quick debugging, smoke tests, and pipeline checks.
They are not the main experiment dataset.

Validate the JSONL files:

```bash
python scripts/validate_data.py
```

Each QA line must contain `id`, `task`, `context`, `question`, and a non-empty
`answers` list. Each summarization line must contain `id`, `task`, `context`, and
`reference`.

## Main Experiment Data

The repository data-preparation script is the single source of truth for main
experiment JSONL files. Kaggle and Colab notebooks should call
`scripts/prepare_data.py` instead of duplicating Qasper or GovReport processing
logic in notebook cells.

Prepare main validation samples from Hugging Face parquet files:

```bash
python scripts/prepare_data.py --qa_samples 1000 --sum_samples 1000 --seed 42 --output_dir data/processed
```

This writes `data/processed/main_qa.jsonl` and `data/processed/main_sum.jsonl`.
Use `--qa_samples 0` or `--sum_samples 0` to write all valid examples for that
task. The script shuffles valid examples with the fixed seed before sample
selection, so repeated runs with the same inputs and seed are deterministic.

Validate prepared main data:

```bash
python scripts/validate_data.py data/processed/main_qa.jsonl data/processed/main_sum.jsonl
```

If datasets cannot be downloaded in a local environment, the script fails with a
clear message. Kaggle/Colab should install the requirements and run the same
command so the notebook uses the repository pipeline directly.

## Running Inference

Run one config:

```bash
python scripts/run_inference.py --config configs/flan_qa.yaml
```

Other included pilot configs:

```bash
python scripts/run_inference.py --config configs/flan_sum.yaml
python scripts/run_inference.py --config configs/longt5_qa.yaml
python scripts/run_inference.py --config configs/longt5_sum.yaml
python scripts/run_inference.py --config configs/tinyllama_qa.yaml
python scripts/run_inference.py --config configs/tinyllama_sum.yaml
python scripts/run_inference.py --config configs/mistral_qa.yaml
python scripts/run_inference.py --config configs/mistral_sum.yaml
```

PowerShell loop for all configs:

```powershell
Get-ChildItem configs\*.yaml | ForEach-Object {
  python scripts\run_inference.py --config $_.FullName
}
```

Bash loop for all configs:

```bash
for cfg in configs/*.yaml; do
  python scripts/run_inference.py --config "$cfg"
done
```

Large models may require a GPU, enough RAM/VRAM, Hugging Face access, and prior
model downloads. The 4-bit Mistral configs require CUDA plus `bitsandbytes`.

## Comparing Metrics

After one or more runs:

```bash
python scripts/compare_metrics.py
```

The script reads `outputs/metrics/*_metrics.csv`, writes
`outputs/metrics/all_metrics_comparison.csv`, task-specific comparison CSVs, a
leaderboard, and optional simple figures in `outputs/figures/` when matplotlib is
installed.

## QA Metrics

QA predictions store:

- `raw_prediction`: exact decoded model output
- `post_prediction`: cleaned answer used for post-processed QA metrics
- `references_json`: JSON list containing all gold answers

Raw QA metrics use `raw_prediction`. Post QA metrics use `post_prediction`.
Exact Match and token-level F1 are computed against all gold answers and take the
best score per sample.

QA post-processing removes common prompt echoes such as `Answer:`, `The answer is`,
and copied instruction fragments. Summarization outputs are not aggressively
post-processed.

## Truncation Strategy

The code enforces `generation.max_input_tokens` before generation. It first builds
the task prompt, then trims only the context if the prompt is too long.

- QA preserves the question and answer instruction near the end of the prompt.
- Summarization preserves the summary instruction and as much document context as
  possible.
- Context truncation is deterministic from the beginning of the context.
- A warning is logged whenever context tokens are removed.
- If the fixed prompt/question alone exceeds the token budget, inference fails
  with a helpful error instead of sending an over-length input to the model.

For fair final experiments, use the same `max_input_tokens` across the model configs
you want to compare.

## Tests

Run lightweight checks:

```bash
python test_load_data.py
python test_model_qa.py
python -m pytest
```

These tests do not download models. They validate JSONL loading, config loading,
data-preparation helper behavior, multiple-answer QA metrics, QA post-processing,
and preservation of all gold answers in prediction outputs.

## Known Limitations

- Pilot data is very small and should not be treated as a final result.
- Inference is zero-shot unless configs and prompts are changed.
- Truncation may remove evidence from long documents.
- Local hardware can limit model size, speed, and quantization support.
- Dataset preparation depends on Hugging Face availability or cached datasets.
