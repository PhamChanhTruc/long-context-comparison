# Long Context Comparison

So sanh hieu nang mo hinh tren bai toan ngu canh dai.  
Compare model performance on long-context NLP tasks.

## Overview | Tong quan

Du an nay danh gia hai nhom mo hinh:
- `encoder-decoder`
- `decoder-only`

Tren hai tac vu:
- `QA`
- `Summarization`

This project benchmarks two model groups:
- `encoder-decoder`
- `decoder-only`

Across two tasks:
- `QA`
- `Summarization`

## Project Structure | Cau truc du an

```text
long-context-comparison/
|-- configs/
|   `-- base.yaml
|-- data/
|   `-- processed/
|       |-- pilot_qa.jsonl
|       `-- pilot_sum.jsonl
|-- outputs/
|   |-- metrics/
|   `-- predictions/
|-- scripts/
|   |-- compare_metrics.py
|   `-- run_inference.py
|-- src/
|   |-- inference.py
|   |-- loaders.py
|   |-- metrics_qa.py
|   |-- metrics_sum.py
|   |-- model_utils.py
|   `-- prompts.py
|-- test_load_data.py
|-- test_model_qa.py
|-- requirements.txt
`-- README.md
```

## Installation | Cai dat

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu chua co `requirements.txt` day du, ban co the cai thu cong:

```powershell
pip install transformers datasets evaluate accelerate sentencepiece rouge_score pandas matplotlib pyyaml
```

If `requirements.txt` is incomplete, you can install the core packages manually:

```powershell
pip install transformers datasets evaluate accelerate sentencepiece rouge_score pandas matplotlib pyyaml
```

## Usage | Cach dung

```powershell
python test_load_data.py
python test_model_qa.py
python scripts\run_inference.py
python scripts\compare_metrics.py
```

## Configuration | Cau hinh

Chinh sua `configs/base.yaml` de chon:
- task (`qa` hoac `summarization`)
- dataset path
- model name
- model architecture (`encoder-decoder` hoac `decoder-only`)
- generation settings

Edit `configs/base.yaml` to select:
- task (`qa` or `summarization`)
- dataset path
- model name
- model architecture (`encoder-decoder` or `decoder-only`)
- generation settings

## Outputs | Dau ra

Thu muc ket qua:
- `outputs/predictions/`: file du doan theo tung model va task
- `outputs/metrics/`: file metric cho tung lan chay va file tong hop

Output folders:
- `outputs/predictions/`: prediction files for each model-task run
- `outputs/metrics/`: per-run metric files and the aggregated comparison file

Vi du file:
- `outputs/predictions/qa_flan-t5-base.csv`
- `outputs/predictions/summarization_TinyLlama-1_1B-Chat-v1_0.csv`
- `outputs/metrics/qa_flan-t5-base_metrics.csv`
- `outputs/metrics/all_metrics_comparison.csv`

## Metrics | Chi so danh gia

QA:
- `raw_exact_match`: do trung khop chinh xac tren dau ra goc
- `raw_f1`: F1 tren dau ra goc
- `post_exact_match`: Exact Match sau hau xu ly cau tra loi ngan
- `post_f1`: F1 sau hau xu ly

Summarization:
- `rouge1`: overlap unigram
- `rouge2`: overlap bigram
- `rougeL`: longest common subsequence
- `rougeLsum`: ROUGE-L cho tom tat muc tai lieu

Efficiency:
- `avg_latency_sec`: thoi gian suy luan trung binh moi mau
- `avg_input_tokens`: so token dau vao trung binh
- `avg_output_tokens`: so token dau ra trung binh

QA:
- `raw_exact_match`: exact string match on raw predictions
- `raw_f1`: token-level F1 on raw predictions
- `post_exact_match`: Exact Match after short-answer post-processing
- `post_f1`: F1 after post-processing

Summarization:
- `rouge1`: unigram overlap
- `rouge2`: bigram overlap
- `rougeL`: longest common subsequence score
- `rougeLsum`: summary-level ROUGE-L

Efficiency:
- `avg_latency_sec`: average inference latency per sample
- `avg_input_tokens`: average number of input tokens
- `avg_output_tokens`: average number of output tokens

## QA Post-processing / Hậu xử lý cho QA

For QA, decoder-only models often produce full-sentence answers such as:  
Trong QA, mô hình decoder-only thường sinh câu trả lời đầy đủ như:

`The capital of France is Paris.`

Instead of the short answer:  
Thay vì câu trả lời ngắn:

`Paris`

To make evaluation fairer, the project includes a post-processing step that extracts shorter answers before computing metrics.  
Để việc đánh giá công bằng hơn, project có thêm bước hậu xử lý nhằm trích xuất câu trả lời ngắn trước khi tính metric.

## Current Pilot Findings / Kết quả pilot hiện tại

Pilot runs suggest:  
Kết quả pilot ban đầu cho thấy:

- Encoder–decoder is faster on CPU  
  Encoder–decoder chạy nhanh hơn trên CPU
- Decoder-only produces more natural answers  
  Decoder-only tạo câu trả lời tự nhiên hơn
- QA results improve significantly after post-processing for decoder-only models  
  Kết quả QA cải thiện rõ sau hậu xử lý với decoder-only
- Decoder-only performs better on pilot summarization ROUGE, but with much higher latency  
  Decoder-only cho ROUGE tốt hơn ở pilot summarization, nhưng độ trễ cao hơn đáng kể

## Workflow / Quy trình làm việc

1. Prepare data in `data/processed/`  
   Chuẩn bị dữ liệu trong `data/processed/`
2. Edit `configs/base.yaml`  
   Chỉnh `configs/base.yaml`
3. Run `run_inference.py`  
   Chạy `run_inference.py`
4. Save predictions and metrics in `outputs/`  
   Lưu predictions và metrics vào `outputs/`
5. Run `compare_metrics.py` for comparison  
   Chạy `compare_metrics.py` để tổng hợp kết quả

## Next Steps / Hướng phát triển tiếp theo

- increase pilot size from 5 to 20, 50, or 100 samples  
  tăng số mẫu pilot từ 5 lên 20, 50 hoặc 100
- move experiments to Colab  
  chuyển thực nghiệm lên Colab
- test longer contexts  
  thử độ dài ngữ cảnh lớn hơn
- add larger models  
  thêm các mô hình lớn hơn
- create comparison plots  
  vẽ biểu đồ so sánh
- write the experimental report section  
  viết chương thực nghiệm cho báo cáo

## Notes / Ghi chú

- The first model download from Hugging Face may take a while.  
  Lần đầu tải model từ Hugging Face có thể mất thời gian.
- If `do_sample=False`, the `temperature` argument may be ignored.  
  Nếu `do_sample=False`, tham số `temperature` có thể bị bỏ qua.
- Decoder-only outputs are often longer, so QA post-processing is important.  
  Đầu ra của decoder-only thường dài hơn, nên hậu xử lý QA là cần thiết.
