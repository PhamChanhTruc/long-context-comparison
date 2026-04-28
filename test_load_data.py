from src.loaders import load_jsonl

qa_data = load_jsonl("data/processed/pilot_qa.jsonl")
sum_data = load_jsonl("data/processed/pilot_sum.jsonl")

print("QA samples:", len(qa_data))
print("First QA sample:", qa_data[0])
print("SUM samples:", len(sum_data))
print("First SUM sample:", sum_data[0])
