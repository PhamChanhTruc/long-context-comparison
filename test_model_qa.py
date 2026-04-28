from src.loaders import load_jsonl
from src.prompts import build_prompt
from src.model_utils import load_model_and_tokenizer, resolve_device

MODEL_NAME = "google/flan-t5-base"
ARCHITECTURE = "encoder-decoder"
TASK_NAME = "qa"

device = resolve_device("auto")
print("Device:", device)

data = load_jsonl("data/processed/pilot_qa.jsonl")
sample = data[0]

print("Loaded sample:")
print(sample)

prompt = build_prompt(sample, TASK_NAME)
print("\nPrompt:\n")
print(prompt)

model, tokenizer = load_model_and_tokenizer(MODEL_NAME, ARCHITECTURE, device)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False
)

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nPrediction:")
print(prediction)

print("\nReference:")
print(sample["answers"])