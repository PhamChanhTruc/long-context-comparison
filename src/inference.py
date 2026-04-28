import time


def run_one_sample(
    sample,
    task_name,
    model_name,
    architecture,
    model,
    tokenizer,
    device,
    max_input_tokens,
    max_new_tokens,
    do_sample,
    temperature,
    build_prompt_fn,
):
    prompt = build_prompt_fn(sample, task_name)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature
    )
    latency_sec = time.perf_counter() - start

    if architecture == "decoder-only":
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    output_tokens = len(tokenizer.encode(prediction, add_special_tokens=False))

    if task_name == "qa":
        reference = sample["answers"][0]
    else:
        reference = sample["reference"]

    return {
        "id": sample["id"],
        "task": task_name,
        "model": model_name,
        "architecture": architecture,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_sec": latency_sec,
        "prediction": prediction,
        "reference": reference,
    }