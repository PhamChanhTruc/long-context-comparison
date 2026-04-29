import time
import torch


def build_decoder_inputs(tokenizer, prompt, device, max_input_tokens):
    # Nếu tokenizer có chat template (ví dụ Mistral Instruct), dùng template đó
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
    else:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

    # truncate thủ công nếu input quá dài
    if encoded["input_ids"].shape[1] > max_input_tokens:
        encoded["input_ids"] = encoded["input_ids"][:, -max_input_tokens:]
        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded["attention_mask"][:, -max_input_tokens:]

    return encoded


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

    if architecture == "decoder-only":
        inputs = build_decoder_inputs(
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_input_tokens=max_input_tokens,
        )

        input_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        latency_sec = time.perf_counter() - start

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        latency_sec = time.perf_counter() - start

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