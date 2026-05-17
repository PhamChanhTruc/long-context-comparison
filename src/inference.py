from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from src.metrics_qa import extract_short_answer


LOGGER = logging.getLogger(__name__)
PromptBuilder = Callable[[dict[str, Any], str, str | None], str]


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    input_tokens: int
    truncated: bool
    truncated_context_tokens: int


def _uses_chat_template(tokenizer: Any, architecture: str) -> bool:
    return architecture == "decoder-only" and bool(getattr(tokenizer, "chat_template", None))


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _prompt_token_ids(tokenizer: Any, prompt: str, architecture: str) -> list[int]:
    if _uses_chat_template(tokenizer, architecture):
        token_ids = tokenizer.apply_chat_template(
            _chat_messages(prompt),
            tokenize=True,
            add_generation_prompt=True,
        )
        if token_ids and isinstance(token_ids[0], list):
            return list(token_ids[0])
        return list(token_ids)
    return tokenizer.encode(prompt, add_special_tokens=True)


def _build_prompt(
    sample: dict[str, Any],
    task_name: str,
    model_name: str,
    build_prompt_fn: PromptBuilder,
    context: str,
) -> str:
    prompt_sample = dict(sample)
    prompt_sample["context"] = context
    return build_prompt_fn(prompt_sample, task_name, model_name)


def _truncate_prompt_context(
    sample: dict[str, Any],
    task_name: str,
    model_name: str,
    tokenizer: Any,
    architecture: str,
    max_input_tokens: int,
    build_prompt_fn: PromptBuilder,
) -> PromptBundle:
    context = str(sample["context"])
    prompt = _build_prompt(sample, task_name, model_name, build_prompt_fn, context)
    input_tokens = len(_prompt_token_ids(tokenizer, prompt, architecture))
    if input_tokens <= max_input_tokens:
        return PromptBundle(
            prompt=prompt,
            input_tokens=input_tokens,
            truncated=False,
            truncated_context_tokens=0,
        )

    empty_prompt = _build_prompt(sample, task_name, model_name, build_prompt_fn, "")
    empty_prompt_tokens = len(_prompt_token_ids(tokenizer, empty_prompt, architecture))
    if empty_prompt_tokens >= max_input_tokens:
        raise ValueError(
            f"max_input_tokens={max_input_tokens} is too small to fit the fixed "
            f"{task_name} prompt/question ({empty_prompt_tokens} tokens)."
        )

    context_token_ids = tokenizer.encode(context, add_special_tokens=False)
    low = 0
    high = len(context_token_ids)
    best_prompt = empty_prompt
    best_len = empty_prompt_tokens
    best_context_len = 0

    while low <= high:
        mid = (low + high) // 2
        candidate_context = tokenizer.decode(
            context_token_ids[:mid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        candidate_prompt = _build_prompt(
            sample,
            task_name,
            model_name,
            build_prompt_fn,
            candidate_context,
        )
        candidate_len = len(_prompt_token_ids(tokenizer, candidate_prompt, architecture))
        if candidate_len <= max_input_tokens:
            best_prompt = candidate_prompt
            best_len = candidate_len
            best_context_len = mid
            low = mid + 1
        else:
            high = mid - 1

    truncated_context_tokens = len(context_token_ids) - best_context_len
    LOGGER.warning(
        "Truncated %s sample %s from %s to %s input tokens; removed %s context tokens.",
        task_name,
        sample.get("id", "<unknown>"),
        input_tokens,
        best_len,
        truncated_context_tokens,
    )
    return PromptBundle(
        prompt=best_prompt,
        input_tokens=best_len,
        truncated=True,
        truncated_context_tokens=truncated_context_tokens,
    )


def _encode_generation_inputs(
    tokenizer: Any,
    prompt: str,
    architecture: str,
    device: str,
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    if _uses_chat_template(tokenizer, architecture):
        try:
            encoded = tokenizer.apply_chat_template(
                _chat_messages(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                _chat_messages(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            encoded = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        if isinstance(encoded, torch.Tensor):
            encoded = {
                "input_ids": encoded,
                "attention_mask": torch.ones_like(encoded),
            }
    else:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)

    input_tokens = encoded["input_ids"].shape[1]
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"Tokenized prompt has {input_tokens} tokens, exceeding "
            f"max_input_tokens={max_input_tokens}. This should have been handled "
            "before generation."
        )
    return {key: value.to(device) for key, value in encoded.items()}


def model_context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates: list[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000_000:
        candidates.append(tokenizer_limit)

    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)

    return min(candidates) if candidates else None


def resolve_effective_max_input_tokens(
    model: Any,
    tokenizer: Any,
    requested_max_input_tokens: int,
) -> int:
    if requested_max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be a positive integer.")

    limit = model_context_limit(model, tokenizer)
    if limit is not None and requested_max_input_tokens > limit:
        LOGGER.warning(
            "Requested max_input_tokens=%s exceeds model/tokenizer limit=%s; using %s.",
            requested_max_input_tokens,
            limit,
            limit,
        )
        return limit
    return requested_max_input_tokens


def build_generation_prompt(
    sample: dict[str, Any],
    task_name: str,
    model_name: str,
    tokenizer: Any,
    architecture: str,
    max_input_tokens: int,
    build_prompt_fn: PromptBuilder,
) -> PromptBundle:
    return _truncate_prompt_context(
        sample=sample,
        task_name=task_name,
        model_name=model_name,
        tokenizer=tokenizer,
        architecture=architecture,
        max_input_tokens=max_input_tokens,
        build_prompt_fn=build_prompt_fn,
    )


def run_one_sample(
    sample: dict[str, Any],
    task_name: str,
    model_name: str,
    architecture: str,
    model: Any,
    tokenizer: Any,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    build_prompt_fn: PromptBuilder,
) -> dict[str, Any]:
    prompt_bundle = build_generation_prompt(
        sample=sample,
        task_name=task_name,
        model_name=model_name,
        tokenizer=tokenizer,
        architecture=architecture,
        max_input_tokens=max_input_tokens,
        build_prompt_fn=build_prompt_fn,
    )
    inputs = _encode_generation_inputs(
        tokenizer=tokenizer,
        prompt=prompt_bundle.prompt,
        architecture=architecture,
        device=device,
        max_input_tokens=max_input_tokens,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    if do_sample:
        gen_kwargs["temperature"] = temperature

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency_sec = time.perf_counter() - start

    if architecture == "decoder-only":
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    else:
        generated_ids = outputs[0]

    raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if task_name == "qa":
        post_prediction = extract_short_answer(raw_prediction)
        references = [str(answer) for answer in sample["answers"]]
        reference = " | ".join(references)
        references_json = json.dumps(references, ensure_ascii=False)
    elif task_name == "summarization":
        post_prediction = raw_prediction.strip()
        reference = str(sample["reference"])
        references_json = ""
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return {
        "id": sample["id"],
        "task": task_name,
        "model_name": model_name,
        "model": model_name,
        "model_type": architecture,
        "architecture": architecture,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "input_tokens": inputs["input_ids"].shape[1],
        "output_tokens": len(generated_ids),
        "latency_sec": latency_sec,
        "truncated": prompt_bundle.truncated,
        "truncated_context_tokens": prompt_bundle.truncated_context_tokens,
        "raw_prediction": raw_prediction,
        "post_prediction": post_prediction,
        "prediction": post_prediction,
        "reference": reference,
        "references_json": references_json,
    }
