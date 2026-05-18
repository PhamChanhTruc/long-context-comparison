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


@dataclass(frozen=True)
class EncodedInputs:
    tensors: dict[str, torch.Tensor]
    input_tokens: int
    truncated: bool


def _uses_chat_template(tokenizer: Any, architecture: str) -> bool:
    return architecture == "decoder-only" and bool(getattr(tokenizer, "chat_template", None))


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


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
    input_tokens = _generation_prompt_token_count(
        tokenizer=tokenizer,
        prompt=prompt,
        architecture=architecture,
        max_input_tokens=max_input_tokens,
    )
    if input_tokens <= max_input_tokens:
        return PromptBundle(
            prompt=prompt,
            input_tokens=input_tokens,
            truncated=False,
            truncated_context_tokens=0,
        )

    empty_prompt = _build_prompt(sample, task_name, model_name, build_prompt_fn, "")
    empty_prompt_tokens = _generation_prompt_token_count(
        tokenizer=tokenizer,
        prompt=empty_prompt,
        architecture=architecture,
        max_input_tokens=max_input_tokens,
    )
    context_token_ids = tokenizer.encode(context, add_special_tokens=False)
    if empty_prompt_tokens >= max_input_tokens:
        LOGGER.warning(
            "Truncated all context for %s sample %s, but the fixed prompt/question "
            "still has %s tokens for max_input_tokens=%s; final tokenizer truncation "
            "will cap the input.",
            task_name,
            sample.get("id", "<unknown>"),
            empty_prompt_tokens,
            max_input_tokens,
        )
        return PromptBundle(
            prompt=empty_prompt,
            input_tokens=empty_prompt_tokens,
            truncated=True,
            truncated_context_tokens=len(context_token_ids),
        )

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
        candidate_len = _generation_prompt_token_count(
            tokenizer=tokenizer,
            prompt=candidate_prompt,
            architecture=architecture,
            max_input_tokens=max_input_tokens,
        )
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


def _input_ids_length(encoded: dict[str, torch.Tensor]) -> int:
    input_ids = encoded["input_ids"]
    if input_ids.ndim == 1:
        return input_ids.shape[0]
    return input_ids.shape[1]


def _normalise_encoded(encoded: Any) -> dict[str, torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        return {
            "input_ids": encoded,
            "attention_mask": torch.ones_like(encoded),
        }
    return dict(encoded)


def _slice_encoded_to_max(
    encoded: dict[str, torch.Tensor],
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            sliced[key] = value[:, :max_input_tokens]
        elif isinstance(value, torch.Tensor) and value.ndim == 1:
            sliced[key] = value[:max_input_tokens]
        else:
            sliced[key] = value
    return sliced


def _tokenize_chat_prompt(
    tokenizer: Any,
    prompt: str,
    truncation: bool,
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if truncation:
        kwargs.update({"truncation": True, "max_length": max_input_tokens})

    try:
        return _normalise_encoded(
            tokenizer.apply_chat_template(_chat_messages(prompt), **kwargs)
        )
    except TypeError:
        kwargs.pop("return_dict", None)

    try:
        return _normalise_encoded(
            tokenizer.apply_chat_template(_chat_messages(prompt), **kwargs)
        )
    except TypeError:
        if truncation:
            kwargs.pop("truncation", None)
            kwargs.pop("max_length", None)
        encoded = _normalise_encoded(
            tokenizer.apply_chat_template(_chat_messages(prompt), **kwargs)
        )
        if truncation:
            return _slice_encoded_to_max(encoded, max_input_tokens)
        return encoded


def _tokenize_plain_prompt(
    tokenizer: Any,
    prompt: str,
    truncation: bool,
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"return_tensors": "pt", "truncation": truncation}
    if truncation:
        kwargs["max_length"] = max_input_tokens

    try:
        return _normalise_encoded(tokenizer(prompt, **kwargs))
    except TypeError:
        if truncation:
            kwargs.pop("max_length", None)
        encoded = _normalise_encoded(tokenizer(prompt, **kwargs))
        if truncation:
            return _slice_encoded_to_max(encoded, max_input_tokens)
        return encoded


def _tokenize_generation_prompt(
    tokenizer: Any,
    prompt: str,
    architecture: str,
    truncation: bool,
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    if _uses_chat_template(tokenizer, architecture):
        return _tokenize_chat_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            truncation=truncation,
            max_input_tokens=max_input_tokens,
        )
    return _tokenize_plain_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        truncation=truncation,
        max_input_tokens=max_input_tokens,
    )


def _generation_prompt_token_count(
    tokenizer: Any,
    prompt: str,
    architecture: str,
    max_input_tokens: int,
) -> int:
    encoded = _tokenize_generation_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        architecture=architecture,
        truncation=False,
        max_input_tokens=max_input_tokens,
    )
    return _input_ids_length(encoded)


def _encode_generation_inputs(
    tokenizer: Any,
    prompt: str,
    architecture: str,
    device: str,
    max_input_tokens: int,
) -> EncodedInputs:
    encoded = _tokenize_generation_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        architecture=architecture,
        truncation=False,
        max_input_tokens=max_input_tokens,
    )

    input_tokens = _input_ids_length(encoded)
    if input_tokens > max_input_tokens:
        LOGGER.warning(
            "Final tokenized prompt has %s tokens, exceeding max_input_tokens=%s; "
            "applying tokenizer truncation before generation.",
            input_tokens,
            max_input_tokens,
        )
        encoded = _tokenize_generation_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            architecture=architecture,
            truncation=True,
            max_input_tokens=max_input_tokens,
        )
        if _input_ids_length(encoded) > max_input_tokens:
            encoded = _slice_encoded_to_max(encoded, max_input_tokens)
        return EncodedInputs(
            tensors={key: value.to(device) for key, value in encoded.items()},
            input_tokens=_input_ids_length(encoded),
            truncated=True,
        )

    return EncodedInputs(
        tensors={key: value.to(device) for key, value in encoded.items()},
        input_tokens=input_tokens,
        truncated=False,
    )


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
    encoded_inputs = _encode_generation_inputs(
        tokenizer=tokenizer,
        prompt=prompt_bundle.prompt,
        architecture=architecture,
        device=device,
        max_input_tokens=max_input_tokens,
    )
    inputs = encoded_inputs.tensors

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
        "input_tokens": encoded_inputs.input_tokens,
        "output_tokens": len(generated_ids),
        "latency_sec": latency_sec,
        "truncated": prompt_bundle.truncated or encoded_inputs.truncated,
        "truncated_context_tokens": prompt_bundle.truncated_context_tokens,
        "raw_prediction": raw_prediction,
        "post_prediction": post_prediction,
        "prediction": post_prediction,
        "reference": reference,
        "references_json": references_json,
    }
