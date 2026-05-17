from __future__ import annotations

import importlib.util

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def resolve_device(device_config: str = "auto") -> str:
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_config == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false.")
    return device_config


def _check_4bit_support(device: str) -> None:
    if device != "cuda":
        raise RuntimeError("4-bit quantization requires a CUDA device.")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "4-bit quantization requires bitsandbytes. Install it with "
            "`pip install bitsandbytes`, or set model.use_4bit: false."
        )


def load_model_and_tokenizer(
    model_name: str,
    architecture: str,
    device: str,
    use_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs: dict[str, object] = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    if architecture == "encoder-decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    elif architecture == "decoder-only":
        if use_4bit:
            _check_4bit_support(device)
            from transformers import BitsAndBytesConfig

            model_kwargs.pop("torch_dtype", None)
            model_kwargs["device_map"] = "auto"
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        raise ValueError(
            "Unsupported architecture. Expected 'encoder-decoder' or 'decoder-only', "
            f"got {architecture!r}."
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cpu":
        model.to(device)

    model.eval()
    return model, tokenizer
