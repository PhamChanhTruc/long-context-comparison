import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def resolve_device(device_config="auto"):
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def load_model_and_tokenizer(
    model_name,
    architecture,
    device,
    use_4bit=False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if architecture == "encoder-decoder":
        model_kwargs = {
            "trust_remote_code": True,
        }
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

    elif architecture == "decoder-only":
        model_kwargs = {
            "trust_remote_code": True,
        }

        if use_4bit and device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cpu":
        model.to(device)

    model.eval()
    return model, tokenizer