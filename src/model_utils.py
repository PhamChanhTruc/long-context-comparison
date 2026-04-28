import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def resolve_device(device_config="auto"):
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def load_model_and_tokenizer(model_name, architecture, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if architecture == "encoder-decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif architecture == "decoder-only":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer