import json

import torch

from src.inference import build_generation_prompt, run_one_sample
from src.metrics_qa import clean_qa_prediction, exact_match, f1_score
from src.prompts import build_prompt


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None
    model_max_length = 512

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inverse_vocab: dict[int, str] = {}
        self._next_id = 10

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._inverse_vocab[self._next_id] = token
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text, add_special_tokens=True):
        token_ids = [self._token_id(token) for token in str(text).split()]
        if add_special_tokens:
            return [self.eos_token_id] + token_ids
        return token_ids

    def __call__(
        self,
        text,
        return_tensors="pt",
        truncation=False,
        max_length=None,
    ):
        assert return_tensors == "pt"
        token_ids = self.encode(text)
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del clean_up_tokenization_spaces
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token_ids = [int(token_id) for token_id in token_ids]
        if token_ids == [9001, 9002, 9003, 9004, 9005]:
            return "Answer: Paris. Do not explain."
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            token = self._inverse_vocab.get(token_id)
            if token:
                tokens.append(token)
        return " ".join(tokens)


class InflatingTokenizer(FakeTokenizer):
    def __call__(
        self,
        text,
        return_tensors="pt",
        truncation=False,
        max_length=None,
    ):
        assert return_tensors == "pt"
        token_ids = self.encode(text)
        token_ids = token_ids + token_ids
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        truncation=False,
        max_length=None,
    ):
        assert tokenize is True
        text = " ".join(str(message["content"]) for message in messages)
        if add_generation_prompt:
            text = f"{text} Assistant:"
        token_ids = self.encode(text)
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        if return_tensors == "pt":
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            if return_dict:
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
            return input_ids
        return token_ids


class FakeModel:
    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[9001, 9002, 9003, 9004, 9005]], dtype=torch.long)


def test_qa_metrics_multiple_answers() -> None:
    assert exact_match("Paris", ["Paris"]) == 1
    assert exact_match("Paris", ["The city of Paris", "Paris"]) == 1
    assert exact_match("London", ["Paris"]) == 0
    assert f1_score("Paris", ["Paris"]) == 1.0
    assert f1_score("Earth", ["the Earth", "Moon"]) == 1.0


def test_post_processing_removes_instruction_echoes() -> None:
    assert clean_qa_prediction("Answer: Paris. Do not explain.") == "Paris"
    assert clean_qa_prediction("The answer is London.") == "London"


def test_prediction_output_keeps_all_gold_answers() -> None:
    sample = {
        "id": "qa_test",
        "task": "qa",
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answers": ["Paris", "City of Paris"],
    }
    output = run_one_sample(
        sample=sample,
        task_name="qa",
        model_name="fake-model",
        architecture="encoder-decoder",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        max_input_tokens=128,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        build_prompt_fn=build_prompt,
    )

    assert output["raw_prediction"] == "Answer: Paris. Do not explain."
    assert output["post_prediction"] == "Paris"
    assert json.loads(output["references_json"]) == ["Paris", "City of Paris"]


def test_long_qa_context_truncates_and_preserves_question() -> None:
    question = "What is the important answer?"
    sample = {
        "id": "qa_long",
        "task": "qa",
        "context": " ".join(f"context_{index}" for index in range(200)),
        "question": question,
        "answers": ["important answer"],
    }
    tokenizer = FakeTokenizer()

    prompt_bundle = build_generation_prompt(
        sample=sample,
        task_name="qa",
        model_name="fake-model",
        tokenizer=tokenizer,
        architecture="decoder-only",
        max_input_tokens=48,
        build_prompt_fn=build_prompt,
    )
    output = run_one_sample(
        sample=sample,
        task_name="qa",
        model_name="fake-model",
        architecture="decoder-only",
        model=FakeModel(),
        tokenizer=tokenizer,
        device="cpu",
        max_input_tokens=48,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        build_prompt_fn=build_prompt,
    )

    assert prompt_bundle.truncated is True
    assert prompt_bundle.input_tokens <= 48
    assert question in prompt_bundle.prompt
    assert output["input_tokens"] <= 48


def test_long_qa_context_truncates_chat_prompt_and_preserves_question() -> None:
    question = "Which phrase must survive?"
    sample = {
        "id": "qa_long_chat",
        "task": "qa",
        "context": " ".join(f"context_{index}" for index in range(200)),
        "question": question,
        "answers": ["survive"],
    }
    tokenizer = FakeChatTokenizer()

    prompt_bundle = build_generation_prompt(
        sample=sample,
        task_name="qa",
        model_name="fake-chat-model",
        tokenizer=tokenizer,
        architecture="decoder-only",
        max_input_tokens=52,
        build_prompt_fn=build_prompt,
    )
    output = run_one_sample(
        sample=sample,
        task_name="qa",
        model_name="fake-chat-model",
        architecture="decoder-only",
        model=FakeModel(),
        tokenizer=tokenizer,
        device="cpu",
        max_input_tokens=52,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        build_prompt_fn=build_prompt,
    )

    assert prompt_bundle.truncated is True
    assert prompt_bundle.input_tokens <= 52
    assert question in prompt_bundle.prompt
    assert output["input_tokens"] <= 52


def test_long_summarization_context_truncates_to_max_input_tokens() -> None:
    sample = {
        "id": "sum_long",
        "task": "summarization",
        "context": " ".join(f"document_{index}" for index in range(200)),
        "reference": "Short summary.",
    }
    output = run_one_sample(
        sample=sample,
        task_name="summarization",
        model_name="fake-model",
        architecture="encoder-decoder",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        max_input_tokens=40,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        build_prompt_fn=build_prompt,
    )

    assert output["input_tokens"] <= 40
    assert output["truncated"] is True


def test_final_encoding_truncation_caps_unexpected_tokenizer_growth() -> None:
    sample = {
        "id": "qa_final_guard",
        "task": "qa",
        "context": "Paris is the capital of France.",
        "question": "What is the capital?",
        "answers": ["Paris"],
    }
    output = run_one_sample(
        sample=sample,
        task_name="qa",
        model_name="fake-model",
        architecture="encoder-decoder",
        model=FakeModel(),
        tokenizer=InflatingTokenizer(),
        device="cpu",
        max_input_tokens=48,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        build_prompt_fn=build_prompt,
    )

    assert output["input_tokens"] <= 48
    assert output["truncated"] is True


def main() -> None:
    test_qa_metrics_multiple_answers()
    test_post_processing_removes_instruction_echoes()
    test_prediction_output_keeps_all_gold_answers()
    test_long_qa_context_truncates_and_preserves_question()
    test_long_qa_context_truncates_chat_prompt_and_preserves_question()
    test_long_summarization_context_truncates_to_max_input_tokens()
    test_final_encoding_truncation_caps_unexpected_tokenizer_growth()
    print("OK: QA metrics, post-processing, and prediction output tests passed.")


if __name__ == "__main__":
    main()
