import json

import torch

from src.inference import run_one_sample
from src.metrics_qa import clean_qa_prediction, exact_match, f1_score
from src.prompts import build_prompt


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None
    model_max_length = 512

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(max(1, len(str(text).split()))))

    def __call__(self, text, return_tensors="pt", truncation=False):
        del truncation
        assert return_tensors == "pt"
        input_ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del token_ids, skip_special_tokens, clean_up_tokenization_spaces
        return "Answer: Paris. Do not explain."


class FakeModel:
    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[7, 8, 9]], dtype=torch.long)


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


def main() -> None:
    test_qa_metrics_multiple_answers()
    test_post_processing_removes_instruction_echoes()
    test_prediction_output_keeps_all_gold_answers()
    print("OK: QA metrics, post-processing, and prediction output tests passed.")


if __name__ == "__main__":
    main()
