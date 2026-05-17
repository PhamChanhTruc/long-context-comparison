from scripts.run_inference import load_config, validate_config
from src.loaders import load_jsonl, validate_dataset


def test_load_and_validate_data() -> None:
    qa_data = load_jsonl("data/processed/pilot_qa.jsonl")
    sum_data = load_jsonl("data/processed/pilot_sum.jsonl")

    validate_dataset(qa_data, task_name="qa")
    validate_dataset(sum_data, task_name="summarization")

    assert len(qa_data) == 5
    assert len(sum_data) == 5
    assert qa_data[1]["answers"] == ["the Earth", "Earth"]


def test_config_loading() -> None:
    config = load_config("configs/flan_qa.yaml")
    validate_config(config)
    assert config["task"]["name"] == "qa"


def main() -> None:
    test_load_and_validate_data()
    test_config_loading()
    print("OK: JSONL loading, schema validation, and config loading passed.")


if __name__ == "__main__":
    main()
