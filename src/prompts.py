def build_prompt(sample, task_name):
    if task_name == "qa":
        return f"""Read the following context and answer the question based only on the provided text.

Context:
{sample['context']}

Question:
{sample['question']}

Answer:
"""
    elif task_name == "summarization":
        return f"""Summarize the following document concisely while preserving the main points.

Document:
{sample['context']}

Summary:
"""
    else:
        raise ValueError(f"Unsupported task_name: {task_name}")