def build_prompt(sample, task_name):
    if task_name == "qa":
        return f"""Read the following context and answer the question using only information from the context.

Give a short answer only.
Do not explain.
Do not repeat the question.
If the answer is not clearly supported by the context, say: unsupported.

Context:
{sample['context']}

Question:
{sample['question']}

Short Answer:
"""
    elif task_name == "summarization":
        return f"""Summarize the following document in 2-4 sentences.

Focus only on the main findings.
Do not copy long parts verbatim.
Write a concise summary.

Document:
{sample['context']}

Summary:
"""
    else:
        raise ValueError(f"Unsupported task_name: {task_name}")