def build_prompt(sample, task_name):
    if task_name == "qa":
        return f"""Read the following context and answer the question using only information from the context.

Instructions:
- Give a short answer only.
- Prefer the shortest exact answer span from the context.
- Do not explain.
- Do not summarize the document.
- Do not repeat the question.
- If the answer is not clearly supported by the context, output exactly: unsupported.

Context:
{sample['context']}

Question:
{sample['question']}

Answer:
"""
    elif task_name == "summarization":
        return f"""Summarize the following document in 2-4 sentences.

Instructions:
- Focus on the main findings only.
- Keep the summary concise and factual.
- Do not copy long parts verbatim.
- Do not add information not supported by the document.

Document:
{sample['context']}

Summary:
"""
    else:
        raise ValueError(f"Unsupported task_name: {task_name}")