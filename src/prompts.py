def build_prompt(sample, task_name, model_name=None):
    model_name = (model_name or "").lower()

    if task_name == "qa":
        # Prompt riêng cho LongT5
        if "long-t5" in model_name or "longt5" in model_name:
            return f"""You are answering a question from a long document.

Read the context carefully and answer the question using only information that is explicitly stated in the context.

Instructions:
- Extract the shortest possible answer span from the context.
- Prefer copying the exact words from the context when possible.
- Answer with a short phrase only.
- Use at most 8 words.
- Do not explain.
- Do not summarize the document.
- Do not repeat the question.
- If the answer is not clearly supported by the context, output exactly: unsupported

Context:
{sample['context']}

Question:
{sample['question']}

Short Answer:
"""
        # Prompt QA chung cho các model khác
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