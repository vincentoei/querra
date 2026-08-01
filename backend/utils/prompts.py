"""Prompt formatting for Text-to-SQL."""

from utils.safety import extract_first_query


SYSTEM = (
    "You are a SQL expert. Given the database schema, write a single correct SQLite SQL query "
    "that answers the question. Output only the SQL query, without explanations or markdown."
)


def _build_user_text(schema: str, question: str, examples: list[dict] | None = None) -> str:
    parts = []
    if examples:
        parts.append("Here are some example question-query pairs:")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:\nQuestion: {ex['question']}\nQuery: {ex['query']}")
        parts.append("Now write the query for the following schema and question.")
    parts.append(f"Schema:\n{schema}\n\nQuestion: {question}\n\nQuery:")
    return "\n\n".join(parts)


def format_zero_shot(tokenizer, schema: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_user_text(schema, question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_few_shot(tokenizer, schema: str, question: str, examples: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_user_text(schema, question, examples)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_sql(output: str) -> str:
    """Strip chat template artifacts and return the first valid SQL query."""
    return extract_first_query(output)
