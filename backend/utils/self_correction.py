"""Self-correction loop for generated SQL."""

import sqlglot

from utils.inference import generate_sql
from utils.postprocess import normalize_sql_to_schema
from utils.prompts import extract_sql


SYSTEM = (
    "You are a SQL expert. The previous SQLite query failed. "
    "Given the database schema, the original question, the failed query, and the error message, "
    "write a corrected SQLite SQL query. Output only the SQL query, without explanations or markdown."
)


def _execution_error(sql: str, db_id: str, db_dir) -> str | None:
    """Return the execution error string, or None if the query executes successfully."""
    from evaluation.metrics import _get_db_path, execute_sql

    db_path = _get_db_path(db_id, db_dir)
    if db_path is None:
        return "No database found"
    try:
        execute_sql(sql, db_path)
        return None
    except Exception as e:
        return str(e)


def _validation_error(sql: str) -> str | None:
    """Return the SQL validation error string, or None if the query parses."""
    try:
        sqlglot.parse(sql, read="sqlite")
        return None
    except Exception as e:
        return str(e)


def build_correction_prompt(tokenizer, schema: str, question: str, sql: str, error: str) -> str:
    """Build a chat prompt asking the model to fix the failed SQL."""
    user_text = (
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"Failed query: {sql}\n\n"
        f"Error: {error}\n\n"
        "Write the corrected SQL query:"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def maybe_correct(
    model,
    tokenizer,
    schema: str,
    question: str,
    sql: str,
    db_id: str,
    db_dir,
    max_retries: int = 2,
):
    """Retry a failed SQL query up to max_retries times.

    Returns the final SQL and the number of retries used (0 if the original was already correct).
    """
    err = _validation_error(sql) or _execution_error(sql, db_id, db_dir)
    if err is None:
        return sql, 0

    for retry in range(max_retries):
        retry_prompt = build_correction_prompt(tokenizer, schema, question, sql, err)
        raw = generate_sql(model, tokenizer, retry_prompt)
        sql = extract_sql(raw)
        sql = normalize_sql_to_schema(sql, schema)
        err = _validation_error(sql) or _execution_error(sql, db_id, db_dir)
        if err is None:
            return sql, retry + 1

    return sql, max_retries
