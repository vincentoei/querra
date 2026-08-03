"""Self-correction loop for generated SQL."""

from db_backends import DatabaseBackend
from utils.inference import generate_sql
from utils.postprocess import normalize_sql_to_schema
from utils.prompts import extract_sql
from utils.safety import is_read_only_sql

SYSTEM = (
    "You are a SQL expert. The previous SQLite query failed. "
    "Given the database schema, the original question, the failed query, and the error message, "
    "write a corrected SQLite SQL query. Output only the SQL query, without explanations or markdown."
)


def _execution_error(sql: str, backend: DatabaseBackend) -> str | None:
    """Return the execution error string, or None if the query executes successfully."""
    try:
        backend.execute(sql)
        return None
    except Exception as e:  # noqa: BLE001 - any backend execution error is the correction signal
        return str(e)


def build_correction_prompt(
    tokenizer, schema: str, question: str, sql: str, error: str
) -> str:
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
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def maybe_correct(
    model,
    tokenizer,
    schema: str,
    question: str,
    sql: str,
    db_id: str | None,
    backend: DatabaseBackend,
    max_retries: int = 2,
):
    """Retry a failed SQL query up to max_retries times.

    Returns the final SQL and the number of retries used (0 if the original was already correct).
    """
    if not is_read_only_sql(sql):
        err = "SQL is not read-only or invalid"
    else:
        err = _execution_error(sql, backend)
    if err is None:
        return sql, 0

    for retry in range(max_retries):
        retry_prompt = build_correction_prompt(tokenizer, schema, question, sql, err)
        raw = generate_sql(model, tokenizer, retry_prompt)
        sql = extract_sql(raw)
        sql = normalize_sql_to_schema(sql, schema)
        if not is_read_only_sql(sql):
            err = "Corrected SQL is not read-only or invalid"
            continue
        err = _execution_error(sql, backend)
        if err is None:
            return sql, retry + 1

    return sql, max_retries
