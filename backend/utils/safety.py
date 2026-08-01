"""Security and validation helpers for generated SQL and database access."""

import re
from pathlib import Path

import sqlglot


DB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_db_id(db_id: str) -> bool:
    """Return True if db_id is a safe directory name component."""
    return bool(DB_ID_PATTERN.fullmatch(db_id))


def validate_db_path(db_id: str, db_dir: Path) -> Path | None:
    """Return a verified database path inside db_dir, or None if invalid."""
    if not validate_db_id(db_id):
        return None
    subdir = db_dir / db_id
    if not subdir.is_dir():
        return None
    # Resolve the directory to defend against symlink tricks.
    resolved = subdir.resolve()
    resolved_root = db_dir.resolve()
    if not str(resolved).startswith(str(resolved_root) + "/") and resolved != resolved_root:
        return None

    candidate = resolved / f"{db_id}.sqlite"
    if candidate.exists():
        return candidate
    candidate = resolved / f"{db_id}.db"
    if candidate.exists():
        return candidate
    files = list(resolved.glob("*.sqlite")) + list(resolved.glob("*.db"))
    return files[0] if files else None


def contains_load_extension(sql: str) -> bool:
    """Detect load_extension(...) function calls in the SQL string."""
    return bool(re.search(r"\bload_extension\s*\(", sql, re.IGNORECASE))


def is_read_only_sql(sql: str) -> bool:
    """Return True if SQL is a valid, read-only SELECT-like query.

    Blocks any statement that is not a SELECT/CTE/UNION/INTERSECT/EXCEPT Query,
    blocks SQL that cannot be parsed, and blocks load_extension(...) calls.
    """
    if contains_load_extension(sql):
        return False
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return False
    if not statements:
        return False
    for stmt in statements:
        if not isinstance(stmt, sqlglot.exp.Query):
            return False
    return True


def extract_first_query(text: str) -> str:
    """Return the first valid SELECT-like query from the model output.

    Finds SQL inside ```sql``` fences (anywhere in the text), or falls back to
    the first line if no valid query is found.
    """
    import re

    text = text.strip()

    # Look for SQL inside markdown code blocks (anywhere in the text).
    code_block = re.search(r"```(?:sql)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block:
        code = code_block.group(1).strip()
    else:
        # Strip leading/trailing fences (legacy edge case).
        if text.startswith("```sql"):
            text = text[6:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        code = text.strip()

    try:
        statements = sqlglot.parse(code, read="sqlite")
    except Exception:
        statements = []

    for stmt in statements:
        if isinstance(stmt, sqlglot.exp.Query):
            return stmt.sql(dialect="sqlite")

    # Fallback to the first line if no valid query is found.
    return code.split("\n")[0].strip()
