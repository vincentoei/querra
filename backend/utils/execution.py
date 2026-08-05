"""Shared SQLite execution utilities for the API and evaluation."""

import sqlite3
from pathlib import Path

from utils.safety import is_read_only_sql, validate_db_path


def get_db_path(db_id: str, db_dir: Path) -> Path | None:
    """Return a verified SQLite database path inside db_dir."""
    return validate_db_path(db_id, db_dir)


def execute_sql(
    sql: str,
    db_path: Path,
    *,
    timeout: float = 10.0,
    max_rows: int | None = None,
    case_sensitive_like: bool = False,
) -> tuple[list[tuple], list[str]]:
    """Execute a read-only SQL query against a SQLite DB and return rows.

    Returns a ``(rows, columns)`` tuple where ``columns`` is the list of
    column names from the cursor description (empty for non-SELECT queries).

    Raises ValueError if the SQL is not read-only, and other exceptions on
    execution errors.
    """
    if not is_read_only_sql(sql):
        raise ValueError("Only read-only SELECT queries are allowed")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA load_extension=OFF")
    conn.row_factory = sqlite3.Row
    try:
        if not case_sensitive_like:
            conn.execute("PRAGMA case_sensitive_like=OFF")
        cur = conn.execute(sql)
        if max_rows is not None:
            rows = cur.fetchmany(max_rows)
        else:
            rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        return [tuple(row) for row in rows], columns
    finally:
        conn.close()
