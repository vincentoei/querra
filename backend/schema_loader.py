"""Introspect a SQLite database and produce a CREATE TABLE schema string."""

import sqlite3
from pathlib import Path
from typing import Any

_TYPE_MAP = {
    "integer": "INTEGER",
    "int": "INTEGER",
    "tinyint": "INTEGER",
    "smallint": "INTEGER",
    "mediumint": "INTEGER",
    "bigint": "INTEGER",
    "real": "REAL",
    "float": "REAL",
    "double": "REAL",
    "numeric": "NUMERIC",
    "decimal": "NUMERIC",
    "boolean": "INTEGER",
    "blob": "BLOB",
}


def _map_sqlite_type(type_str: str) -> str:
    t = type_str.strip().split("(")[0].lower()
    return _TYPE_MAP.get(t, "TEXT")


def _table_names(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]


def _table_info(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
    return [dict(row) for row in cursor.fetchall()]


def _foreign_keys(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    cursor = conn.execute(f'PRAGMA foreign_key_list("{table_name}")')
    return [dict(row) for row in cursor.fetchall()]


def build_schema_from_sqlite(db_path: Path | str) -> str:
    """Return a CREATE TABLE schema string for the SQLite database at db_path."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        statements = []
        fk_comments = []
        for table_name in _table_names(conn):
            columns = _table_info(conn, table_name)
            if not columns:
                continue

            pk_cols = [col["name"] for col in columns if col["pk"]]
            # If no explicit PK, SQLite has rowid; do not add synthetic PK.
            col_defs = []
            for col in columns:
                line = f"    {col['name']} {_map_sqlite_type(col['type'])}"
                if col["notnull"]:
                    line += " NOT NULL"
                col_defs.append(line)
            if pk_cols:
                col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

            create = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
            statements.append(create)

            for fk in _foreign_keys(conn, table_name):
                fk_comments.append(
                    f"-- {table_name}.{fk['from']} -> {fk['table']}.{fk['to']}"
                )

        result = "\n\n".join(statements)
        if fk_comments:
            result += "\n\n" + "\n".join(fk_comments)
        return result
    finally:
        conn.close()


def validate_db_file(db_path: Path | str) -> None:
    """Raise ValueError if the file is not a readable SQLite database."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise ValueError(f"Not a file: {db_path}")
    header = db_path.read_bytes()[:16]
    if header != b"SQLite format 3\x00":
        raise ValueError(f"Not a valid SQLite database: {db_path}")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute("SELECT 1")
        conn.close()
    except sqlite3.Error as e:
        raise ValueError(f"Not a valid SQLite database: {e}")
