"""SQLite registry for registered databases."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DatabaseRecord:
    db_id: str
    display_name: str
    backend_type: str
    db_path: str | None
    connection_env: str | None
    schema_text: str
    description: str
    created_at: datetime
    updated_at: datetime


class DatabaseRegistry:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS databases (
                    db_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    backend_type TEXT NOT NULL DEFAULT 'sqlite',
                    db_path TEXT,
                    connection_env TEXT,
                    schema_text TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    db_id TEXT NOT NULL,
                    question TEXT,
                    generated_sql TEXT,
                    edited_sql TEXT,
                    execution_result TEXT,
                    execution_error TEXT,
                    latency_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (db_id) REFERENCES databases(db_id)
                )
                """
            )
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Migrate older registry schemas to the current layout."""
        cur = conn.execute("PRAGMA table_info(databases)")
        columns = {row[1] for row in cur.fetchall()}
        required = {"backend_type", "connection_env"}
        if required.issubset(columns):
            return

        # Legacy table: recreate with the new schema and migrate data.
        conn.execute("ALTER TABLE databases RENAME TO databases_old")
        conn.execute(
            """
            CREATE TABLE databases (
                db_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                backend_type TEXT NOT NULL DEFAULT 'sqlite',
                db_path TEXT,
                connection_env TEXT,
                schema_text TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO databases
            (db_id, display_name, backend_type, db_path, connection_env, schema_text, description, created_at, updated_at)
            SELECT db_id, display_name, 'sqlite', db_path, NULL, schema_text, description, created_at, updated_at
            FROM databases_old
            """
        )
        conn.execute("DROP TABLE databases_old")

    def list_databases(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT db_id, display_name, backend_type, description, created_at, updated_at FROM databases ORDER BY display_name"
            ).fetchall()
            return [dict(row) for row in rows]

    def list_databases_admin(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT db_id, display_name, backend_type, db_path, connection_env, schema_text, description, created_at, updated_at FROM databases ORDER BY display_name"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_database(self, db_id: str) -> DatabaseRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM databases WHERE db_id = ?", (db_id,)
            ).fetchone()
            if row is None:
                return None
            return DatabaseRecord(**dict(row))

    def register_database(
        self,
        db_id: str,
        display_name: str,
        schema_text: str,
        description: str = "",
        backend_type: str = "sqlite",
        db_path: str | None = None,
        connection_env: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO databases
                (db_id, display_name, backend_type, db_path, connection_env, schema_text, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(db_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    backend_type=excluded.backend_type,
                    db_path=excluded.db_path,
                    connection_env=excluded.connection_env,
                    schema_text=excluded.schema_text,
                    description=excluded.description,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    db_id,
                    display_name,
                    backend_type,
                    db_path,
                    connection_env,
                    schema_text,
                    description,
                ),
            )
            conn.commit()

    def delete_database(self, db_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM databases WHERE db_id = ?", (db_id,))
            conn.commit()
            return cursor.rowcount > 0

    def update_schema(self, db_id: str, schema_text: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE databases SET schema_text = ?, updated_at = CURRENT_TIMESTAMP WHERE db_id = ?",
                (schema_text, db_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def record_query(
        self,
        db_id: str,
        question: str | None,
        generated_sql: str | None,
        edited_sql: str | None,
        execution_result: str | None,
        execution_error: str | None,
        latency_ms: float | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_history
                (db_id, question, generated_sql, edited_sql, execution_result, execution_error, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    db_id,
                    question,
                    generated_sql,
                    edited_sql,
                    execution_result,
                    execution_error,
                    latency_ms,
                ),
            )
            conn.commit()

    def get_history(
        self, db_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if db_id:
                rows = conn.execute(
                    "SELECT * FROM query_history WHERE db_id = ? ORDER BY created_at DESC LIMIT ?",
                    (db_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
