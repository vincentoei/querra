"""Database backend abstraction for SQLite, PostgreSQL, and MySQL."""

import base64
from abc import ABC, abstractmethod
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import sqlglot

from registry import DatabaseRecord
from schema_loader import build_schema_from_sqlite, validate_db_file
from utils.execution import execute_sql
from utils.safety import is_read_only_sql

_CANONICAL_TYPES = {
    "TEXT",
    "INTEGER",
    "REAL",
    "NUMERIC",
    "BOOLEAN",
    "DATE",
}


def _canonical_type(data_type: str) -> str:
    t = data_type.lower().split("(")[0].strip()
    if t in ("text", "varchar", "character varying", "char", "nvarchar", "clob"):
        return "TEXT"
    if t in (
        "integer",
        "int",
        "smallint",
        "bigint",
        "mediumint",
        "tinyint",
        "serial",
        "bigserial",
        "smallserial",
    ):
        return "INTEGER"
    if t in ("real", "float", "double", "double precision", "numeric", "decimal"):
        return (
            "REAL"
            if t in ("real", "float", "double", "double precision")
            else "NUMERIC"
        )
    if t in ("boolean", "bool"):
        return "BOOLEAN"
    if t in (
        "date",
        "datetime",
        "timestamp",
        "timestamp without time zone",
        "timestamp with time zone",
        "time",
    ):
        return "DATE"
    return "TEXT"


def _serialize_value(value: Any) -> Any:
    """Convert non-JSON-native values to strings."""
    if value is None:
        return None
    if isinstance(value, (datetime, date, time, UUID, Decimal)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_serialize_value(v) for v in value)
    return value


def _serialize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_serialize_value(c) for c in row) for row in rows]


def _build_schema_text(
    table_columns: dict[str, list[tuple[str, str]]],
    primary_keys: dict[str, set[str]],
    foreign_keys: list[tuple[str, str, str, str]],
) -> str:
    """Build a SQLite-style CREATE TABLE schema string from introspection data."""
    statements = []
    for table_name in sorted(table_columns):
        cols = table_columns[table_name]
        col_defs = [f"    {name} {_canonical_type(t)}" for name, t in cols]
        pk = sorted(primary_keys.get(table_name, []))
        if pk:
            col_defs.append(f"    PRIMARY KEY ({', '.join(pk)})")
        statements.append(
            f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n)"
        )

    fk_lines = [
        f"-- {table}.{col} -> {ref_table}.{ref_col}"
        for table, col, ref_table, ref_col in foreign_keys
    ]
    result = "\n\n".join(statements)
    if fk_lines:
        result += "\n\n" + "\n".join(fk_lines)
    return result


def _transpile_to_dialect(sql: str, dialect: str) -> str:
    """Transpile SQLite SQL to the target dialect."""
    if dialect == "sqlite":
        return sql
    try:
        statements = sqlglot.transpile(sql, read="sqlite", write=dialect)
    except Exception as e:
        raise ValueError(f"Failed to transpile SQL to {dialect}: {e}") from e
    if not statements:
        raise ValueError("Transpilation produced no SQL")
    return "; ".join(statements)


class DatabaseBackend(ABC):
    """Abstract interface for database introspection and execution."""

    backend_type: str
    dialect: str

    @abstractmethod
    def get_schema(self) -> str:
        """Return a SQLite-style CREATE TABLE schema string for the model."""
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> None:
        """Validate the connection. Raise ValueError on failure."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, sql: str, max_rows: int | None = None) -> list[tuple]:
        """Execute a canonical SQLite SELECT query and return serialized rows."""
        raise NotImplementedError

    @staticmethod
    def from_registry(record: DatabaseRecord) -> "DatabaseBackend":
        """Create the appropriate backend from a registry record."""
        if record.backend_type == "sqlite":
            return SQLiteBackend(record.db_path)
        if record.backend_type == "postgres":
            dsn = _dsn_from_env(record.connection_env)
            return PostgresBackend(dsn)
        if record.backend_type == "mysql":
            dsn = _dsn_from_env(record.connection_env)
            return MySQLBackend(dsn)
        raise ValueError(f"Unsupported backend type: {record.backend_type}")


def _dsn_from_env(env_var: str | None) -> str:
    if not env_var:
        raise ValueError("connection_env is required for this backend")
    value = (
        env_var
        if env_var.startswith(
            ("postgresql://", "postgres://", "mysql://", "mysql+pymysql://")
        )
        else None
    )
    if not value:
        value = _env_lookup(env_var)
    if not value:
        raise ValueError(f"Environment variable {env_var} is not set or empty")
    return value


def _env_lookup(name: str) -> str | None:
    import os

    return os.environ.get(name)


class SQLiteBackend(DatabaseBackend):
    backend_type = "sqlite"
    dialect = "sqlite"

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def validate(self) -> None:
        validate_db_file(self.db_path)

    def get_schema(self) -> str:
        return build_schema_from_sqlite(self.db_path)

    def execute(self, sql: str, max_rows: int | None = None) -> list[tuple]:
        if not is_read_only_sql(sql):
            raise ValueError("Only read-only SELECT queries are allowed")
        return execute_sql(sql, self.db_path, max_rows=max_rows)


class PostgresBackend(DatabaseBackend):
    backend_type = "postgres"
    dialect = "postgres"

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def validate(self) -> None:
        try:
            with self._connect() as conn:
                conn.read_only = True
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception as e:
            raise ValueError(f"PostgreSQL connection failed: {e}") from e

    def get_schema(self) -> str:
        import psycopg

        table_columns: dict[str, list[tuple[str, str]]] = {}
        primary_keys: dict[str, set[str]] = {}
        foreign_keys: list[tuple[str, str, str, str]] = []

        try:
            with self._connect() as conn:
                conn.read_only = True
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name, column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                        ORDER BY table_name, ordinal_position
                        """
                    )
                    for table_name, column_name, data_type in cur:
                        table_columns.setdefault(table_name, []).append(
                            (column_name, data_type)
                        )

                    cur.execute(
                        """
                        SELECT tc.table_name, kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        WHERE tc.constraint_type = 'PRIMARY KEY'
                            AND tc.table_schema = current_schema()
                        """
                    )
                    for table_name, column_name in cur:
                        primary_keys.setdefault(table_name, set()).add(column_name)

                    cur.execute(
                        """
                        SELECT tc.table_name, kcu.column_name,
                               ccu.table_name AS foreign_table_name,
                               ccu.column_name AS foreign_column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage ccu
                            ON ccu.constraint_name = tc.constraint_name
                            AND ccu.table_schema = tc.table_schema
                        WHERE tc.constraint_type = 'FOREIGN KEY'
                            AND tc.table_schema = current_schema()
                        """
                    )
                    for table_name, column_name, ref_table, ref_col in cur:
                        foreign_keys.append(
                            (table_name, column_name, ref_table, ref_col)
                        )
        except psycopg.Error as e:
            raise ValueError(f"PostgreSQL schema introspection failed: {e}") from e

        return _build_schema_text(table_columns, primary_keys, foreign_keys)

    def execute(self, sql: str, max_rows: int | None = None) -> list[tuple]:
        if not is_read_only_sql(sql):
            raise ValueError("Only read-only SELECT queries are allowed")
        target_sql = _transpile_to_dialect(sql, self.dialect)

        import psycopg

        try:
            with self._connect() as conn:
                conn.read_only = True
                with conn.cursor() as cur:
                    cur.execute(target_sql)
                    rows = (
                        cur.fetchmany(max_rows)
                        if max_rows is not None
                        else cur.fetchall()
                    )
                    return _serialize_rows(rows)
        except psycopg.Error as e:
            raise ValueError(f"PostgreSQL execution failed: {e}") from e


class MySQLBackend(DatabaseBackend):
    backend_type = "mysql"
    dialect = "mysql"

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._connect_kwargs = self._parse_dsn(dsn)

    @staticmethod
    def _parse_dsn(dsn: str) -> dict[str, Any]:
        parsed = urlparse(dsn)
        if parsed.scheme not in ("mysql", "mysql+pymysql"):
            raise ValueError(f"Invalid MySQL DSN scheme: {parsed.scheme}")
        return {
            "host": parsed.hostname,
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip("/"),
            "charset": "utf8mb4",
        }

    def _connect(self):
        import pymysql

        return pymysql.connect(**self._connect_kwargs)

    def validate(self) -> None:
        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            finally:
                conn.close()
        except Exception as e:
            raise ValueError(f"MySQL connection failed: {e}") from e

    def get_schema(self) -> str:
        import pymysql

        table_columns: dict[str, list[tuple[str, str]]] = {}
        primary_keys: dict[str, set[str]] = {}
        foreign_keys: list[tuple[str, str, str, str]] = []

        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT DATABASE()")
                    database = cur.fetchone()[0]
                    if not database:
                        raise ValueError("MySQL database not selected in DSN")

                    cur.execute(
                        """
                        SELECT table_name, column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = %s
                        ORDER BY table_name, ordinal_position
                        """,
                        (database,),
                    )
                    for table_name, column_name, data_type in cur:
                        table_columns.setdefault(table_name, []).append(
                            (column_name, data_type)
                        )

                    cur.execute(
                        """
                        SELECT table_name, column_name
                        FROM information_schema.key_column_usage
                        WHERE table_schema = %s AND constraint_name = 'PRIMARY'
                        """,
                        (database,),
                    )
                    for table_name, column_name in cur:
                        primary_keys.setdefault(table_name, set()).add(column_name)

                    cur.execute(
                        """
                        SELECT table_name, column_name, referenced_table_name, referenced_column_name
                        FROM information_schema.key_column_usage
                        WHERE table_schema = %s AND referenced_table_name IS NOT NULL
                        """,
                        (database,),
                    )
                    foreign_keys.extend(cur)
            finally:
                conn.close()
        except pymysql.Error as e:
            raise ValueError(f"MySQL schema introspection failed: {e}") from e

        return _build_schema_text(table_columns, primary_keys, foreign_keys)

    def execute(self, sql: str, max_rows: int | None = None) -> list[tuple]:
        if not is_read_only_sql(sql):
            raise ValueError("Only read-only SELECT queries are allowed")
        target_sql = _transpile_to_dialect(sql, self.dialect)

        import pymysql

        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SET SESSION TRANSACTION READ ONLY")
                    cur.execute(target_sql)
                    rows = (
                        cur.fetchmany(max_rows)
                        if max_rows is not None
                        else cur.fetchall()
                    )
                    return _serialize_rows(rows)
            finally:
                conn.close()
        except pymysql.Error as e:
            raise ValueError(f"MySQL execution failed: {e}") from e


if __name__ == "__main__":
    # Basic schema text builder sanity check.
    schema = _build_schema_text(
        {
            "users": [("id", "bigint"), ("name", "varchar")],
            "orders": [("id", "integer"), ("user_id", "integer"), ("total", "numeric")],
        },
        {"users": {"id"}, "orders": {"id"}},
        [("orders", "user_id", "users", "id")],
    )
    assert "CREATE TABLE users" in schema
    assert "CREATE TABLE orders" in schema
    assert "PRIMARY KEY (id)" in schema
    assert "orders.user_id -> users.id" in schema
    print("Schema builder sanity check passed.")
