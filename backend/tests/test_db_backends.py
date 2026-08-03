"""Tests for database backend abstraction and transpilation."""

import sqlite3
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from db_backends import (
    DatabaseBackend,
    MySQLBackend,
    PostgresBackend,
    SQLiteBackend,
    _build_schema_text,
    _canonical_type,
    _serialize_value,
    _transpile_to_dialect,
)
from registry import DatabaseRecord


def test_canonical_type_mapping():
    assert _canonical_type("varchar") == "TEXT"
    assert _canonical_type("INTEGER") == "INTEGER"
    assert _canonical_type("bigint") == "INTEGER"
    assert _canonical_type("numeric") == "NUMERIC"
    assert _canonical_type("real") == "REAL"
    assert _canonical_type("boolean") == "BOOLEAN"
    assert _canonical_type("timestamp") == "DATE"
    assert _canonical_type("geometry") == "TEXT"


def test_serialize_value():
    assert _serialize_value(None) is None
    assert _serialize_value("hello") == "hello"
    assert _serialize_value(123) == 123
    assert _serialize_value(Decimal("10.5")) == "10.5"
    assert (
        _serialize_value(UUID("12345678-1234-5678-1234-567812345678"))
        == "12345678-1234-5678-1234-567812345678"
    )
    assert (
        _serialize_value(datetime(2024, 1, 1, 12, 0, 0))  # noqa: DTZ001 - naive datetime serialization test
        == "2024-01-01T12:00:00"
    )
    assert _serialize_value(date(2024, 1, 1)) == "2024-01-01"
    assert _serialize_value(b"abc") == "YWJj"


def test_build_schema_text():
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
    assert "id INTEGER" in schema
    assert "name TEXT" in schema
    assert "PRIMARY KEY (id)" in schema
    assert "orders.user_id -> users.id" in schema


def test_transpile_to_dialect():
    sql = "SELECT name FROM singer WHERE age > 30 LIMIT 10"
    assert _transpile_to_dialect(sql, "sqlite") == sql
    pg = _transpile_to_dialect(sql, "postgres")
    assert "SELECT" in pg
    assert "LIMIT 10" in pg
    mysql = _transpile_to_dialect(sql, "mysql")
    assert "SELECT" in mysql
    assert "LIMIT 10" in mysql


def test_sqlite_backend_from_registry(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO t (name) VALUES ('x')")
    conn.commit()
    conn.close()

    record = DatabaseRecord(
        db_id="test",
        display_name="Test",
        backend_type="sqlite",
        db_path=str(db_path),
        connection_env=None,
        schema_text="",
        description="",
        created_at=None,
        updated_at=None,
    )
    backend = DatabaseBackend.from_registry(record)
    assert isinstance(backend, SQLiteBackend)
    schema = backend.get_schema()
    assert "CREATE TABLE t" in schema
    rows = backend.execute("SELECT name FROM t")
    assert rows == [("x",)]


def test_sqlite_backend_blocks_destructive(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite3.connect(db_path).execute("CREATE TABLE t (id INTEGER)").close()
    backend = SQLiteBackend(db_path)
    with pytest.raises(ValueError):
        backend.execute("DROP TABLE t")


def test_postgres_backend_dsn_from_env(monkeypatch):
    monkeypatch.setenv("TEST_PG_URL", "postgresql://user:pass@localhost/db")
    record = DatabaseRecord(
        db_id="pg",
        display_name="PG",
        backend_type="postgres",
        db_path=None,
        connection_env="TEST_PG_URL",
        schema_text="",
        description="",
        created_at=None,
        updated_at=None,
    )
    backend = DatabaseBackend.from_registry(record)
    assert isinstance(backend, PostgresBackend)


def test_mysql_backend_dsn_from_env(monkeypatch):
    monkeypatch.setenv("TEST_MYSQL_URL", "mysql://user:pass@localhost/db")
    record = DatabaseRecord(
        db_id="my",
        display_name="MySQL",
        backend_type="mysql",
        db_path=None,
        connection_env="TEST_MYSQL_URL",
        schema_text="",
        description="",
        created_at=None,
        updated_at=None,
    )
    backend = DatabaseBackend.from_registry(record)
    assert isinstance(backend, MySQLBackend)
    assert backend._connect_kwargs["host"] == "localhost"
    assert backend._connect_kwargs["database"] == "db"
    assert backend._connect_kwargs["user"] == "user"
    assert backend._connect_kwargs["password"] == "pass"


def test_mysql_backend_parses_dsn_with_plus_scheme():
    backend = MySQLBackend("mysql+pymysql://u:p@host:3307/mydb")
    assert backend._connect_kwargs["host"] == "host"
    assert backend._connect_kwargs["port"] == 3307
    assert backend._connect_kwargs["database"] == "mydb"


def test_mysql_backend_rejects_bad_scheme():
    with pytest.raises(ValueError):
        MySQLBackend("postgresql://u:p@host/db")
