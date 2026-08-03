"""Tests for SQLite schema introspection."""

import sqlite3

import pytest

from schema_loader import build_schema_from_sqlite, validate_db_file


def test_build_schema(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)"
    )
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), total REAL)"
    )
    conn.commit()
    conn.close()

    schema = build_schema_from_sqlite(db_path)
    assert "CREATE TABLE users" in schema
    assert "CREATE TABLE orders" in schema
    assert "PRIMARY KEY (id)" in schema
    assert (
        "users.user_id -> orders.id" in schema or "orders.user_id -> users.id" in schema
    )


def test_validate_db_file(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    validate_db_file(db_path)

    bad = tmp_path / "bad.txt"
    bad.write_text("not a db")
    with pytest.raises(ValueError):
        validate_db_file(bad)
