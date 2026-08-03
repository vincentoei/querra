"""Shared pytest fixtures for the Querra backend."""

import os
import sqlite3
from pathlib import Path

import pytest

from registry import DatabaseRegistry

TEST_DIR = Path("/tmp/opencode/querra_tests")
TEST_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("ALLOWED_DB_DIRS", str(TEST_DIR))
os.environ.setdefault("QUERRA_DB", str(TEST_DIR / "querra_test.db"))

TEST_DB = TEST_DIR / "test_company.db"


def _create_test_db() -> Path:
    conn = sqlite3.connect(TEST_DB)
    conn.execute("DROP TABLE IF EXISTS employees")
    conn.execute("DROP TABLE IF EXISTS departments")
    conn.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER, department_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, location TEXT)"
    )
    conn.execute(
        "INSERT INTO employees (name, age, department_id) VALUES ('Alice', 30, 1), ('Bob', 25, 2)"
    )
    conn.execute(
        "INSERT INTO departments (name, location) VALUES ('Engineering', 'HQ'), ('Sales', 'Remote')"
    )
    conn.commit()
    conn.close()
    return TEST_DB


@pytest.fixture(scope="session", autouse=True)
def test_db() -> Path:
    return _create_test_db()


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-Admin-API-Key": "test-admin-key"}


@pytest.fixture(autouse=True)
def clean_registry():
    db = DatabaseRegistry(os.environ["QUERRA_DB"])
    with db._connect() as conn:
        conn.execute("DELETE FROM databases")
        conn.execute("DELETE FROM query_history")
        conn.commit()
    yield
