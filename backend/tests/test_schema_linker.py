"""Tests for schema linking (table-level and column-level)."""

from utils.schema import build_schema
from utils.schema_linker import SchemaLinker


def _sample_tables() -> dict:
    return {
        "company_db": {
            "table_names_original": ["employees", "departments", "projects"],
            "column_names_original": [
                [0, "id"],
                [0, "name"],
                [0, "age"],
                [0, "salary"],
                [0, "department_id"],
                [1, "id"],
                [1, "name"],
                [1, "location"],
                [2, "id"],
                [2, "title"],
                [2, "budget"],
            ],
            "column_types": [
                "number",
                "text",
                "number",
                "number",
                "number",
                "number",
                "text",
                "text",
                "number",
                "text",
                "number",
            ],
            "primary_keys": [0, 5, 8],
            "foreign_keys": [[4, 5]],
        }
    }


def test_table_level_link_keyword():
    tables = _sample_tables()
    linker = SchemaLinker(tables, model_name=None)
    selected = linker.link("company_db", "What are the employee names?", top_k=0)
    assert "employees" in selected
    assert "departments" in selected  # FK closure pulls in departments


def test_column_level_preserves_key_columns():
    tables = _sample_tables()
    linker = SchemaLinker(tables, model_name=None)
    _schema, selected_tables, selected_columns = linker.build_schema_column_level(
        "company_db",
        "What is the title of each project?",
        table_top_k=0,
        column_top_k=0,
    )
    assert "projects" in selected_tables
    assert "title" in selected_columns["projects"]
    assert "id" in selected_columns["projects"]  # PK preserved


def test_column_level_includes_foreign_key_columns():
    tables = _sample_tables()
    linker = SchemaLinker(tables, model_name=None)
    selected_tables, selected_columns = linker.link_columns(
        "company_db",
        "What are the employee names?",
        table_top_k=0,
        column_top_k=0,
    )
    assert "employees" in selected_tables
    assert "department_id" in selected_columns["employees"]  # FK preserved
    assert "id" in selected_columns["employees"]  # PK preserved


def test_column_level_keyword_selection():
    tables = _sample_tables()
    linker = SchemaLinker(tables, model_name=None)
    selected_tables, selected_columns = linker.link_columns(
        "company_db",
        "What is the salary of employees?",
        table_top_k=0,
        column_top_k=0,
    )
    assert "employees" in selected_tables
    assert "salary" in selected_columns["employees"]
    assert "age" not in selected_columns["employees"]


def test_build_schema_with_selected_columns():
    tables_entry = _sample_tables()["company_db"]
    selected_columns = {"employees": {"name", "salary"}, "departments": {"name"}}
    schema = build_schema(
        tables_entry,
        selected_tables=set(selected_columns),
        selected_columns=selected_columns,
    )
    assert "CREATE TABLE employees" in schema
    assert "name" in schema
    assert "salary" in schema
    assert "age" not in schema  # not selected and not a key column
    assert "department_id" in schema  # FK preserved
    assert "CREATE TABLE departments" in schema
    assert "CREATE TABLE projects" not in schema


def test_build_schema_column_level_empty_selection():
    tables_entry = _sample_tables()["company_db"]
    selected_columns = {"employees": set()}
    schema = build_schema(
        tables_entry,
        selected_tables={"employees"},
        selected_columns=selected_columns,
    )
    assert "CREATE TABLE employees" in schema
    assert "id" in schema  # PK preserved
    assert "department_id" in schema  # FK preserved


def test_build_schema_column_level_missing_table():
    """Tables not present in selected_columns should render all columns."""
    tables_entry = _sample_tables()["company_db"]
    schema = build_schema(
        tables_entry,
        selected_tables={"employees", "departments"},
        selected_columns={"employees": {"name"}},
    )
    assert "CREATE TABLE employees" in schema
    assert "name" in schema
    assert "CREATE TABLE departments" in schema
    assert (
        "location" in schema
    )  # all columns because departments not in selected_columns
