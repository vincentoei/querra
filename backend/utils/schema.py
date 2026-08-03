"""Build schema strings from Spider tables.json metadata."""

import json
from pathlib import Path
from typing import Any

TYPE_MAP = {
    "number": "INTEGER",
    "text": "TEXT",
    "time": "TEXT",
    "boolean": "INTEGER",
    "others": "TEXT",
}


def _map_type(t: str) -> str:
    return TYPE_MAP.get(t.lower(), "TEXT")


def build_schema(
    tables_entry: dict[str, Any],
    selected_tables: set[str] | None = None,
    include_fks: bool = True,
) -> str:
    """Return a CREATE TABLE style schema string for a Spider DB.

    If selected_tables is given, only those tables are included.
    Foreign-key comments are only emitted when both tables are selected.
    """
    table_names = tables_entry["table_names_original"]
    column_names = tables_entry["column_names_original"]
    column_types = tables_entry["column_types"]
    primary_keys = set(tables_entry["primary_keys"])
    foreign_keys = tables_entry["foreign_keys"]

    name_to_idx = {name: i for i, name in enumerate(table_names)}
    selected_indices: set[int] | None = None
    if selected_tables is not None:
        selected_indices = {
            name_to_idx[name] for name in selected_tables if name in name_to_idx
        }

    # Group columns by table, skipping wildcard and sqlite_sequence.
    table_columns: dict[int, list[tuple[int, str, str]]] = {}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1 or col_name == "*":
            continue
        if table_idx < 0 or table_idx >= len(table_names):
            continue
        if selected_indices is not None and table_idx not in selected_indices:
            continue
        table_name = table_names[table_idx]
        if table_name == "sqlite_sequence":
            continue
        table_columns.setdefault(table_idx, []).append(
            (col_idx, col_name, _map_type(column_types[col_idx]))
        )

    statements = []
    for table_idx in sorted(table_columns.keys()):
        table_name = table_names[table_idx]
        if table_name == "sqlite_sequence":
            continue
        cols = table_columns[table_idx]
        col_defs = [f"    {col_name} {col_type}" for _, col_name, col_type in cols]
        pk_cols = [c[1] for c in cols if c[0] in primary_keys]
        if pk_cols:
            col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
        statements.append(
            f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n)"
        )

    if not include_fks:
        return "\n\n".join(statements)

    fk_lines = []
    for col_a, col_b in foreign_keys:
        table_a = column_names[col_a][0]
        table_b = column_names[col_b][0]
        name_a = column_names[col_a][1]
        name_b = column_names[col_b][1]
        if table_a == -1 or table_b == -1:
            continue
        if selected_indices is not None and (
            table_a not in selected_indices or table_b not in selected_indices
        ):
            continue
        fk_lines.append(
            f"-- {table_names[table_a]}.{name_a} -> {table_names[table_b]}.{name_b}"
        )

    return "\n\n".join(statements) + ("\n\n" + "\n".join(fk_lines) if fk_lines else "")


def build_schema_for_tables(
    tables_entry: dict[str, Any], selected_tables: set[str], include_fks: bool = True
) -> str:
    """Convenience wrapper for building a schema from a selected set of tables."""
    return build_schema(
        tables_entry, selected_tables=selected_tables, include_fks=include_fks
    )


def load_tables(path: Path | str) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return {entry["db_id"]: entry for entry in data}
