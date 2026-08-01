"""Schema-aware post-processing for generated SQL."""

import re
from difflib import get_close_matches

import sqlglot


def _build_identifier_map(schema: str) -> dict[str, str]:
    """Return a lowercase -> exact-case mapping for all table/column identifiers in the schema."""
    mapping: dict[str, str] = {}
    # The schema contains multiple CREATE TABLE statements separated by blank lines.
    # sqlglot needs them parsed individually, so split on CREATE TABLE boundaries.
    segments = [s.strip() for s in re.split(r"(?=CREATE TABLE)", schema.strip()) if s.strip()]

    for segment in segments:
        try:
            statements = sqlglot.parse(segment, read="sqlite")
        except Exception:
            continue

        for stmt in statements:
            if not isinstance(stmt, sqlglot.exp.Create):
                continue
            schema_expr = stmt.this
            if not isinstance(schema_expr, sqlglot.exp.Schema):
                continue

            table_node = schema_expr.args.get("this")
            if isinstance(table_node, sqlglot.exp.Table):
                name = table_node.name
                mapping[name.lower()] = name

            for col_def in schema_expr.expressions:
                if isinstance(col_def, sqlglot.exp.ColumnDef):
                    col_name = col_def.name
                    mapping[col_name.lower()] = col_name

    return mapping


def _exact_identifier(name: str, identifier_map: dict[str, str]) -> str | None:
    """Find the schema identifier matching `name`, using exact lowercase then fuzzy fallback."""
    key = name.lower()
    if key in identifier_map:
        return identifier_map[key]
    if len(key) < 3:
        return None
    matches = get_close_matches(key, identifier_map.keys(), n=1, cutoff=0.8)
    if matches:
        return identifier_map[matches[0]]
    return None


def normalize_sql_to_schema(sql: str, schema: str) -> str:
    """Normalize identifier casing in `sql` to match the exact names in `schema`."""
    identifier_map = _build_identifier_map(schema)
    if not identifier_map:
        return sql

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception:
        # If we cannot parse the SQL, return it unchanged.
        return sql

    for stmt in statements:
        # Normalize real table names in FROM / JOIN clauses.
        for table in stmt.find_all(sqlglot.exp.Table):
            exact = _exact_identifier(table.name, identifier_map)
            if exact:
                table.set("this", sqlglot.exp.to_identifier(exact))

        # Normalize column names (but not their table aliases).
        for column in stmt.find_all(sqlglot.exp.Column):
            exact = _exact_identifier(column.name, identifier_map)
            if exact:
                column.set("this", sqlglot.exp.to_identifier(exact))

    return "; ".join(stmt.sql(dialect="sqlite") for stmt in statements)
