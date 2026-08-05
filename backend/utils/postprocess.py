"""Schema-aware post-processing for generated SQL."""

import re
from difflib import get_close_matches

import sqlglot


def _build_identifier_map(schema: str, dialect: str = "sqlite") -> dict[str, str]:
    """Return a lowercase -> exact-case mapping for all table/column identifiers in the schema."""
    mapping: dict[str, str] = {}
    # The schema contains multiple CREATE TABLE statements separated by blank lines.
    # sqlglot needs them parsed individually, so split on CREATE TABLE boundaries.
    segments = [
        s.strip() for s in re.split(r"(?=CREATE TABLE)", schema.strip()) if s.strip()
    ]

    for segment in segments:
        try:
            statements = sqlglot.parse(segment, read=dialect)
        except sqlglot.errors.SqlglotError:
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


def normalize_sql_to_schema(sql: str, schema: str, dialect: str = "sqlite") -> str:
    """Normalize identifier casing in `sql` to match the exact names in `schema`."""
    identifier_map = _build_identifier_map(schema, dialect=dialect)
    if not identifier_map:
        return sql

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
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


def _build_table_column_map(
    schema: str, dialect: str = "sqlite"
) -> dict[str, dict[str, str]]:
    """Return {table_lower: {col_lower: exact_case}} for the schema."""
    mapping: dict[str, dict[str, str]] = {}
    segments = [
        s.strip() for s in re.split(r"(?=CREATE TABLE)", schema.strip()) if s.strip()
    ]
    for segment in segments:
        try:
            statements = sqlglot.parse(segment, read=dialect)
        except sqlglot.errors.Error:
            continue
        for stmt in statements:
            if not isinstance(stmt, sqlglot.exp.Create):
                continue
            schema_expr = stmt.this
            if not isinstance(schema_expr, sqlglot.exp.Schema):
                continue
            table_node = schema_expr.args.get("this")
            if not isinstance(table_node, sqlglot.exp.Table):
                continue
            table_name = table_node.name
            table_map: dict[str, str] = {}
            for col_def in schema_expr.expressions:
                if isinstance(col_def, sqlglot.exp.ColumnDef):
                    col_name = col_def.name
                    table_map[col_name.lower()] = col_name
            mapping[table_name.lower()] = table_map
    return mapping


def _collect_aliases(
    stmt: sqlglot.exp.Expression,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return alias maps for tables in FROM/JOIN.

    Returns:
        alias_to_table: {alias_lower: table_name}
        table_to_aliases: {table_lower: [original_alias_names]}
    """
    alias_to_table: dict[str, str] = {}
    table_to_aliases: dict[str, list[str]] = {}
    for table in stmt.find_all(sqlglot.exp.Table):
        table_name = table.name
        alias_node = table.args.get("alias")
        if alias_node is None:
            # No explicit alias; the table name itself is the usable reference.
            alias_name = table_name
        else:
            alias_name = (
                alias_node.name if hasattr(alias_node, "name") else str(alias_node)
            )
        alias_lower = alias_name.lower()
        alias_to_table[alias_lower] = table_name
        table_to_aliases.setdefault(table_name.lower(), []).append(alias_name)
    return alias_to_table, table_to_aliases


def fix_qualified_alias_references(
    sql: str, schema: str, dialect: str = "sqlite"
) -> str:
    """Fix qualified column references that point to the wrong table alias.

    Example: if `Album` is aliased `T1` and `Artist` is aliased `T2`, the
    generated `T2.Title` is rewritten to `T1.Title` when `Title` exists only
    in `Album`.
    """
    table_columns = _build_table_column_map(schema, dialect=dialect)
    if not table_columns:
        return sql

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return sql

    fixed_any = False
    for stmt in statements:
        alias_to_table, table_to_aliases = _collect_aliases(stmt)
        if not alias_to_table:
            continue

        for column in stmt.find_all(sqlglot.exp.Column):
            col_table = column.table
            if not col_table:
                continue
            current_table = alias_to_table.get(col_table.lower())
            if not current_table:
                continue
            current_cols = table_columns.get(current_table.lower(), {})
            if column.name.lower() in current_cols:
                # Column exists on the aliased table; nothing to fix.
                continue

            # Find which table in the schema actually has this column.
            candidate_tables = [
                table_name
                for table_name, cols in table_columns.items()
                if column.name.lower() in cols
            ]
            # Restrict to tables that are actually referenced in the query.
            candidate_aliases = []
            for table_name in candidate_tables:
                candidate_aliases.extend(table_to_aliases.get(table_name, []))

            if len(candidate_aliases) != 1:
                # Ambiguous or no fixable alias; leave as-is.
                continue

            correct_alias = candidate_aliases[0]
            column.set("table", sqlglot.exp.to_identifier(correct_alias))
            fixed_any = True

    if not fixed_any:
        return sql

    return "; ".join(stmt.sql(dialect="sqlite") for stmt in statements)


def normalize_quoted_string_literals(
    sql: str, schema: str, dialect: str = "sqlite"
) -> str:
    """Convert double-quoted unknown identifiers to single-quoted literals.

    Generated SQL often uses double quotes around string values (e.g.
    `"Aerosmith"`), but SQLite treats double quotes as identifiers. sqlglot
    then reports them as unknown columns. This helper converts them to single
    quotes when the quoted text is not a known table or column name.
    """
    identifier_map = _build_identifier_map(schema, dialect=dialect)
    if not identifier_map:
        return sql

    def _replace(match: re.Match[str]) -> str:
        text = match.group(1)
        if text.lower() in identifier_map:
            return match.group(0)
        return f"'{text}'"

    return re.sub(r'"([^"]+)"', _replace, sql)


def postprocess_sql(sql: str, schema: str, dialect: str = "sqlite") -> str:
    """Normalize identifier casing and fix qualified alias references."""
    sql = normalize_quoted_string_literals(sql, schema, dialect=dialect)
    sql = normalize_sql_to_schema(sql, schema, dialect=dialect)
    sql = fix_qualified_alias_references(sql, schema, dialect=dialect)
    return sql
