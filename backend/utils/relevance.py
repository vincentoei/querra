"""Question-schema relevance checking."""

import logging
import re

import sqlglot
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# Same embedding model used elsewhere to avoid loading multiple models.
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _extract_tables_and_columns(
    schema_text: str, dialect: str = "sqlite"
) -> dict[str, list[str]]:
    """Parse a CREATE TABLE schema string and return {table: [columns]}."""
    tables: dict[str, list[str]] = {}
    try:
        parsed = sqlglot.parse(schema_text, dialect=dialect)
    except sqlglot.errors.ParseError as e:
        logger.warning("Failed to parse schema with sqlglot: %s", e)
        parsed = []

    for stmt in parsed:
        if stmt is None:
            continue
        try:
            table_expr = getattr(stmt.this, "this", None)
            table_name = table_expr.name if hasattr(table_expr, "name") else None
            if not table_name:
                continue
            columns = []
            for col_def in stmt.find_all(sqlglot.exp.ColumnDef):
                col_name = col_def.this.name if hasattr(col_def.this, "name") else None
                if col_name:
                    columns.append(col_name)
            tables[table_name] = columns
        except (AttributeError, TypeError) as e:
            logger.warning("Schema extraction skipped a statement: %s", e)
            continue

    if tables:
        return tables

    # Fallback regex parser.
    current_table: str | None = None
    for line in schema_text.splitlines():
        line = line.strip()
        table_match = re.match(r"CREATE TABLE\s+(\w+)", line, re.IGNORECASE)
        if table_match:
            current_table = table_match.group(1)
            tables[current_table] = []
            continue
        if current_table and line and not line.startswith("--"):
            col_match = re.match(r"(\w+)", line)
            if col_match:
                tables[current_table].append(col_match.group(1))
    return tables


def _build_schema_candidates(tables: dict[str, list[str]]) -> list[str]:
    """Build one candidate text per table describing the table and its columns."""
    candidates = []
    for table_name, columns in tables.items():
        col_text = ", ".join(columns) if columns else "no columns"
        candidates.append(f"Table {table_name} has columns: {col_text}")
    return candidates


def _matches_whole_word(name: str, text_lower: str) -> bool:
    """Return True if `name` appears as a whole word in `text_lower`."""
    key = re.escape(name.lower())
    return bool(re.search(r"\b" + key + r"\b", text_lower))


def _keyword_matches_schema(question: str, tables: dict[str, list[str]]) -> bool:
    """Return True if the question literally mentions any table or column."""
    q = question.lower()
    for table_name, columns in tables.items():
        if _matches_whole_word(table_name, q):
            return True
        for col in columns:
            if _matches_whole_word(col, q):
                return True
    return False


def check_question_schema_relevance(
    question: str,
    schema_text: str,
    threshold: float = 0.35,
    dialect: str = "sqlite",
) -> tuple[bool, float, str | None]:
    """Return (is_related, score, warning_message).

    A question is considered related if it literally mentions any schema table or
    column, or if its embedding is sufficiently similar to at least one
    table/column candidate from the schema.
    """
    if not question.strip():
        return False, 0.0, "Please enter a question."

    tables = _extract_tables_and_columns(schema_text, dialect=dialect)
    if not tables:
        return True, 1.0, None

    # Keyword rescue: avoid false positives on short/specific questions that
    # name a schema object but score below the embedding threshold.
    if _keyword_matches_schema(question, tables):
        return True, 1.0, None

    candidates = _build_schema_candidates(tables)
    model = _get_embed_model()

    try:
        question_emb = model.encode(question, convert_to_tensor=True)
        candidate_embs = model.encode(candidates, convert_to_tensor=True)
        similarities = util.cos_sim(question_emb, candidate_embs)[0]
        max_score = float(similarities.max())
    except (RuntimeError, ValueError) as e:
        logger.warning("Embedding similarity failed: %s", e)
        return True, 1.0, None

    if max_score >= threshold:
        return True, max_score, None

    return (
        False,
        max_score,
        (
            "This question may not match the selected database. "
            "Please check that your question refers to data in this database."
        ),
    )


def validate_sql_identifiers(
    schema_text: str, sql: str, dialect: str = "sqlite"
) -> tuple[bool, list[str]]:
    """Return (all_valid, list_of_unknown_identifiers).

    Parses the generated SQL and checks that every table and column it
    references exists in the schema.
    """
    tables = _extract_tables_and_columns(schema_text, dialect=dialect)
    known_tables = set(tables.keys())
    known_columns: set[str] = set()
    for cols in tables.values():
        known_columns.update(cols)

    unknown: set[str] = set()
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        logger.warning("Failed to parse generated SQL for identifier validation: %s", e)
        return False, ["failed to parse generated SQL"]

    for table in parsed.find_all(sqlglot.exp.Table):
        name = table.name
        if name and name not in known_tables:
            unknown.add(name)

    for column in parsed.find_all(sqlglot.exp.Column):
        name = column.name
        if name and name not in known_columns:
            unknown.add(name)

    return not unknown, sorted(unknown)
