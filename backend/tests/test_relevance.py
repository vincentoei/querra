"""Tests for question-schema relevance checking."""

import pytest

from utils.relevance import (
    _extract_tables_and_columns,
    check_question_schema_relevance,
    validate_sql_identifiers,
)

SCHEMA = """CREATE TABLE singer (
    singer_id INTEGER,
    name TEXT,
    age INTEGER
);

CREATE TABLE album (
    album_id INTEGER,
    singer_id INTEGER,
    title TEXT
);"""


def test_extract_tables_and_columns():
    tables = _extract_tables_and_columns(SCHEMA)
    assert set(tables.keys()) == {"singer", "album"}
    assert tables["singer"] == ["singer_id", "name", "age"]
    assert tables["album"] == ["album_id", "singer_id", "title"]


def test_extract_tables_and_columns_regex_fallback():
    # sqlglot should parse this, but ensure regex fallback also works if needed.
    text = "CREATE TABLE foo (a INTEGER, b TEXT)\n\nCREATE TABLE bar (c INTEGER)"
    tables = _extract_tables_and_columns(text)
    assert set(tables.keys()) == {"foo", "bar"}


@pytest.mark.slow
@pytest.mark.skip(
    "Downloads sentence-transformers model; run manually for full relevance checks."
)
def test_check_question_schema_relevance_related():
    related, score, warning = check_question_schema_relevance(
        "What are the names of singers older than 30?", SCHEMA, threshold=0.35
    )
    assert related is True
    assert warning is None
    assert score > 0.35


@pytest.mark.slow
@pytest.mark.skip(
    "Downloads sentence-transformers model; run manually for full relevance checks."
)
def test_check_question_schema_relevance_unrelated():
    related, score, warning = check_question_schema_relevance(
        "What is the weather in Paris today?", SCHEMA, threshold=0.35
    )
    assert related is False
    assert warning is not None
    assert score < 0.35


def test_check_question_schema_relevance_keyword_match():
    # Short/specific questions that name a schema object should pass via keyword
    # rescue without loading the embedding model.
    related, score, warning = check_question_schema_relevance(
        "What album does Aerosmith has?", SCHEMA, threshold=0.35
    )
    assert related is True
    assert warning is None
    assert score == 1.0


def test_check_question_schema_relevance_empty_question():
    related, score, warning = check_question_schema_relevance("", SCHEMA)
    assert related is False
    assert warning == "Please enter a question."
    assert score == 0.0


def test_validate_sql_identifiers_valid():
    valid, unknown = validate_sql_identifiers(
        SCHEMA,
        "SELECT name FROM singer WHERE age > 30",
    )
    assert valid is True
    assert unknown == []


def test_validate_sql_identifiers_unknown_table():
    valid, unknown = validate_sql_identifiers(
        SCHEMA,
        "SELECT name FROM artist WHERE age > 30",
    )
    assert valid is False
    assert "artist" in unknown


def test_validate_sql_identifiers_unknown_column():
    valid, unknown = validate_sql_identifiers(
        SCHEMA,
        "SELECT height FROM singer",
    )
    assert valid is False
    assert "height" in unknown


def test_validate_sql_identifiers_parse_error():
    valid, unknown = validate_sql_identifiers(
        SCHEMA,
        "SELECT {invalid",
    )
    assert valid is False
    assert unknown == ["failed to parse generated SQL"]
