"""Tests for schema-aware SQL post-processing."""

from utils.postprocess import (
    _build_table_column_map,
    fix_qualified_alias_references,
    normalize_quoted_string_literals,
    normalize_sql_to_schema,
    postprocess_sql,
)

SCHEMA = """
CREATE TABLE Artist (
    ArtistId INTEGER,
    Name TEXT,
    PRIMARY KEY (ArtistId)
);

CREATE TABLE Album (
    AlbumId INTEGER,
    Title TEXT,
    ArtistId INTEGER,
    PRIMARY KEY (AlbumId)
);

CREATE TABLE Track (
    TrackId INTEGER,
    Name TEXT,
    AlbumId INTEGER,
    PRIMARY KEY (TrackId)
);
"""


def test_build_table_column_map():
    mapping = _build_table_column_map(SCHEMA)
    assert "artist" in mapping
    assert "album" in mapping
    assert "track" in mapping
    assert mapping["artist"]["name"] == "Name"
    assert mapping["album"]["title"] == "Title"


def test_normalize_sql_to_schema_casing():
    sql = "select artistid, name from artist"
    normalized = normalize_sql_to_schema(sql, SCHEMA)
    assert "ArtistId" in normalized
    assert "Name" in normalized
    assert "Artist" in normalized


def test_fix_alias_reference_swapped_aliases():
    sql = (
        "SELECT T2.Title FROM Album AS T1 "
        "JOIN Artist AS T2 ON T1.ArtistId = T2.ArtistId "
        'WHERE T2.Name = "Aerosmith"'
    )
    fixed = fix_qualified_alias_references(sql, SCHEMA)
    assert "T1.Title" in fixed
    assert "T2.Title" not in fixed
    assert "T2.Name" in fixed


def test_fix_alias_reference_no_change_when_correct():
    sql = (
        "SELECT T1.Title FROM Album AS T1 "
        "JOIN Artist AS T2 ON T1.ArtistId = T2.ArtistId "
        'WHERE T2.Name = "Aerosmith"'
    )
    fixed = fix_qualified_alias_references(sql, SCHEMA)
    assert "T1.Title" in fixed
    assert "T2.Name" in fixed


def test_fix_alias_reference_ambiguous_column():
    """Columns present in multiple tables should not be rewritten."""
    # Name exists in both Artist and Track.
    sql = (
        "SELECT T1.Name FROM Artist AS T1 JOIN Track AS T2 ON T1.ArtistId = T2.AlbumId"
    )
    fixed = fix_qualified_alias_references(sql, SCHEMA)
    assert "T1.Name" in fixed


def test_postprocess_sql_combined():
    sql = (
        "select t2.title from album as t1 "
        "join artist as t2 on t1.artistid = t2.artistid "
        'where t2.name = "Aerosmith"'
    )
    processed = postprocess_sql(sql, SCHEMA)
    # Alias case is preserved from the input; the fix changes only the alias
    # that references the wrong table.
    assert "t1.Title" in processed
    assert "t2.Name" in processed
    assert "Artist" in processed
    assert "Album" in processed


def test_fix_alias_reference_invalid_parse():
    """Malformed SQL is returned unchanged."""
    sql = "SELECT FROM WHERE"
    fixed = fix_qualified_alias_references(sql, SCHEMA)
    assert fixed == sql


def test_fix_alias_reference_no_qualified_columns():
    sql = "SELECT Title FROM Album"
    fixed = fix_qualified_alias_references(sql, SCHEMA)
    assert fixed == sql


def test_normalize_quoted_string_literals():
    sql = 'SELECT * FROM Album WHERE Title = "Aerosmith"'
    normalized = normalize_quoted_string_literals(sql, SCHEMA)
    assert "'Aerosmith'" in normalized
    assert '"Aerosmith"' not in normalized


def test_preserve_known_double_quoted_identifiers():
    # If the quoted text matches a known column, keep double quotes.
    sql = 'SELECT "Name" FROM Artist'
    normalized = normalize_quoted_string_literals(sql, SCHEMA)
    assert '"Name"' in normalized


def test_postprocess_sql_with_double_quoted_literal_and_alias_fix():
    sql = (
        "SELECT T2.Title FROM Album AS T1 "
        "JOIN Artist AS T2 ON T1.ArtistId = T2.ArtistId "
        'WHERE T2.Name = "Aerosmith"'
    )
    processed = postprocess_sql(sql, SCHEMA)
    assert "T1.Title" in processed
    assert "'Aerosmith'" in processed
    assert '"Aerosmith"' not in processed
