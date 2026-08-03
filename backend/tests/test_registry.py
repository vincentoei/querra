"""Tests for the database registry."""

from registry import DatabaseRegistry


def test_register_and_get(tmp_path):
    db = DatabaseRegistry(tmp_path / "reg.db")
    db.register_database(
        db_id="demo",
        display_name="Demo DB",
        db_path="/tmp/demo.sqlite",
        schema_text="CREATE TABLE t (id INTEGER);",
        description="test",
    )
    record = db.get_database("demo")
    assert record is not None
    assert record.display_name == "Demo DB"
    assert record.schema_text == "CREATE TABLE t (id INTEGER);"


def test_delete(tmp_path):
    db = DatabaseRegistry(tmp_path / "reg.db")
    db.register_database(
        db_id="temp",
        display_name="Temp",
        db_path="/tmp/t.sqlite",
        schema_text="CREATE TABLE t (id INTEGER);",
    )
    assert db.delete_database("temp") is True
    assert db.get_database("temp") is None
    assert db.delete_database("temp") is False


def test_list_databases(tmp_path):
    db = DatabaseRegistry(tmp_path / "reg.db")
    db.register_database(
        db_id="a",
        display_name="A",
        db_path="/tmp/a.sqlite",
        schema_text="CREATE TABLE a (id INTEGER);",
    )
    db.register_database(
        db_id="b",
        display_name="B",
        db_path="/tmp/b.sqlite",
        schema_text="CREATE TABLE b (id INTEGER);",
    )
    rows = db.list_databases()
    assert len(rows) == 2
    assert {r["db_id"] for r in rows} == {"a", "b"}


def test_query_history(tmp_path):
    db = DatabaseRegistry(tmp_path / "reg.db")
    db.register_database(
        db_id="h",
        display_name="H",
        db_path="/tmp/h.sqlite",
        schema_text="CREATE TABLE h (id INTEGER);",
    )
    db.record_query(
        db_id="h",
        question="count rows",
        generated_sql="SELECT count(*) FROM h",
        edited_sql=None,
        execution_result="[(5,)]",
        execution_error=None,
        latency_ms=12.5,
    )
    history = db.get_history(db_id="h")
    assert len(history) == 1
    assert history[0]["question"] == "count rows"
