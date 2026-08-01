"""Evaluation metrics for Text-to-SQL."""

import sys
from pathlib import Path

import re
import sqlglot
import sqlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.execution import execute_sql, get_db_path


def normalize_sql(sql: str) -> str:
    """Normalize SQL string for exact match comparison."""
    try:
        parsed = sqlparse.format(
            sql,
            keyword_case="lower",
            identifier_case="lower",
            strip_comments=True,
            reindent=True,
        )
    except Exception:
        parsed = sql.lower()
    # Remove all whitespace and backticks.
    parsed = parsed.replace("`", "")
    parsed = re.sub(r"\s+", "", parsed)
    return parsed


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def execution_match(pred: str, gold: str, db_id: str, db_dir: Path) -> bool:
    db_path = get_db_path(db_id, db_dir)
    if db_path is None:
        return False
    try:
        pred_rows = execute_sql(pred, db_path)
    except Exception:
        return False
    try:
        gold_rows = execute_sql(gold, db_path)
    except Exception:
        # If gold fails, assume mismatch.
        return False
    # Multiset comparison via sorted rows.
    try:
        return sorted(pred_rows) == sorted(gold_rows)
    except Exception:
        return False


def _extract_components(sql: str) -> dict:
    """Extract simple component sets from SQL via sqlglot."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return {}

    def sql_str(node):
        return node.sql(dialect="sqlite").lower()

    components = {
        "select": set(),
        "from": set(),
        "where": set(),
        "group_by": set(),
        "order_by": set(),
        "join": set(),
    }

    for select in parsed.find_all(sqlglot.exp.Select):
        for proj in select.expressions:
            components["select"].add(sql_str(proj))

        # Tables in the FROM clause.
        from_expr = select.args.get("from_")
        if from_expr:
            for table in from_expr.find_all(sqlglot.exp.Table):
                components["from"].add(table.name.lower())

        # Join tables.
        for join in select.args.get("joins", []):
            for table in join.find_all(sqlglot.exp.Table):
                components["join"].add(table.name.lower())

        # WHERE clause.
        where_expr = select.args.get("where")
        if where_expr:
            components["where"].add(sql_str(where_expr))

        # GROUP BY.
        group_expr = select.args.get("group")
        if group_expr:
            components["group_by"].update(sql_str(g) for g in group_expr.expressions)

        # ORDER BY.
        order_expr = select.args.get("order")
        if order_expr:
            components["order_by"].update(sql_str(o) for o in order_expr.expressions)

    return components


def component_match(pred: str, gold: str) -> dict[str, bool]:
    """Compare SQL clause presence and content.

    For each clause:
    - True if the clause is absent in both, or present in both with matching content.
    - False if the clause is present in one but not the other, or present in both but mismatched.
    """
    pred_comp = _extract_components(pred)
    gold_comp = _extract_components(gold)
    keys = ["select", "from", "where", "group_by", "order_by", "join"]
    result = {}
    for k in keys:
        pred_present = bool(pred_comp.get(k))
        gold_present = bool(gold_comp.get(k))
        if not pred_present and not gold_present:
            result[k] = True
        elif pred_present != gold_present:
            result[k] = False
        else:
            result[k] = pred_comp.get(k) == gold_comp.get(k)
    return result
