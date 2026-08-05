"""Compare full-schema vs schema-linked inference on the Spider dev set."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import sqlglot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

ROOT = Path(__file__).resolve().parent.parent


def _extract_gold_tables(sql: str) -> set[str]:
    """Extract table names from the gold SQL query."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.Error:
        return set()
    return {t.name for t in parsed.find_all(sqlglot.exp.Table)}


def _run_eval(
    name: str,
    output: str,
    *,
    use_schema_linking: bool,
    use_column_level_linking: bool,
    limit: int | None,
    self_correct: bool,
    max_retries: int,
    top_k: int,
    column_top_k: int,
):
    cmd = [
        sys.executable,
        "scripts/run_baseline.py",
        "--adapter_path",
        str(settings.adapter_dir),
        "--mode",
        "zero",
        "--output",
        output,
    ]
    if use_column_level_linking:
        cmd.append("--use_column_level_linking")
        cmd.extend(["--column_level_top_k", str(column_top_k)])
    elif use_schema_linking:
        cmd.append("--use_schema_linking")
        cmd.extend(["--schema_linking_top_k", str(top_k)])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if self_correct:
        cmd.append("--self_correct")
        cmd.extend(["--max_retries", str(max_retries)])

    print(f"\n=== Running {name} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _compute_table_recall(predictions: list[dict]) -> float:
    recalls = []
    for p in predictions:
        gold_tables = _extract_gold_tables(p["gold"])
        selected = set(p.get("selected_tables", []))
        if not gold_tables:
            continue
        recalls.append(len(gold_tables & selected) / len(gold_tables))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _extract_gold_columns(sql: str) -> set[str]:
    """Extract column names from the gold SQL query."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.Error:
        return set()
    columns: set[str] = set()
    for col in parsed.find_all(sqlglot.exp.Column):
        if col.name:
            columns.add(col.name)
    return columns


def _compute_column_recall(predictions: list[dict]) -> float:
    recalls = []
    for p in predictions:
        gold_columns = _extract_gold_columns(p["gold"])
        selected: set[str] = set()
        for cols in p.get("selected_columns", {}).values():
            selected.update(cols)
        if not gold_columns:
            continue
        recalls.append(len(gold_columns & selected) / len(gold_columns))
    return sum(recalls) / len(recalls) if recalls else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit dev examples for quick testing"
    )
    parser.add_argument(
        "--no_self_correct", action="store_true", help="Disable self-correction"
    )
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--top_k", type=int, default=settings.schema_linking_top_k_tables
    )
    parser.add_argument(
        "--column_top_k", type=int, default=settings.schema_linking_top_k_columns
    )
    args = parser.parse_args()

    self_correct = not args.no_self_correct

    full_output = "outputs/schema_linking_full.json"
    linked_output = "outputs/schema_linking_linked.json"
    column_output = "outputs/schema_linking_column.json"

    _run_eval(
        "Full schema",
        full_output,
        use_schema_linking=False,
        use_column_level_linking=False,
        limit=args.limit,
        self_correct=self_correct,
        max_retries=args.max_retries,
        top_k=args.top_k,
        column_top_k=args.column_top_k,
    )
    _run_eval(
        "Schema-linked (table-level)",
        linked_output,
        use_schema_linking=True,
        use_column_level_linking=False,
        limit=args.limit,
        self_correct=self_correct,
        max_retries=args.max_retries,
        top_k=args.top_k,
        column_top_k=args.column_top_k,
    )
    _run_eval(
        "Schema-linked (column-level)",
        column_output,
        use_schema_linking=False,
        use_column_level_linking=True,
        limit=args.limit,
        self_correct=self_correct,
        max_retries=args.max_retries,
        top_k=args.top_k,
        column_top_k=args.column_top_k,
    )

    full_data = json.loads((ROOT / full_output).read_text(encoding="utf-8"))
    linked_data = json.loads((ROOT / linked_output).read_text(encoding="utf-8"))
    column_data = json.loads((ROOT / column_output).read_text(encoding="utf-8"))

    full_preds = full_data["predictions"]
    linked_preds = linked_data["predictions"]
    column_preds = column_data["predictions"]
    full_metrics = full_data["metrics"]
    linked_metrics = linked_data["metrics"]
    column_metrics = column_data["metrics"]

    full_tokens = sum(p["schema_tokens"] for p in full_preds) / len(full_preds)
    linked_tokens = sum(p["schema_tokens"] for p in linked_preds) / len(linked_preds)
    column_tokens = sum(p["schema_tokens"] for p in column_preds) / len(column_preds)
    table_recall = _compute_table_recall(linked_preds)
    column_table_recall = _compute_table_recall(column_preds)
    column_recall = _compute_column_recall(column_preds)

    results = {
        "limit": args.limit,
        "self_correct": self_correct,
        "top_k": args.top_k,
        "column_top_k": args.column_top_k,
        "full_schema": {
            "exact_match": full_metrics["exact_match"],
            "execution_accuracy": full_metrics["execution_accuracy"],
            "avg_latency": full_metrics["avg_latency"],
            "avg_schema_tokens": full_tokens,
        },
        "linked_schema": {
            "exact_match": linked_metrics["exact_match"],
            "execution_accuracy": linked_metrics["execution_accuracy"],
            "avg_latency": linked_metrics["avg_latency"],
            "avg_schema_tokens": linked_tokens,
        },
        "column_linked_schema": {
            "exact_match": column_metrics["exact_match"],
            "execution_accuracy": column_metrics["execution_accuracy"],
            "avg_latency": column_metrics["avg_latency"],
            "avg_schema_tokens": column_tokens,
        },
        "table_recall": table_recall,
        "column_linked_table_recall": column_table_recall,
        "column_recall": column_recall,
    }

    print("\n=== Schema Linking Evaluation ===")
    print(
        f"{'Metric':<35} {'Full Schema':>15} {'Table Linked':>15} {'Column Linked':>15}"
    )
    print(
        f"{'Exact Match':<35} {full_metrics['exact_match']:>15.3f} {linked_metrics['exact_match']:>15.3f} {column_metrics['exact_match']:>15.3f}"
    )
    print(
        f"{'Execution Accuracy':<35} {full_metrics['execution_accuracy']:>15.3f} {linked_metrics['execution_accuracy']:>15.3f} {column_metrics['execution_accuracy']:>15.3f}"
    )
    print(
        f"{'Avg Latency (s)':<35} {full_metrics['avg_latency']:>15.3f} {linked_metrics['avg_latency']:>15.3f} {column_metrics['avg_latency']:>15.3f}"
    )
    print(
        f"{'Avg Schema Tokens':<35} {full_tokens:>15.1f} {linked_tokens:>15.1f} {column_tokens:>15.1f}"
    )
    print(
        f"{'Table Recall':<35} {'—':>15} {table_recall:>15.3f} {column_table_recall:>15.3f}"
    )
    print(f"{'Column Recall':<35} {'—':>15} {'—':>15} {column_recall:>15.3f}")

    out_path = ROOT / "outputs" / "schema_linking_eval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
