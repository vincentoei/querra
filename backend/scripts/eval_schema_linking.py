"""Compare full-schema vs schema-linked inference on the Spider dev set."""

import argparse
import json
import subprocess
from pathlib import Path

import sqlglot

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SCHEMA_LINKING_TOP_K_TABLES


ROOT = Path(__file__).resolve().parent.parent


def _extract_gold_tables(sql: str) -> set[str]:
    """Extract table names from the gold SQL query."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    return {t.name for t in parsed.find_all(sqlglot.exp.Table)}


def _run_eval(
    name: str,
    output: str,
    *,
    use_schema_linking: bool,
    limit: int | None,
    self_correct: bool,
    max_retries: int,
    top_k: int,
):
    cmd = [
        sys.executable,
        "scripts/run_baseline.py",
        "--adapter_path",
        "models/best-model",
        "--mode",
        "zero",
        "--output",
        output,
    ]
    if use_schema_linking:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit dev examples for quick testing")
    parser.add_argument("--no_self_correct", action="store_true", help="Disable self-correction")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=SCHEMA_LINKING_TOP_K_TABLES)
    args = parser.parse_args()

    self_correct = not args.no_self_correct

    full_output = "outputs/schema_linking_full.json"
    linked_output = "outputs/schema_linking_linked.json"

    _run_eval(
        "Full schema",
        full_output,
        use_schema_linking=False,
        limit=args.limit,
        self_correct=self_correct,
        max_retries=args.max_retries,
        top_k=args.top_k,
    )
    _run_eval(
        "Schema-linked",
        linked_output,
        use_schema_linking=True,
        limit=args.limit,
        self_correct=self_correct,
        max_retries=args.max_retries,
        top_k=args.top_k,
    )

    full_data = json.loads((ROOT / full_output).read_text(encoding="utf-8"))
    linked_data = json.loads((ROOT / linked_output).read_text(encoding="utf-8"))

    full_preds = full_data["predictions"]
    linked_preds = linked_data["predictions"]
    full_metrics = full_data["metrics"]
    linked_metrics = linked_data["metrics"]

    full_tokens = sum(p["schema_tokens"] for p in full_preds) / len(full_preds)
    linked_tokens = sum(p["schema_tokens"] for p in linked_preds) / len(linked_preds)
    table_recall = _compute_table_recall(linked_preds)

    results = {
        "limit": args.limit,
        "self_correct": self_correct,
        "top_k": args.top_k,
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
        "table_recall": table_recall,
    }

    print("\n=== Schema Linking Evaluation ===")
    print(f"{'Metric':<30} {'Full Schema':>15} {'Linked Schema':>15}")
    print(
        f"{'Exact Match':<30} {full_metrics['exact_match']:>15.3f} {linked_metrics['exact_match']:>15.3f}"
    )
    print(
        f"{'Execution Accuracy':<30} {full_metrics['execution_accuracy']:>15.3f} {linked_metrics['execution_accuracy']:>15.3f}"
    )
    print(
        f"{'Avg Latency (s)':<30} {full_metrics['avg_latency']:>15.3f} {linked_metrics['avg_latency']:>15.3f}"
    )
    print(f"{'Avg Schema Tokens':<30} {full_tokens:>15.1f} {linked_tokens:>15.1f}")
    print(f"{'Table Recall (linked)':<30} {'—':>15} {table_recall:>15.3f}")

    out_path = ROOT / "outputs" / "schema_linking_eval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
