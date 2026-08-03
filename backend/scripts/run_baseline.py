"""Run base-model or prompt-engineered baseline on Spider dev."""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BASE_MODEL,
    DB_DIR,
    PROCESSED_DEV,
    PROCESSED_TRAIN,
    SCHEMA_LINKING_EMBED_MODEL,
    SCHEMA_LINKING_EMBEDDINGS_CACHE,
    SCHEMA_LINKING_TOP_K_TABLES,
    TABLES_FILE,
)
from db_backends import SQLiteBackend
from evaluation.few_shot_retriever import FewShotRetriever
from evaluation.metrics import component_match, exact_match, execution_match
from utils.execution import get_db_path
from utils.inference import generate_sql, load_model_and_tokenizer
from utils.postprocess import normalize_sql_to_schema
from utils.prompts import extract_sql, format_few_shot, format_zero_shot
from utils.schema import load_tables
from utils.schema_linker import SchemaLinker
from utils.self_correction import maybe_correct


def load_examples(path: Path, limit: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    if limit:
        examples = examples[:limit]
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--mode", choices=["zero", "few"], default="zero")
    parser.add_argument("--split", default=str(PROCESSED_DEV))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--few_shot_k", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--self_correct",
        action="store_true",
        help="Retry failed queries with the error message",
    )
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--use_schema_linking",
        action="store_true",
        help="Select relevant tables before prompt construction",
    )
    parser.add_argument(
        "--schema_linking_top_k", type=int, default=SCHEMA_LINKING_TOP_K_TABLES
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="text-to-sql-qlora")
    args = parser.parse_args()

    run_name = f"{args.mode}_shot_{Path(args.model).name}"
    if args.use_wandb:
        if not HAS_WANDB:
            raise ImportError(
                "wandb is required for --use_wandb. Install with: uv sync --extra train"
            )
        wandb.init(project=args.wandb_project, name=run_name)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, adapter_path=args.adapter_path
    )

    examples = load_examples(Path(args.split), args.limit)

    linker = None
    if args.use_schema_linking:
        print("Loading schema linker...")
        tables = load_tables(TABLES_FILE)
        linker = SchemaLinker(
            tables, SCHEMA_LINKING_EMBED_MODEL, SCHEMA_LINKING_EMBEDDINGS_CACHE
        )
        if SCHEMA_LINKING_EMBEDDINGS_CACHE.exists():
            print("Loading schema embeddings cache...")
            linker.load_cache()
        else:
            print("Building schema embeddings cache...")
            linker.build_cache()

    retriever = None
    if args.mode == "few":
        retriever = FewShotRetriever(PROCESSED_TRAIN)
        cache = PROCESSED_TRAIN.with_suffix(".pkl")
        if cache.exists():
            print("Loading few-shot index...")
            retriever.load_index(cache)
        else:
            print("Building few-shot index...")
            retriever.build_index(cache)

    predictions = []
    exact = 0
    exec_acc = 0
    component_totals = {
        k: 0 for k in ["select", "from", "where", "group_by", "order_by", "join"]
    }
    latencies = []

    for i, ex in enumerate(examples):
        schema, question, query, db_id = (
            ex["schema"],
            ex["question"],
            ex["query"],
            ex["db_id"],
        )
        selected_tables: list[str] = []
        if linker:
            schema, selected_tables_set = linker.build_schema(
                db_id, question, top_k=args.schema_linking_top_k
            )
            selected_tables = sorted(selected_tables_set)
        schema_tokens = len(tokenizer(schema, add_special_tokens=False)["input_ids"])

        if args.mode == "few":
            few_examples = retriever.retrieve(question, k=args.few_shot_k)
            prompt = format_few_shot(tokenizer, schema, question, few_examples)
        else:
            prompt = format_zero_shot(tokenizer, schema, question)

        start = time.time()
        raw = generate_sql(model, tokenizer, prompt)
        pred = extract_sql(raw)
        pred = normalize_sql_to_schema(pred, schema)

        em = exact_match(pred, query)
        xm = execution_match(pred, query, db_id, DB_DIR)
        cm = component_match(pred, query)

        retries = 0
        if args.self_correct and not xm:
            db_path = get_db_path(db_id, DB_DIR)
            if db_path:
                backend = SQLiteBackend(db_path)
                corrected, retries = maybe_correct(
                    model,
                    tokenizer,
                    schema,
                    question,
                    pred,
                    db_id,
                    backend,
                    max_retries=args.max_retries,
                )
                if corrected != pred:
                    pred = corrected
                    em = exact_match(pred, query)
                    xm = execution_match(pred, query, db_id, DB_DIR)
                    cm = component_match(pred, query)

        latencies.append(time.time() - start)

        exact += int(em)
        exec_acc += int(xm)
        for k, v in cm.items():
            component_totals[k] += int(v)

        predictions.append(
            {
                "db_id": db_id,
                "question": question,
                "gold": query,
                "pred": pred,
                "exact_match": em,
                "execution_match": xm,
                "component_match": cm,
                "latency": latencies[-1],
                "retries": retries,
                "schema_used": schema,
                "selected_tables": selected_tables,
                "schema_tokens": schema_tokens,
            }
        )
        if (i + 1) % 10 == 0:
            print(
                f"[{i + 1}/{len(examples)}] EM={exact / (i + 1):.3f} "
                f"Exec={exec_acc / (i + 1):.3f} AvgLat={sum(latencies) / len(latencies):.2f}s"
            )

    n = len(examples)
    metrics = {
        "exact_match": exact / n,
        "execution_accuracy": exec_acc / n,
        "avg_latency": sum(latencies) / n,
    }
    metrics.update({f"component_{k}": v / n for k, v in component_totals.items()})
    print(metrics)

    if args.use_wandb:
        if not HAS_WANDB:
            raise ImportError(
                "wandb is required for --use_wandb. Install with: uv sync --extra train"
            )
        wandb.log(metrics)
        wandb.log({"predictions": wandb.Table(dataframe=_to_df(predictions))})

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)
        print(f"Saved results to {out_path}")

    if args.use_wandb and HAS_WANDB:
        wandb.finish()


def _to_df(predictions):
    import pandas as pd

    rows = []
    for p in predictions:
        row = dict(p)
        row.pop("component_match")
        for k, v in p["component_match"].items():
            row[f"cm_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
