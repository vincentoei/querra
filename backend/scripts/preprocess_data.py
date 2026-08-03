"""Preprocess Spider into instruction-formatted JSONL files."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DB_DIR,
    DEV_FILE,
    PROCESSED_DEV,
    PROCESSED_DIR,
    PROCESSED_TRAIN,
    SOURCE_DB_DIR,
    TABLES_FILE,
    TRAIN_FILE,
)
from utils.schema import build_schema, load_tables


def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables(TABLES_FILE)

    # Copy SQLite database files.
    if SOURCE_DB_DIR.exists():
        for db_path in SOURCE_DB_DIR.iterdir():
            if db_path.is_dir():
                dest = DB_DIR / db_path.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(db_path, dest)
        print(f"Copied {len(list(SOURCE_DB_DIR.iterdir()))} databases to {DB_DIR}")

    def process_split(src_path: Path, dst_path: Path) -> None:
        examples = load_questions(src_path)
        with open(dst_path, "w", encoding="utf-8") as out:
            for ex in examples:
                db_id = ex["db_id"]
                schema = build_schema(tables[db_id])
                record = {
                    "db_id": db_id,
                    "schema": schema,
                    "question": ex["question"],
                    "query": ex["query"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} examples to {dst_path}")

    process_split(TRAIN_FILE, PROCESSED_TRAIN)
    process_split(DEV_FILE, PROCESSED_DEV)


if __name__ == "__main__":
    main()
