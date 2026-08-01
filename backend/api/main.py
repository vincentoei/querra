"""FastAPI inference service for Text-to-SQL."""

import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

import sqlglot
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    DB_DIR,
    PROCESSED_TRAIN,
    SCHEMA_LINKING_EMBEDDINGS_CACHE,
    SCHEMA_LINKING_EMBED_MODEL,
    SCHEMA_LINKING_TOP_K_TABLES,
    TABLES_FILE,
)
from evaluation.few_shot_retriever import FewShotRetriever
from utils.inference import generate_sql, load_model_and_tokenizer
from utils.postprocess import normalize_sql_to_schema
from utils.prompts import extract_sql, format_few_shot, format_zero_shot
from utils.schema import load_tables
from utils.schema_linker import SchemaLinker
from utils.self_correction import maybe_correct


class GenerateRequest(BaseModel):
    schema_str: str = Field(..., description="CREATE TABLE style schema string")
    question: str = Field(..., description="Natural language question")
    db_id: str | None = Field(None, description="Optional Spider database ID to execute against")
    execute: bool = Field(True, description="Run the generated SQL against the DB")
    use_few_shot: bool = Field(False, description="Use retrieved few-shot examples in the prompt")
    few_shot_k: int = Field(3, ge=0, le=10, description="Number of few-shot examples (only if use_few_shot=True)")
    use_schema_linking: bool = Field(False, description="Select relevant tables from the db_id schema before generating SQL")


class GenerateResponse(BaseModel):
    sql: str
    valid: bool
    execution_result: list | None = None
    execution_error: str | None = None
    latency: float


_state = {}


def _get_schema_linker() -> SchemaLinker:
    if "schema_linker" not in _state:
        print("Loading schema linker...")
        tables = load_tables(TABLES_FILE)
        linker = SchemaLinker(
            tables, SCHEMA_LINKING_EMBED_MODEL, SCHEMA_LINKING_EMBEDDINGS_CACHE
        )
        if SCHEMA_LINKING_EMBEDDINGS_CACHE.exists():
            linker.load_cache()
        else:
            linker.build_cache()
        _state["schema_linker"] = linker
        print("Schema linker loaded.")
    return _state["schema_linker"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.environ.get("MODEL_NAME", BASE_MODEL)
    adapter_path = os.environ.get("ADAPTER_PATH", str(ADAPTER_DIR))
    if not Path(adapter_path).exists():
        adapter_path = None
    print(f"Loading model {model_name} (adapter={adapter_path})...")
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_path=adapter_path)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    print("Model loaded.")

    print("Loading few-shot retriever...")
    retriever = FewShotRetriever(PROCESSED_TRAIN)
    cache = PROCESSED_TRAIN.with_suffix(".pkl")
    if cache.exists():
        retriever.load_index(cache)
    else:
        retriever.build_index(cache)
    _state["retriever"] = retriever
    print("Retriever loaded.")
    yield
    _state.clear()


app = FastAPI(title="Querra - Text-to-SQL Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "model": os.environ.get("MODEL_NAME", BASE_MODEL),
    }


def _validate_sql(sql: str) -> bool:
    try:
        sqlglot.parse(sql, read="sqlite")
        return True
    except Exception:
        return False


def _execute_sql(sql: str, db_id: str, db_dir: Path = DB_DIR) -> list:
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        db_path = db_dir / db_id / f"{db_id}.db"
    if not db_path.exists():
        files = list((db_dir / db_id).glob("*.sqlite")) + list(
            (db_dir / db_id).glob("*.db")
        )
        if not files:
            raise FileNotFoundError(f"No database found for {db_id}")
        db_path = files[0]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
        return [tuple(row) for row in rows]
    finally:
        conn.close()


def _try_execute(sql: str, db_id: str) -> tuple[list | None, str | None]:
    try:
        return _execute_sql(sql, db_id), None
    except Exception as e:
        return None, str(e)


def _block_destructive(sql: str) -> bool:
    upper = sql.upper()
    for keyword in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"):
        if keyword in upper:
            return True
    return False


@app.post("/generate-sql", response_model=GenerateResponse)
async def generate_sql_endpoint(req: GenerateRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model, tokenizer = _state["model"], _state["tokenizer"]
    schema = req.schema_str
    if req.use_schema_linking and req.db_id:
        schema, _ = _get_schema_linker().build_schema(
            req.db_id, req.question, top_k=SCHEMA_LINKING_TOP_K_TABLES
        )
    if req.use_few_shot and "retriever" in _state:
        examples = _state["retriever"].retrieve(req.question, k=req.few_shot_k)
        prompt = format_few_shot(tokenizer, schema, req.question, examples)
    else:
        prompt = format_zero_shot(tokenizer, schema, req.question)
    start = time.time()
    raw = generate_sql(model, tokenizer, prompt)
    sql = extract_sql(raw)
    sql = normalize_sql_to_schema(sql, schema)

    if _block_destructive(sql):
        latency = time.time() - start
        return GenerateResponse(
            sql=sql,
            valid=False,
            execution_error="Destructive queries are blocked",
            latency=latency,
        )

    execution_result = None
    execution_error = None
    if req.execute and req.db_id:
        valid = _validate_sql(sql)
        if not valid:
            execution_error = "Generated SQL failed validation"
        else:
            execution_result, execution_error = _try_execute(sql, req.db_id)

        # Self-correction on validation or execution failure.
        if execution_error:
            sql, _ = maybe_correct(
                model, tokenizer, schema, req.question, sql, req.db_id, DB_DIR, max_retries=2
            )
            sql = normalize_sql_to_schema(sql, schema)
            if _block_destructive(sql):
                execution_error = "Destructive query in corrected SQL"
            else:
                valid = _validate_sql(sql)
                if not valid:
                    execution_error = "Corrected SQL still failed validation"
                else:
                    execution_result, execution_error = _try_execute(sql, req.db_id)

    latency = time.time() - start
    return GenerateResponse(
        sql=sql,
        valid=_validate_sql(sql),
        execution_result=execution_result,
        execution_error=execution_error,
        latency=latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
