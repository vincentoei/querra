"""FastAPI inference service for Text-to-SQL."""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    DB_DIR,
    EXECUTE_TIMEOUT,
    MAX_RESULT_ROWS,
    PROCESSED_TRAIN,
    SCHEMA_LINKING_EMBEDDINGS_CACHE,
    SCHEMA_LINKING_EMBED_MODEL,
    SCHEMA_LINKING_TOP_K_TABLES,
    TABLES_FILE,
)
from evaluation.few_shot_retriever import FewShotRetriever
from utils.execution import execute_sql, get_db_path
from utils.inference import generate_sql, load_model_and_tokenizer
from utils.postprocess import normalize_sql_to_schema
from utils.prompts import extract_sql, format_few_shot, format_zero_shot
from utils.safety import is_read_only_sql, validate_db_id
from utils.schema import load_tables
from utils.schema_linker import SchemaLinker
from utils.self_correction import maybe_correct


class GenerateRequest(BaseModel):
    schema_str: str = Field(..., description="CREATE TABLE style schema string")
    question: str = Field(..., description="Natural language question")
    db_id: str | None = Field(None, description="Optional Spider database ID to execute against")
    execute: bool = Field(False, description="Run the generated SQL against the DB")
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


def _get_retriever() -> FewShotRetriever:
    if "retriever" not in _state:
        print("Loading few-shot retriever...")
        retriever = FewShotRetriever(PROCESSED_TRAIN)
        cache = PROCESSED_TRAIN.with_suffix(".pkl")
        if cache.exists():
            retriever.load_index(cache)
        else:
            retriever.build_index(cache)
        _state["retriever"] = retriever
        print("Retriever loaded.")
    return _state["retriever"]


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
    yield
    _state.clear()


app = FastAPI(title="Querra - Text-to-SQL Assistant", lifespan=lifespan)

_raw_cors = os.environ.get("CORS_ORIGINS", "*")
_cors_list = [o.strip() for o in _raw_cors.split(",") if o.strip()]
if _cors_list == ["*"]:
    _cors_origins = ["*"]
    _cors_credentials = False
elif _cors_list:
    _cors_origins = _cors_list
    _cors_credentials = True
else:
    _cors_origins = []
    _cors_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
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


@app.post("/generate-sql", response_model=GenerateResponse)
async def generate_sql_endpoint(req: GenerateRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model, tokenizer = _state["model"], _state["tokenizer"]
    schema = req.schema_str

    if req.db_id and not validate_db_id(req.db_id):
        raise HTTPException(status_code=400, detail="Invalid db_id")

    if req.use_schema_linking and req.db_id:
        schema, _ = await asyncio.to_thread(
            _get_schema_linker().build_schema,
            req.db_id,
            req.question,
            top_k=SCHEMA_LINKING_TOP_K_TABLES,
        )

    if req.use_few_shot:
        examples = await asyncio.to_thread(
            _get_retriever().retrieve, req.question, req.few_shot_k
        )
        prompt = format_few_shot(tokenizer, schema, req.question, examples)
    else:
        prompt = format_zero_shot(tokenizer, schema, req.question)

    start = time.perf_counter()
    raw = await asyncio.to_thread(generate_sql, model, tokenizer, prompt)
    sql = extract_sql(raw)
    sql = normalize_sql_to_schema(sql, schema)

    if not is_read_only_sql(sql):
        latency = time.perf_counter() - start
        return GenerateResponse(
            sql=sql,
            valid=False,
            execution_error="Query is not read-only or invalid",
            latency=latency,
        )

    execution_result = None
    execution_error = None
    if req.execute and req.db_id:
        db_path = get_db_path(req.db_id, DB_DIR)
        if db_path is None:
            execution_error = "Database not found"
        else:
            try:
                execution_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        execute_sql,
                        sql,
                        db_path,
                        max_rows=MAX_RESULT_ROWS,
                        case_sensitive_like=False,
                    ),
                    timeout=EXECUTE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                execution_error = f"Query execution timed out after {EXECUTE_TIMEOUT}s"
            except Exception as e:
                execution_error = str(e)

        if execution_error:
            # Self-correction runs model generation + execution in a background thread.
            corrected_sql, _ = await asyncio.to_thread(
                maybe_correct,
                model,
                tokenizer,
                schema,
                req.question,
                sql,
                req.db_id,
                DB_DIR,
                max_retries=2,
            )
            if corrected_sql != sql:
                sql = corrected_sql
                sql = normalize_sql_to_schema(sql, schema)
                if not is_read_only_sql(sql):
                    execution_error = "Corrected query is not read-only or invalid"
                else:
                    db_path = get_db_path(req.db_id, DB_DIR)
                    if db_path is None:
                        execution_error = "Database not found"
                    else:
                        try:
                            execution_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    execute_sql,
                                    sql,
                                    db_path,
                                    max_rows=MAX_RESULT_ROWS,
                                    case_sensitive_like=False,
                                ),
                                timeout=EXECUTE_TIMEOUT,
                            )
                            execution_error = None
                        except asyncio.TimeoutError:
                            execution_error = f"Query execution timed out after {EXECUTE_TIMEOUT}s"
                        except Exception as e:
                            execution_error = str(e)

    latency = time.perf_counter() - start
    return GenerateResponse(
        sql=sql,
        valid=is_read_only_sql(sql),
        execution_result=execution_result,
        execution_error=execution_error,
        latency=latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
