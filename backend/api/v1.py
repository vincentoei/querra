"""Public API v1 routes for Querra."""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

from api.security import require_admin_api_key
from api.state import get_model, get_tokenizer
from config import (
    EXECUTE_TIMEOUT,
    MAX_RESULT_ROWS,
    PROCESSED_TRAIN,
    SCHEMA_LINKING_EMBED_MODEL,
    SCHEMA_LINKING_EMBEDDINGS_CACHE,
    SCHEMA_LINKING_TOP_K_TABLES,
    TABLES_FILE,
    UPLOAD_DIR,
)
from db_backends import DatabaseBackend
from evaluation.few_shot_retriever import FewShotRetriever
from registry import DatabaseRecord, DatabaseRegistry
from schema_loader import validate_db_file
from utils.inference import generate_sql
from utils.postprocess import normalize_sql_to_schema
from utils.prompts import extract_sql, format_few_shot, format_zero_shot
from utils.safety import is_read_only_sql, validate_admin_db_path, validate_db_id
from utils.schema import load_tables
from utils.schema_linker import SchemaLinker
from utils.self_correction import maybe_correct

router = APIRouter(prefix="/api/v1")

# FastAPI dependency singletons to avoid B008 lint warnings on default args.
_ADMIN_AUTH = Depends(require_admin_api_key)
_UPLOAD_FILE = File(...)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: DatabaseRegistry | None = None


def get_registry() -> DatabaseRegistry:
    global _registry
    if _registry is None:
        from config import QUERRA_DB

        _registry = DatabaseRegistry(QUERRA_DB)
    return _registry


# ---------------------------------------------------------------------------
# Schema linker / few-shot retriever (lazy)
# ---------------------------------------------------------------------------

_linker: SchemaLinker | None = None
_retriever: FewShotRetriever | None = None


def _get_schema_linker() -> SchemaLinker | None:
    global _linker
    if _linker is None:
        if not TABLES_FILE.exists():
            return None
        try:
            tables = load_tables(TABLES_FILE)
            _linker = SchemaLinker(
                tables, SCHEMA_LINKING_EMBED_MODEL, SCHEMA_LINKING_EMBEDDINGS_CACHE
            )
            if SCHEMA_LINKING_EMBEDDINGS_CACHE.exists():
                _linker.load_cache()
            else:
                _linker.build_cache()
        except Exception:
            logger.exception("Failed to initialize schema linker")
            _linker = None
    return _linker


def _get_retriever() -> FewShotRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FewShotRetriever(PROCESSED_TRAIN)
        cache = PROCESSED_TRAIN.with_suffix(".pkl")
        if cache.exists():
            _retriever.load_index(cache)
        else:
            _retriever.build_index(cache)
    return _retriever


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DatabaseSummary(BaseModel):
    db_id: str
    display_name: str
    backend_type: str
    description: str | None
    created_at: str | None
    updated_at: str | None


class DatabaseDetail(BaseModel):
    db_id: str
    display_name: str
    backend_type: str
    db_path: str | None
    connection_env: str | None
    schema_text: str
    description: str | None
    created_at: str | None
    updated_at: str | None


class RegisterDatabaseRequest(BaseModel):
    db_id: str
    display_name: str
    backend_type: str = "sqlite"
    db_path: str | None = None
    connection_env: str | None = None
    description: str = ""

    @field_validator("db_id")
    @classmethod
    def _validate_db_id(cls, v: str) -> str:
        if not validate_db_id(v):
            raise ValueError("db_id must be alphanumeric with - and _ only")
        return v

    @field_validator("backend_type")
    @classmethod
    def _validate_backend_type(cls, v: str) -> str:
        if v not in ("sqlite", "postgres", "mysql"):
            raise ValueError("backend_type must be sqlite, postgres, or mysql")
        return v

    @model_validator(mode="after")
    def _validate_connection(self):
        if self.backend_type == "sqlite":
            if not self.db_path:
                raise ValueError("db_path is required for sqlite backend")
            if self.connection_env:
                raise ValueError("connection_env is not used for sqlite backend")
        else:
            if not self.connection_env:
                raise ValueError(
                    "connection_env is required for postgres and mysql backends"
                )
            if self.db_path:
                raise ValueError("db_path is not used for postgres and mysql backends")
        return self


class GenerateRequest(BaseModel):
    db_id: str | None = Field(None, description="Registered database ID to use")
    schema_str: str | None = Field(
        None, description="Direct schema string (alternative to db_id)"
    )
    question: str = Field(..., description="Natural language question")
    execute: bool = Field(False, description="Run the generated SQL against the DB")
    use_few_shot: bool = Field(False, description="Use retrieved few-shot examples")
    few_shot_k: int = Field(3, ge=0, le=10)
    use_schema_linking: bool = Field(
        False, description="Select relevant tables (Spider DBs only)"
    )
    self_correct: bool = Field(True, description="Retry on execution failure")
    max_retries: int = Field(2, ge=0, le=5)

    @field_validator("db_id")
    @classmethod
    def _validate_db_id(cls, v: str | None) -> str | None:
        if v is not None and not validate_db_id(v):
            raise ValueError("db_id must be alphanumeric with - and _ only")
        return v


class GenerateResponse(BaseModel):
    sql: str
    valid: bool
    execution_result: list | None = None
    execution_error: str | None = None
    latency: float


class ExecuteRequest(BaseModel):
    db_id: str = Field(..., description="Registered database ID")
    sql: str = Field(..., description="SQL query to execute")

    @field_validator("db_id")
    @classmethod
    def _validate_db_id(cls, v: str) -> str:
        if not validate_db_id(v):
            raise ValueError("db_id must be alphanumeric with - and _ only")
        return v


class ExecuteResponse(BaseModel):
    sql: str
    valid: bool
    execution_result: list | None = None
    execution_error: str | None = None
    latency: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allowed_db_dirs() -> list[str]:
    from config import ALLOWED_DB_DIRS, DATA_DIR

    raw = ALLOWED_DB_DIRS.strip()
    if raw:
        return [d.strip() for d in raw.split(",") if d.strip()]
    return [str(DATA_DIR)]


def _resolve_schema_and_backend(
    req: GenerateRequest,
) -> tuple[str, DatabaseBackend | None]:
    """Return (schema, backend) for the request."""
    if req.db_id:
        record = get_registry().get_database(req.db_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Database not found: {req.db_id}"
            )
        try:
            backend = DatabaseBackend.from_registry(record)
            schema = backend.get_schema()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return schema, backend
    if req.schema_str:
        return req.schema_str, None
    raise HTTPException(status_code=400, detail="Provide either db_id or schema_str")


def _maybe_schema_link(db_id: str, question: str, schema: str) -> str:
    """Reduce schema for Spider-style databases if a linker is available."""
    linker = _get_schema_linker()
    if linker is None or db_id not in linker.tables:
        return schema
    try:
        reduced, _ = linker.build_schema(
            db_id, question, top_k=SCHEMA_LINKING_TOP_K_TABLES
        )
        return reduced
    except Exception:
        logger.exception("Schema linking failed for db_id=%s", db_id)
        return schema


def _record_query(
    db_id: str,
    question: str,
    sql: str,
    execution_result: list | None,
    execution_error: str | None,
    latency: float,
    edited_sql: str | None = None,
) -> None:
    try:
        get_registry().record_query(
            db_id=db_id,
            question=question,
            generated_sql=sql,
            edited_sql=edited_sql,
            execution_result=repr(execution_result)
            if execution_result is not None
            else None,
            execution_error=execution_error,
            latency_ms=latency * 1000,
        )
    except Exception:
        # Never fail the request because of logging.
        logger.exception("Failed to record query history")


async def _run_backend_query(
    backend: DatabaseBackend,
    sql: str,
    max_rows: int = MAX_RESULT_ROWS,
) -> tuple[list | None, str | None]:
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(backend.execute, sql, max_rows),
            timeout=EXECUTE_TIMEOUT,
        )
        return rows, None
    except TimeoutError:
        return None, f"Query execution timed out after {EXECUTE_TIMEOUT}s"
    except Exception as e:  # noqa: BLE001 - backend may raise DB-specific errors
        return None, str(e)


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@router.get("/databases", response_model=list[DatabaseSummary])
async def list_databases():
    return get_registry().list_databases()


@router.get("/databases/{db_id}/schema", response_model=str)
async def get_database_schema(db_id: str):
    if not validate_db_id(db_id):
        raise HTTPException(status_code=400, detail="Invalid db_id")
    record = get_registry().get_database(db_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        backend = DatabaseBackend.from_registry(record)
        return backend.get_schema()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate-sql", response_model=GenerateResponse)
async def generate_sql_endpoint(req: GenerateRequest):
    try:
        model = get_model()
        tokenizer = get_tokenizer()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded")

    schema, backend = _resolve_schema_and_backend(req)

    if req.use_schema_linking:
        schema = _maybe_schema_link(req.db_id or "", req.question, schema)

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
        if req.db_id:
            _record_query(
                req.db_id,
                req.question,
                sql,
                None,
                "Query is not read-only or invalid",
                latency,
            )
        return GenerateResponse(
            sql=sql,
            valid=False,
            execution_error="Query is not read-only or invalid",
            latency=latency,
        )

    result: list | None = None
    error: str | None = None
    if req.execute and backend is not None:
        result, error = await _run_backend_query(backend, sql)

        if error and req.self_correct:
            corrected, _ = await asyncio.to_thread(
                maybe_correct,
                model,
                tokenizer,
                schema,
                req.question,
                sql,
                req.db_id,
                backend,
                max_retries=req.max_retries,
            )
            if corrected != sql:
                sql = corrected
                sql = normalize_sql_to_schema(sql, schema)
                if not is_read_only_sql(sql):
                    error = "Corrected query is not read-only or invalid"
                else:
                    result, error = await _run_backend_query(backend, sql)

    latency = time.perf_counter() - start
    if req.db_id:
        _record_query(req.db_id, req.question, sql, result, error, latency)
    return GenerateResponse(
        sql=sql,
        valid=is_read_only_sql(sql),
        execution_result=result,
        execution_error=error,
        latency=latency,
    )


@router.post("/execute-sql", response_model=ExecuteResponse)
async def execute_sql_endpoint(req: ExecuteRequest):
    if not is_read_only_sql(req.sql):
        raise HTTPException(status_code=400, detail="Query is not read-only or invalid")
    record = get_registry().get_database(req.db_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        backend = DatabaseBackend.from_registry(record)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start = time.perf_counter()
    result, error = await _run_backend_query(backend, req.sql)
    latency = time.perf_counter() - start
    _record_query(req.db_id, None, req.sql, result, error, latency, edited_sql=req.sql)
    return ExecuteResponse(
        sql=req.sql,
        valid=is_read_only_sql(req.sql),
        execution_result=result,
        execution_error=error,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@router.post("/admin/databases", status_code=status.HTTP_201_CREATED)
async def register_database(
    req: RegisterDatabaseRequest,
    _=_ADMIN_AUTH,
):
    if req.backend_type == "sqlite":
        try:
            db_path = validate_admin_db_path(req.db_path, _allowed_db_dirs())
            validate_db_file(db_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        record = DatabaseRecord(
            db_id=req.db_id,
            display_name=req.display_name,
            backend_type="sqlite",
            db_path=str(db_path),
            connection_env=None,
            schema_text="",
            description=req.description,
            created_at=None,
            updated_at=None,
        )
        try:
            backend = DatabaseBackend.from_registry(record)
            schema_text = backend.get_schema()
        except Exception as e:  # noqa: BLE001 - DB backends may raise diverse errors
            raise HTTPException(
                status_code=400, detail=f"Failed to introspect SQLite database: {e}"
            )

        get_registry().register_database(
            db_id=req.db_id,
            display_name=req.display_name,
            backend_type="sqlite",
            db_path=str(db_path),
            schema_text=schema_text,
            description=req.description,
        )
        return {"db_id": req.db_id, "backend_type": "sqlite", "db_path": str(db_path)}

    # PostgreSQL / MySQL
    record = DatabaseRecord(
        db_id=req.db_id,
        display_name=req.display_name,
        backend_type=req.backend_type,
        db_path=None,
        connection_env=req.connection_env,
        schema_text="",
        description=req.description,
        created_at=None,
        updated_at=None,
    )
    try:
        backend = DatabaseBackend.from_registry(record)
        backend.validate()
        schema_text = backend.get_schema()
    except Exception as e:  # noqa: BLE001 - DB backends may raise diverse errors
        raise HTTPException(
            status_code=400,
            detail=f"Failed to connect to {req.backend_type} database: {e}",
        )

    get_registry().register_database(
        db_id=req.db_id,
        display_name=req.display_name,
        backend_type=req.backend_type,
        connection_env=req.connection_env,
        schema_text=schema_text,
        description=req.description,
    )
    return {
        "db_id": req.db_id,
        "backend_type": req.backend_type,
        "connection_env": req.connection_env,
    }


@router.post("/admin/databases/upload", status_code=status.HTTP_201_CREATED)
async def upload_database(
    db_id: str = Form(...),
    display_name: str = Form(...),
    file: UploadFile = _UPLOAD_FILE,
    description: str = Form(""),
    _=_ADMIN_AUTH,
):
    if not validate_db_id(db_id):
        raise HTTPException(status_code=400, detail="Invalid db_id")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{db_id}.sqlite"
    try:
        content = await file.read()
        await asyncio.to_thread(dest.write_bytes, content)
        validate_db_file(dest)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - catch any upload processing error
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to process upload: {e}")

    try:
        record = DatabaseRecord(
            db_id=db_id,
            display_name=display_name,
            backend_type="sqlite",
            db_path=str(dest),
            connection_env=None,
            schema_text="",
            description=description,
            created_at=None,
            updated_at=None,
        )
        schema_text = DatabaseBackend.from_registry(record).get_schema()
    except Exception as e:  # noqa: BLE001 - DB backends may raise diverse errors
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to introspect uploaded database: {e}"
        )

    get_registry().register_database(
        db_id=db_id,
        display_name=display_name,
        backend_type="sqlite",
        db_path=str(dest),
        schema_text=schema_text,
        description=description,
    )
    return {"db_id": db_id, "backend_type": "sqlite", "db_path": str(dest)}


@router.get("/admin/databases", response_model=list[DatabaseDetail])
async def admin_list_databases(_=_ADMIN_AUTH):
    return get_registry().list_databases_admin()


@router.get("/admin/databases/{db_id}", response_model=DatabaseDetail)
async def admin_get_database(db_id: str, _=_ADMIN_AUTH):
    if not validate_db_id(db_id):
        raise HTTPException(status_code=400, detail="Invalid db_id")
    record = get_registry().get_database(db_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Database not found")
    return {
        "db_id": record.db_id,
        "display_name": record.display_name,
        "backend_type": record.backend_type,
        "db_path": record.db_path,
        "connection_env": record.connection_env,
        "schema_text": record.schema_text,
        "description": record.description,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


@router.delete("/admin/databases/{db_id}", status_code=status.HTTP_204_NO_CONTENT)
async def admin_delete_database(db_id: str, _=_ADMIN_AUTH):
    if not validate_db_id(db_id):
        raise HTTPException(status_code=400, detail="Invalid db_id")
    deleted = get_registry().delete_database(db_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Database not found")


@router.post("/admin/databases/{db_id}/refresh-schema")
async def refresh_database_schema(db_id: str, _=_ADMIN_AUTH):
    if not validate_db_id(db_id):
        raise HTTPException(status_code=400, detail="Invalid db_id")
    record = get_registry().get_database(db_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        backend = DatabaseBackend.from_registry(record)
        schema_text = backend.get_schema()
    except Exception as e:  # noqa: BLE001 - DB backends may raise diverse errors
        raise HTTPException(
            status_code=400, detail=f"Failed to introspect database: {e}"
        )
    get_registry().update_schema(db_id, schema_text)
    return {"db_id": db_id, "schema_text": schema_text}
