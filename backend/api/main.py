"""FastAPI inference service for Querra Text-to-SQL."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from api.state import _state
from config import ADAPTER_DIR, BASE_MODEL

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("SKIP_MODEL_LOAD", "0") == "1":
        logger.info("SKIP_MODEL_LOAD=1: model loading disabled")
    else:
        from utils.inference import load_model_and_tokenizer

        model_name = os.environ.get("MODEL_NAME", BASE_MODEL)
        adapter_path = os.environ.get("ADAPTER_PATH", str(ADAPTER_DIR))
        if not Path(adapter_path).exists():
            adapter_path = None
        logger.info("Loading model %s (adapter=%s)", model_name, adapter_path)
        model, tokenizer = load_model_and_tokenizer(
            model_name, adapter_path=adapter_path
        )
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        logger.info("Model loaded")
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

_max_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
_window_seconds = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
app.add_middleware(
    RateLimitMiddleware,
    max_requests=_max_requests,
    window_seconds=_window_seconds,
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "model": os.environ.get("MODEL_NAME", BASE_MODEL),
    }


from api.v1 import router as v1_router

app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
