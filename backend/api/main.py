"""FastAPI inference service for Querra Text-to-SQL."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from api.state import _state
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.skip_model_load:
        logger.info("SKIP_MODEL_LOAD=1: model loading disabled")
    else:
        from utils.inference import load_model_and_tokenizer

        model_name = settings.model_name
        adapter_path = settings.effective_adapter_path
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

_cors_list = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
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

app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "model": settings.model_name,
    }


from api.v1 import router as v1_router

app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
