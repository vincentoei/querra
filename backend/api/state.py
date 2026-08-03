"""Shared application state for the FastAPI service."""

from typing import Any

_state: dict[str, Any] = {}


def get_model() -> Any:
    """Return the loaded model, or raise if not loaded."""
    model = _state.get("model")
    if model is None:
        raise RuntimeError("Model not loaded")
    return model


def get_tokenizer() -> Any:
    tokenizer = _state.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")
    return tokenizer


def get_state() -> dict[str, Any]:
    return _state
