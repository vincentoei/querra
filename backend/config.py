"""Shared path and runtime configuration."""

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


# Load .env into os.environ, but only override values that are unset or empty.
# This preserves test overrides and deployment env vars while still filling in
# local dev values from the .env file.
_env_vars = dotenv_values(PROJECT_ROOT / ".env")
for _key, _value in _env_vars.items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "spider_extracted" / "spider_data"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "databases"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_TRAIN = PROCESSED_DIR / "train.jsonl"
PROCESSED_DEV = PROCESSED_DIR / "dev.jsonl"
TABLES_FILE = RAW_DIR / "tables.json"
TRAIN_FILE = RAW_DIR / "train_spider.json"
DEV_FILE = RAW_DIR / "dev.json"
SOURCE_DB_DIR = RAW_DIR / "database"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or backend/.env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Paths
    querra_db: Path = DATA_DIR / "querra.db"

    # Model / inference
    base_model: str = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
    model_name: str = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
    adapter_dir: Path = MODEL_DIR / "best-model"
    adapter_path: str | None = None
    max_seq_length: int = 2048
    max_result_rows: int = 100
    execute_timeout: float = 10.0

    # LoRA defaults
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default=["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 1e-4
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # Few-shot / schema linking
    few_shot_k: int = 3
    schema_linking_top_k_tables: int = 3
    schema_linking_top_k_columns: int = 5
    schema_linking_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    question_schema_relevance_threshold: float = 0.35

    # Security
    admin_api_key: str
    allowed_db_dirs: str = str(DATA_DIR)

    # Runtime toggles
    skip_model_load: bool = False
    use_cpu: bool = False
    hf_token: str | None = None
    wandb_api_key: str | None = None
    cors_origins: str = "*"
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    @field_validator("target_modules", mode="before")
    @classmethod
    def _parse_target_modules(cls, value):
        """Allow comma-separated target_modules from environment variables."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def schema_linking_embeddings_cache(self) -> Path:
        return PROCESSED_DIR / "schema_embeddings.pkl"

    @property
    def effective_adapter_path(self) -> str | None:
        """Return the adapter path to use, falling back to adapter_dir."""
        if self.adapter_path is not None:
            return self.adapter_path
        path = self.adapter_dir
        if path.exists():
            return str(path)
        return None


settings = Settings()
