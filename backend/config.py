"""Shared path and model configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "spider_extracted" / "spider_data"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "databases"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_TRAIN = PROCESSED_DIR / "train.jsonl"
PROCESSED_DEV = PROCESSED_DIR / "dev.jsonl"
TABLES_FILE = RAW_DIR / "tables.json"
TRAIN_FILE = RAW_DIR / "train_spider.json"
DEV_FILE = RAW_DIR / "dev.json"
SOURCE_DB_DIR = RAW_DIR / "database"

BASE_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
ADAPTER_DIR = MODEL_DIR / "best-model"

MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

FEW_SHOT_K = 3
