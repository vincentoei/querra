# Querra — Text-to-SQL with QLoRA

Querra is an end-to-end text-to-SQL assistant that fine-tunes an open-source LLM to convert natural-language questions into executable SQL. It demonstrates the full lifecycle of a production-oriented LLM feature: data preprocessing, QLoRA fine-tuning, execution-based evaluation, safety hardening, and a Next.js frontend.

This project is designed as a portfolio case study. The codebase is intentionally small and modular so that readers can trace every decision from the raw dataset to the deployed API.

---

## What it does

1. Accept a natural-language question and a database schema.
2. Generate a SQL query using a fine-tuned `Qwen2.5-Coder-3B-Instruct` model.
3. Validate the SQL, block destructive queries, and execute it against the database.
4. Return the SQL, execution result, latency, and warnings if the question looks unrelated to the schema.

The headline metric is **execution accuracy**: does the generated query return the same result set as the gold query? Exact string match is too strict — two SQL strings can differ and still be correct.

---

## Results

Spider dev set (1034 examples):

| Model | Exact Match | Execution Accuracy | Avg Latency |
|---|---|---:|---:|
| Base zero-shot | 0.148 | 0.497 | 1.175 s |
| Prompt-engineered few-shot | 0.272 | 0.581 | 1.045 s |
| QLoRA fine-tuned (1 epoch) | 0.413 | **0.716** | 1.469 s |
| + greedy decoding + schema-aware post-processing + self-correction | 0.389 | **0.731** | 1.490 s |

Fine-tuning improved execution accuracy from ~50% (base) to ~73%. Inference-time optimizations added another ~1.5 percentage points without retraining.

See `docs/portfolio_case_study.md` for the full methodology, ablations, and error analysis.

---

## Tech stack

- **Frontend**
  - Next.js
  - React
  - TypeScript
  - Tailwind CSS
  - shadcn/ui

- **Backend**
  - Python
  - FastAPI
  - Uvicorn
  - Pydantic
  - SQLGlot

- **ML / DL**
  - PyTorch
  - Pandas
  - NumPy
  - Weights & Biases (W&B)

- **LLM / Generative AI**
  - Hugging Face Transformers
  - Qwen2.5-Coder-3B-Instruct
  - PEFT (LoRA, QLoRA)
  - bitsandbytes
  - sentence-transformers
  - Schema linking

- **Databases**
  - SQLite

- **DevOps & CI/CD**
  - Docker

---

## Quick start

### Backend

```bash
cd backend
cp .env.example .env
# Fill in HF_TOKEN, ADMIN_API_KEY, and optional WANDB_API_KEY
uv run --extra serve --extra db python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The backend automatically loads variables from `backend/.env` via `pydantic-settings`.

Register a SQLite database:

```bash
curl -X POST http://localhost:8000/api/v1/admin/databases \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "db_id": "chinook_1",
    "display_name": "Chinook Sample",
    "backend_type": "sqlite",
    "db_path": "/home/vincentoei/projects/querra/backend/data/databases/chinook_1/chinook_1.sqlite"
  }'
```

Register a PostgreSQL database (connection string lives in an env var):

```bash
export SUPABASE_<project>_POSTGRES_URL="postgresql://user:password@host:5432/db"

curl -X POST http://localhost:8000/api/v1/admin/databases \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "db_id": "<project>_pg",
    "display_name": "My Postgres",
    "backend_type": "postgres",
    "connection_env": "SUPABASE_<project>_POSTGRES_URL"
  }'
```

Replace `<project>` with a short identifier for the project (e.g. `bodify`).

Register a MySQL database (connection string lives in an env var):

```bash
export MYSQL_<project>_URL="mysql://user:password@host:3306/db_name"

curl -X POST http://localhost:8000/api/v1/admin/databases \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "db_id": "<project>_mysql",
    "display_name": "My MySQL",
    "backend_type": "mysql",
    "connection_env": "MYSQL_<project>_URL"
  }'
```

Replace `<project>` with a short identifier for the project (e.g. `shop`).

### Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend expects the backend at
`http://localhost:8000`; override with `NEXT_PUBLIC_API_URL` in
`frontend/.env.local`.

---

## Project structure

```text
backend/
  api/            # FastAPI routes and middleware
  data/           # Raw data, processed data, registry DB
  evaluation/     # Exact-match, execution-accuracy, component-match metrics
  models/         # LoRA adapters (best-model is the production adapter)
  scripts/        # Preprocessing, evaluation, schema-linking comparison
  training/       # QLoRA training script
  utils/          # Inference, post-processing, safety, schema linking, relevance
  tests/          # pytest suite
frontend/
  app/            # Next.js app router pages
  components/     # UI components and query form
docs/             # Case study, architecture, API reference, deployment guide
```

---

## Key design decisions

- **3B over 7B:** The 7B model was ~20× too slow to train on an RTX 5060 (8 GB VRAM). The 3B model keeps the same pipeline and fits comfortably.
- **Execution accuracy as headline metric:** Measures semantic correctness better than string match.
- **Review-first UI:** Users see generated SQL before running it, with warnings for low schema relevance or unknown identifiers.
- **Warn, don't block:** The relevance guardrail shows a warning but lets the user proceed, because no heuristic is perfect.

---

## License

MIT
