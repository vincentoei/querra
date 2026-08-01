"""Schema-aware RAG: select relevant tables before prompt construction."""

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, util

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SCHEMA_LINKING_EMBED_MODEL
from utils.schema import _map_type, build_schema_for_tables


class SchemaLinker:
    """Retrieve relevant tables from a database schema for a given question.

    Uses keyword matching plus sentence-transformer embedding similarity, then
    expands the selection via foreign-key closure so JOINs are not broken.
    """

    def __init__(
        self,
        tables: dict[str, dict[str, Any]],
        model_name: str | None = SCHEMA_LINKING_EMBED_MODEL,
        cache_path: Path | str | None = None,
    ):
        self.tables = tables
        self.model = SentenceTransformer(model_name) if model_name else None
        self.cache_path = Path(cache_path) if cache_path else None
        self._candidates: dict[str, list[tuple[str, str]]] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    def _build_candidates(self, db_id: str) -> list[tuple[str, str]]:
        """Return [(table_name, candidate_text)] for every table in the DB."""
        entry = self.tables[db_id]
        table_names = entry["table_names_original"]
        column_names = entry["column_names_original"]
        column_types = entry["column_types"]

        table_cols: dict[int, list[tuple[str, str]]] = {}
        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1 or col_name == "*":
                continue
            if table_idx < 0 or table_idx >= len(table_names):
                continue
            table_name = table_names[table_idx]
            if table_name == "sqlite_sequence":
                continue
            table_cols.setdefault(table_idx, []).append(
                (col_name, _map_type(column_types[col_idx]))
            )

        candidates = []
        for table_idx in sorted(table_cols.keys()):
            table_name = table_names[table_idx]
            cols = table_cols[table_idx]
            col_text = ", ".join(f"{name} ({ctype})" for name, ctype in cols)
            candidates.append(
                (table_name, f"Table {table_name} has columns: {col_text}.")
            )
        return candidates

    def _ensure_embeddings(self, db_id: str) -> None:
        """Load from cache or encode table candidates for the DB."""
        if db_id in self._embeddings:
            return
        if self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
                if db_id in self._embeddings:
                    return
            except Exception:
                pass
        candidates = self._build_candidates(db_id)
        self._candidates[db_id] = candidates
        if candidates and self.model is not None:
            texts = [text for _, text in candidates]
            self._embeddings[db_id] = self.model.encode(
                texts, show_progress_bar=False, convert_to_tensor=False
            )
        else:
            self._embeddings[db_id] = np.zeros((len(candidates), 0))
        if self.cache_path:
            self.save_cache()

    def build_cache(self) -> None:
        """Encode and cache embeddings for every database in tables."""
        for db_id in self.tables:
            self._ensure_embeddings(db_id)
        if self.cache_path:
            self.save_cache()

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump((self._candidates, self._embeddings), f)

    def load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        with open(self.cache_path, "rb") as f:
            self._candidates, self._embeddings = pickle.load(f)

    @staticmethod
    def _matches_name(name: str, question_lower: str) -> bool:
        """Whole-word match of a table/column name inside the question."""
        key = re.escape(name.lower())
        return bool(re.search(r"\b" + key + r"\b", question_lower))

    def _keyword_selected(self, db_id: str, question: str) -> set[str]:
        """Select tables whose name or columns appear literally in the question."""
        entry = self.tables[db_id]
        table_names = entry["table_names_original"]
        column_names = entry["column_names_original"]
        q = question.lower()
        selected: set[str] = set()

        for table_name in table_names:
            if table_name == "sqlite_sequence":
                continue
            if self._matches_name(table_name, q):
                selected.add(table_name)

        for table_idx, col_name in column_names:
            if table_idx == -1 or col_name == "*":
                continue
            table_name = table_names[table_idx]
            if self._matches_name(col_name, q):
                selected.add(table_name)

        return selected

    def _foreign_key_closure(
        self, selected: set[str], entry: dict[str, Any]
    ) -> set[str]:
        """Expand selected tables to include FK-linked tables."""
        table_names = entry["table_names_original"]
        column_names = entry["column_names_original"]
        foreign_keys = entry["foreign_keys"]
        name_to_idx = {name: i for i, name in enumerate(table_names)}
        selected_indices = {
            name_to_idx[n] for n in selected if n in name_to_idx
        }

        changed = True
        while changed:
            changed = False
            for col_a, col_b in foreign_keys:
                table_a = column_names[col_a][0]
                table_b = column_names[col_b][0]
                if table_a == -1 or table_b == -1:
                    continue
                if table_a in selected_indices and table_b not in selected_indices:
                    selected_indices.add(table_b)
                    changed = True
                elif table_b in selected_indices and table_a not in selected_indices:
                    selected_indices.add(table_a)
                    changed = True

        return {table_names[i] for i in selected_indices}

    def link(
        self,
        db_id: str,
        question: str,
        top_k: int = 3,
        use_keywords: bool = True,
    ) -> set[str]:
        """Return the set of selected table names for the question."""
        entry = self.tables.get(db_id)
        if not entry:
            return set()

        selected: set[str] = set()
        if use_keywords:
            selected.update(self._keyword_selected(db_id, question))

        candidates = self._build_candidates(db_id)
        if top_k > 0 and candidates and self.model is not None:
            self._ensure_embeddings(db_id)
            embeddings = self._embeddings.get(db_id)
            if embeddings is not None and embeddings.shape[0] > 0:
                query_emb = self.model.encode(
                    question, show_progress_bar=False, convert_to_tensor=False
                )
                scores = util.cos_sim(query_emb, embeddings)[0]
                scores = np.asarray(scores).flatten()
                k = min(top_k, len(candidates))
                top_indices = np.argsort(scores)[::-1][:k]
                for i in top_indices:
                    selected.add(candidates[i][0])

        selected = self._foreign_key_closure(selected, entry)

        if not selected:
            # Fallback to the full schema if nothing was selected.
            selected = {
                name for name in entry["table_names_original"] if name != "sqlite_sequence"
            }

        return selected

    def build_schema(
        self,
        db_id: str,
        question: str,
        top_k: int = 3,
        use_keywords: bool = True,
        include_fks: bool = True,
    ) -> tuple[str, set[str]]:
        """Return (filtered_schema_string, selected_table_names)."""
        entry = self.tables.get(db_id)
        if not entry:
            return "", set()
        selected = self.link(db_id, question, top_k=top_k, use_keywords=use_keywords)
        schema = build_schema_for_tables(entry, selected, include_fks=include_fks)
        return schema, selected


if __name__ == "__main__":
    # Keyword + FK-closure sanity check; no embedding model needed.
    tables = {
        "test_db": {
            "table_names_original": ["users", "orders", "products"],
            "column_names_original": [
                [0, "id"],
                [0, "name"],
                [1, "order_id"],
                [1, "user_id"],
                [1, "product_id"],
                [2, "id"],
                [2, "title"],
            ],
            "column_types": ["number", "text", "number", "number", "number", "number", "text"],
            "primary_keys": [0, 2, 5],
            "foreign_keys": [[3, 0], [4, 5]],
        }
    }
    linker = SchemaLinker(tables, model_name=None)
    selected = linker.link("test_db", "What products did users order?", top_k=0)
    assert "users" in selected, selected
    assert "orders" in selected, selected
    assert "products" in selected, selected
    print("Schema linker sanity check passed.")
