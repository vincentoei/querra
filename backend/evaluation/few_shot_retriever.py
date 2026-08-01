"""Embedding-based few-shot example retrieval for prompt engineering."""

import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util


class FewShotRetriever:
    def __init__(self, train_path: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.train_path = Path(train_path)
        self.model = SentenceTransformer(model_name)
        self.examples: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, cache_path: Path | None = None) -> None:
        with open(self.train_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f]
        questions = [ex["question"] for ex in self.examples]
        self.embeddings = self.model.encode(
            questions, convert_to_tensor=False, show_progress_bar=True
        )
        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((self.examples, self.embeddings), f)

    def load_index(self, cache_path: Path) -> None:
        with open(cache_path, "rb") as f:
            self.examples, self.embeddings = pickle.load(f)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        if self.embeddings is None:
            raise RuntimeError("Index not built or loaded")
        query_emb = self.model.encode(question, convert_to_tensor=False)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        scores = np.asarray(scores).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        return [self.examples[i] for i in top_k]
