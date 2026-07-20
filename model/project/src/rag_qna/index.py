from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_qna.documents import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class VectorIndex:
    def __init__(self, chunks: list[DocumentChunk], embeddings: np.ndarray, model_name: str):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embeddings = _normalize(embeddings.astype(np.float32))
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @classmethod
    def from_chunks(cls, chunks: list[DocumentChunk], model_name: str = DEFAULT_EMBEDDING_MODEL) -> "VectorIndex":
        model = SentenceTransformer(model_name)
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        index = cls(chunks=chunks, embeddings=embeddings, model_name=model_name)
        index.model = model
        return index

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        metadata_path = index_dir / "chunks.json"
        embeddings_path = index_dir / "embeddings.npy"
        config_path = index_dir / "config.json"

        if not metadata_path.exists() or not embeddings_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Index files are missing in: {index_dir}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**row) for row in rows]
        embeddings = np.load(embeddings_path)
        return cls(chunks=chunks, embeddings=embeddings, model_name=config["embedding_model"])

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (index_dir / "config.json").write_text(
            json.dumps({"embedding_model": self.model_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(index_dir / "embeddings.npy", self.embeddings)

    def search(self, query: str, top_k: int = 4) -> list[tuple[DocumentChunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = _normalize(query_embedding.astype(np.float32))[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[index], float(scores[index])) for index in top_indices]


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, a_min=1e-12, a_max=None)

