import json
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import VECTOR_STORE_PATH, EMBEDDING_MODEL


class VectorIndex:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index_path = Path(VECTOR_STORE_PATH) / "faiss.index"
        self.metadata_path = Path(VECTOR_STORE_PATH) / "metadata.json"
        self.index = None
        self.metadata = []

    def add_texts(self, texts, metadata_list):
        if not texts:
            return

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings).astype("float32")
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def search(self, query, k=5):
        if self.index is None:
            self.load()
        if self.index is None:
            return []

        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    "distance": float(dist),
                    "metadata": self.metadata[int(idx)]
                })
        return results
