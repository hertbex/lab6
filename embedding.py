from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> None:
        self._encoder = SentenceTransformer(model)
        self.normalize = normalize

    def embed(self, text: str) -> np.ndarray:
        vec = self._encoder.encode(text, normalize_embeddings=self.normalize)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vectors = self._encoder.encode(texts, normalize_embeddings=self.normalize)
        return np.asarray(vectors, dtype=np.float32)


if __name__ == "__main__":
    embeddings = Embeddings()

    vector = embeddings.embed("Дерево отрезков можно расширить для поиска по векторам.")

    print(type(vector))
    print(vector.shape)
    print(vector[:10])
