import numpy as np
from sentence_transformers import SentenceTransformer

class LemmaVectorizer:
    def __init__(self, keyed_vectors:dict=None):
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.keyed_vectors = keyed_vectors if keyed_vectors else dict()

    def __call__(self, X: str) -> np.ndarray:
        v = self.keyed_vectors.get(X)#.lemma_.lower())
        if v is not None:
            return np.asarray(v, dtype=float)

        # if X not in self.keyed_vectors, add it
        self.keyed_vectors[X] = self.embedding_model.encode(X)
        return self.keyed_vectors[X]