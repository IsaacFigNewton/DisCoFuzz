from typing import Tuple, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA

# Import DisCoFuzz package classes
from discofuzz.constants import *
from discofuzz.config import *

class BaseEmbeddingModel:
    def __init__(self,
            sentence_transformer: Optional[str] = None,
            dim_reduc: Optional[PCA] = None,
        ) -> None:
        self.sentence_transformer = SentenceTransformer(sentence_transformer or DEFAULTS["sentence_transformer"])
        self.dim_reduc = dim_reduc or PCA(n_components=DEFAULTS["n_components"])
    
    def embed_reshape_if_needed(self, X: List[str]) -> np.ndarray:
        embeddings = self.sentence_transformer.encode(X)
        if embeddings.ndim == 1:
            return embeddings.reshape(1, -1)
        return embeddings

    def fit_transform(self, X: List[str]) -> np.ndarray:
        if not isinstance(X, list):
            raise ValueError(f"X was of type {type(X)}, was expecting list of strings.")
        return self.dim_reduc.fit_transform(self.embed_reshape_if_needed(X))

    def encode(self, X: List[str]) -> np.ndarray:
        if not isinstance(X, list):
            raise ValueError(f"X was of type {type(X)}, was expecting list of strings.")
        return self.dim_reduc.transform(self.embed_reshape_if_needed(X))