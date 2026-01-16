from typing import Tuple, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA

# Import DisCoFuzz package classes
from discofuzz.constants import *
from discofuzz.config import *

class BaseEmbeddingModel:
    def __init__(self,
            sentence_transformer: Optional[str],
            dim_reduc: Optional[PCA],
        ) -> None:
        self.sentence_transformer = SentenceTransformer(sentence_transformer or DEFAULTS["sentence_transformer"])
        self.dim_reduc = dim_reduc or PCA(n_components=DEFAULTS["n_components"])
    
    def fit_transform(self, X: List[str]) -> np.ndarray:
        return self.dim_reduc.fit_transform(self.sentence_transformer.encode(X))

    def encode(self, text: List[str]) -> np.ndarray:
        return self.dim_reduc.transform(self.sentence_transformer.encode(text))