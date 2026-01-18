from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import spacy
import numpy as np
from spacy.tokens import Token, Doc
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import seaborn as sns
import wget as wget
import zipfile

from .config import *

# Import DisCoFuzz package classes
from .constants import *
from .BaseEmbeddingModel import BaseEmbeddingModel
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

class EvalHarness:
    def __init__(self,
            embedding_model: BaseEmbeddingModel,
            spacy_model: Any,
            fuzzifier: FuzzyFourierTensorTransformer
        ):
        """
        :param spacy_model: Description
        :type spacy_model: Optional[str]

        :param embedding_model: Description
        :type embedding_model: Optional[str]
        
        """
        # use the GPU for TensorFlow operations if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {gpus}")
        
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.fuzzifier = fuzzifier
        self.sent_embeddings: List[np.ndarray] = list()
        self.fuzzy_sent_embeddings: List[List[tf.Tensor]] = list()
        self.tok_embeddings: List[np.ndarray] = list()
        self.fuzzy_tok_embeddings: List[List[tf.Tensor]] = list()

    
    def fit(self, X: pd.DataFrame):
        # get sentence embedding baseline
        sents = list()
        for i in [1, 2]:
            # get a list of the sentences to embed
            sents.extend(X[f"sent_{i}"].to_list())
        sbert_sent_vects = self.embedding_model.fit_transform(sents)
        self.sent_embeddings = [
            sbert_sent_vects[:len(X)],
            sbert_sent_vects[len(X):]
        ]

        # get fuzzified sentence embedding baseline
        for i in [1, 2]:
            self.fuzzy_sent_embeddings.append(
                pd.Series([np.array(emb) for emb in self.sent_embeddings[i-1]]).apply(self.fuzzifier.fuzzify)
            )
        
        # get average of individual token embeddings
        mean_tok_vects = dict()
        for j in [1, 2]:
            mean_tok_vects[j] = list()
            for i, row in X.iterrows():
                token_embs = [
                    self.embedding_model.encode([token.text])
                    for token in self.spacy_model(row[f"sent_{j}"])
                    if not token.is_punct
                ]
                if token_embs:
                    # Concatenate all token embeddings and take mean
                    all_embs = np.concatenate(token_embs, axis=0)
                    mean_tok_vects[j].append(np.mean(all_embs, axis=0))
                else:
                    mean_tok_vects[j].append(np.zeros(DEFAULTS["n_components"]))
        self.tok_embeddings = [np.array(mean_tok_vects[1]), np.array(mean_tok_vects[2])]

        # get fuzzified mean token embedding baseline
        for i in [1, 2]:
            self.fuzzy_tok_embeddings.append(
                pd.Series([np.array(emb) for emb in self.tok_embeddings[i-1]]).apply(self.fuzzifier.fuzzify)
            )
        
    def get_sbert_sentence_baseline(self) -> np.ndarray:
        # Calculate similarity - returns diagonal of similarity matrix
        return np.diag(cosine_similarity(self.sent_embeddings[0], self.sent_embeddings[1]))
    

    def get_sbert_token_baseline(self) -> np.ndarray:
        # Add SBERT token-level baseline - returns diagonal of similarity matrix
        return np.diag(cosine_similarity(self.tok_embeddings[0], self.tok_embeddings[1]))
    
