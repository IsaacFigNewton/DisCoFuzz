from typing import Optional, Dict
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import nltk
from .types import TokenDataclass

nltk.download("wordnet")
from nltk.corpus import wordnet as wn

from sentence_transformers import SentenceTransformer
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer
from .wn_lemma_vect_enrichment.FuzzyLemmaEnricher import FuzzyLemmaEnricher

class TensorStore:
    def __init__(self,
            embedding_model: Optional[SentenceTransformer]=None,
            fuzzifier:Optional[FuzzyFourierTensorTransformer]=None,
            dim_reduc=None,
            cache_embeddings:bool=True,
            wn_lemma_defaults:bool=True
        ):
        self.embedding_model = embedding_model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.fuzzifier = fuzzifier or FuzzyFourierTensorTransformer()
        self.dim_reduc = dim_reduc or PCA(n_components=128)
        self.fitted = False
        
        self.cache_embeddings = cache_embeddings
        self.keyed_tensors:Dict[str, Dict[str, tf.Tensor]] = dict()

        self.lemma_enricher = None
        if wn_lemma_defaults:
            self.lemma_enricher = FuzzyLemmaEnricher(
                embedding_model=self.embedding_model,
                fuzzifier=self.fuzzifier,
                dim_reduc=self.dim_reduc,
            )
            self.keyed_tensors = self.lemma_enricher.get_lemma_embeddings()
            self.dim_reduc = self.lemma_enricher.dim_reduc
            self.fitted = True

    def _fuzzify_dim_reduced_vect(self, vect: np.ndarray):
        embedding = vect.squeeze()
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
        return tf.convert_to_tensor(self.fuzzifier.fuzzify(embedding), dtype=tf.complex64)

    def _embed_text(self, text:str):
        embedding = self.embedding_model.encode(text)
        embedding = self.dim_reduc.transform(embedding.reshape(1, -1))
        return self._fuzzify_dim_reduced_vect(embedding)


    def fit(self, X:np.ndarray, y=None):
        if not self.fitted:
            self.dim_reduc.fit_transform(X)
            self.fitted = True
        else:
            raise Exception("Cannot re-fit the dimensionality reduction model that has already been fit.")


    def __call__(self, tok: TokenDataclass) -> tf.Tensor:
        """
        take a string,
        embed it,
        store it if cache_embeddings=True
        return the tensor
        """
        if not self.fitted:
            raise Exception("TensorStore.dim_reduc must be fit prior to calling.")

        # if there's a cached embedding for the input text
        if self.keyed_tensors.get(tok.text.lower()) is not None:
            return self.keyed_tensors[tok.text.lower()][tok.pos_]
        # otherwise embed it
        embedding = self._embed_text(tok.text.lower())
        # store it if desired
        if self.cache_embeddings:
            self.keyed_tensors[tok.text.lower()][tok.pos_] = embedding
        # return embedding
        return embedding