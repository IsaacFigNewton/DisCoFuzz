from typing import Optional, Dict
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import nltk
from .types import TokenDataclass

nltk.download("wordnet")
from nltk.corpus import wordnet as wn

from sentence_transformers import SentenceTransformer
from .BaseEmbeddingModel import BaseEmbeddingModel

from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer
from .wn_lemma_vect_enrichment.FuzzyLemmaEnricher import FuzzyLemmaEnricher

class TensorStore:
    def __init__(self,
            embedding_model: BaseEmbeddingModel,
            fuzzifier:Optional[FuzzyFourierTensorTransformer],
            cache_embeddings:bool=True,
        ):
        self.embedding_model = embedding_model
        self.fuzzifier = fuzzifier or FuzzyFourierTensorTransformer()
        
        self.cache_embeddings = cache_embeddings
        self.keyed_tensors:Dict[str, Dict[str, tf.Tensor]] = dict()

        self.lemma_enricher = None

    def _fuzzify_dim_reduced_vect(self, vect: np.ndarray):
        embedding = vect.squeeze()
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
        return tf.convert_to_tensor(self.fuzzifier.fuzzify(embedding), dtype=tf.complex64)

    def _embed_text(self, text:str) -> tf.Tensor:
        return self._fuzzify_dim_reduced_vect(self.embedding_model.encode(text))

    def populate_with_wn_defaults(self):
        self.lemma_enricher = FuzzyLemmaEnricher(
            embedding_model=self.embedding_model,
            fuzzifier=self.fuzzifier,
        )
        self.keyed_tensors = self.lemma_enricher.get_lemma_embeddings()


    def __call__(self, tok: TokenDataclass) -> tf.Tensor:
        """
        take a string,
        embed it,
        store it if cache_embeddings=True
        return the tensor
        """
        pos = tok.pos_
        # clean the SpaCy POS tags
        if pos in {"PROPN", "NOUN", "PRON"}:
            pos = "NOUN"
        
        # if there's a cached embedding for the input text
        if self.keyed_tensors.get(tok.text.lower()) is not None:
            # if the embedding for the lemma with this particular POS is not in the dict
            if pos not in self.keyed_tensors[tok.text.lower()]:
                # otherwise embed it
                embedding = self._embed_text(tok.text.lower())
                self.keyed_tensors[tok.text.lower()][pos] = embedding
            return self.keyed_tensors[tok.text.lower()][pos]

        # otherwise embed it
        embedding = self._embed_text(tok.text.lower())
        # store it if desired
        if self.cache_embeddings:
            self.keyed_tensors[tok.text.lower()] = dict()
            self.keyed_tensors[tok.text.lower()][pos] = embedding
        # return embedding
        return embedding