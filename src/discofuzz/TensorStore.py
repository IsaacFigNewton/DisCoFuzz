from typing import Optional, Dict
import numpy as np
from spacy.tokens import Token
import tensorflow as tf

from sentence_transformers import SentenceTransformer
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

class TensorStore:
    def __init__(self,
            embedding_model: Optional[SentenceTransformer]=None,
            fuzzifier:Optional[FuzzyFourierTensorTransformer]=None,
            cache_embeddings:bool=True
        ):
        self.embedding_model = embedding_model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.fuzzifier = fuzzifier or FuzzyFourierTensorTransformer()
        self.cache_embeddings = cache_embeddings
        self.keyed_tensors:Dict[str, tf.Tensor] = dict()

    def __call__(self, text: str) -> tf.Tensor:
        """
        take a string,
        embed it,
        store it if cache_embeddings=True
        return the tensor
        """
        # if there's a cached embedding for the input text
        if self.keyed_tensors.get(text) is not None:
            return self.keyed_tensors[text]
        # otherwise embed it
        embedding = self.embedding_model.encode(text)
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
        embedding = tf.convert_to_tensor(self.fuzzifier.fuzzify(embedding), dtype=tf.complex64)
        # store it if desired
        if self.cache_embeddings:
            self.keyed_tensors[text] = embedding
        # return embedding
        return embedding