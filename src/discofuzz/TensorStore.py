from typing import Optional, Dict
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import nltk
import pandas as pd

nltk.download("wordnet")
from nltk.corpus import wordnet as wn

from sentence_transformers import SentenceTransformer
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

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
        self.keyed_tensors:Dict[str, tf.Tensor] = dict()
        
        if wn_lemma_defaults:
            self._init_wordnet_embeddings()


    def _fuzzify_dim_reduced_vect(self, vect: np.ndarray):
        embedding = vect.squeeze()
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
        return tf.convert_to_tensor(self.fuzzifier.fuzzify(embedding), dtype=tf.complex64)

    def _embed_text(self, text:str):
        embedding = self.embedding_model.encode(text)
        embedding = self.dim_reduc.transform(embedding.reshape(1, -1))
        return self._fuzzify_dim_reduced_vect(embedding)

    def _init_wordnet_embeddings(self):
        print("Initializing TensorStore instance with wordnet lemma embeddings as defaults...")
        print("Embedding all the wordnet lemmas...")
        lemma_vects = self.embedding_model.encode(list(wn.all_lemma_names()))
        # fit and reduce lemma vects
        print("Performing dimensionality reduction on all the wordnet lemmas...")
        lemma_vects_reduced = self.dim_reduc.fit_transform(lemma_vects)
        self.fitted = True
        
        # store the fuzzified, dimensionality-reduced lemma embeddings to lemma_tensors
        print("Fuzzifying all the dimensionality-reduced wordnet lemmas...")
        lemma_tensors = dict()
        for lemma, vect in zip(wn.all_lemma_names(), lemma_vects_reduced):
            lemma_tensors[lemma] = self._fuzzify_dim_reduced_vect(vect)
        
        # embed each wordnet synset as the union of its lemmas' tensors
        print("Getting fuzzy tensor embedding for all the wordnet synsets...")
        for synset in wn.all_eng_synsets():
            s_name = synset.name()
            s_lemmas = [str(l.name()) for l in synset.lemmas()]

            # # set lemma tensor to union of lemma embeddings
            # self.keyed_tensors[s_name] = self.fuzzifier.iterated_union([
            #     lemma_tensors[l]
            #     for l in s_lemmas
            # ])
            # set lemma tensor to mean of lemma embeddings
            s_lemma_tens = [
                lemma_tensors[l]
                for l in s_lemmas
                if l in lemma_tensors
            ]
            if len(s_lemma_tens) > 0:
                self.keyed_tensors[s_name] = self.fuzzifier.iterated_union(s_lemma_tens)


    def fit(self, X:np.ndarray, y=None):
        if not self.fitted:
            self.dim_reduc.fit_transform(X)
            self.fitted = True
        else:
            raise Exception("Cannot re-fit the dimensionality reduction model that has already been fit.")


    def __call__(self, text: str) -> tf.Tensor:
        """
        take a string,
        embed it,
        store it if cache_embeddings=True
        return the tensor
        """
        if not self.fitted:
            raise Exception("TensorStore.dim_reduc must be fit prior to calling.")

        # if there's a cached embedding for the input text
        if self.keyed_tensors.get(text) is not None:
            return self.keyed_tensors[text]
        # otherwise embed it
        embedding = self._embed_text(text)
        # store it if desired
        if self.cache_embeddings:
            self.keyed_tensors[text] = embedding
        # return embedding
        return embedding