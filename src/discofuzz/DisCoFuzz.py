from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import spacy
import numpy as np
from spacy.tokens import Token, Doc
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .config import DEFAULTS, COL_PREFIXES

# Import DisCoFuzz package classes
from .constants import *

from .BaseEmbeddingModel import BaseEmbeddingModel
from .embedding_composition_classes import (
    DepTreeBuilder,
    SpacyDependencyComposer
)
from .TensorStore import TensorStore
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

class DisCoFuzz:
    def __init__(self,
            embedding_model: BaseEmbeddingModel,
            fuzzification_kernel_size: Optional[int],
            spacy_model: Optional[str],
            enrich_lemmas_with_wn: Optional[bool],
            keep_branch_json: Optional[bool],
        ):
        """
        Creates a new DisCoFuzz instance
        
        :param n_components: Description
        :type n_components: Optional[int]

        :param fuzzification_kernel_size: Description
        :type fuzzification_kernel_size: Optional[int]

        :param spacy_model: Description
        :type spacy_model: Optional[str]

        :param embedding_model: Description
        :type embedding_model: Optional[str]
        
        :param enrich_lemmas_with_wn: Whether or not to refine lemma embeddings
            by using WordNet lexical relations associated with it.
        :type enrich_lemmas_with_wn: Optional[bool]
        
        :param enrich_lemmas_with_wn: Whether or not
            to keep dependency parse branches' nested Tuple[JSON] representations
            when aligning a dataset with the harness.
        :type enrich_lemmas_with_wn: Optional[bool]
        """
        # use the GPU for TensorFlow operations if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {gpus}")

        self.embedding_model = embedding_model
        self.fuzzification_kernel_size = fuzzification_kernel_size or DEFAULTS["fuzzification_kernel_size"]
        self.spacy_model = spacy.load(spacy_model or DEFAULTS["spacy_model"])
        self.enrich_lemmas_with_wn = enrich_lemmas_with_wn or DEFAULTS["enrich_lemmas_with_wn"]
        self.keep_branch_json = keep_branch_json or DEFAULTS["keep_branch_json"]

        self.fuzzifier = FuzzyFourierTensorTransformer(
            kernel_size=        self.fuzzification_kernel_size
        )
        self.lemma_vectorizer = TensorStore(
            embedding_model=    self.embedding_model,
            fuzzifier=          self.fuzzifier,
        )
        self.tree_builder = DepTreeBuilder(
            spacy_model=        spacy_model,
            lemma_vectorizer=   self.lemma_vectorizer
        )
    

    @staticmethod
    def _get_fuzzy_emb_col(s: str, i: int):
        return f"sent_{i}_fuzzy_{s}"


    def _build_tensor_trees(self, dataset: pd.DataFrame) -> Dict[str, pd.Series]:
        # for all sentences...
        branch_embedding_dict = dict()
        for i in [1, 2]:
            # extract all relevant dependency branches
            branch = dataset.apply(
                lambda x: self.tree_builder.extract_branch(
                    x[f"sent_{i}"],
                    x[f"tok_idx_{i}"]
                ),
                axis=1
            )
            # optionally include tokens' pre-embedding JSON representations
            if self.keep_branch_json:
                branch_embedding_dict[f"{COL_PREFIXES['branch']}_text_{i}"] = branch
            # get the dependency tree branch's associated FUZZY embedding
            branch_embedding_dict[f"{COL_PREFIXES['branch']}_embedding_{i}"] = branch.apply(self.tree_builder.get_branch_tuple_embedding)
        
        return branch_embedding_dict


    def fit(self, X: pd.DataFrame):
        # enrich the tensor store with dimensionality-reduced+fuzzified wordnet info
        if self.enrich_lemmas_with_wn:
            self.lemma_vectorizer.populate_with_wn_defaults()
        return X
    

    def _align(self, X:pd.DataFrame) -> pd.DataFrame:        
        # extract dependency trees for the sentences,
        #   embed each token in the trees,
        #   and store the branches as nested tuples of tensors
        for col_name, col_values in self._build_tensor_trees(X).items():
            if col_name in X:
                print(f"WARNING: {col_name} already exists in X. Skipping for now...")
            else:
                X[col_name] = pd.Series(col_values)
        
        # get fuzzified embedding baseline
        for i in [1, 2]:
            X[self._get_fuzzy_emb_col("None", i)] = X[f"sent_{i}_embedding"].apply(lambda x: self.lemma_vectorizer._fuzzify_dim_reduced_vect(x))

        return X


    def _get_branch_embedding_args(self, X:pd.DataFrame) -> Tuple[pd.Series, pd.Series]: 
        # combine the branch embeddings and POS tags into args to pass to each embedding composition model
        tup_emb_args = list()
        for i in [1, 2]:
            tup_emb_args.append(X[[f"{COL_PREFIXES['branch']}_embedding_{i}", "pos"]].apply(lambda x: tuple(x), axis=1))
        
        return tup_emb_args[0], tup_emb_args[1]


    def predict(self, X: pd.DataFrame, strategy:str):
        # add the dependency parse branches/tensor trees to the dataframe
        X = self._align(X)
        tup_emb_args = self._get_branch_embedding_args(X)

        composer = SpacyDependencyComposer(strategy, self.fuzzifier)
        for i in [1, 2]:
            # compose embeddings
            X[self._get_fuzzy_emb_col(strategy, i)] = tup_emb_args[i-1].apply(lambda x: composer(x[0], x[1]))
        
        return X
        
        
    def predict_batch(self, X: pd.DataFrame, strategies: List[str]):
        # add the dependency parse branches/tensor trees to the dataframe
        X = self._align(X)
        tup_emb_args = self._get_branch_embedding_args(X)

        # get compositional embeddings for glosses using different strategies
        for s in strategies:
            print(f"\tComposing embeddings with {s} approach...")
            composer = SpacyDependencyComposer(s, self.fuzzifier)
            for i in [1, 2]:
                # compose embeddings
                X[self._get_fuzzy_emb_col(s, i)] = tup_emb_args[i-1].apply(lambda x: composer(x[0], x[1]))
        
        return X