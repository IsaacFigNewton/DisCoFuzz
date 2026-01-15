from typing import Optional, Dict, List
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import nltk
import pandas as pd

nltk.download("wordnet")
from nltk.corpus import wordnet as wn

from sentence_transformers import SentenceTransformer
from ..fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

LEMMA_ENRICHMENT_STRATEGIES = [
    "union-of-hyper-hypo-intersections"
]

class FuzzyLemmaEnricher:
    def __init__(self,
            embedding_model: SentenceTransformer,
            fuzzifier: FuzzyFourierTensorTransformer,
            dim_reduc=None,
        ):
        self.embedding_model = embedding_model
        self.fuzzifier = fuzzifier
        self.dim_reduc = dim_reduc
        
        self.keyed_tensors:Dict[str, Dict[str, tf.Tensor]] = dict()

    def _clean_lem_name(self, lemma):
        return lemma.name().replace("_", " ")
    
    def _clean_syn_name(self, synset):
        return synset.name()\
                        .split(".")[0]\
                        .replace("_", " ")

    def _fuzzify_dim_reduced_vect(self, vect: np.ndarray):
        embedding = vect.squeeze()
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
        return tf.convert_to_tensor(self.fuzzifier.fuzzify(embedding), dtype=tf.complex64)


    def _get_all_synset_tensors(self,
                                strategy=LEMMA_ENRICHMENT_STRATEGIES[0]
                            ) -> Dict[str, tf.Tensor]:
        match strategy:
            # embed each lemma as the union of the intersection of its synsets' hyper and hyponyms' fuzzy lemma tensors
            case LEMMA_ENRICHMENT_STRATEGIES[0]:
                # get all synsets' hypernyms' and hyponyms' cleaned names
                synset_info = {
                    synset.name(): {
                        "hypernyms": [self._clean_syn_name(h) for h in synset.hypernyms()],
                        "hyponyms": [self._clean_syn_name(h) for h in synset.hypernyms()],
                    }
                    for synset in wn.all_eng_synsets()
                    if synset is not None
                }

                # embed/fuzzify each synset's hyper and hyponyms
                synset_info = {
                    s: {
                        # intersect the fuzzy sets for all a synset's hypernyms
                        "hypernyms": self.fuzzifier.iterated_intersection([
                            self.keyed_tensors[l]
                            for l in syn_dict["hypernyms"]
                            if l in self.keyed_tensors
                        ]),
                        # union the fuzzy sets for all a synset's hyponyms
                        "hyponyms": self.fuzzifier.iterated_union([
                            self.keyed_tensors[l]
                            for l in syn_dict["hyponyms"]
                            if l in self.keyed_tensors
                        ])
                    }
                    for s, syn_dict in synset_info.items()
                }

                # intersect the intersection of the hypernyms' sets with the union of the hyponyms' sets
                synset_tensors = {
                    s: self.fuzzifier.intersection(
                        syn_dict["hypernyms"],
                        syn_dict["hyponyms"]
                    )
                    for s, syn_dict in synset_info.items()
                }
                
                return {s: t for s, t in synset_tensors.items() if t is not None}
            
            case _:
                raise ValueError(f"Invalid strategy provided: {strategy}.\nExpected one of {LEMMA_ENRICHMENT_STRATEGIES}")


    def _enrich_tensor_store(self,
            lemma_tens: Dict[str, tf.Tensor],
            synset_tens: Dict[str, tf.Tensor]
        ) -> Dict[str, Dict[str, tf.Tensor]]:
        # TODO: check that keys of synset_tens correspond to real synsets

        # map lemmas to dicts of synsets and their associated embeddings
        lemma_syn_tens_map = {
            l: {
                s: synset_tens[s.name()]
                for s in wn.synsets(l)
                if s is not None
            }
            for l in lemma_tens.keys()
        }
        # map lemma+POS tag to list of synset tensors
        lemma_pos_tens_map = {
            l: {
                # get the mean of all noun synsets' tensors,
                #   use the lemma's base (unenriched) tensor as a fallback
                "n": tf.reduce_mean(tf.convert_to_tensor([
                    t
                    for s, t in lem_dict.items()
                    if ".n." in s
                ]), axis=0) or lemma_tens[l],
                # get the mean of all verb synsets' tensors,
                #   use the lemma's base (unenriched) tensor as a fallback
                "v": tf.reduce_mean(tf.convert_to_tensor([
                    t
                    for s, t in lem_dict.items()
                    if ".v." in s
                ]), axis=0) or lemma_tens[l]
            }
            for l, lem_dict in lemma_syn_tens_map.items()
        }

        return {
            l: pos_tens_dict
            for l, pos_tens_dict in lemma_pos_tens_map.items()
            if pos_tens_dict is not None
        }


    def get_lemma_embeddings(self, strategy:str = LEMMA_ENRICHMENT_STRATEGIES[0]) -> Dict[str, Dict[str, tf.Tensor]]:
        print("Initializing TensorStore instance with wordnet lemma embeddings as defaults...")
        print("Embedding all the wordnet lemmas...")
        all_lemmas = [
            l.replace("_", " ")
            for l in wn.all_lemma_names()
        ]
        lemma_vects = self.embedding_model.encode(all_lemmas)
        # fit and reduce lemma vects
        print("Performing dimensionality reduction on all the wordnet lemmas...")
        lemma_vects_reduced = self.dim_reduc.fit_transform(lemma_vects)
        self.fitted = True
        
        # store the fuzzified, dimensionality-reduced lemma embeddings to lemma_tensors
        lemma_tensors = dict()
        print("Fuzzifying all the dimensionality-reduced wordnet lemmas...")
        for lemma, vect in zip(all_lemmas, lemma_vects_reduced):
            lemma_tensors[lemma] = self._fuzzify_dim_reduced_vect(vect)

        print("Getting fuzzy tensor embeddings for all the wordnet synsets...")
        synset_tensors = self._get_all_synset_tensors(strategy=strategy)
        
        self.keyed_tensors = self._enrich_tensor_store(
            lemma_tens=lemma_tensors,
            synset_tens=synset_tensors
        )

        return self.keyed_tensors