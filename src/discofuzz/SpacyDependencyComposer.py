import numpy as np
import spacy
import tensorflow as tf
from .constants import STRATEGIES
from .LemmaVectorizer import LemmaVectorizer
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

class SpacyDependencyComposer:
  modifiers = {"v", "a"}

  def __init__(self,
      strategy:str=None,
      fuzzifier=None
    ):
    if strategy not in STRATEGIES:
      raise ValueError(f"Unknown strategy: {strategy}")
    self.strategy = strategy
    self.fuzzifier = FuzzyFourierTensorTransformer() if fuzzifier is None else fuzzifier

  def _compose_tok_embedding(self, branch: tuple | np.ndarray, root_pos: str = 'n') -> tf.Tensor:

    if isinstance(branch, tuple):
      # get all the childrens'embeddings
      child_embeddings = [
          self._compose_tok_embedding(e)
          for e in branch
      ]

      # compose child embeddings based on strategy
      match self.strategy:

        case "mean":
          return tf.reduce_mean(tf.stack(child_embeddings), axis=0)

        case "intersection+mean":
          child_embeddings_intersected = [
              self.fuzzifier.intersection(current_tok_tens, c)
              for c in child_embeddings
          ]
          return tf.reduce_mean(tf.stack(child_embeddings_intersected), axis=0)

        case "intersection+union":
          child_embeddings_intersected = [
              self.fuzzifier.intersection(current_tok_tens, c)
              for c in child_embeddings
          ]
          return self.fuzzifier.iterated_union(child_embeddings_intersected)

        case "intersection+intersection":
          child_embeddings_intersected = [
              self.fuzzifier.intersection(current_tok_tens, c)
              for c in child_embeddings
          ]
          return self.fuzzifier.iterated_intersection(child_embeddings_intersected)

        case "selective_intersection+mean":
          # if the current token is a modifier of some kind, intersect it with all its children
          if root_pos in self.modifiers:
            child_embeddings_intersected = [
                self.fuzzifier.intersection(current_tok_tens, c)
                for c in child_embeddings
            ]
            return tf.reduce_mean(tf.stack(child_embeddings_intersected), axis=0)
          # otherwise, just return the mean of its children
          return tf.reduce_mean(tf.stack(child_embeddings), axis=0)

        case "selective_intersection+union":
          # if the current token is a modifier of some kind, intersect it with all its children
          if root_pos in self.modifiers:
            child_embeddings_intersected = [
                self.fuzzifier.intersection(current_tok_tens, c)
                for c in child_embeddings
            ]
            # return the union of these intersected children
            return self.fuzzifier.iterated_union(child_embeddings_intersected)
          # otherwise, just return the union of its children
          return self.fuzzifier.iterated_union(tf.stack(child_embeddings))

        case "selective_intersection+intersection+mean":
          # if the current token is a modifier of some kind, intersect it with all its children
          if root_pos in self.modifiers:
            child_embeddings_intersected = [
                self.fuzzifier.intersection(current_tok_tens, c)
                for c in child_embeddings
            ]
            # then intersect those intersections
            return self.fuzzifier.iterated_intersection(child_embeddings_intersected)
          # otherwise, just return the mean of its children
          return tf.reduce_mean(tf.stack(child_embeddings), axis=0)

        case _:
          raise ValueError(f"Unknown strategy: {self.strategy}")

    # if the token is a leaf
    else:
      return current_tok_tens

  def __call__(self,
      branch: tuple,
      root_pos: str
  ) -> tf.Tensor:
      # baseline
      if self.strategy is None:
          root_emb = [e for e in branch if type(e) == np.ndarray]
          return root_emb[0]

      return self._compose_tok_embedding(branch, root_pos)