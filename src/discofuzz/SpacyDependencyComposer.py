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

  def _compose_tok_embedding(self, branch: tuple | tf.Tensor, root_pos: str = 'n', is_root_call: bool = True) -> tf.Tensor:

    # Base case: if branch is a tensor, return it
    if isinstance(branch, tf.Tensor):
      return branch

    if isinstance(branch, tuple):
      # Handle empty tuple (no children)
      if len(branch) == 0:
        return None

      # get all the childrens' embeddings
      child_embeddings = [
          self._compose_tok_embedding(e, root_pos, False)
          for e in branch
      ]

      # For the root call, the structure is ((lefts), (root,), (rights))
      # We need to get the root from index 1 before filtering
      if is_root_call and len(child_embeddings) >= 2:
        current_tok_tens = child_embeddings[1]
        # Filter out None values after extracting root
        child_embeddings = [e for e in child_embeddings if e is not None]
      else:
        # Filter out None values
        child_embeddings = [e for e in child_embeddings if e is not None]
        # If no valid embeddings, return None
        if len(child_embeddings) == 0:
          return None
        current_tok_tens = child_embeddings[0]

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
          return self.fuzzifier.iterated_union(child_embeddings)

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

  def __call__(self,
      branch: tuple,
      root_pos: str
  ) -> tf.Tensor:
      # Handle invalid input
      if not isinstance(branch, tuple):
          raise TypeError(f"Expected branch to be a tuple, got {type(branch).__name__}: {branch}")

      # baseline - extract root token from ((lefts), (root,), (rights)) structure
      if self.strategy is None:
          def find_tensor(obj):
              if isinstance(obj, tf.Tensor):
                  return obj
              elif isinstance(obj, tuple):
                  for item in obj:
                      result = find_tensor(item)
                      if result is not None:
                          return result
              return None

          # The root is in branch[1], which is a tuple containing the root tensor
          if len(branch) >= 2 and isinstance(branch[1], tuple) and len(branch[1]) > 0:
              return find_tensor(branch[1])
          # Fallback: search entire structure
          return find_tensor(branch)

      return self._compose_tok_embedding(branch, root_pos)