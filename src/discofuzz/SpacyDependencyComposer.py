from typing import Optional
import numpy as np
import spacy
import tensorflow as tf
from .constants import STRATEGIES
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

class SpacyDependencyComposer:
  modifiers = {"v", "a"}

  def __init__(self,
      strategy:str,
      fuzzifier:Optional[FuzzyFourierTensorTransformer]=None
    ):
    if strategy not in STRATEGIES:
      raise ValueError(f"Unknown strategy: {strategy}")
    self.strategy = strategy
    self.fuzzifier = FuzzyFourierTensorTransformer() if fuzzifier is None else fuzzifier


  def _compose_tok_embedding(self,
      branch: tuple | tf.Tensor,
      root_pos: Optional[str] = None
    ) -> tf.Tensor | None:

    # Base case: if branch is a tensor, return it
    if isinstance(branch, tf.Tensor):
      return branch

    if isinstance(branch, tuple):
      # get all the childrens' embeddings
      child_embeddings = [
          self._compose_tok_embedding(e)
          for e in branch
          if e is not None
      ]

      # For the root call, the structure is ((root,), (lefts), (rights))
      # if there're left or right children
      if len(child_embeddings) > 1:
        current_tok_tens = child_embeddings[0]
        child_embeddings = child_embeddings[1:]
      # if there is only a root
      elif len(child_embeddings) == 1:
         return child_embeddings[0]
      # if there're no valid children (like an empty left or right branch)
      else:
        return None
      
      child_embeddings = [
        c
        for c in child_embeddings
        if c is not None
      ]

      # compose child embeddings based on strategy
      match self.strategy:

        case "mean":
          return tf.reduce_mean(tf.stack([current_tok_tens]+child_embeddings), axis=0)

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

      return self._compose_tok_embedding(branch, root_pos)