import spacy
import tensorflow as tf
from .constants import STRATEGIES
from .LemmaVectorizer import LemmaVectorizer
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

class SpacyDependencyComposer:
  def __init__(self,
      strategy:str=None,
      spacy_model=None,
      lemma_vectorizer=None,
      fuzzifier=None
    ):
    if strategy not in STRATEGIES:
      raise ValueError(f"Unknown strategy: {strategy}")
    self.strategy = strategy
    self.nlp = spacy.load("en_core_web_sm") if spacy_model is None else spacy_model
    self.lemma_vectorizer = LemmaVectorizer() if lemma_vectorizer is None else lemma_vectorizer
    self.fuzzifier = FuzzyFourierTensorTransformer() if fuzzifier is None else fuzzifier

  def _compose_tok_embedding(self, token) -> tf.Tensor:
    # lemma_vectorizer returns a numpy array, so convert it to tf.Tensor for fuzzification
    current_tok_tens = self.lemma_vectorizer(token.lemma_.lower())
    current_tok_tens = tf.convert_to_tensor(current_tok_tens, dtype=tf.float32)
    current_tok_tens = self.fuzzifier.fuzzify(current_tok_tens)

    if token.children:
      # get all the childrens'embeddings
      child_embeddings = [
          self._compose_tok_embedding(c)
          for c in token.children
      ]

      if not child_embeddings:
        # If no valid children, return the current token's embedding
        return current_tok_tens

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
          if token.pos_ in {"VERB", "ADJ", "ADV"}:
            child_embeddings_intersected = [
                self.fuzzifier.intersection(current_tok_tens, c)
                for c in child_embeddings
            ]
            return tf.reduce_mean(tf.stack(child_embeddings_intersected), axis=0)
          # otherwise, just return the mean of its children
          return tf.reduce_mean(tf.stack(child_embeddings), axis=0)

        case "selective_intersection+union":
          # if the current token is a modifier of some kind, intersect it with all its children
          if token.pos_ in {"VERB", "ADJ", "ADV"}:
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
          if token.pos_ in {"VERB", "ADJ", "ADV"}:
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
      text: str,
  ) -> tf.Tensor:
      # baseline
      if self.strategy is None:
          return self.fuzzifier.fuzzify(self.lemma_vectorizer(text))

      doc = self.nlp(text)
      root_embeddings = []
      for sent in doc.sents:
          root_emb = self._compose_tok_embedding(sent.root)
          if root_emb is not None:
            root_embeddings.append(root_emb)

      if not root_embeddings:
          return None

      # get the average
      return tf.reduce_mean(tf.stack(root_embeddings), axis=0)