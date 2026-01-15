import tensorflow as tf
from spacy.tokens import Doc, Token
import json

from ..TensorStore import TensorStore

class DepTreeBuilder:
    def __init__(self, spacy_model, lemma_vectorizer: TensorStore):
        self.spacy_model = spacy_model
        self.lemma_vectorizer = lemma_vectorizer

    def _build_branch(self, tok: Token):
        tok_lefts = [json.dumps(vars(t)) for t in tok.lefts]
        tok_rights = [json.dumps(vars(t)) for t in tok.rights if not t.is_punct]
        branch = [json.dumps(vars(tok))]
        if len(tok_lefts) > 0:
            branch.append(tuple(tok_lefts))
        if len(tok_rights) > 0:
            branch.append(tuple(tok_rights))
        branch = tuple(branch)

        # if the current token has a parent,
        #   add it to the branch
        if tok.has_head:
            branch = (str(tok.head.lemma_), tuple(branch))

        # store in dataframe
        return branch

    def extract_branch(self, row, i:int):
        # build out branch
        doc = self.spacy_model(row[f"sent_{i}"])
        tok = doc[row[f"tok_idx_{i}"]]
        return self._build_branch(tok)
    
    def extract_tree(self, row, i:int):
        # build out branch
        doc = self.spacy_model(row[f"sent_{i}"])
        tok = doc.sents[0].root
        return self._build_branch(tok)
    
    def get_branch_tuple_embedding(self,
        branch: tuple|str
    ) -> tuple|tf.Tensor:
        if isinstance(branch, tuple):
            return tuple([
                self.get_branch_tuple_embedding(child)
                for child in branch
            ])
        elif isinstance(branch, str):
            tok = Token(**json.loads(branch))
            return self.lemma_vectorizer(tok).numpy()