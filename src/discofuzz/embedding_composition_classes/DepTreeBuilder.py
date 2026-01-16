import tensorflow as tf
from spacy.tokens import Doc, Token
import json

from ..types import TokenDataclass
from ..TensorStore import TensorStore

class DepTreeBuilder:
    def __init__(self, spacy_model, lemma_vectorizer: TensorStore):
        self.spacy_model = spacy_model
        self.lemma_vectorizer = lemma_vectorizer

    @staticmethod
    def _token_to_dict(tok: Token):
        return {
            "text": tok.text,
            "pos_": tok.pos_
        }
    
    @staticmethod
    def _token_to_str(tok: Token):
        return json.dumps({
            "text": tok.text,
            "pos_": tok.pos_
        }, indent=4)
    

    def _build_branch(self, tok: Token):
        tok_lefts = [self._token_to_str(t) for t in tok.lefts]
        tok_rights = [self._token_to_str(t) for t in tok.rights if not t.is_punct]
        branch = [self._token_to_str(tok)]
        if len(tok_lefts) > 0:
            branch.append(tuple(tok_lefts))
        if len(tok_rights) > 0:
            branch.append(tuple(tok_rights))
        branch = tuple(branch)

        # if the current token has a parent,
        #   add it to the branch
        if tok.has_head:
            branch = (self._token_to_str(tok.head), tuple(branch))

        # store in dataframe
        return branch

    def extract_branch(self, sentence: str, tok_idx: int):
        # build out branch
        doc = self.spacy_model(sentence)
        tok = doc[tok_idx]
        return self._build_branch(tok)
    
    def extract_tree(self, sentence: str):
        # build out branch
        doc = self.spacy_model(sentence)
        tok = list(doc.sents)[0].root
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

            tok = TokenDataclass(**json.loads(branch))
            return self.lemma_vectorizer(tok).numpy()