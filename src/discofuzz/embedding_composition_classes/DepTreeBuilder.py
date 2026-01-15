import tensorflow as tf

from ..TensorStore import TensorStore

class DepTreeBuilder:
    def __init__(self, spacy_model, lemma_vectorizer: TensorStore):
        self.spacy_model = spacy_model
        self.lemma_vectorizer = lemma_vectorizer

    def extract_branch(self, row, i:int):
        # build out branch
        doc = self.spacy_model(row[f"sent_{i}"])
        tok = doc[row[f"tok_idx_{i}"]]

        tok_lefts = [str(t.lemma_) for t in tok.lefts]
        tok_rights = [str(t.lemma_) for t in tok.rights if not t.is_punct]
        branch = [str(tok.lemma_)]
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
    
    def get_branch_tuple_embedding(self,
        branch: tuple|str
    ) -> tuple|tf.Tensor:
        if isinstance(branch, tuple):
            return tuple([
                self.get_branch_tuple_embedding(child)
                for child in branch
            ])
        elif isinstance(branch, str):
            return self.lemma_vectorizer(branch).numpy()