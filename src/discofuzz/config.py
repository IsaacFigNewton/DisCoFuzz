# Import DisCoFuzz package classes
from discofuzz.constants import *

COL_PREFIXES = {
    "branch": "branch_tuple"
}

DEFAULTS = {
    "dataset_path": ".",
    "sample_size": 1000,

    "sim_metrics_enum": [
        SIMILARITY_METRICS.COS,
        SIMILARITY_METRICS.W1,
        SIMILARITY_METRICS.W2,
        SIMILARITY_METRICS.Q
    ],
    "n_components": 64,
    "fuzzification_kernel_size": 16,
    "spacy_model": "en_core_web_sm",
    "sentence_transformer": "all-MiniLM-L6-v2",
    "enrich_lemmas_with_wn": True,
    "keep_branch_json": False,
}

def get_fuzzy_emb_col(s: str, i: int):
    return f"sent_{i}_fuzzy_{s}"