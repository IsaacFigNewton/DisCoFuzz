import pandas as pd
import tensorflow as tf
import keras
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
    "enrich_lemmas_with_wn": False,
    "keep_branch_json": False,
}

def get_fuzzy_emb_col(s: str, i: int):
    return f"sent_{i}_fuzzy_{s}"

def normalize_about_median(data: tf.Tensor) -> tf.Tensor:
    min = tf.math.reduce_min(data, axis=0, keepdims=True)
    min = tf.broadcast_to(min, tf.shape(data))
    max = tf.math.reduce_max(data, axis=0, keepdims=True)
    max = tf.broadcast_to(max, tf.shape(data))
    med = keras.ops.median(data, axis=0, keepdims=True)
    med = tf.broadcast_to(med, tf.shape(data))
    return (data - med) / (max - min)