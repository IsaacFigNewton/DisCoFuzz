from typing import Dict, Any
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

def fmt_fuzzy_emb_col(s: str, i: int):
    return f"sent_{i}_fuzzy_{s}"
def fmt_fuzzy_sim_metric_col(s:str, metric:str):
    return f"fuzzy_{s}_{metric}"
def fmt_dim_reduc_sim_col(prefix: str, dimensionality: int):
    return f"{prefix}_sim_components={dimensionality}"

def get_dim_reduc_sim_cols(df: pd.DataFrame, prefix: str):
    return [fmt_dim_reduc_sim_col(prefix, int(col)) for col in df.columns]

def parse_params_from_str(params: str) -> Dict[str, Any]:
    # parse various params from string
    fuzzy = 'fuzzy_' in params
    params = params.replace('fuzzy_', '').replace('_pred', '')
    metric_end_idx = params.index("_sim_")
    sim_metric = params[:metric_end_idx].split("_")[-1]
    strategy_end_idx = metric_end_idx - len(sim_metric) - 1
    n_components_str = params.split('_')[-1]
    # return a dict of them
    return {
        'fuzzy':                fuzzy,
        'strategy':             params[:strategy_end_idx],
        'similarity_metric':    sim_metric,
        'n_components':         int(n_components_str.split("=")[-1]),
        'model':                params[:metric_end_idx]
    }

def normalize_about_median(data: tf.Tensor, axis: int = 0) -> tf.Tensor:
    min = tf.math.reduce_min(data, axis=axis, keepdims=True)
    min = tf.broadcast_to(min, tf.shape(data))
    max = tf.math.reduce_max(data, axis=axis, keepdims=True)
    max = tf.broadcast_to(max, tf.shape(data))
    med = keras.ops.mean(data, axis=axis, keepdims=True)
    med = tf.broadcast_to(med, tf.shape(data))
    return (data - med) / (max - min)