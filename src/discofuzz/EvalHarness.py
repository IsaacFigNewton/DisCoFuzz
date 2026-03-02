from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import spacy
import numpy as np
from spacy.tokens import Token, Doc
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import seaborn as sns
import wget as wget
import zipfile

from .config import *

# Import DisCoFuzz package classes
from .constants import SIMILARITY_METRICS
from .BaseEmbeddingModel import BaseEmbeddingModel
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer
from .EvalVisualizationsMixin import EvalVisualizationsMixin

class EvalHarness:
    def __init__(self,
            sim_metrics: List[SIMILARITY_METRICS],
            composition_strategies: List[str],
            embedding_model: BaseEmbeddingModel,
            spacy_model: Any,
            fuzzifier: FuzzyFourierTensorTransformer
        ):
        """
        :param spacy_model: Description
        :type spacy_model: Optional[str]

        :param embedding_model: Description
        :type embedding_model: Optional[str]
        
        """
        # use the GPU for TensorFlow operations if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {gpus}")
        
        self.sim_metrics = sim_metrics
        self.composition_strategies = sorted(composition_strategies + ["baseline_sent", "baseline_tok"])

        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.fuzzifier = fuzzifier
        self.sent_embeddings = list()
        self.tok_embeddings = list()
        self.fuzzy_sent_embeddings: List[pd.Series] = list()
        self.fuzzy_tok_embeddings: List[pd.Series] = list()

    
    def fit(self, X: pd.DataFrame):
        # get sentence embedding baseline
        sents = list()
        for i in [1, 2]:
            # get a list of the sentences to embed
            sents.extend(X[f"sent_{i}"].to_list())
        sbert_sent_vects = self.embedding_model.fit_transform(sents)
        self.sent_embeddings = [
            sbert_sent_vects[:len(X)],
            sbert_sent_vects[len(X):]
        ]

        # get average of individual token embeddings
        mean_tok_vects = dict()
        for j in [1, 2]:
            mean_tok_vects[j] = list()
            for i, row in X.iterrows():
                token_embs = [
                    self.embedding_model.encode([token.text])
                    for token in self.spacy_model(row[f"sent_{j}"])
                    if not token.is_punct
                ]
                if token_embs:
                    # Concatenate all token embeddings and take mean
                    all_embs = np.concatenate(token_embs, axis=0)
                    mean_tok_vects[j].append(np.mean(all_embs, axis=0))
                else:
                    mean_tok_vects[j].append(np.zeros(DEFAULTS["n_components"]))
        self.tok_embeddings = [np.array(mean_tok_vects[1]), np.array(mean_tok_vects[2])]

        # get fuzzy baselines
        for i in [1, 2]:
            # get fuzzified sentence embedding baseline
            sent_embeddings = pd.Series([np.array(emb) for emb in self.sent_embeddings[i-1]])
            fuzzified_sent_embeddings = sent_embeddings.apply(self.fuzzifier.fuzzify)
            self.fuzzy_sent_embeddings.append(fuzzified_sent_embeddings)
            # get fuzzified mean token embedding baseline
            tok_embeddings = pd.Series([np.array(emb) for emb in self.tok_embeddings[i-1]])
            fuzzified_tok_embeddings = tok_embeddings.apply(self.fuzzifier.fuzzify)
            self.fuzzy_tok_embeddings.append(fuzzified_tok_embeddings)
        
        # convert to tensors for easier processing later
        self.sent_embeddings = tf.convert_to_tensor(self.sent_embeddings)
        self.tok_embeddings = tf.convert_to_tensor(self.tok_embeddings)


    def _cosine_similarity_all_prefixes(self, a: tf.Tensor, b: tf.Tensor):
        """
        Computes cosine similarities between 'a' and 'b'
        for the first n components in [1, d].

        Args:
            a: Tensor of shape (batch_size, d)
            b: Tensor of shape (batch_size, d)
            eps: Small constant for numerical stability

        Returns:
            Tensor of shape (batch_size, d), where:
            output[:, n-1] = cosine similarities of all samples using first n components
        """
        print(a.shape)
        # Prefix dot products
        prefix_dot = tf.cumsum(a * b, axis=1)
        # Prefix norms
        prefix_norm_a = tf.sqrt(tf.cumsum(a**2, axis=1))
        prefix_norm_b = tf.sqrt(tf.cumsum(b**2, axis=1))
        # Cosine similarity for each prefix
        cosine_sim = prefix_dot / (prefix_norm_a * prefix_norm_b)
        return cosine_sim

    def get_sbert_sentence_baseline(self) -> tf.Tensor:
        cos_sims = self._cosine_similarity_all_prefixes(self.sent_embeddings[0], self.sent_embeddings[1])
        # normalize the similarities evaluated within each dim-reduced subspace
        return normalize_about_median(cos_sims, axis=0)
    

    def get_sbert_token_baseline(self) -> tf.Tensor:
        cos_sims = self._cosine_similarity_all_prefixes(self.tok_embeddings[0], self.tok_embeddings[1])
        # normalize the similarities evaluated within each dim-reduced subspace
        return normalize_about_median(cos_sims, axis=0)
    

    def get_similarities(self, X: pd.DataFrame):
        # get fuzzified baseline embeddings
        for i in [1, 2]:
            X[fmt_fuzzy_emb_col("baseline_sent", i)] = self.fuzzy_sent_embeddings[i-1]
            X[fmt_fuzzy_emb_col("baseline_tok", i)] = self.fuzzy_tok_embeddings[i-1]

        # get baseline embeddings' (non-fuzzy) cosine similarities
        llm_sent_baseline_df = pd.DataFrame(self.get_sbert_sentence_baseline().numpy())
        llm_sent_baseline_df.columns = get_dim_reduc_sim_cols(llm_sent_baseline_df, "baseline_sent_cos")
        llm_tok_baseline_df = pd.DataFrame(self.get_sbert_token_baseline().numpy())
        llm_tok_baseline_df.columns = get_dim_reduc_sim_cols(llm_tok_baseline_df, "baseline_tok_cos")

        # get embedding similarities across all metrics
        normed_sims_dfs = list()
        for sim_metric in self.sim_metrics:
            print(f"\n\t=== Computing similarities with {sim_metric.value} metric ===")
            for s in self.composition_strategies:
                print(f"\t\tGetting compositional embedding relatedness scores for {s} approach...")
                try:
                    sims = self.fuzzifier.similarity(
                        tf.convert_to_tensor(X[fmt_fuzzy_emb_col(s, 1)].tolist()),
                        tf.convert_to_tensor(X[fmt_fuzzy_emb_col(s, 2)].tolist()),
                        method=sim_metric,
                    )
                except Exception as e:
                    raise e
                
                # normalize similarity scores
                # shape = (batch_size, n_components, kernel_size)
                normalized_fuzzy_sims = normalize_about_median(tf.convert_to_tensor(sims), axis=0)
                normed_fuzzy_sims_df = pd.DataFrame(normalized_fuzzy_sims.numpy())
                normed_fuzzy_sims_df.columns = get_dim_reduc_sim_cols(normed_fuzzy_sims_df, fmt_fuzzy_sim_metric_col(s, sim_metric.value))
                normed_sims_dfs.append(normed_fuzzy_sims_df)
        
        return pd.concat([llm_sent_baseline_df, llm_tok_baseline_df]+normed_sims_dfs, axis=1,)


    def classify_similarities(self, X: pd.DataFrame):
        # Vectorized thresholding across the entire DataFrame
        y_pred = X > 0
        y_pred = y_pred.add_suffix("_pred")
        return y_pred


    def _get_baselines(self, scores: pd.DataFrame, dimensionality: int) -> dict:
        # mask for baseline rows
        score_cols = ["accuracy", "precision", "recall", "f1_score"]
        mask = (~scores["fuzzy"]) & (scores["n_components"] == (dimensionality - 1))
        baselines_df = scores.loc[mask, ["model"]+score_cols].copy()
        # ensure missing values become None (instead of NaN) in the final dict
        baselines_df = baselines_df.where(pd.notna(baselines_df), None)
        # convert to dict keyed by model
        return baselines_df\
            .set_index("model")[score_cols]\
            .to_dict(orient="index")


    def score(self,
            X: pd.DataFrame,
            y: pd.Series,
            plot: bool = True
        ) -> pd.DataFrame:
        # Create evaluation metrics dataframe for ALL similarity metrics
        metrics_data = []
        for col in X.columns:
            if col == "is_related":
                continue
            y_pred = X[col].astype(int)
            params = parse_params_from_str(col)
            metrics = {
                'accuracy':             accuracy_score(y, y_pred),
                'precision':            precision_score(y, y_pred, zero_division=0),
                'recall':               recall_score(y, y_pred, zero_division=0),
                'f1_score':             f1_score(y, y_pred, zero_division=0)
            }
            params.update(metrics)
            metrics_data.append(params)

        scores = pd.DataFrame(metrics_data)
        scores = scores.sort_values(
            ['f1_score', 'similarity_metric', ],
            ascending=[False, True]
        ).reset_index(drop=True)

        return scores


    def visualize_similarities(self, X: pd.DataFrame, dimensionality: int):
        n = len(self.sim_metrics)
        fig, axes, axes_flat, nrows, ncols = EvalVisualizationsMixin._make_grid(n, ncols=2, w=5, h=4, sharey=True)

        baseline_col = fmt_dim_reduc_sim_col("baseline_sent_cos", dimensionality - 1)

        # Collect legend items across subplots (so we can do one legend below)
        all_handles, all_labels = [], []
        seen = set()

        for metric_idx, sim_metric in enumerate(self.sim_metrics):
            ax = axes_flat[metric_idx]

            metric_cols = [
                col for col in X.columns
                if ("fuzzy_" in col and sim_metric.value in col and str(dimensionality - 1) in col)
            ]

            colors = EvalVisualizationsMixin._colormap_list(len(metric_cols), "viridis")

            for i, col in enumerate(metric_cols):
                label = (
                    col.replace("fuzzy_", "")
                       .replace(f"_{sim_metric.value}_sim_components={dimensionality - 1}", "")
                )
                h = ax.scatter(
                    x=X[baseline_col],
                    y=X[col],
                    color=colors[i],
                    label=label,
                    alpha=0.6
                )
                if label not in seen:
                    seen.add(label)
                    all_handles.append(h)
                    all_labels.append(label)

            EvalVisualizationsMixin._show_axis_labels(
                ax, metric_idx, nrows, ncols,
                xlabel="sentence embedding cosine similarity",
                ylabel="fuzzy compositional similarity",
                show_left_y_only=True,
                show_bottom_x_only=True
            )

            ax.set_title(f"Sentence Embedding vs. {sim_metric.value}", fontsize=14)
            ax.grid(alpha=0.3)

        EvalVisualizationsMixin._hide_unused_axes(axes_flat, n)
        EvalVisualizationsMixin._legend_below(fig, all_handles, all_labels, y=-0.02, ncol_cap=5, fontsize=9)
        EvalVisualizationsMixin._finish(fig, bottom_space=0.08, top_space=1.0)


    def visualize_metric_by_sim_n_components(self, df: pd.DataFrame, strategies=None, metric: str = "f1_score"):
        required = {'strategy', 'similarity_metric', 'n_components', metric, 'fuzzy'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        baseline_strats = ['baseline_sent', 'baseline_tok']
        base = df[
            (~df['fuzzy']) &
            (df['similarity_metric'] == 'cos') &
            (df['strategy'].isin(baseline_strats))
        ].copy()

        base_grouped = (
            base.groupby(['strategy', 'n_components'], as_index=False)
                .agg({metric: 'mean'})
                .sort_values(['strategy', 'n_components'])
        )

        dmain = df if strategies is None else df[df['strategy'].isin(strategies)]
        dmain_grouped = (
            dmain.groupby(['similarity_metric', 'strategy', 'n_components'], as_index=False)
                .agg({metric: 'mean'})
                .sort_values(['similarity_metric', 'strategy', 'n_components'])
        )

        sim_metrics = sorted(dmain_grouped['similarity_metric'].unique())
        if len(sim_metrics) != 4:
            print(f"Warning: Found {len(sim_metrics)} similarity metrics (expected 4).")

        fig, axes, axes_flat, nrows, ncols = EvalVisualizationsMixin._make_grid(4, ncols=2, w=4, h=3, sharex=True, sharey=True)

        for idx, sim_metric in enumerate(sim_metrics[:4]):
            ax = axes_flat[idx]
            metric_data = dmain_grouped[dmain_grouped['similarity_metric'] == sim_metric]

            for strategy_name, group in metric_data.groupby('strategy'):
                ax.plot(
                    group['n_components'],
                    group[metric],
                    label=strategy_name
                )

            if not base_grouped.empty:
                for bname, bgrp in base_grouped.groupby('strategy'):
                    ax.plot(
                        bgrp['n_components'],
                        bgrp[metric],
                        linestyle='--',
                        label=f"{bname} (non-fuzzy, cos)"
                    )

            EvalVisualizationsMixin._show_axis_labels(
                ax, idx, nrows, ncols,
                xlabel="n_components",
                ylabel="F1 Score"
            )
            ax.set_title(sim_metric)
            ax.grid(True)

        # Single legend below (unique entries)
        handles, labels = EvalVisualizationsMixin._unique_legend_from_axes(axes_flat[:len(sim_metrics[:4])])
        EvalVisualizationsMixin._legend_below(fig, handles, labels, y=-0.02, ncol_cap=3, fontsize=9)
        EvalVisualizationsMixin._finish(fig, bottom_space=0.08, top_space=1.0)


    def visualize_scores(self, scores: pd.DataFrame, dimensionality: int, only_acc_f1: bool = True):
        metric_names = ['accuracy', 'f1_score'] if only_acc_f1 else ['accuracy', 'precision', 'recall', 'f1_score']
        n = len(metric_names)

        ncols = 2
        fig, axes, axes_flat, nrows, ncols = EvalVisualizationsMixin._make_grid(
            n, ncols=ncols,
            w=6 if only_acc_f1 else 10,
            h=6 if only_acc_f1 else 8,
            sharey=True
        )
        fig.suptitle('Evaluation Metrics by Strategy and Similarity Metric', fontsize=18, y=0.98)

        baselines = self._get_baselines(scores, dimensionality)

        cmap = plt.cm.viridis
        sim_metric_colors = {
            sim_metric: cmap(i / max(len(self.sim_metrics) - 1, 1))
            for i, sim_metric in enumerate(self.sim_metrics)
        }

        bar_width = 0.25
        x_pos = np.arange(len(self.composition_strategies))

        for idx, metric in enumerate(metric_names):
            ax = axes_flat[idx]

            # Bars
            for i, sim_metric in enumerate(self.sim_metrics):
                strat_vals = []
                for s in self.composition_strategies:
                    m = (
                        (scores['strategy'] == s) &
                        (scores['similarity_metric'] == sim_metric.value) &
                        (scores['n_components'] == dimensionality - 1)
                    )
                    strat_vals.append(float(scores.loc[m, metric].iloc[0]))

                offset = (i - len(self.sim_metrics) / 2 + 0.5) * bar_width
                ax.barh(
                    x_pos + offset,
                    strat_vals,
                    bar_width,
                    label=sim_metric.value,
                    color=sim_metric_colors[sim_metric],
                    alpha=0.8
                )

            # Baselines
            for i, (label, values) in enumerate(baselines.items()):
                v = values.get(metric)
                if v is None:
                    continue
                color = tuple(0.7 * c for c in cmap(i / max(len(baselines) - 1, 1)))
                ax.axvline(
                    x=v, color=color,
                    linestyle=':', linewidth=2,
                    label=label, alpha=0.8, zorder=0
                )

            ax.set_yticks(x_pos)

            # y tick labels only on left column (shared helper)
            EvalVisualizationsMixin._show_axis_labels(
                ax, idx, nrows, ncols,
                xlabel=metric.capitalize(),
                ylabel="Strategy"
            )

            if (idx % ncols) == 0:
                ax.set_yticklabels([s.replace('_', ' ') for s in self.composition_strategies], fontsize=9)

            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3)

        handles, labels = EvalVisualizationsMixin._unique_legend_from_axes(axes_flat[:n])
        EvalVisualizationsMixin._legend_below(fig, handles, labels, y=-0.02, ncol_cap=3, fontsize=10)

        bottom_space = 0.12 if only_acc_f1 else 0.08
        EvalVisualizationsMixin._finish(fig, bottom_space=bottom_space, top_space=0.95)
    
    
    def plot_confusion_matrices(self,
            X: pd.DataFrame,
            y: pd.Series,
            dimensionality: int,
            n_cols: int = 2,
        ):
        # Create confusion matrices for all metrics
        for sim_metric in self.sim_metrics:
            # filter to just the columns with the current sim metric
            metric_cols = [
                c
                for c in X.columns
                if sim_metric.value in c and str(dimensionality-1) in c
            ]
            # Calculate grid size
            n_rows = int(np.ceil(len(metric_cols) / n_cols))
            
            plt.figure(figsize=(5*n_cols, 5*n_rows))
            plt.suptitle(f"Confusion Matrices for different embedding composition methods using {sim_metric.value} similarity\n", fontsize=16)
            
            for i, col in enumerate(metric_cols):
                # Calculate confusion matrix
                cm = confusion_matrix(
                    y,
                    X[col].astype(int)
                )

                # Plot confusion matrix
                plt.subplot(n_rows, n_cols, i+1)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='viridis',
                    xticklabels=['Unrelated', 'Related'],
                    yticklabels=['Unrelated', 'Related']
                )
                params = parse_params_from_str(col)
                plt.title(f"{'fuzzy_' if params['fuzzy'] else ''}{params['strategy']}")
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

            plt.tight_layout()
            plt.show()