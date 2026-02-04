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
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import wget as wget
import zipfile

from .config import *

# Import DisCoFuzz package classes
from .constants import SIMILARITY_METRICS
from .BaseEmbeddingModel import BaseEmbeddingModel
from .fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer

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
        self.sent_embeddings: List[np.ndarray] = list()
        self.fuzzy_sent_embeddings: List[pd.Series] = list()
        self.tok_embeddings: List[np.ndarray] = list()
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
            fuzzified_sent_embeddings = pd.Series([np.array(emb) for emb in self.sent_embeddings[i-1]]).apply(self.fuzzifier.fuzzify)
            self.fuzzy_sent_embeddings.append(fuzzified_sent_embeddings)
            # get fuzzified mean token embedding baseline
            fuzzified_tok_embeddings = pd.Series([np.array(emb) for emb in self.tok_embeddings[i-1]]).apply(self.fuzzifier.fuzzify)
            self.fuzzy_tok_embeddings.append(fuzzified_tok_embeddings)
    

    def get_sbert_sentence_baseline(self) -> pd.Series:
        # Calculate similarity - returns diagonal of similarity matrix
        cos_sims = cosine_similarity(self.sent_embeddings[0], self.sent_embeddings[1])
        return normalize_about_median(pd.Series(np.diag(cos_sims)))
    

    def get_sbert_token_baseline(self) -> pd.Series:
        # Add SBERT token-level baseline - returns diagonal of similarity matrix
        cos_sims = cosine_similarity(self.tok_embeddings[0], self.tok_embeddings[1])
        return normalize_about_median(pd.Series(np.diag(cos_sims)))
    

    def get_similarities(self, X: pd.DataFrame):
        # get fuzzified baseline embeddings
        for i in [1, 2]:
            X[get_fuzzy_emb_col("baseline_sent", i)] = self.fuzzy_sent_embeddings[i-1]
            X[get_fuzzy_emb_col("baseline_tok", i)] = self.fuzzy_tok_embeddings[i-1]

        # get embedding similarities across all metrics
        sims_df = pd.DataFrame()
        for sim_metric in self.sim_metrics:
            print(f"\n\t=== Computing similarities with {sim_metric.value} metric ===")
            for s in self.composition_strategies:
                print(f"\t\tGetting compositional embedding relatedness scores for {s} approach...")
                sims = list()
                for i, row in X.iterrows():
                    try:
                        sims.append(self.fuzzifier.similarity(
                            row[get_fuzzy_emb_col(s, 1)],
                            row[get_fuzzy_emb_col(s, 2)],
                            method=sim_metric,
                        ))
                    except Exception as e:
                        print(row)
                        raise e
                
                col = f"fuzzy_{s}_{sim_metric.value}_sim"
                # normalize similarity scores
                sims_df[col] = normalize_about_median(pd.Series(sims))
        
        # get baseline embeddings' (non-fuzzy) cosine similarities
        sims_df["baseline_sent_cos_sim"] = normalize_about_median(self.get_sbert_sentence_baseline())
        sims_df["baseline_tok_cos_sim"] = normalize_about_median(self.get_sbert_token_baseline())
        
        return sims_df


    def visualize_similarities(self, X: pd.DataFrame):
        # Create subplots for each similarity metric
        fig, axes = plt.subplots(
            1,
            len(self.sim_metrics),
            figsize=(8*len(self.sim_metrics), 6)
        )
        if len(self.sim_metrics) == 1:
            axes = [axes]

        for metric_idx, sim_metric in enumerate(self.sim_metrics):
            ax = axes[metric_idx]
            
            # Get columns for this metric
            metric_cols = [
                col
                for col in X.columns
                if sim_metric.value in col
            ]
            cmap = plt.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, len(metric_cols)))
            
            for i, col in enumerate(metric_cols):
                ax.scatter(
                    x=X["baseline_sent_cos_sim"],
                    y=X[col],
                    color=colors[i],
                    label=col.replace(f"fuzzy_", "").replace(f"_{sim_metric.value}_sim", ""),
                    alpha=0.5
                )
            
            ax.set_xlabel("sentence embedding cosine similarity", fontsize=12)
            ax.set_ylabel(f"{sim_metric.value} fuzzy compositional similarity", fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            # ax.set_yscale("log")
            ax.set_title(f"Sentence Embedding vs. Fuzzy Compositional Similarity ({sim_metric.value})", fontsize=14)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


    def classify_similarities(self, X: pd.DataFrame):
        y_pred = pd.DataFrame()
        for c in X.columns:
            # use a simple threshold for classification
            y_pred[f"{c}_pred"] = X[c] > 0
        return y_pred


    def plot_confusion_matrices(self,
            X: pd.DataFrame,
            y: pd.Series,
            n_cols: int = 2
        ):    
        # Create confusion matrices for all metrics
        for sim_metric in self.sim_metrics:
            # filter to just the columns with the current sim metric
            metric_cols = [c for c in X.columns if sim_metric.value in c]
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
                plt.title(f'{col.replace(f"_{sim_metric.value}_sim_pred", "").replace("_", " ")}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

            plt.tight_layout()
            plt.show()


    def _get_baselines(self, scores: pd.DataFrame):
        baselines = dict()
        baseline_inclusion_reqs = lambda x: "baseline_" in x and "fuzzy" not in x
        for i, row in scores.iterrows():
            if baseline_inclusion_reqs(row['model']):
                # print(row)
                baselines[row['model']] = {
                    'accuracy': row['accuracy'] if len(row) > 0 else None,
                    'precision': row['precision'] if len(row) > 0 else None,
                    'recall': row['recall'] if len(row) > 0 else None,
                    'f1_score': row['f1_score'] if len(row) > 0 else None
                }
        return baselines


    def _visualize_scores(self, scores: pd.DataFrame):
        # Create combined bar graphs with different colors for each similarity metric
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        # Get baseline values
        baselines = self._get_baselines(scores)

        # Create one figure with 4 subplots (one for each metric)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Evaluation Metrics by Strategy and Similarity Metric', fontsize=18, y=0.995)

        # Create a viridis color map for similarity metrics
        cmap = plt.cm.viridis
        sim_metric_colors = {
            sim_metric: cmap(i / max(len(self.sim_metrics) - 1, 1))
            for i, sim_metric in enumerate(self.sim_metrics)
        }

        for idx, metric in enumerate(metric_names):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data for grouped bar chart
            bar_width = 0.25
            x_pos = np.arange(len(self.composition_strategies))
            
            # Plot bars for each similarity metric
            for i, sim_metric in enumerate(self.sim_metrics):
                values = []
                for s in self.composition_strategies:
                    strat_mask = (scores['strategy'] == s) & (scores['similarity_metric'] == sim_metric.value)
                    metric_value = float(scores[strat_mask][metric].iloc[0])
                    values.append(metric_value)
                    
                
                # Plot bars with offset
                offset = (i - len(self.sim_metrics)/2 + 0.5) * bar_width
                bars = ax.barh(
                    x_pos + offset,
                    values,
                    bar_width,
                    label=sim_metric.value,
                    color=sim_metric_colors[sim_metric],
                    alpha=0.8
                )

            
            # Add baseline lines
            for i, (label, values) in enumerate(baselines.items()):
                color = cmap(i / max(len(baselines) - 1, 1))
                color = tuple([0.7*v for v in color])
                if values[metric] is not None:
                    ax.axvline(
                        x=values[metric],
                        color=color,
                        linestyle=':',
                        linewidth=2,
                        label=label,
                        alpha=0.8,
                        zorder=0
                    )
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels([s.replace('_', ' ') for s in self.composition_strategies], fontsize=9)
            ax.set_xlabel(metric.capitalize(), fontsize=12)
            ax.set_ylabel('Strategy', fontsize=12)
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3)
            ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.show()


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
            
            # Extract metric name from column
            metric_name = col.replace("_sim_pred", "").split("_")[-1]
            strategy = col.replace(f'_{metric_name}_sim_pred', '').replace('fuzzy_', '')
            
            metrics_data.append({
                'strategy': strategy,
                'similarity_metric': metric_name,
                'model': col.replace('_pred', ''),
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0)
            })

        scores = pd.DataFrame(metrics_data)
        scores = scores.sort_values(
            ['f1_score', 'similarity_metric', ],
            ascending=[False, True]
        ).reset_index(drop=True)

        self._visualize_scores(scores)
        return scores