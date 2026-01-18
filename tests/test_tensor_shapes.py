"""
Unit tests for tensor shape handling in DisCoFuzz package.
Tests ensure that tensor operations handle various input shapes correctly.
"""

import unittest
import numpy as np
import tensorflow as tf
from discofuzz.fuzzy_classes.FuzzyFourierTensorTransformer import FuzzyFourierTensorTransformer
from discofuzz.BaseEmbeddingModel import BaseEmbeddingModel
from discofuzz.EvalHarness import EvalHarness
from discofuzz.DisCoFuzz import DisCoFuzz
import pandas as pd


class TestTensorShapes(unittest.TestCase):
    """Test tensor shape handling throughout the DisCoFuzz pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.fuzzifier = FuzzyFourierTensorTransformer(sigma=0.1, kernel_size=8)
        self.embedding_model = BaseEmbeddingModel()

    def test_fuzzify_1d_array(self):
        """Test fuzzify with 1D numpy array."""
        embedding = np.random.randn(64).astype(np.float32)
        result = self.fuzzifier.fuzzify(embedding)

        self.assertEqual(len(result.shape), 2, "Fuzzified result should be 2D")
        self.assertEqual(result.shape[0], 64, "First dimension should match embedding size")
        self.assertEqual(result.shape[1], 8, "Second dimension should match kernel size")

    def test_fuzzify_2d_array_single_row(self):
        """Test fuzzify with 2D numpy array of shape (1, n)."""
        embedding = np.random.randn(1, 64).astype(np.float32)
        result = self.fuzzifier.fuzzify(embedding)

        self.assertEqual(len(result.shape), 2, "Fuzzified result should be 2D")
        self.assertEqual(result.shape[0], 64, "First dimension should match embedding size")
        self.assertEqual(result.shape[1], 8, "Second dimension should match kernel size")

    def test_fuzzify_tensor(self):
        """Test fuzzify with TensorFlow tensor."""
        embedding = tf.random.normal([64], dtype=tf.float32)
        result = self.fuzzifier.fuzzify(embedding)

        self.assertEqual(len(result.shape), 2, "Fuzzified result should be 2D")
        self.assertEqual(result.shape[0], 64, "First dimension should match embedding size")
        self.assertEqual(result.shape[1], 8, "Second dimension should match kernel size")

    def test_fuzzify_invalid_shape(self):
        """Test that fuzzify raises error for invalid shapes."""
        # 3D array should raise error
        embedding = np.random.randn(2, 64, 1).astype(np.float32)
        with self.assertRaises(ValueError):
            self.fuzzifier.fuzzify(embedding)

    def test_intersection_shapes(self):
        """Test intersection operation maintains correct shapes."""
        A = np.random.randn(64).astype(np.float32)
        B = np.random.randn(64).astype(np.float32)

        fuzz_A = self.fuzzifier.fuzzify(A)
        fuzz_B = self.fuzzifier.fuzzify(B)

        result = self.fuzzifier.intersection(fuzz_A, fuzz_B)

        self.assertEqual(result.shape, fuzz_A.shape, "Intersection result should match input shape")

    def test_union_shapes(self):
        """Test union operation maintains correct shapes."""
        A = np.random.randn(64).astype(np.float32)
        B = np.random.randn(64).astype(np.float32)

        fuzz_A = self.fuzzifier.fuzzify(A)
        fuzz_B = self.fuzzifier.fuzzify(B)

        result = self.fuzzifier.union(fuzz_A, fuzz_B)

        self.assertEqual(result.shape, fuzz_A.shape, "Union result should match input shape")

    def test_similarity_computation(self):
        """Test similarity computation returns scalar."""
        A = np.random.randn(64).astype(np.float32)
        B = np.random.randn(64).astype(np.float32)

        fuzz_A = self.fuzzifier.fuzzify(A)
        fuzz_B = self.fuzzifier.fuzzify(B)

        from discofuzz.constants import SIMILARITY_METRICS
        # Test with a valid similarity metric
        sim = self.fuzzifier.similarity(fuzz_A, fuzz_B, SIMILARITY_METRICS.COS)

        self.assertIsInstance(float(sim), float, "Similarity should be a scalar")

    def test_embedding_model_output_shapes(self):
        """Test that embedding model produces correct output shapes."""
        # Need enough sentences for PCA (more than n_components=64)
        sentences = [f"This is test sentence number {i}." for i in range(100)]
        embeddings = self.embedding_model.fit_transform(sentences)

        self.assertEqual(len(embeddings.shape), 2, "Embeddings should be 2D")
        self.assertEqual(embeddings.shape[0], 100, "Should have one embedding per sentence")
        self.assertEqual(embeddings.shape[1], 64, "Embedding dimension should be 64 after PCA")

    def test_eval_harness_baseline_shapes(self):
        """Test that EvalHarness produces correct baseline shapes."""
        # Create test data with enough samples for PCA
        test_data = pd.DataFrame({
            'sent_1': [f'The cat sat on the mat number {i}.' for i in range(100)],
            'sent_2': [f'The dog ran quickly number {i}.' for i in range(100)],
            'target_word': ['cat'] * 100
        })

        # Create new embedding model for this test to avoid interference
        embedding_model = BaseEmbeddingModel()
        model = DisCoFuzz(embedding_model)
        eval_harness = EvalHarness(
            embedding_model,
            model.spacy_model,
            self.fuzzifier
        )

        # Fit the eval harness
        eval_harness.fit(test_data)

        # Check sentence baseline shapes
        sent_baseline = eval_harness.get_sbert_sentence_baseline()
        self.assertEqual(len(sent_baseline), 100, "Should have one similarity per pair")
        self.assertEqual(len(sent_baseline.shape), 1, "Baseline should be 1D array")

        # Check token baseline shapes
        tok_baseline = eval_harness.get_sbert_token_baseline()
        self.assertEqual(len(tok_baseline), 100, "Should have one similarity per pair")
        self.assertEqual(len(tok_baseline.shape), 1, "Baseline should be 1D array")

        # Check fuzzy embeddings
        for fuzzy_emb in eval_harness.fuzzy_sent_embeddings:
            self.assertEqual(len(fuzzy_emb), 100, "Should have embeddings for each sentence")

    def test_batch_fuzzification(self):
        """Test that batch fuzzification works correctly."""
        embeddings = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        fuzzified = [self.fuzzifier.fuzzify(emb) for emb in embeddings]

        self.assertEqual(len(fuzzified), 5, "Should have 5 fuzzified embeddings")
        for fuzz in fuzzified:
            self.assertEqual(fuzz.shape, (64, 8), "Each fuzzified embedding should have shape (64, 8)")


if __name__ == '__main__':
    unittest.main()
