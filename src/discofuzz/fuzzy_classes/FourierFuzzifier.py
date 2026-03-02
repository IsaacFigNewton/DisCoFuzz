import tensorflow as tf
import numpy as np
from enum import Enum
from typing import Union, Tuple
from scipy.special import expit
import ot
from ..constants import SIMILARITY_METRICS
from .FuzzyFourierSetMixin import FuzzyFourierSetMixin

class FourierFuzzifier(FuzzyFourierSetMixin):
    """TensorFlow-accelerated version of FourierFuzzifier with set operations"""

    def __init__(self, sigma:float, kernel_size:int):
        if kernel_size < 1:
            raise ValueError("Kernel size must be at least 1")
        super().__init__(sigma, kernel_size)

    def _get_w2_sim_2D(self, x: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        a = x[0].numpy()
        b = x[1].numpy()
        freqs = tf.range(0, self.kernel_size, dtype=tf.float32)
        u, v = np.meshgrid(freqs, freqs)

        # Wasserstein-2 metric
        cost = (u - v)**2
        cost = np.ascontiguousarray(cost, dtype='float64')

        component_costs = []
        for i in range(a.shape[0]):
            plan = ot.emd(
                np.ascontiguousarray(a[i, :]),
                np.ascontiguousarray(b[i, :]),
                cost,
                check_marginals=False
            )
            component_costs.append(tf.convert_to_tensor(np.sum(plan * cost)))
        
        return tf.convert_to_tensor(component_costs)                  # (N,)


    def similarity_batch(
        self,
        a: tf.Tensor,
        b: tf.Tensor,
        method: SIMILARITY_METRICS,
        cum: bool,
    ) -> tf.Tensor:
        """
        a, b: (batch_size, n_components, kernel_size)
        Returns:
          - If cum == True (default): (batch_size, n_components) where out[:, n-1] uses first n components (prefix-aggregated).
          - If cum == False: (batch_size, 1) with similarity aggregated across ALL components.
        """
        if a is None or b is None:
            raise ValueError("Inputs must be tensors, got None")
        if len(a.shape) != 3:
            raise ValueError(f"a must have shape (B,N,K), got {a.shape}")
        if len(b.shape) != 3:
            raise ValueError(f"b must have shape (B,N,K), got {b.shape}")
        if a.shape != b.shape:
            raise ValueError(f"Inputs must have same shape, got {a.shape} vs {b.shape}")

        match method:
            case SIMILARITY_METRICS.COS:
                a_abs = tf.abs(a)
                b_abs = tf.abs(b)
                # per-component energies: (B, N)
                ab_comp = tf.reduce_sum(a_abs * b_abs, axis=2)
                aa_comp = tf.reduce_sum(a_abs * a_abs, axis=2)
                bb_comp = tf.reduce_sum(b_abs * b_abs, axis=2)

                if cum:
                    # prefix-aggregate across components -> (B, N)
                    numerator = tf.cumsum(ab_comp, axis=1)
                    denominator_a = tf.sqrt(tf.cumsum(aa_comp, axis=1))
                    denominator_b = tf.sqrt(tf.cumsum(bb_comp, axis=1))
                    similarity = numerator / (denominator_a * denominator_b + 1e-10)  # (B,N)
                    return similarity
                else:
                    # aggregate across components -> (B, 1)
                    numerator = tf.reduce_sum(ab_comp, axis=1, keepdims=True)
                    denominator_a = tf.sqrt(tf.reduce_sum(aa_comp, axis=1, keepdims=True))
                    denominator_b = tf.sqrt(tf.reduce_sum(bb_comp, axis=1, keepdims=True))
                    similarity = numerator / (denominator_a * denominator_b + 1e-10)  # (B,1)
                    return similarity

            case SIMILARITY_METRICS.W1:
                # get component-wise integrals (B, N)
                psi_batch = tf.map_fn(self._get_cdf_batch, a - b)                       # (B, N, K)
                integrate_cdf_batch = tf.map_fn(self._integrate_batch, psi_batch)       # (B, N)
                abs_diff_batch = tf.abs(integrate_cdf_batch)                            # (B, N)

                if cum:
                    # prefix-sum across components -> (B, N)
                    w1_dist = tf.cumsum(abs_diff_batch, axis=1)                         # (B,N)
                else:
                    # aggregated W1 across all components -> (B, 1)
                    w1_dist = tf.reduce_sum(abs_diff_batch, axis=1, keepdims=True)      # (B,1)

                return 1.0 - tf.math.log1p(w1_dist)

            case SIMILARITY_METRICS.W2:
                a_norm = tf.map_fn(self._normalize_batch, a)                             # (B, N, K)
                b_norm = tf.map_fn(self._normalize_batch, b)                             # (B, N, K)
                # component distances -> expect (B, N)
                component_distances = tf.map_fn(
                    self._get_w2_sim_2D,
                    (tf.abs(a_norm), tf.abs(b_norm)),
                    dtype=(tf.float64, tf.float64),
                    fn_output_signature=tf.float64
                )                                                                        # (B, N)

                if cum:
                    w2_dist = tf.cumsum(component_distances, axis=1)                     # (B,N)
                else:
                    w2_dist = tf.reduce_sum(component_distances, axis=1, keepdims=True)  # (B,1)

                return 1.0 - tf.math.log1p(w2_dist)

            case SIMILARITY_METRICS.Q:
                psi_batch = tf.map_fn(self._get_cdf_batch, a - b)                       # (B, N, K)
                psi_batch_npsd = self.get_npsd_batch(psi_batch)                         # assume (B, N, K) -> npsd per component
                p_x = tf.map_fn(self._integrate_batch, psi_batch_npsd)                  # (B, N)
                abs_diff_batch = tf.abs(p_x)                                            # (B, N)

                if cum:
                    q_dist = tf.cumsum(abs_diff_batch, axis=1)                          # (B,N)
                else:
                    q_dist = tf.reduce_sum(abs_diff_batch, axis=1, keepdims=True)       # (B,1)

                return 1.0 - q_dist

            case _:
                raise ValueError(f"Unknown method: {method}")