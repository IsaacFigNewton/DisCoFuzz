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
    ) -> tf.Tensor:
        """
        a, b: (batch_size, n_components, kernel_size)
        Returns: (batch_size, n_components) where out[:, n-1] uses first n components.
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
                # numerator = aggregated spectra of convolution of a and b
                ab = a * b
                aa = a * a
                bb = b * b
                ab_components_total_power = tf.reduce_sum(ab, axis=2)  # (B, N)
                aa_components_total_power = tf.reduce_sum(aa, axis=2)  # (B, N)
                bb_components_total_power = tf.reduce_sum(bb, axis=2)  # (B, N)
                
                # Prefix aggregate across components' energies so prefix n uses first n components
                numerator = tf.cumsum(ab_components_total_power, axis=1)                # (B, N)
                denominator_a = tf.sqrt(tf.cumsum(aa_components_total_power, axis=1))   # (B, N)
                denominator_b = tf.sqrt(tf.cumsum(bb_components_total_power, axis=1))   # (B, N)
                # similarity = correllation coefficient between the two npsd's
                similarity = numerator / (denominator_a * denominator_b + 1e-10)
                return 1 - tf.abs(similarity).numpy()
            
            case SIMILARITY_METRICS.W1:
                # Modified Wasserstein-1 earthmover's distance of probability distributions
                #   = sum of absolute values of integrals of differences in components' CDFs
                #   = sum of magnitudes of W1 EMDs
                # using tf.map_fn to get cdf of each component in 2D slices of tensor (different samples)
                psi_batch = tf.map_fn(self._get_cdf_batch, a - b)                       # (B, N, K)
                # integrate cdfs for each component in each sample
                integrate_cdf_batch = tf.map_fn(self._integrate_batch, psi_batch)       # (B, N)
                abs_diff_batch = tf.abs(integrate_cdf_batch)                            # (B, N)
                w1_dist = tf.cumsum(abs_diff_batch).numpy()                             # (B, N)
                # do 1-log1p(w1) since EMD is inversely proportional to distributions' similarities
                return 1-np.log1p(w1_dist)
            
            case SIMILARITY_METRICS.W2:
                a = tf.map_fn(self._normalize_batch, a)                                 # (B, N, K)
                b = tf.map_fn(self._normalize_batch, b)                                 # (B, N, K)
                # get W2 distances for each 2D slice of the batch
                component_distances = tf.map_fn(
                    self._get_w2_sim_2D,
                    (tf.abs(a), tf.abs(b)),
                    dtype=(tf.float64, tf.float64),
                    fn_output_signature=tf.float64
                )                                                                       # (B, N)
                aggregated_costs = tf.cumsum(component_distances, axis=1)               # (B, N)
                return 1-tf.abs(aggregated_costs**2)                                    # (B, N)
            
            case SIMILARITY_METRICS.Q:
                psi_batch = tf.map_fn(self._get_cdf_batch, a - b)                       # (B, N, K)
                # Get the PDF associated with the wave function described by psi
                #   p(x) = integral(0, 1, |psi|^2)
                # this metric is different from the Wasserstein-1 metric
                #   only in that it squares psi prior to integration
                psi_batch_npsd = self.get_npsd_batch(psi_batch)
                p_x = tf.map_fn(self._integrate_batch, psi_batch_npsd)                  # (B, N)
                
                # do 1-log1p(w1**2) since distance is inversely proportional to distributions' similarities
                return 1-tf.cumsum(tf.abs(p_x), axis=1).numpy()                                 # (B, N)

            case _:
                raise ValueError(f"Unknown method: {method}")