import tensorflow as tf
import numpy as np
from enum import Enum
from typing import Union
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
    

    def get_p_wasserstein_distance_batch(self, a: tf.Tensor, b: tf.Tensor, p: int) -> float:
        a_cdf = self._get_cdf_batch(a)
        b_cdf = self._get_cdf_batch(b)
        
        # calculate inverse cdf of each tensor column,
        #   such that a_cdf(a_cdf_inv(x)) = 1
        # these calculations are considered so trivial
        #   that I leave them to whoever reads this code to solve
        a_inv_cdf = tf.zeros_like(a)
        b_inv_cdf = tf.zeros_like(b)
        abs_inv_cdf_diff = tf.abs(a_inv_cdf - b_inv_cdf)
        
        # see https://en.wikipedia.org/wiki/Wasserstein_metric for full definition of Wp metric
        return self._integrate_batch(abs_inv_cdf_diff**p)**(1.0/p)


    
    def similarity(self, a: tf.Tensor, b: tf.Tensor, method: str) -> Union[float, np.ndarray]:
        """
        Compute similarity as ot similarity in frequency domain.
        a, b: shape (kernel_size,)
        Returns: scalar similarity
        """
        if not a or not b:
          return None

        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if len(tf.shape(b)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(b)}")

        return self.similarity_batch(
            tf.expand_dims(a, axis=0),
            tf.expand_dims(b, axis=0),
            method=method
        )


    def similarity_batch(self,
            a: tf.Tensor,
            b: tf.Tensor,
            method: str,
        ) -> np.ndarray:
        """
        Batch computation of pairwise similarities.
        a, b: shape (batch_size, kernel_size)
        Returns: shape (batch_size,) - similarity for each pair
        """
        if a is None or b is None:
            raise ValueError(f"Inputs must be tensors, got None")
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if len(tf.shape(b)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(b)}")

        match method:

            case SIMILARITY_METRICS.COS:
                # numerator = aggregated spectra of convolution of a and b
                numerator = tf.reduce_sum(a * b)
                denominator_a = tf.sqrt(tf.reduce_sum(a * a))
                denominator_b = tf.sqrt(tf.reduce_sum(b * b))
                # similarity = correllation coefficient between the two npsd's
                similarity = numerator / (denominator_a * denominator_b + 1e-10)
                return 1 - tf.abs(similarity).numpy()
            
            case SIMILARITY_METRICS.W1:
                # Modified Wasserstein-1 earthmover's distance of probability distributions
                #   = sum of absolute values of integrals of differences in components' CDFs
                #   = sum of magnitudes of W1 EMDs
                psi = self._get_cdf_batch(a - b)
                abs_diff = tf.abs(self._integrate_batch(psi))
                w1_dist = tf.reduce_sum(abs_diff).numpy()
                # do 1-log1p(w1) since EMD is inversely proportional to distributions' similarities
                return 1-np.log1p(w1_dist)
            
            case SIMILARITY_METRICS.W2:
                a = self._normalize_batch(a)
                b = self._normalize_batch(b)
                a = tf.abs(a).numpy()
                b = tf.abs(b).numpy()
                freqs = tf.range(0, self.kernel_size, dtype=tf.float32)
                u, v = np.meshgrid(freqs, freqs)

                # Wasserstein-2 metric
                cost = (u - v)**2
                cost = np.ascontiguousarray(cost, dtype='float64')

                total_cost = 0
                for i in range(a.shape[0]):
                    plan = ot.emd(
                        np.ascontiguousarray(a[i, :]),
                        np.ascontiguousarray(b[i, :]),
                        cost,
                        check_marginals=False
                    )
                    total_cost += np.sum(plan * cost)
                
                return 1-np.abs(total_cost**2)# / a.shape[0]))
            
            case SIMILARITY_METRICS.Q:
                psi = self._get_cdf_batch(a - b)
                # Get the PDF associated with the wave function described by psi
                #   p(x) = integral(0, 1, |psi|^2)
                # this metric is different from the Wasserstein-1 metric
                #   only in that it squares psi prior to integration
                p_x = self._integrate_batch(self.get_npsd_batch(psi))
                
                # do 1-log1p(w1**2) since distance is inversely proportional to distributions' similarities
                return 1-tf.reduce_sum(tf.abs(p_x)).numpy()

            case _:
                raise ValueError(f"Unknown method: {method}")