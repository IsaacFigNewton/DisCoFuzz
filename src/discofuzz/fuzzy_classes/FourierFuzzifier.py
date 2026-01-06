import tensorflow as tf
import numpy as np
from typing import Union
from scipy.special import expit
import ot
from .FuzzyFourierSetMixin import FuzzyFourierSetMixin

class FourierFuzzifier(FuzzyFourierSetMixin):
    """TensorFlow-accelerated version of FourierFuzzifier with set operations"""

    def __init__(self, sigma:float, kernel_size:int):
        if kernel_size < 1:
            raise ValueError("Kernel size must be at least 1")
        super().__init__(sigma, kernel_size)

    def get_npsd_batch(self, a: tf.Tensor, global_npsd:bool=False) -> tf.Tensor:
        # normalize the power spectral densities
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        
        norms = tf.math.reduce_sum(tf.abs(a), axis=1)

        # if we want to normalize against the global power spectral densities
        #   default is just component-wise power spectra normalization
        if global_npsd:
            # aggregate power spectral densities
            norms = tf.math.reduce_sum(norms, axis=0)
            # broadcast it to use with 'a'
            norms = tf.expand_dims(norms, axis=0)
        
        norms = tf.expand_dims(norms, axis=1)

        norms = tf.broadcast_to(
            norms,
            [tf.shape(a)[0], tf.shape(a)[1]]
        )
        return tf.abs(a) / norms
    
    
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

            case "cos":
                a_npsd = self.get_npsd_batch(a, global_npsd=True)
                b_npsd = self.get_npsd_batch(b, global_npsd=True)
                # numerator = aggregated hadamard product of a_npsd and b_npsd
                numerator = tf.reduce_sum(a_npsd * b_npsd)
                denominator_a = tf.sqrt(tf.reduce_sum(a_npsd * a_npsd))
                denominator_b = tf.sqrt(tf.reduce_sum(b_npsd * b_npsd))
                # similarity = correllation coefficient between the two npsd's
                similarity = numerator / (denominator_a * denominator_b + 1e-10)
                return similarity.numpy()
            
            case "npsd-ot":
                # get normalized power density spectra
                a = self.get_npsd_batch(a, global_npsd=True).numpy()
                b = self.get_npsd_batch(b, global_npsd=True).numpy()
                freqs = tf.range(0, self.kernel_size, dtype=tf.float32)
                u, v = np.meshgrid(freqs, freqs)

                # custom cost metric does ~0.1% worse than Wasserstein-2 cost metric
                # cost = np.exp(np.abs(u - v))

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
                
                return 1-np.log1p(np.abs(total_cost))# / a.shape[0]))

            case "p-ot":
                # Modified Wasserstein-1 earthmover's distance of probability distributions
                #   = absolute value of double integral of difference in CDFs
                abs_diff = tf.abs(self._integrate_batch(self._get_cdf_batch(a - b)))
                return 1-np.log1p(tf.reduce_sum(abs_diff).numpy())

            case _:
                raise ValueError(f"Unknown method: {method}")