import tensorflow as tf
import numpy as np
from typing import List, Union
import ot

class FourierFuzzifier(FuzzyFourierSetMixin):
    """TensorFlow-accelerated version of FourierFuzzifier with set operations"""

    def __init__(self, sigma: float, kernel_size: int):
        if kernel_size < 1:
            raise ValueError("Kernel size must be at least 1")
        super().__init__(sigma, kernel_size)

    def get_npsd_batch(self, a: tf.Tensor) -> tf.Tensor:
        # normalize the power spectral densities
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        norms = tf.expand_dims(tf.math.reduce_sum(tf.abs(a), axis=1), axis=1)
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

    def similarity_batch(self, a: tf.Tensor, b: tf.Tensor, method: str) -> Union[float, np.ndarray]:
        """
        Batch computation of pairwise similarities.
        a, b: shape (batch_size, kernel_size)
        Returns: shape (batch_size,) - similarity for each pair
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if len(tf.shape(b)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(b)}")

        flatten_layer = tf.keras.layers.Flatten()
        freqs = tf.range(0, self.kernel_size, dtype=tf.float32)

        if method == "npsd-ot":
            # get normalized power density spectra
            a_npsd = self.get_npsd_batch(a)
            b_npsd = self.get_npsd_batch(b)

            # # energy is proportional to frequency^2 * amplitude^2
            # a_energy = a_npsd * a_npsd * freqs * freqs
            # b_energy = b_npsd * b_npsd * freqs * freqs
            # # Normalize to probability distributions
            # a_energy = a_energy / tf.reduce_sum(a_energy, axis=1, keepdims=True)
            # b_energy = b_energy / tf.reduce_sum(b_energy, axis=1, keepdims=True)

            max_freq = self.kernel_size - 1
            a_npsd = a_npsd.numpy()
            b_npsd = b_npsd.numpy()

            u, v = np.meshgrid(freqs, freqs)
            # normalize Wasserstein-2 metric
            cost = np.exp(u - v)
            # cost = (u - v)**2
            cost = np.ascontiguousarray(cost, dtype='float64')

            total_cost = 0
            for i in range(a.shape[0]):
                plan = ot.emd(
                    np.ascontiguousarray(a_npsd[i, :]),
                    np.ascontiguousarray(b_npsd[i, :]),
                    cost,
                    check_marginals=False
                )
                total_cost += np.sum(plan * cost)

            return 1-np.log1p(np.abs(total_cost / a.shape[0]))

        elif method == "p-ot":
            # Wasserstein-1 earthmover's distance of probability distributions
            #   = integrate(tf.abs(antiderivative(pdf_1) - antiderivative(pdf_2)))
            # get cdfs of the distributions
            a_cdf = self._get_cdf_batch(a)
            b_cdf = self._get_cdf_batch(b)
            # get the absolute value of their difference
            diff = tf.cast(tf.abs(a_cdf - b_cdf), dtype=tf.complex64)
            # integrate their absolute difference
            abs_diff = tf.abs(self._integrate_batch(diff))
            # print(abs_diff[:5])
            return 1-np.log1p(tf.reduce_sum(abs_diff).numpy())

        else:
            raise ValueError(f"Unknown method: {method}")