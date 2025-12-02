import tensorflow as tf
import numpy as np
from typing import Union
import ot
from .FuzzyFourierSetMixin import FuzzyFourierSetMixin

class FourierFuzzifier(FuzzyFourierSetMixin):
    """TensorFlow-accelerated version of FourierFuzzifier with set operations"""

    def __init__(self, sigma:float, kernel_size:int, dft_kernel_size:int=10):
        if kernel_size < 1:
            raise ValueError("Kernel size must be at least 1")
        super().__init__(sigma, kernel_size)
        self.dft_kernel_size = dft_kernel_size

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


    def _get_dft(self, a: tf.Tensor) -> tf.Tensor:
        """
        gets the first self.dft_kernel_size x self.dft_kernel_size complex coefficients in the DFT of a tensor
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        
        # Cast to complex type if needed and compute DFT along the last dimension
        a_complex = tf.cast(a, tf.complex64)
        dft_full = tf.signal.fft(a_complex)
        
        # Extract the first dft_kernel_size x dft_kernel_size coefficients
        dft_slice = dft_full[:, :self.dft_kernel_size, :self.dft_kernel_size]

        return dft_slice


    def similarity_batch(self,
            a: tf.Tensor,
            b: tf.Tensor,
            method: str,
            dft_reduc: bool=False
        ) -> Union[float, np.ndarray]:
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

        flatten_layer = tf.keras.layers.Flatten()

        if method == "npsd-ot":

            if dft_reduc:
                a = self._get_dft(a)
                a = flatten_layer(a).numpy()
                b = self._get_dft(b).numpy()
                b = flatten_layer(b).numpy()
                freqs = tf.ones((1, self.dft_kernel_size**2), dtype=tf.float32)
                u, v = np.meshgrid(freqs, freqs)

                # normalize Wasserstein-2 metric
                cost = np.exp(np.abs(u - v))
                # cost = (u - v)**2
                cost = np.ascontiguousarray(cost, dtype='float64')

                plan = ot.emd(
                    np.ascontiguousarray(a[i, :]),
                    np.ascontiguousarray(b[i, :]),
                    cost,
                    check_marginals=False
                )
                total_cost = np.sum(plan * cost)
            
            else:
                # get normalized power density spectra
                a = self.get_npsd_batch(a).numpy()
                b = self.get_npsd_batch(b).numpy()
                freqs = tf.range(0, self.kernel_size, dtype=tf.float32)
            
                u, v = np.meshgrid(freqs, freqs)

                # normalize Wasserstein-2 metric
                cost = np.exp(np.abs(u - v))
                # cost = (u - v)**2
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