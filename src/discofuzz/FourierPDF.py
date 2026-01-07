from typing import Tuple
import tensorflow as tf

class FourierPDF:
    def __init__(self, kernel_size:int):
        self.kernel_size = kernel_size
        # get frequencies
        self.k_values = tf.cast(
            tf.range(0, kernel_size),
            tf.complex64
        )
        # pre-compute partial divisor for faster integration
        #   shape=(, self.kernel_size)
        # add 1e-20 to avoid division by 0
        self.divisor = tf.expand_dims(
            self.k_values+1e-20,
            axis=0
        )

    def _get_DC_AC_divisor_batch(self, a:tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(a)[0]

        # Create indices for the k=0 term for each item in the batch
        row_indices = tf.range(batch_size)
        col_indices = tf.zeros(batch_size, dtype=tf.int32)
        indices = tf.stack([row_indices, col_indices], axis=1)

        # get DC/constant terms (first column)
        DC = a[:, 0]
        # get AC term (just set first column to 0s)
        AC = tf.tensor_scatter_nd_update(
            a,
            indices,
            tf.zeros(batch_size, dtype=tf.complex64)
        )
        
        # get divisors
        harmonics = tf.broadcast_to(
            self.divisor,
            tf.shape(a),
        )

        return DC, AC, harmonics, indices

    def _get_cdf_batch(self, a: tf.Tensor) -> tf.Tensor:
        """
        Batch integration.
        a: shape (batch_size, kernel_size)
        Returns: shape (batch_size,)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensors must be complex64, received {a.dtype}")

        DC, AC, harmonics, indices = self._get_DC_AC_divisor_batch(a)
        AC = AC / harmonics

        # add the k=0 terms back in
        return tf.tensor_scatter_nd_update(
            AC,
            indices,
            a[:, 0]
        )

    def _integrate_batch(self, a: tf.Tensor, ub:float=1, lb:float=0) -> tf.Tensor:
        """
        Batch integration.
        a: shape (batch_size, kernel_size)
        Returns: shape (batch_size,)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensors must be complex64, received {a.dtype}")

        DC, AC, harmonics, _ = self._get_DC_AC_divisor_batch(a)

        upper_bound = tf.exp(harmonics * ub)
        lower_bound = tf.exp(harmonics * lb)
        diff = upper_bound - lower_bound

        integrals_k_nonzero = tf.reduce_sum(
            (AC / harmonics) * diff,
            axis=1
        )

        # Combine k=0 and k>0 terms
        #   returns rank-1 tensor with just components' fourier series' integrals
        return DC + integrals_k_nonzero

    def _integrate(self, a: tf.Tensor, ub:float=1, lb:float=0) -> tf.Tensor:
        """
        Single integration helper.
        a: shape (kernel_size)
        Returns: scalar
        """
        if len(tf.shape(a)) != 1:
            raise ValueError(f"Input tensor must have shape (kernel_size,), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensors must be complex64, received {a.dtype}")

        a_batch = tf.expand_dims(a, axis=0)
        result_batch = self._integrate_batch(a_batch, ub, lb)
        return tf.squeeze(result_batch)

    def _normalize_batch(self, a: tf.Tensor) -> tf.Tensor:
        """
        Batch normalization of probability density functions.
        a: shape (batch_size, kernel_size)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensor must be complex64, received {a.dtype}")

        total_integral = self._integrate_batch(a)  # (batch_size,)
        total_integral = tf.expand_dims(total_integral, 1)  # (batch_size, 1)

        # Avoid division by zero
        total_integral = tf.where(
            tf.abs(total_integral) > 1e-10,
            total_integral,
            tf.ones_like(total_integral)
        )

        return a / total_integral

    def _normalize(self, a: tf.Tensor) -> tf.Tensor:
        """
        Single normalization helper for normalizing probability density functions.
        a: shape (kernel_size,)
        Returns: shape (kernel_size,)
        """
        if len(tf.shape(a)) != 1:
            raise ValueError(f"Input tensor must have shape (kernel_size,), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensor must be complex64, received {a.dtype}")

        a_batch = tf.expand_dims(a, axis=0)
        result_batch = self._normalize_batch(a_batch)
        return tf.squeeze(result_batch, axis=0)

    def _convolve_batch(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Batch convolution using FFT.
        a, b: shape (batch_size, kernel_size)
        Returns: shape (batch_size, kernel_size)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if len(tf.shape(b)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(b)}")

        # Batch FFT convolution
        A_fft = tf.signal.fft(tf.cast(a, tf.complex64))
        B_fft = tf.signal.fft(tf.cast(b, tf.complex64))
        C_fft = A_fft * B_fft
        C = tf.signal.ifft(C_fft)

        return tf.cast(C, tf.complex64)

    def _convolve(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Single convolution helper.
        a, b: shape (kernel_size,)
        Returns: shape (kernel_size,)
        """
        a_batch = tf.expand_dims(a, axis=0)
        b_batch = tf.expand_dims(b, axis=0)
        result_batch = self._convolve_batch(a_batch, b_batch)
        return tf.squeeze(result_batch, axis=0)

    def _differentiate_batch(self, a: tf.Tensor) -> tf.Tensor:
        """
        Batch differentiation.
        a: shape (batch_size, kernel_size)
        Returns: shape (batch_size, kernel_size)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")

        return a * tf.exp(1j * self.k_values)

    def _differentiate(self, a: tf.Tensor) -> tf.Tensor:
        """
        Single differentiation helper.
        a: shape (kernel_size,)
        Returns: shape (kernel_size,)
        """
        a_batch = tf.expand_dims(a, axis=0)
        result_batch = self._differentiate_batch(a_batch)
        return tf.squeeze(result_batch, axis=0)