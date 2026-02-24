from typing import Tuple
import tensorflow as tf
import numpy as np

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
        # replace 0th term with 1 to avoid division by 0
        self.divisor = tf.expand_dims(
            tf.concat([tf.constant([1], dtype=tf.complex64), self.k_values[1:]], axis=0),
            axis=0
        )

        # used as ramp for integration
        # expand along batch axis
        self.sawtooth = 1 / (1j * self.k_values) * (-1)**(self.k_values + 1)
        # set the DC term of the sawtooth signal to 0 to get rid of nan
        self.sawtooth = tf.tensor_scatter_nd_update(
            self.sawtooth,
            indices=[[0]],
            updates=tf.constant([0], dtype=tf.complex64)
        )
        self.sawtooth = tf.expand_dims(
            self.sawtooth,
            axis=0
        )


    def get_npsd_batch(self, a: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.abs(a)**2, dtype=tf.complex64)


    def evaluate_batch(self,
            a: tf.Tensor,
            resolution: int = 200,
            x: tf.Tensor|None = None
        ) -> tf.Tensor:
        """
        Compute fx(x) for 'resolution' # samples,
            for each distribution in the batch from their Fourier coefficients.
        """
        if x is None:
            x = tf.linspace(0.0, 1.0, resolution)
        x = tf.cast(x, dtype=tf.complex64)
        # get a matrix of shape (kernel_size, resolution)
        #   for evaluating a range of sample points along the signal
        x_k = tf.matmul(self.k_values[:, None], x[None, :])
        basis = tf.math.exp(1j * x_k)

        # (batch_size, kernel_size) x (kernel_size, resolution)
        #   = (batch_size, resolution)
        return tf.matmul(a, basis)


    def _get_cdf_batch(self, a: tf.Tensor) -> tf.Tensor:
        """
        Batch integration.
        a: shape (batch_size, kernel_size)
        Returns: shape (batch_size, kernel_size)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensors must be complex64, received {a.dtype}")

        batch_size = tf.shape(a)[0]

        # Create indices for the k=0 term for each item in the batch
        row_indices = tf.range(batch_size)
        col_indices = tf.zeros(batch_size, dtype=tf.int32)
        indices = tf.stack([row_indices, col_indices], axis=1)

        # get antiderivative of k=0 term
        # get DC/constant terms (first column)
        DC = tf.broadcast_to(
            a[:, 0][:, None],
            tf.shape(a)
        )
        # build ramp terms (for integral)
        ramp = tf.broadcast_to(
            self.sawtooth,
            tf.shape(a)
        )

        # get antiderivatives for all k >= 1
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
        AC = AC / (1j * harmonics)

        # add the k=0 terms' antiderivatives (ramp) back into AC terms
        return DC * ramp + AC

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

        bounds = tf.constant([lb, ub])
        cdf = self._get_cdf_batch(a)
        cdf_lb_ub = self.evaluate_batch(cdf, x=bounds)
        # integral is just cdf evaluated at 1 - cdf evaluated at 0
        return cdf_lb_ub[:, 1] - cdf_lb_ub[:, 0]


    def _normalize_batch(self, a: tf.Tensor) -> tf.Tensor:
        """
        Batch normalization of probability density functions.
        Returns Normalized Power Spectral Densities (NPSD) of input pdfs
        a: shape (batch_size, kernel_size)
        """
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        if a.dtype != tf.complex64:
            raise ValueError(f"Input tensor must be complex64, received {a.dtype}")
        # normalize the power spectral densities
        if len(tf.shape(a)) != 2:
            raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
        
        # norms = tf.math.reduce_sum(tf.abs(a)**2, axis=1)
        norms = self._integrate_batch(a)
        norms = tf.broadcast_to(norms[:, None], tf.shape(a))

        return a / norms


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

        return a * 1j * self.k_values - tf.broadcast_to(self.sawtooth, tf.shape(a))