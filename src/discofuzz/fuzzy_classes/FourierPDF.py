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

        # init negative k values for padding (purely for convolution)
        #   tf.shape(neg_k) = (1, self.kernel_size)
        self.neg_k_values = -1 * (self.kernel_size - self.k_values)

        # pre-compute partial divisor for faster integration
        #   shape=(, self.kernel_size)
        # add 1e-20 to avoid division by 0
        self.divisor = tf.expand_dims(
            self.k_values+1e-20,
            axis=0
        )

    def evaluate_batch(self,
            a: tf.Tensor,
            resolution: int = 200
        ) -> tf.Tensor:
        """
        Compute fx(x) for 'resolution' # samples,
            for each distribution in the batch from their Fourier coefficients.
        """
        x = tf.cast(tf.linspace(0.0, 1.0, resolution), dtype=tf.complex64)
        # get a matrix of shape (kernel_size, resolution)
        #   for evaluating a range of sample points along the signal
        x_k = tf.matmul(self.k_values[:, None], x[None, :])
        basis = tf.math.exp(1j * x_k)

        # (batch_size, kernel_size) x (kernel_size, resolution)
        #   = (batch_size, resolution)
        print("shape of tensor evaluated from 0 to 1: ", tf.matmul(a, basis).shape)
        return tf.matmul(a, basis)


    def _get_DC_AC_divisor_batch(self, a:tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Docstring for _get_DC_AC_divisor_batch
        
        :param self: Description
        :param a: Description
        :type a: tf.Tensor
        :return: Description
        :rtype: Tuple[Tensor, Tensor, Tensor, Tensor]
        """
        
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
        Returns: shape (batch_size, kernel_size)
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
            DC
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

        # normalize each row's power spectral densities
        totals = tf.reduce_sum(tf.abs(a)**2, axis=1)
        total_integral = tf.expand_dims(tf.cast(totals, dtype=tf.complex64), axis=1)

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


    @tf.function
    def _rowwise_complex_conv(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Row-wise 1D convolution (depthwise / per-row kernel) supporting real or complex.

        - a: [batch, 3k]
        - b: [batch, 3k]
        Convolve each row of a with the associated row of b, then keep
        output indices [k-1 : 2k-1] (k values).

        For complex inputs, computes:
        (a_r + i a_i) * (b_r + i b_i)
        = (a_r*b_r - a_i*b_i) + i(a_r*b_i + a_i*b_r)
        where * denotes the same row-wise real conv.
        """
        # b = b[:, :(1 - self.kernel_size)]

        def _rowwise_conv_real(a_real: tf.Tensor, b_real: tf.Tensor) -> tf.Tensor:
            # a_real, b_real: [B, 3k] real dtype
            a_t = tf.transpose(a_real)  # [3k, B]
            b_t = tf.transpose(b_real)  # [3k, B]

            main = tf.expand_dims(a_t, axis=0)  # [1, 3k, B]  (width, channels=B)

            # Diagonal kernels: [filter_width=3k, in_channels=B, out_channels=B]
            kernels = tf.linalg.diag(b_t)

            y_g = tf.nn.conv1d(
                main,
                filters=kernels,
                stride=1,
                padding="SAME",
            )  # [1, out_w, B]

            y = tf.transpose(tf.squeeze(y_g, axis=0))  # [B, out_w]

            tf.print("out_w =", tf.shape(y)[1])
            return y  # [B, k]

        # Fast path for real inputs
        if not a.dtype.is_complex and not b.dtype.is_complex:
            return _rowwise_conv_real(a, b)

        # Complex path: promote both to complex, then split
        # (also handles the case where one is complex and the other is real)
        complex_dtype = a.dtype if a.dtype.is_complex else b.dtype
        a_c = tf.cast(a, complex_dtype)
        b_c = tf.cast(b, complex_dtype)

        a_r = tf.math.real(a_c)
        a_i = tf.math.imag(a_c)
        b_r = tf.math.real(b_c)
        b_i = tf.math.imag(b_c)

        # Compute the 4 real convolutions (each returns [B, k])
        rr = _rowwise_conv_real(a_r, b_r)
        ii = _rowwise_conv_real(a_i, b_i)
        ri = _rowwise_conv_real(a_r, b_i)
        ir = _rowwise_conv_real(a_i, b_r)

        real_out = rr - ii
        imag_out = ri + ir

        print("shape of convolved output: ", tf.shape(real_out))
        return tf.complex(real_out, imag_out)


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

        # # Batch FFT convolution
        # A_fft = tf.signal.fft(tf.cast(a, tf.complex64))
        # B_fft = tf.signal.fft(tf.cast(b, tf.complex64))
        # C_fft = A_fft * B_fft
        # C = tf.signal.ifft(C_fft)
        # return tf.cast(C, tf.complex64)
        
        # get paddings
        # a_conv_a = self._rowwise_complex_conv(a, a)
        a_conv_b = self._rowwise_complex_conv(a, b)
        b_conv_a = self._rowwise_complex_conv(b, a)
        # b_conv_b = self._rowwise_complex_conv(b, b)
        return -1 / (a_conv_b + b_conv_a)


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

        return a * 1j * self.k_values

    def _differentiate(self, a: tf.Tensor) -> tf.Tensor:
        """
        Single differentiation helper.
        a: shape (kernel_size,)
        Returns: shape (kernel_size,)
        """
        a_batch = tf.expand_dims(a, axis=0)
        result_batch = self._differentiate_batch(a_batch)
        return tf.squeeze(result_batch, axis=0)