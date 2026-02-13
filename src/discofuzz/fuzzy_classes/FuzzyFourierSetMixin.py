import tensorflow as tf
import numpy as np
from .FourierPDF import FourierPDF

class FuzzyFourierSetMixin(FourierPDF):
  def __init__(self, sigma: float, kernel_size: int):
      if kernel_size < 1:
          raise ValueError("Kernel size must be at least 1")
      super().__init__(kernel_size)
      self.sigma = tf.constant(sigma, dtype=tf.complex64)

      # Pre-compute Fourier coefficients for all k values
      # C_k = e^{-\frac{a^{2}k^{2}}{2}} and keep mu portion separate for now
      c_k = tf.exp(-0.5 * (self.sigma ** 2) * (self.k_values ** 2))
      self.fourier_coeffs = c_k / (2 * np.pi)


  def _get_gaussian_at_mu_batch(self, mu: tf.Tensor) -> tf.Tensor:
        """
        Batch computation of Fourier series for Gaussians centered at multiple mu values.
        mu: shape (batch_size,)
        Returns: shape (batch_size, kernel_size)
        """
        # mu part of C_n = e^{-ikb}
        #   combine mu.shape = (batch_size,) with self.k_values.T.shape = (,self.kernel_size)
        #   to get tensor of shape (batch_size, self.kernel_size)
        mu = tf.cast(mu, tf.complex64)[:, None]              # (B,1)
        k  = tf.cast(self.k_values, tf.complex64)[None, :]   # (1,K)
        mu_c_k = tf.exp(-1j * (mu * k))                      # (B,K)
        return self._normalize_batch(mu_c_k * self.fourier_coeffs)



  def fuzzify(self, component: float) -> tf.Tensor:
      """
      Convert a real-valued component to a Fourier series representation of a Gaussian.
      component: scalar float
      Returns: shape (kernel_size,)
      """
      mu = tf.constant([component], dtype=tf.float32)
      return self._get_gaussian_at_mu_batch(mu)


  def negation_batch(self, a: tf.Tensor) -> tf.Tensor:
      """
      Batch fuzzy negation: NOT(a) = 1 - a
      a: shape (batch_size, kernel_size)
      Returns: shape (batch_size, kernel_size)
      """
      if len(tf.shape(a)) != 2:
          raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
      batch_size = tf.shape(a)[0]

      # Create batch of constant function 1
      # base tensor of same shape as a
      ones = tf.zeros_like(a)
      # indices at which to place 1's
      indices = tf.stack([
          tf.range(batch_size),
          tf.zeros(batch_size, dtype=tf.int32)
      ], axis=1)
      # ones to be inserted at said indices
      updates = tf.ones([batch_size], dtype=tf.complex64)
      # update the base tensor with 1's
      ones = tf.tensor_scatter_nd_update(ones, indices, updates)

      # get normalized negation
      return self._normalize_batch(ones - a)


  def intersection_batch(self, a: tf.Tensor, b: tf.Tensor, normalize: bool = True) -> tf.Tensor:
      """
      Batch fuzzy intersection using product.
      a, b: shape (batch_size, kernel_size)
      Returns: shape (batch_size, kernel_size)
      """
      if len(tf.shape(a)) != 2:
          raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
      result = self._convolve_batch(a, b)
      if normalize:
          result = self._normalize_batch(result)
      return result


  def union_batch(self, a: tf.Tensor, b: tf.Tensor, normalize: bool = True) -> tf.Tensor:
      """
      Batch fuzzy union: a + b - a*b
      a, b: shape (batch_size, kernel_size)
      Returns: shape (batch_size, kernel_size)
      """
      if len(tf.shape(a)) != 2:
          raise ValueError(f"Input tensor must have shape (batch_size, kernel_size), received tensor of shape {tf.shape(a)}")
      convolved = self._convolve_batch(a, b)
      result = a + b - convolved
      if normalize:
          result = self._normalize_batch(result)
      return result