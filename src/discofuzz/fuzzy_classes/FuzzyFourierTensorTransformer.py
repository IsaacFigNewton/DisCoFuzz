from typing import List
import numpy as np
import tensorflow as tf
from ..constants import SIMILARITY_METRICS
from .FourierFuzzifier import FourierFuzzifier

class FuzzyFourierTensorTransformer:
    """
    TensorFlow-accelerated fuzzy tensor transformer.
    All operations are vectorized and GPU-compatible.
    """

    def __init__(self, sigma: float = 0.1, kernel_size: int = 16):
        self.fuzzifier = FourierFuzzifier(sigma, kernel_size)
        self.kernel_size = kernel_size

    @tf.function
    def fuzzify(self, A: tf.Tensor|np.ndarray) -> tf.Tensor:
        """
        Vectorized fuzzification.
        A: (d) or (1, d)
        Returns: shape (d, kernel_size)
        """
        if isinstance(A, np.ndarray):
            A = tf.convert_to_tensor(A, dtype=tf.complex64)

        # Ensure A is 1D by squeezing all dimensions of size 1
        A = tf.squeeze(A)

        # Verify the result is 1D
        if len(tf.shape(A)) != 1:
            raise ValueError(f"Input tensor must be 1D after squeezing, received tensor of shape {tf.shape(A)}")

        # Reshape to (batch_size, d, kernel_size)
        return tf.cast(self.fuzzifier._get_gaussian_at_mu_batch(A), dtype=tf.complex64)


    @tf.function
    def intersection(self, A: tf.Tensor, B: tf.Tensor, normalize: bool = True) -> tf.Tensor:
        """
        Vectorized fuzzy intersection.
        A, B: shape (d,kernel_size)
        """
        if not len(A.shape) == 2:
          raise ValueError(f"A must be rank 2 tensors. Expected A.shape == 2, but got A.shape == {A.shape}")
        if not len(B.shape) == 2:
          raise ValueError(f"B must be rank 2 tensors. Expected A.shape == 2, but got A.shape == {B.shape}")

        result = self.fuzzifier._convolve_batch(A, B)

        if normalize:
            result = self.fuzzifier._normalize_batch(result)

        return result


    @tf.function
    def iterated_intersection(self, vects: List[tf.Tensor]) -> tf.Tensor:
        """
        Efficiently compute intersection over multiple tensors.
        vects: shape (n_vects, d,kernel_size)
        """
        # if there's only 1 tensor to get the intersection
        if len(vects) == 1:
            return vects[0]

        result = vects[0]
        for v in vects:
          # only include vect in intersection if it's the correct shape
          if len(v.shape) == 3:
              result = self.intersection(result, v, normalize=False)

        # Normalize the final result
        return self.fuzzifier._normalize_batch(result)

    @tf.function
    def union(self, A: tf.Tensor, B: tf.Tensor, normalize: bool = True) -> tf.Tensor:
        """
        Vectorized fuzzy union: A + B - A*B
        A, B: shape (d,kernel_size)
        """
        convolved = self.intersection(A, B, normalize=False)
        result = A + B - convolved

        if normalize:
            shape = tf.shape(result)
            if len(result.shape) == 3:
                result = self.fuzzifier._normalize_batch(result)
            else:
                result_flat = tf.reshape(result, [-1, self.kernel_size])
                result_flat = self.fuzzifier._normalize_batch(result_flat)
                result = tf.reshape(result_flat, shape)

        return result

    @tf.function
    def iterated_union(self, vects: List[tf.Tensor]) -> tf.Tensor:
        """
        Efficiently compute union over multiple tensors.
        vects: shape (n_vects, d,kernel_size)
        """
        # if there's only 1 tensor to get the union
        if len(vects) == 1:
            return vects[0]
        
        result = vects[0]
        for v in vects:
          # only include vect in union if it's the correct shape
          if len(v.shape) == 3:
              result = self.union(result, v, normalize=False)
        # Normalize the final result
        return self.fuzzifier._normalize_batch(result)

    def similarity(self,
            A: tf.Tensor,
            B: tf.Tensor,
            method: SIMILARITY_METRICS
        ) -> float:
        """
        Vectorized similarity computation.
        A, B: shape (d,kernel_size)
        Returns: scalar similarity between the normalized power spectral densities of A, B
        """
        
        if A is None or B is None:
            raise ValueError(f"Inputs must be tensor, got None")
        if not len(A.shape) == 2:
          print(A)
          raise ValueError(f"A must be rank 2 tensor. Expected A.shape == 2, but got A.shape == {A.shape}")
        if not len(B.shape) == 2:
          raise ValueError(f"B must be rank 2 tensor. Expected A.shape == 2, but got A.shape == {B.shape}")

        return self.fuzzifier.similarity_batch(A, B, method)