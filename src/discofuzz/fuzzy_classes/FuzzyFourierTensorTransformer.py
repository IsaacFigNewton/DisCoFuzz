import tensorflow as tf
from .FourierFuzzifier import FourierFuzzifier

class FuzzyFourierTensorTransformer:
    """
    TensorFlow-accelerated fuzzy tensor transformer.
    All operations are vectorized and GPU-compatible.
    """

    def __init__(self, sigma: float = 0.1, kernel_size: int = 8):
        self.fuzzifier = FourierFuzzifier(sigma, kernel_size)
        self.kernel_size = kernel_size

    @tf.function
    def fuzzify(self, A: tf.Tensor) -> tf.Tensor:
        """
        Vectorized fuzzification.
        A: (d)
        Returns: shape (d, kernel_size)
        """
        if len(tf.shape(A)) != 1:
            raise ValueError(f"Input tensor must have shape (d), received tensor of shape {tf.shape(A)}")

        # Reshape to (batch_size, d, kernel_size)
        return self.fuzzifier._get_gaussian_at_mu_batch(A)


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
    def iterated_intersection(self, vects: tf.Tensor) -> tf.Tensor:
        """
        Efficiently compute intersection over multiple tensors.
        vects: shape (n_vects, d,kernel_size)
        """

        # if there's only 1 tensor to get the intersection
        if tf.shape(vects)[0] == 1:
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
    def iterated_union(self, vects: tf.Tensor) -> tf.Tensor:
        """
        Efficiently compute union over multiple tensors.
        vects: shape (n_vects, d,kernel_size)
        """

        # if there's only 1 tensor to get the union
        if tf.shape(vects)[0] == 1:
            return vects[0]

        result = vects[0]
        for v in vects:
          # only include vect in union if it's the correct shape
          if len(v.shape) == 3:
              result = self.union(result, v, normalize=False)

        # Normalize the final result
        return self.fuzzifier._normalize_batch(result)

    def similarity(self, A: tf.Tensor, B: tf.Tensor, method:str = "p-ot") -> float:
        """
        Vectorized similarity computation.
        A, B: shape (d,kernel_size)
        Returns: scalar similarity between the normalized power spectral densities of A, B
        """
        if not len(A.shape) == 2:
          raise ValueError(f"A must be rank 2 tensors. Expected A.shape == 2, but got A.shape == {A.shape}")
        if not len(B.shape) == 2:
          raise ValueError(f"B must be rank 2 tensors. Expected A.shape == 2, but got A.shape == {B.shape}")

        if method == "cos":
            a_npsd = self.fuzzifier.get_npsd_batch(A)
            b_npsd = self.fuzzifier.get_npsd_batch(B)
            # numerator = aggregated hadamard product of a_npsd and b_npsd
            numerator = tf.reduce_sum(a_npsd * b_npsd)
            denominator_a = tf.sqrt(tf.reduce_sum(a_npsd * a_npsd))
            denominator_b = tf.sqrt(tf.reduce_sum(b_npsd * b_npsd))
            # similarity = correllation coefficient between the two npsd's
            similarity = numerator / (denominator_a * denominator_b + 1e-10)
            return similarity.numpy()

        else:
            return self.fuzzifier.similarity_batch(A, B, method)