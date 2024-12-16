import torch
import torch.nn as nn

class AllZeroEncoder(nn.Module):
    r"""AllZeroEncoder(k, n, dtype=tf.float32, **kwargs)
    Dummy encoder that always outputs the all-zero codeword of length ``n``.

    Note that this encoder is a dummy encoder and does NOT perform real
    encoding!

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        k: int
            Defining the number of information bit per codeword.

        n: int
            Defining the desired codeword length.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        inputs: [...,k], tf.float32
            2+D tensor containing arbitrary values (not used!).

    Output
    ------
        : [...,n], tf.float32
            2+D tensor containing all-zero codewords.

    Raises
    ------
        AssertionError
            ``k`` and ``n`` must be positive integers and ``k`` must be smaller
            (or equal) than ``n``.

    Note
    ----
        As the all-zero codeword is part of any linear code, it is often used
        to simulate BER curves of arbitrary (LDPC) codes without the need of
        having access to the actual generator matrix. However, this `"all-zero
        codeword trick"` requires symmetric channels (such as BPSK), otherwise
        scrambling is required (cf. [Pfister]_ for further details).

        This encoder is a dummy encoder that is needed for some all-zero
        codeword simulations independent of the input. It does NOT perform
        real encoding although the information bits are taken as input.
        This is just to ensure compatibility with other encoding layers.
    """

    def __init__(self, k, n, dtype=torch.float32):
        super().__init__()

        assert isinstance(k, (int, float)), "k must be a number."
        assert isinstance(n, (int, float)), "n must be a number."

        k = int(k)
        n = int(n)

        assert k >= 0, "k cannot be negative."
        assert n >= 0, "n cannot be negative."
        assert n >= k, "Invalid coderate (>1)."

        self._k = k
        self._n = n
        self._coderate = k / n

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        return self._k

    @property
    def n(self):
        return self._n

    @property
    def coderate(self):
        return self._coderate

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build."""
        pass

    def forward(self, inputs):
        """Encoding function that outputs the all-zero codeword.

        This function returns the all-zero codeword of shape `[..., n]`.
        Note that this encoder is a dummy encoder and does NOT perform real
        encoding!

        Args:
            inputs (tf.float32): Tensor of arbitrary shape.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Note:
            This encoder is a dummy encoder that is needed for some all-zero
            codeword simulations independent of the input. It does NOT perform
            real encoding although the information bits are taken as input.
            This is just to ensure compatibility with other encoding layers.
        """
        # keep shape of first dimensions
        # return an all-zero tensor of shape [..., n]
        output_shape = list(inputs.shape[:-1]) + [self._n]
        return torch.zeros(output_shape, dtype=torch.float32)
