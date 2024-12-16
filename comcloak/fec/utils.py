import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import warnings
from importlib_resources import files, as_file
from comcloak.fec.ldpc import codes
from comcloak.utils import log2
# from comcloak.nr.utils import generate_prng_seq as generate_prng_seq_utils

def generate_prng_seq(length, c_init):
    r"""Implements pseudo-random sequence generator as defined in Sec. 5.2.1
    in [3GPP38211]_ based on a length-31 Gold sequence.

    Parameters
    ----------
    length: int
        Desired output sequence length.

    c_init: int
        Initialization sequence of the PRNG. Must be in the range of 0 to
        :math:`2^{32}-1`.

    Output
    ------
    :[``length``], ndarray of 0s and 1s
        Containing the scrambling sequence.

    Note
    ----
    The initialization sequence ``c_init`` is application specific and is
    usually provided be higher layer protocols.
    """

    # check inputs for consistency
    assert(length%1==0), "length must be a positive integer."
    length = int(length)
    assert(length>0), "length must be a positive integer."

    assert(c_init%1==0), "c_init must be integer."
    c_init = int(c_init)
    assert(c_init<2**32), "c_init must be in [0, 2^32-1]."
    assert(c_init>=0), "c_init must be in [0, 2^32-1]."

    # internal parameters
    n_seq = 31 # length of gold sequence
    n_c = 1600 # defined in 5.2.1 in 38.211

    # init sequences
    c = np.zeros(length)
    x1 = np.zeros(length + n_c + n_seq)
    x2 = np.zeros(length + n_c + n_seq)

    #int2bin
    bin_ = format(c_init, f'0{n_seq}b')
    c_init = [int(x) for x in bin_[-n_seq:]] if n_seq else []
    c_init = np.flip(c_init) # reverse order

    # init x1 and x2
    x1[0] = 1
    x2[0:n_seq] = c_init

    # and run the generator
    for idx in range(length + n_c):
        x1[idx+31] = np.mod(x1[idx+3] + x1[idx], 2)
        x2[idx+31] = np.mod(x2[idx+3] + x2[idx+2] + x2[idx+1] + x2[idx], 2)

    # update output sequence
    for idx in range(length):
        c[idx] = np.mod(x1[idx+n_c] + x2[idx+n_c], 2)

    return c

def llr2mi(llr, s=None, reduce_dims=True):
    # pylint: disable=line-too-long
    r"""Implements an approximation of the mutual information based on LLRs.

    The function approximates the mutual information for given ``llr`` as
    derived in [Hagenauer]_ assuming an `all-zero codeword` transmission

    .. math::

        I \approx 1 - \sum \operatorname{log_2} \left( 1 + \operatorname{e}^{-\text{llr}} \right).

    This approximation assumes that the following `symmetry condition` is fulfilled

    .. math::

        p(\text{llr}|x=0) = p(\text{llr}|x=1) \cdot \operatorname{exp}(\text{llr}).

    For `non-all-zero` codeword transmissions, this methods requires knowledge
    about the signs of the original bit sequence ``s`` and flips the signs
    correspondingly (as if the all-zero codeword was transmitted).

    Please note that we define LLRs as :math:`\frac{p(x=1)}{p(x=0)}`, i.e.,
    the sign of the LLRs differ to the solution in [Hagenauer]_.

    Input
    -----
        llr : torch.float32
            Tensor of arbitrary shape containing LLR-values.

        s : None or torch.float32
            Tensor of same shape as llr containing the signs of the
            transmitted sequence (assuming BPSK), i.e., +/-1 values.

        reduce_dims : bool
            Defaults to True. If True, all dimensions are
            reduced and the return is a scalar. Otherwise, `reduce_mean` is
            only taken over the last dimension.

    Output
    ------
        mi : torch.float32
            A scalar tensor (if ``reduce_dims`` is True) or a tensor of same
            shape as ``llr`` apart from the last dimensions that is removed.
            It contains the approximated value of the mutual information.

    Raises
    ------
        TypeError
            If dtype of ``llr`` is not a real-valued float.

    """

    if s is None:
        s = torch.ones_like(llr)

    if llr.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("Dtype of llr must be a real-valued float.")

    # ensure that both tensors are compatible
    # s = tf.cast(s, llr.dtype)
    s = s.type(llr.dtype)

    # scramble sign as if all-zero cw was transmitted
    llr_zero = torch.multiply(s, llr)
    llr_zero = torch.clip(llr_zero, -20., 20.)  # clip for stability
    x = log2(1. + torch.exp(1. * llr_zero))
    if reduce_dims:
        # x = 1. - tf.reduce_mean(x)
        x = torch.mean(x)
    else:
        # x = 1. - tf.reduce_mean(x, axis=-1)
        x = 1. - torch.mean(x, dim=-1)
    return x

def load_parity_check_examples(pcm_id, verbose=False):
    # pylint: disable=line-too-long
    """Utility function to load example codes stored in sub-folder LDPC/codes.

    The following codes are available

    - 0 : `(7,4)`-Hamming code of length `k=4` information bits and codeword    length `n=7`.

    - 1 : `(63,45)`-BCH code of length `k=45` information bits and codeword    length `n=63`.

    - 2 : (127,106)-BCH code of length `k=106` information bits and codeword    length `n=127`.

    - 3 : Random LDPC code with regular variable node degree 3 and check node degree 6 of length `k=50` information bits and codeword length         `n=100`.

    - 4 : 802.11n LDPC code of length of length `k=324` information bits and    codeword length `n=648`.

    Input
    -----
        pcm_id : int
            An integer defining which matrix id to load.

        verbose : bool
            Defaults to False. If True, the code parameters are
            printed.

    Output
    ------
        pcm: ndarray of `zeros` and `ones`
            An ndarray containing the parity check matrix.

        k : int
            An integer defining the number of information bits.

        n : int
            An integer defining the number of codeword bits.

        coderate : float
            A float defining the coderate (assuming full rank of
            parity-check matrix).
    """

    source = files(codes).joinpath("example_codes.npy")
    with as_file(source) as code:
        pcms = np.load(code, allow_pickle=True)

    pcm = np.array(pcms[pcm_id]) # load parity-check matrix
    n = int(pcm.shape[1]) # number of codeword bits (codeword length)
    k = int(n - pcm.shape[0]) # number of information bits k per codeword
    coderate = k / n

    if verbose:
        print(f"\nn: {n}, k: {k}, coderate: {coderate:.3f}")
    return pcm, k, n, coderate

def int_mod_2(x):
    r"""Efficient implementation of modulo 2 operation for integer inputs.

    This function assumes integer inputs or implicitly casts to int.

    Remark: the function `tf.math.mod(x, 2)` is placed on the CPU and, thus,
    causes unnecessary memory copies.

    Parameters
    ----------
    x: tf.Tensor
        Tensor to which the modulo 2 operation is applied.

    """

    # x_int32 = tf.cast(x, tf.int32)
    # y_int32 = tf.bitwise.bitwise_and(x_int32, tf.constant(1, tf.int32))
    # return tf.cast(y_int32, x.dtype)
    x_int32 = x.type(torch.int32)
    y_int32 = torch.Tensor.bitwise_and(x_int32, torch.tensor(1, dtype=torch.int32))
    return y_int32.type(x.dtype)

def j_fun_inv_torch(mi, verify_inputs=True):
    """
    Calculates the inverse J-function in PyTorch.
    
    Args:
        mi (torch.Tensor): Tensor of arbitrary shape.
        verify_inputs (bool): If True, ensures mi is within valid range.

    Returns:
        torch.Tensor: Tensor of the same shape as mi.
    """
    assert isinstance(verify_inputs, bool), "verify_inputs must be bool."
    
    if verify_inputs:
        # Ensure mi is in (0, 1) range
        mi = torch.clamp(mi, 1e-10, 1.0)
    else:
        if torch.any(mi < 0) or torch.any(mi > 1):
            raise ValueError("mi must be in the range [0, 1].")

    h1 = 0.3073
    h2 = 0.8935
    h3 = 1.1064

    # J-function inverse calculation
    mu = 0.5 * ((-1 / h1) * torch.log2(1 - mi ** (1 / h3))) ** (1 / h2)
    return torch.clamp(mu, max=20)  # Clip the output to mu_max = 20

class GaussianPriorSource(nn.Module):
    """
    Generates fake LLRs as if all-zero codeword was transmitted
    over a Bi-AWGN channel.

    Args:
        specified_by_mi (bool): If True, interpret noise variance as mutual information.
        dtype (torch.dtype): Data type for internal calculations.
    """
    def __init__(self, specified_by_mi=False, dtype=torch.float32):
        super().__init__()
        
        if dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("Only float dtypes are supported.")
        
        self.specified_by_mi = specified_by_mi
        self.dtype = dtype

    def forward(self, inputs):
        """
        Generate Gaussian distributed fake LLRs.

        Args:
            output_shape (tuple): Desired shape of the output tensor.
            noise_var (torch.Tensor): Noise variance or mutual information (if specified_by_mi=True).

        Returns:
            torch.Tensor: LLRs with the given shape and properties.
        """
        output_shape, noise_var = inputs
        # output_shape = torch.tensor(output_shape)
        noise_var = torch.tensor(noise_var)
        if self.specified_by_mi:
            # Interpret noise_var as mutual information
            mi_a = torch.clamp(noise_var, min=1e-7, max=1.0)
            mu_llr = j_fun_inv_torch(mi_a)
            sigma_llr = torch.sqrt(2 * mu_llr)
        else:
            # Interpret noise_var as noise variance
            noise_var = torch.clamp(noise_var, min=1e-7)
            sigma_llr = torch.sqrt(4 / noise_var)
            mu_llr = sigma_llr ** 2 / 2

        # Convert to the specified dtype
        mu_llr = mu_llr.to(self.dtype)
        sigma_llr = sigma_llr.to(self.dtype)

        # Generate LLRs with Gaussian approximation
        llr = torch.normal(
            mean=-1.0 * mu_llr,
            std=sigma_llr,
            size=output_shape,
            dtype=self.dtype,
            device=noise_var.device
        )
        return llr


