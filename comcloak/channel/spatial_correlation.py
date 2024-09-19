"""Various classes for spatially correlated flat-fading channels."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from comcloak.utils import expand_to_rank,matrix_sqrt

class SpatialCorrelation(ABC):
   # pylint: disable=line-too-long
    r"""Abstract class that defines an interface for spatial correlation functions.

    The :class:`~sionna.channel.FlatFadingChannel` model can be configured with a
    spatial correlation model.

    Input
    -----
    h : tf.complex
        Tensor of arbitrary shape containing spatially uncorrelated
        channel coefficients

    Output
    ------
    h_corr : tf.complex
        Tensor of the same shape and dtype as ``h`` containing the spatially
        correlated channel coefficients.
    """
    @abstractmethod
    def __call__(self, h, *args, **kwargs):
        return NotImplemented

class KroneckerModel(SpatialCorrelation):
    """Kronecker model for spatial correlation in PyTorch.

    Parameters
    ----------
    r_tx : [..., K, K], torch.complex
        Tensor containing the transmit correlation matrices.

    r_rx : [..., M, M], torch.complex
        Tensor containing the receive correlation matrices.

    Input
    -----
    h : [..., M, K], torch.complex
        Tensor containing spatially uncorrelated
        channel coefficients.

    Output
    ------
    h_corr : [..., M, K], torch.complex
        Tensor containing the spatially
        correlated channel coefficients.
    """
    def __init__(self, r_tx=None, r_rx=None):
        super().__init__()
        self.r_tx = r_tx
        self.r_rx = r_rx

    @property
    def r_tx(self):
        r"""Tensor containing the transmit correlation matrices.

        Note
        ----
        If you want to set this property in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
        """
        return self._r_tx
    
    @r_tx.setter
    def r_tx(self, value):
        self._r_tx = value
        if self._r_tx is not None:
            self._r_tx_sqrt = matrix_sqrt(value)
        else:
            self._r_tx_sqrt = None

    @property
    def r_rx(self):
        r"""Tensor containing the receive correlation matrices.

        Note
        ----
        If you want to set this property in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
        """
        return self._r_rx
    
    @r_rx.setter
    def r_rx(self, value):
        self._r_rx = value
        if self._r_rx is not None:
            self._r_rx_sqrt = matrix_sqrt(value)
        else:
            self._r_rx_sqrt = None
    
    def __call__(self, h, *args, **kwargs):
        if self._r_tx_sqrt is not None:
            r_tx_sqrt = expand_to_rank(self._r_tx_sqrt.to(h.device),h.dim(),0)
            h = torch.matmul(h, r_tx_sqrt.conj().transpose(-2, -1))

        
        if self._r_rx_sqrt is not None:
            r_rx_sqrt = expand_to_rank(self._r_rx_sqrt.to(h.device), h.dim(),0)
            h = torch.matmul(r_rx_sqrt, h)
        
        return h
""" # 测试例子
def test_kronecker_model():
    tf.random.set_seed(GLOBAL_SEED_NUMBER)

    # A = torch.tensor([[4, 1+1j], [1-1j, 3]], dtype=torch.complex64)
    # B = torch.tensor([[2, 0], [0, 2]], dtype=torch.complex64)
    
    # 创建 Hermitian 正半定矩阵
    A = torch.tensor([[4, 1+1j], [1-1j, 3]], dtype=torch.complex64)
    B = torch.tensor([[2, 0], [0, 2]], dtype=torch.complex64)    

    # 创建 KroneckerModel 实例
    model = KroneckerModel(r_tx=A, r_rx=B)

    # 创建未相关的通道系数矩阵
    h_tf_real = tf.random.normal(shape=[2,2],dtype=tf.float32)
    h_tf_img = tf.random.normal(shape=[2,2],dtype=tf.float32)
    h_np_real = h_tf_real.numpy()
    h_np_img = h_tf_img.numpy()

    h_torch_real = torch.tensor(h_np_real, dtype=torch.float32)
    h_torch_img = torch.tensor(h_np_img, dtype=torch.float32)
    h = torch.complex(h_torch_real,h_torch_img)
    
    # h = torch.randn(2, 2, dtype=torch.complex64)

    # 生成相关的通道系数矩阵
    h_corr = model(h)

    print("Uncorrelated channel coefficients:\n", h)
    print("Correlated channel coefficients:\n", h_corr)

test_kronecker_model() """

class PerColumnModel(nn.Module):
    r"""Per-column model for spatial correlation.

    Given a batch of matrices :math:`\mathbf{H}\in\mathbb{C}^{M\times K}`
    and correlation matrices :math:`\mathbf{R}_k\in\mathbb{C}^{M\times M}`, k=1,\dots,K,
    this function will generate the output :math:`\mathbf{H}_\text{corr}\in\mathbb{C}^{M\times K}`,
    with columns

    .. math::

        \mathbf{h}^\text{corr}_k = \mathbf{R}^{\frac12}_k \mathbf{h}_k,\quad k=1, \dots, K

    where :math:`\mathbf{h}_k` is the kth column of :math:`\mathbf{H}`.
    Note that all :math:`\mathbf{R}_k\in\mathbb{C}^{M\times M}` must
    be positive semi-definite.

    This model is typically used to simulate a MIMO channel between multiple
    single-antenna users and a base station with multiple antennas.
    The resulting SIMO channel for each user has a different spatial correlation.

    Parameters
    ----------
    r_rx : [..., M, M], torch.complex
        Tensor containing the receive correlation matrices. If
        the rank of `r_rx` is smaller than that of the input `h`,
        it will be broadcast. For a typically use of this model, `r_rx`
        has shape [..., K, M, M], i.e., a different correlation matrix for each
        column of `h`.

    Input
    -----
    h : [..., M, K], torch.complex
        Tensor containing spatially uncorrelated
        channel coefficients.

    Output
    ------
    h_corr : [..., M, K], torch.complex
        Tensor containing the spatially
        correlated channel coefficients.
    """
    def __init__(self, r_rx):
        super().__init__()
        self.r_rx = r_rx

    @property
    def r_rx(self):
        """Tensor containing the receive correlation matrices.

        Note
        ----
        If you want to set this property in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
        """
        return self._r_rx
    
    @r_rx.setter
    def r_rx(self, value):
        self._r_rx = value
        if self._r_rx is not None:
            self._r_rx_sqrt = matrix_sqrt(value)
    
    def __call__(self, h):
        if self._r_rx is not None:
            h = h.transpose(-2, -1)
            h = h.unsqueeze(-1)
            r_rx_sqrt = expand_to_rank(self._r_rx_sqrt, h.dim(), 0)
            h = torch.matmul(r_rx_sqrt, h)
            h = h.squeeze(-1)
            h = h.transpose(-2, -1)

        return h      
"""
from abc import ABC, abstractmethod
import torch
import tensorflow as tf
import numpy as np
import torch.nn as nn

 from my_code.mysionna.utils import expand_to_rank,matrix_sqrt
from my_code.mysionna.constants import GLOBAL_SEED_NUMBER    
def test_PerColumnModel():
    tf.random.set_seed(GLOBAL_SEED_NUMBER)
    # Define input parameters
    M, K = 4, 3  # Dimensions of the matrices
    h_tf = tf.complex(tf.random.normal([M, K]), tf.random.normal([M, K]))  # Random complex matrix
    r_rx_tf = tf.complex(tf.eye(M, batch_shape=[K]), tf.zeros([K, M, M]))  # Identity matrices as correlation matrices

    r_rx_np = r_rx_tf.numpy()
    r_rx = torch.from_numpy(r_rx_np)

    h_np = h_tf.numpy()
    h = torch.from_numpy(h_np)
    # Initialize the PerColumnModel
    model = PerColumnModel(r_rx)

    # Get the correlated channel coefficients
    h_corr = model(h)

    # Print the input and output matrices
    print("Input h:\n", h)
    print("Output h_corr:\n", h_corr)

test_PerColumnModel()
     """