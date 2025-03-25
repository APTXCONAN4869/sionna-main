import torch
import torch.nn as nn
import torch.nn.functional as F

from comcloak.utils import expand_to_rank,complex_normal
from comcloak.constants import GLOBAL_SEED_NUMBER

# import tensorflow as tf
# from tensorflow.keras.layers import Layer
# from comcloak.utils import expand_to_rank,complex_normal

class AWGN(nn.Module):
    r"""AWGN(dtype=torch.complex64, **kwargs)

    Add complex AWGN to the inputs with a certain variance.

    This class inherits from the PyTorch `nn.Module` class and can be used as layer in
    a PyTorch model.

    This layer adds complex AWGN noise with variance ``no`` to the input.
    The noise has variance ``no/2`` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Example
    --------

    Setting-up:

    >>> awgn_channel = AWGN()

    Running:

    >>> # x is the channel input
    >>> # no is the noise variance
    >>> y = awgn_channel((x, no))

    Parameters
    ----------
        dtype : Complex torch.dtype
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `torch.complex64`.

    Input
    -----

        (x, no) :
            Tuple:

        x :  Tensor, torch.complex
            Channel input

        no : Scalar or Tensor, torch.float
            Scalar or tensor whose shape can be broadcast to the shape of ``x``.
            The noise power ``no`` is per complex dimension. If ``no`` is a
            scalar, noise of the same variance will be added to the input.
            If ``no`` is a tensor, it must have a shape that can be broadcast to
            the shape of ``x``. This allows, e.g., adding noise of different
            variance to each example in a batch. If ``no`` has a lower rank than
            ``x``, then ``no`` will be broadcast to the shape of ``x`` by adding
            dummy dimensions after the last axis.

    Output
    -------
        y : Tensor with same shape as ``x``, torch.complex
            Channel output
    """

    def __init__(self, dtype=torch.complex64, **kwargs):
        super().__init__(**kwargs)
        if dtype == torch.complex32:
            self._real_dtype =  torch.float16
        if dtype == torch.complex64:
            self._real_dtype = torch.float32
        if dtype == torch.complex128:
            self._real_dtype = torch.float64

    def forward(self, inputs):
        x, no = inputs


        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(x.shape, dtype=x.dtype)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, x.ndim, axis=-1)
        
        # Apply variance scaling
        no = no.to(noise.device)
        x = x.to(noise.device)
        no = no.to(self._real_dtype)
        noise *= torch.sqrt(no).to(noise.dtype)

        # Add noise to input
        y = x + noise

        return y

"""
def test_awgn():
    torch.manual_seed(42)  # 设置随机种子

    # 创建 AWGN 层实例
    awgn_layer = AWGN()

    # 定义输入数据 x (batch_size=5, num_symbols=13, complex dimension=4)
    x_real = torch.randn(5, 13, 4, dtype=torch.float32)
    x_imag = torch.randn(5, 13, 4, dtype=torch.float32)
    x = torch.complex(x_real, x_imag)

    # 定义噪声功率 no (形状可以广播到 x)
    no = torch.tensor([0.1], dtype=torch.float32)

    # 调用 forward 方法
    y = awgn_layer((x, no))

    # 打印输出
    print("Input x shape:", x.shape)
    print("Noise power no shape:", no.shape)
    print("Output y shape:", y.shape)
    print("Output y:", y)

# 运行测试函数
test_awgn()

"""    