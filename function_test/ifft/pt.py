import torch
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 PyTorch 代码
def ifft(tensor, axis=-1):
    r"""Computes the normalized IDFT along a specified axis.

    This operation computes the normalized one-dimensional discrete inverse
    Fourier transform (IDFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{X}\in\mathbb{C}^N`, the IDFT
    :math:`\mathbf{x}\in\mathbb{C}^N` is computed as

    .. math::
        x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
            j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.

    Input
    -----
    tensor : torch.Tensor
        Tensor of arbitrary shape.

    axis : int
        Indicates the dimension along which the IDFT is taken.

    Output
    ------
    : torch.Tensor
        Tensor of the same dtype and shape as ``tensor``.
    """
    fft_size = torch.tensor(tensor.shape[axis], dtype=tensor.dtype)
    scale = torch.sqrt(fft_size)

    if axis not in [-1, tensor.ndim]:
        output = torch.fft.ifft(torch.swapaxes(tensor, axis, -1))
        output = torch.swapaxes(output, axis, -1)
    else:
        output = torch.fft.ifft(tensor)

    return scale * output


# 创建测试张量,注意输入张量的要求
shape = (2, 3, 4, 5, 6, 7)
real_part = torch.randn(shape, dtype=torch.float32)
imag_part = torch.randn(shape, dtype=torch.float32)
tensor_torch = torch.complex(real_part, imag_part)

np.save(os.path.join(current_dir, 'tensor_torch.npy'), tensor_torch.numpy())
# 测试 PyTorch 的函数
num_dims = 2
axis = -1
output_ifft = ifft(tensor_torch, axis)

target_rank = 5
# output_functionB = functionB(tensor_torch, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'ifft_pt.npy'), output_ifft.numpy())
# np.save(os.path.join(current_dir, 'functionB_pt.npy'), output_functionB.numpy())
