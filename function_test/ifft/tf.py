import tensorflow as tf
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 TensorFlow 代码
from tensorflow.experimental.numpy import swapaxes
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
    tensor : tf.complex
        Tensor of arbitrary shape.

    axis : int
        Indicates the dimension along which the IDFT is taken.

    Output
    ------
    : tf.complex
        Tensor of the same dtype and shape as ``tensor``.
    """
    fft_size = tf.cast(tf.shape(tensor)[axis], tensor.dtype)
    scale = tf.sqrt(fft_size)

    if axis not in [-1, tensor.shape.rank]:
        output =  tf.signal.ifft(swapaxes(tensor, axis, -1))
        output = swapaxes(output, axis, -1)
    else:
        output = tf.signal.ifft(tensor)

    return scale * output




# 创建测试张量（从文件中读取 PyTorch 的输入）
complex_tensor_np = np.load(os.path.join(current_dir, 'tensor_torch.npy'))
real_part = np.real(complex_tensor_np)
imag_part = np.imag(complex_tensor_np)
tensor_tf = tf.complex(real_part, imag_part)


# 测试 TensorFlow 的函数
num_dims = 2
axis = -1
output_ifft = ifft(tensor_tf, axis)

target_rank = 5
# output_functionB = functionB(tensor_tf, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'ifft_tf.npy'), output_ifft.numpy())
# np.save(os.path.join(current_dir, 'functionB_tf.npy'), output_functionB.numpy())
