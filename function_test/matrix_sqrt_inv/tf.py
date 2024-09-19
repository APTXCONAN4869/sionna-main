import sys
sys.path.insert(0, 'D:\sionna-main')
import tensorflow as tf
import numpy as np
import os
import sionna as sn
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 TensorFlow 代码
def matrix_sqrt_inv(tensor):
    r""" Computes the inverse square root of a Hermitian matrix.

    Given a batch of Hermitian positive definite matrices
    :math:`\mathbf{A}`, with square root matrices :math:`\mathbf{B}`,
    such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`, the function
    returns :math:`\mathbf{B}^{-1}`, such that
    :math:`\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor`` containing
        the inverse matrix square root of its last two dimensions.

    Note:
        If you want to use this function in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
    """
    # if sn.config.xla_compat and not tf.executing_eagerly():
    #     s, u = tf.linalg.eigh(tensor)

    #     # Compute 1/sqrt of eigenvalues
    #     s = tf.abs(s)
    #     tf.debugging.assert_positive(s, "Input must be positive definite.")
    #     s = 1/tf.sqrt(s)
    #     s = tf.cast(s, u.dtype)

    #     # Matrix multiplication
    #     s = tf.expand_dims(s, -2)
    #     return tf.matmul(u*s, u, adjoint_b=True)
    # else:
    #     # print(tf.linalg.inv(tf.linalg.sqrtm(tensor)))
    #     return tf.linalg.inv(tf.linalg.sqrtm(tensor))
    s, u = tf.linalg.eigh(tensor)

    # Compute 1/sqrt of eigenvalues
    s = tf.abs(s)
    tf.debugging.assert_positive(s, "Input must be positive definite.")
    s = 1/tf.sqrt(s)
    s = tf.cast(s, u.dtype)

    # Matrix multiplication
    s = tf.expand_dims(s, -2)
    print(tf.matmul(u*s, u, adjoint_b=True))
    print(tf.linalg.inv(tf.linalg.sqrtm(tensor)))
    return tf.linalg.inv(tf.linalg.sqrtm(tensor))




# 创建测试张量（从文件中读取 PyTorch 的输入）
tensor_torch = np.load(os.path.join(current_dir, 'tensor_torch.npy'))
tensor_tf = tf.convert_to_tensor(tensor_torch, dtype = tf.complex64)

# 测试 TensorFlow 的函数
num_dims = 2
axis = 1
output_matrix_sqrt_inv = matrix_sqrt_inv(tensor_tf)


# 保存输出到文件
np.save(os.path.join(current_dir, 'matrix_sqrt_inv_tf.npy'), output_matrix_sqrt_inv.numpy())
