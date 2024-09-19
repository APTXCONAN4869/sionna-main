import tensorflow as tf
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 TensorFlow 代码
def matrix_inv(tensor):
    r""" Computes the inverse of a Hermitian matrix.

    Given a batch of Hermitian positive definite matrices
    :math:`\mathbf{A}`, the function
    returns :math:`\mathbf{A}^{-1}`, such that
    :math:`\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor``, containing
        the inverse of its last two dimensions.

    Note:
        If you want to use this function in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
    """
    if tensor.dtype in [tf.complex64, tf.complex128] \
                    and sn.config.xla_compat \
                    and not tf.executing_eagerly():
        s, u = tf.linalg.eigh(tensor)

        # Compute inverse of eigenvalues
        s = tf.abs(s)
        tf.debugging.assert_positive(s, "Input must be positive definite.")
        s = 1/s
        s = tf.cast(s, u.dtype)

        # Matrix multiplication
        s = tf.expand_dims(s, -2)
        return tf.matmul(u*s, u, adjoint_b=True)
    else:
        return tf.linalg.inv(tensor)



# 创建测试张量（从文件中读取 PyTorch 的输入）
tensor_torch = np.load(os.path.join(current_dir, 'tensor_torch.npy'))
tensor_tf = tf.constant(tensor_torch)

# 测试 TensorFlow 的函数
num_dims = 2
axis = 1
output_matrix_inv = matrix_inv(tensor_tf)

target_rank = 5
# output_functionB = functionB(tensor_tf, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'matrix_inv_tf.npy'), output_matrix_inv.numpy())
# np.save(os.path.join(current_dir, 'functionB_tf.npy'), output_functionB.numpy())
