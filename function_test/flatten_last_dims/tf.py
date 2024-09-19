import tensorflow as tf
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 TensorFlow 代码
def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last ``num_dims`` dimensions of a ``tensor``.
    It is a simplified version of the function ``flatten_dims``.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of ``tensor``.

    Returns:
        A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
        dimensions, but the same number of elements.
    """
    msg = "`num_dims` must be >= 2"
    tf.debugging.assert_greater_equal(num_dims, 2, msg)

    msg = "`num_dims` must <= rank(`tensor`)"
    tf.debugging.assert_less_equal(num_dims, tf.rank(tensor), msg)

    if num_dims==len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tf.shape(tensor)
        last_dim = tf.reduce_prod(tensor.shape[-num_dims:])
        new_shape = tf.concat([shape[:-num_dims], [last_dim]], 0)

    return tf.reshape(tensor, new_shape)


# 创建测试张量（从文件中读取 PyTorch 的输入）
tensor_torch = np.load(os.path.join(current_dir, 'tensor_torch.npy'))
tensor_tf = tf.constant(tensor_torch)

# 测试 TensorFlow 的函数
num_dims = 2
# axis = 1
output_flatten_last_dims = flatten_last_dims(tensor_tf, num_dims)

# target_rank = 5
# output_functionB = functionB(tensor_tf, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'flatten_last_dims_tf.npy'), output_flatten_last_dims.numpy())
# np.save(os.path.join(current_dir, 'functionB_tf.npy'), output_functionB.numpy())
