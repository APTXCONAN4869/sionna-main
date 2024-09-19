import tensorflow as tf
import numpy as np
def insert_dims_tf(tensor, num_dims, axis=-1):
    msg = "`num_dims` must be nonnegative."
    tf.debugging.assert_greater_equal(num_dims, 0, msg)

    rank = tf.rank(tensor)
    msg = "`axis` is out of range `[-(D+1), D]`)"
    tf.debugging.assert_less_equal(axis, rank, msg)
    tf.debugging.assert_greater_equal(axis, -(rank+1), msg)

    axis = axis if axis >= 0 else rank + axis + 1
    shape = tf.shape(tensor)
    new_shape = tf.concat([shape[:axis],
                           tf.ones([num_dims], tf.int32),
                           shape[axis:]], 0)
    output = tf.reshape(tensor, new_shape)

    return output

def expand_to_rank_tf(tensor, target_rank, axis=-1):
    num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    output = insert_dims_tf(tensor, num_dims, axis)

    return output

# 创建测试张量（从文件中读取 PyTorch 的输入）
tensor_torch = np.load('tensor_torch.npy')
tensor_tf = tf.constant(tensor_torch)

# 测试 TensorFlow 的函数
num_dims = 2
axis = 1
output_tf_insert_dims = insert_dims_tf(tensor_tf, num_dims, axis)

target_rank = 5
output_tf_expand_to_rank = expand_to_rank_tf(tensor_tf, target_rank, axis)

# 保存输出到文件
np.save('output_tf_insert_dims.npy', output_tf_insert_dims.numpy())
np.save('output_tf_expand_to_rank.npy', output_tf_expand_to_rank.numpy())
