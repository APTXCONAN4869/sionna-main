import sys
sys.path.insert(0, 'D:\sionna-main')
import tensorflow as tf
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 TensorFlow 代码
def functionA(tensor, num_dims, axis=-1):
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

def functionB(tensor, target_rank, axis=-1):
    num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    output = functionA(tensor, num_dims, axis)

    return output

# 创建测试张量（从文件中读取 PyTorch 的输入）
tensor_torch = np.load(os.path.join(current_dir, 'tensor_torch.npy'))
tensor_tf = tf.constant(tensor_torch)

# 测试 TensorFlow 的函数
num_dims = 2
axis = 1
output_functionA = functionA(tensor_tf, num_dims, axis)

target_rank = 5
output_functionB = functionB(tensor_tf, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'functionA_tf.npy'), output_functionA.numpy())
np.save(os.path.join(current_dir, 'functionB_tf.npy'), output_functionB.numpy())
