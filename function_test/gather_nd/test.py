import torch
import tensorflow as tf
import numpy as np

# 创建一个4维张量 (2, 3, 4, 5) 的例子
data_tf = tf.reshape(tf.range(2 * 3 * 4 * 5), (2, 3, 4, 5))
data_torch = torch.tensor(data_tf.numpy())

# 索引示例，类似于 tf.gather_nd 的索引
indices = np.array([
    [[0, 1, 2, 3], [1, 2, 3, 4]],    # 索引出 [data[0, 1, :, :, :], data[1, 2, :, :, :]]
    [[0, 0, 1, 2], [1, 1, 2, 3]],    # 索引出 [data[0, 0, :, :, :], data[1, 1, :, :, :]]
])
indices_tf = tf.constant(indices)
indices_torch = torch.tensor(indices)

# TensorFlow 结果
output_tf = tf.gather_nd(data_tf, indices_tf)

# 测试你的基于 PyTorch 的 gather_nd 函数

def gather_nd_pytorch(params, indices):
    '''
    ND_example
    params: tensor shaped [n(1), ..., n(d)] --> d-dimensional tensor
    indices: tensor shaped [m(1), ..., m(i-1), m(i)] --> multidimensional list of i-dimensional indices, m(i) <= d

    returns: tensor shaped [m(1), ..., m(i-1), n(m(i)+1), ..., n(d)] m(i) < d
             tensor shaped [m(1), ..., m(i-1)] m(i) = d
    '''
    indices_shape = indices.shape
    flattened_indices = indices.view(-1, indices.shape[-1])
    processed_tensors = []
    for coordinates in flattened_indices:
        sub_tensor = params[(*coordinates,)] 
        processed_tensors.append(sub_tensor)
    output_shape = indices_shape[:-1] + sub_tensor.shape
    output = torch.stack(processed_tensors).reshape(output_shape)


    return output

# 使用你的 gather_nd_pytorch 函数
output_torch = gather_nd_pytorch(data_torch, indices_torch)

# 打印和比较结果
print("TensorFlow Output:\n", output_tf.numpy())
print("PyTorch Output:\n", output_torch.numpy())

# 检查输出是否相同
if np.array_equal(output_tf.numpy(), output_torch.numpy()):
    print("The outputs are identical!")
else:
    print("The outputs are different.")
