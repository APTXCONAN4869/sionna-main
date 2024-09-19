import torch
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 PyTorch 代码
def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last `num_dims` dimensions of a `tensor`.
    It is a simplified version of the function `flatten_dims`.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of `tensor`.

    Returns:
        A tensor of the same type as `tensor` with `num_dims`-1 lesser
        dimensions, but the same number of elements.
    """
    assert num_dims >= 2, "`num_dims` must be >= 2"
    assert num_dims <= len(tensor.shape), "`num_dims` must <= rank(`tensor`)"

    if num_dims == len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tensor.shape
        last_dim = torch.prod(torch.tensor(shape[-num_dims:]))
        new_shape = list(shape[:-num_dims]) + [last_dim.item()]

    return tensor.reshape(new_shape)


# 创建测试张量
tensor_torch = torch.randn(2, 3)
np.save(os.path.join(current_dir, 'tensor_torch.npy'), tensor_torch.numpy())
# 测试 PyTorch 的函数
num_dims = 2
# axis = 1
output_flatten_last_dims = flatten_last_dims(tensor_torch, num_dims)

# target_rank = 5
# output_functionB = functionB(tensor_torch, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'flatten_last_dims_pt.npy'), output_flatten_last_dims.numpy())
# np.save(os.path.join(current_dir, 'functionB_pt.npy'), output_functionB.numpy())
