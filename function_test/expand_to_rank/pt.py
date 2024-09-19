import torch
import numpy as np
#
def insert_dims_torch(tensor, num_dims, axis=-1):
    assert num_dims >= 0, "`num_dims` must be nonnegative."
    
    rank = tensor.dim()
    assert -(rank + 1) <= axis <= rank, "`axis` is out of range `[-(D+1), D]`"

    if axis < 0:
        axis += rank + 1

    for _ in range(num_dims):
        tensor = tensor.unsqueeze(axis)

    return tensor

def expand_to_rank_torch(tensor, target_rank, axis=-1):
    current_rank = tensor.dim()
    num_dims = max(target_rank - current_rank, 0)
    output = insert_dims_torch(tensor, num_dims, axis)

    return output

# 创建测试张量
tensor_torch = torch.randn(2, 3)
np.save('tensor_torch.npy',tensor_torch.numpy())
# 测试 PyTorch 的函数
num_dims = 2
axis = 1
output_torch_insert_dims = insert_dims_torch(tensor_torch, num_dims, axis)

target_rank = 5
output_torch_expand_to_rank = expand_to_rank_torch(tensor_torch, target_rank, axis)

# 保存输出到文件
np.save('output_torch_insert_dims.npy', output_torch_insert_dims.numpy())
np.save('output_torch_expand_to_rank.npy', output_torch_expand_to_rank.numpy())
