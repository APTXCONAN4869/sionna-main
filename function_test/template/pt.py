import torch
import numpy as np
import os
import sys
sys.path.insert(0, 'D:\sionna-main')
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 PyTorch 代码
def functionA(tensor, num_dims, axis=-1):
    assert num_dims >= 0, "`num_dims` must be nonnegative."
    
    rank = tensor.dim()
    assert -(rank + 1) <= axis <= rank, "`axis` is out of range `[-(D+1), D]`"

    if axis < 0:
        axis += rank + 1

    for _ in range(num_dims):
        tensor = tensor.unsqueeze(axis)

    return tensor

def functionB(tensor, target_rank, axis=-1):
    current_rank = tensor.dim()
    num_dims = max(target_rank - current_rank, 0)
    output = functionA(tensor, num_dims, axis)

    return output

# 创建测试张量,注意输入张量的要求
tensor_torch = torch.randn(2, 3)
np.save(os.path.join(current_dir, 'tensor_torch.npy'), tensor_torch.numpy())
# 测试 PyTorch 的函数
num_dims = 2
axis = 1
output_functionA = functionA(tensor_torch, num_dims, axis)

target_rank = 5
output_functionB = functionB(tensor_torch, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'functionA_pt.npy'), output_functionA.numpy())
np.save(os.path.join(current_dir, 'functionB_pt.npy'), output_functionB.numpy())
