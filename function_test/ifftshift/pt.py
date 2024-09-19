import torch
import numpy as np
import os
from numpy.fft import ifftshift
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 PyTorch 代码


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
