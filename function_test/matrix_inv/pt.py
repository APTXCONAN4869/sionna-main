import torch
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 PyTorch 代码
def matrix_inv(tensor):
    """
    Computes the inverse of a Hermitian matrix.

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
        A tensor of the same shape and type as tensor, containing
        the inverse of its last two dimensions.
    """
    if tensor.is_complex():
        s, u = torch.linalg.eigh(tensor)
        
        # Compute inverse of eigenvalues
        s = s.abs()
        assert torch.all(s > 0), "Input must be positive definite."
        s = 1 / s
        s = s.to(u.dtype)
        
        # Matrix multiplication
        s = s.unsqueeze(-2)
        return torch.matmul(u * s, u.conj().transpose(-2, -1))
    else:
        return torch.inverse(tensor)




# 创建测试张量
tensor_torch = torch.randn(3, 3)
np.save(os.path.join(current_dir, 'tensor_torch.npy'), tensor_torch.numpy())
# 测试 PyTorch 的函数
num_dims = 2
axis = 1
output_matrix_inv = matrix_inv(tensor_torch)

target_rank = 5
# output_functionB = functionB(tensor_torch, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'matrix_inv_pt.npy'), output_matrix_inv.numpy())
# np.save(os.path.join(current_dir, 'functionB_pt.npy'), output_functionB.numpy())
