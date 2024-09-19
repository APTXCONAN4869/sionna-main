import torch
import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 PyTorch 代码
def create_hermitian_matrix(n):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (A + A.conj().T) / 2

def matrix_sqrt_inv(tensor):
    r"""
    Computes the inverse square root of a Hermitian matrix.

    Given a batch of Hermitian positive definite matrices
    :math:`\mathbf{A}`, with square root matrices :math:`\mathbf{B}`,
    such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`, the function
    returns :math:`\mathbf{B}^{-1}`, such that
    :math:`\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor`` containing
        the inverse matrix square root of its last two dimensions.
    """
    # Compute the eigenvalues and eigenvectors
    s, u = torch.linalg.eigh(tensor)
    
    # Compute 1/sqrt of eigenvalues
    s = torch.abs(s)
    if torch.any(s <= 0):
        raise ValueError("Input must be positive definite.")
    s = 1 / torch.sqrt(s)
    
    # Matrix multiplication
    s = s.unsqueeze(-2)
    return torch.matmul(u * s, u.transpose(-2, -1).conj())
    

        



# 创建测试张量,注意输入张量的要求
tensor_torch = create_hermitian_matrix(4)
np.save(os.path.join(current_dir, 'tensor_torch.npy'), torch.tensor(tensor_torch, dtype=torch.complex64).numpy())
# 测试 PyTorch 的函数
num_dims = 2
axis = 1
with torch.set_grad_enabled(False):
    output_matrix_sqrt_inv = matrix_sqrt_inv(torch.tensor(tensor_torch, dtype=torch.complex64))

# 保存输出到文件
np.save(os.path.join(current_dir, 'matrix_sqrt_inv_pt.npy'), output_matrix_sqrt_inv.numpy())
