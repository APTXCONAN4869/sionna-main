import torch
import numpy as np

import os
# tensor1 = torch.tensor([[[0, 0, 0],[1, 2+1j, 3]]])
# arr = np.zeros((1, 120, 1))
# arr_transposed = arr.transpose(-1, -2)  # Shape becomes (1, 1, 120)
# print(arr_transposed.shape)  # Output: (1, 1, 120)

# tensor2 = torch.tensor([0, 1, 0])
# output = torch.nan_to_num(tensor1 / tensor2, nan=0.0, posinf=0.0, neginf=0.0)
# result = torch.where(tensor2 != 0, tensor1 / tensor2, torch.zeros_like(tensor1))
# print(result)

# print("Current directory:", os.path.dirname(__file__))
# # 使用相对路径，假设脚本与数据文件位于不同的文件夹中
# data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'myfile.npy')

# # 加载.npy文件
# # data = np.load(data_path)
# print(data_path)
# import torch

# # 创建嵌套张量
# nested_tensor = torch.nested.nested_tensor([torch.tensor([1, 2]), torch.tensor([3, 4, 5])])

# print(nested_tensor)


# nested_tensor1 = torch.nested.nested_tensor([torch.tensor([1, 2]), torch.tensor([3, 4, 5])])
# nested_tensor2 = torch.nested.nested_tensor([torch.tensor([6, 7]), torch.tensor([8, 9, 10])])

# # 对嵌套张量进行加法操作
# result = nested_tensor1 + nested_tensor2
# print(result)

# nested_tensor1 = torch.nested.nested_tensor([torch.tensor([1, 2]), torch.tensor([3, 4])])
# nested_tensor2 = torch.nested.nested_tensor([torch.tensor([1]), torch.tensor([5, 6])])

# # 进行广播加法操作
# result = nested_tensor1 + nested_tensor2
# print(result)

import torch
from typing import Callable, List
def ragged_reduce_prod(flat_values: torch.Tensor, row_splits: torch.Tensor) -> torch.Tensor:
    """
    Computes the product along each row of a ragged tensor.

    Args:
        flat_values (torch.Tensor): The flat values of the ragged tensor.
        row_splits (torch.Tensor): The row splits defining the ragged structure.

    Returns:
        torch.Tensor: A 1D tensor containing the product of each row.
    """
    # Initialize an empty list to store the row-wise products
    row_products = []

    # Iterate through the row_splits to compute row products
    for start, end in zip(row_splits[:-1], row_splits[1:]):
        row = flat_values[start:end]  # Extract the row
        if row.numel() > 0:
            row_products.append(row.prod(dim=1, keepdim=True))  # Compute the product of the row
        else:
            row_products.append(torch.tensor(1.0))  # Handle empty rows (product is 1)

    # Convert the list of row products to a tensor
    return torch.stack(row_products)

def gather_pytorch(input_data, indices=None, batch_dims=0, axis=0):
    input_data = torch.tensor(input_data)
    indices = torch.tensor(indices)
    if batch_dims == 0:
        if axis < 0:
            axis = len(input_data.shape) + axis
        data = torch.index_select(input_data, axis, indices.flatten())
        shape_input = list(input_data.shape)
        # shape_ = delete(shape_input, axis)
        # 连接列表
        shape_output = shape_input[:axis] + \
            list(indices.shape) + shape_input[axis + 1:]
        data_output = data.reshape(shape_output)
        return data_output
    else:
        data_output = []
        for data,ind in zip(input_data, indices):
            r = gather_pytorch(data, ind, batch_dims=batch_dims-1)
            data_output.append(r)
        return torch.stack(data_output)
# sign_node=np.array([[-1.]])
# sign_val=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# test = gather_pytorch(sign_node, sign_val)

# flat_values = torch.tensor([[-1.0], [-1.0], [1.0], [1.0], [1.0], [1.0], [-1.0], [1.0], [-1.0], [-1.0]])
# # flat_values = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0])

# # tensor = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# # input = tensor
# # indices = [tensor, tensor, tensor]
# # test = gather_pytorch(input, indices, axis=None)
# row_splits = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
# # if flat_values.ndim == 2 and flat_values.size(1) == 1:
# #     # If input is 2D with shape [N, 1], return a single 2D value
# #     result = torch.tensor([[ragged_reduce_prod(flat_values, row_splits)]])
# # else:
#     # Otherwise, return the flattened minimum value
#     # result = ragged_reduce_prod(flat_values, row_splits)
# result = ragged_reduce_prod(flat_values, row_splits)
# print("Result:", result)  # Output: tensor([24., 1.])

# import torch

# flat_values = torch.tensor([-16.635532, -16.635532, -6.2010665, -6.2010665])
# row_splits = torch.tensor([0, 2, 4])

# # 模拟 RaggedTensor
# ragged = []
# for i in range(len(row_splits) - 1):
#     segment = flat_values[row_splits[i]:row_splits[i + 1]]
#     if i == 0:
#         segment = segment.view(-1, 2)
#     ragged.append(segment)

# # Reduce sum 操作
# reduced = [r.sum(dim=0) if r.ndim > 1 else r.sum().unsqueeze(0) for r in ragged]

# # 输出
# print(reduced)

import tensorflow as tf

sign_val = tf.constant([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],  # 第一批次
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]  # 第二批次
], dtype=tf.float32)

print(sign_val.shape)  # (2, 3, 4)
result = tf.reduce_prod(sign_val)
print(result.shape)
result = tf.reduce_prod(sign_val, axis=0)
print(result.shape)
result = tf.reduce_prod(sign_val, axis=1)
print(result.shape)
result = tf.reduce_prod(sign_val, axis=2)
print(result.shape)
