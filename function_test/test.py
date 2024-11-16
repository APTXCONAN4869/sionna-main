import torch
import numpy as np
# tensor1 = torch.tensor([[[0, 0, 0],[1, 2+1j, 3]]])
arr = np.zeros((1, 120, 1))
arr_transposed = arr.transpose(-1, -2)  # Shape becomes (1, 1, 120)
print(arr_transposed.shape)  # Output: (1, 1, 120)

# tensor2 = torch.tensor([0, 1, 0])
# output = torch.nan_to_num(tensor1 / tensor2, nan=0.0, posinf=0.0, neginf=0.0)
# result = torch.where(tensor2 != 0, tensor1 / tensor2, torch.zeros_like(tensor1))
# print(result)