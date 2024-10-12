import torch

tensor1 = torch.tensor([[[0, 0, 0],[1, 2+1j, 3]]])
tensor2 = torch.tensor([0, 1, 0])
output = torch.nan_to_num(tensor1 / tensor2, nan=0.0, posinf=0.0, neginf=0.0)
result = torch.where(tensor2 != 0, tensor1 / tensor2, torch.zeros_like(tensor1))
print(result)