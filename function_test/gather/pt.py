import torch
import numpy as np
import os


input =[ [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]],
 
         [[[7, 7, 7], [8, 8, 8]],
         [[9, 9, 9], [10, 10, 10]],
         [[11, 11, 11], [12, 12, 12]]],
 
        [[[13, 13, 13], [14, 14, 14]],
         [[15, 15, 15], [16, 16, 16]],
         [[17, 17, 17], [18, 18, 18]]]
         ]
input = torch.tensor(input)
print(input.shape)
# Create an index tensor with the same shape as the input
# index = torch.tensor([0, 2]).view(2, 1, 1, 1).expand(-1, 3, 2, 3)
indices = torch.tensor([[0,2],[0,1]])
output = torch.index_select(input, dim=0, index=indices)
print(output)
