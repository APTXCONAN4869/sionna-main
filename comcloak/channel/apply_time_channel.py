import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comcloak.channel.awgn import AWGN
from comcloak.utils import insert_dims

# import tensorflow as tf

from comcloak.constants import GLOBAL_SEED_NUMBER

def gather_pytorch(input_data,  axis=0, indices=None):
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


class ApplyTimeChannel(nn.Module):
    def __init__(self, num_time_samples, l_tot, add_awgn=True, dtype=torch.complex64):
        super(ApplyTimeChannel, self).__init__()
        self.add_awgn = add_awgn
        self.dtype = dtype

        # Generate Toeplitz matrix for gathering
        first_column = np.concatenate([np.arange(0, num_time_samples), np.full([l_tot - 1], num_time_samples)])
        first_row = np.concatenate([[0], np.full([l_tot - 1], num_time_samples)])
        self._g = torch.tensor(scipy.linalg.toeplitz(first_column, first_row), dtype=torch.long)

        if self.add_awgn:
            self.awgn = AWGN(dtype=dtype)

    def forward(self, inputs):
        if self.add_awgn:
            x, h_time, no = inputs
        else:
            x, h_time = inputs

        # Prepare the channel input for broadcasting and matrix multiplication
        x = F.pad(x, (0, 1))
        x = insert_dims(x, 2, axis=1)     # Add singleton dimension at the end

        # Gather operation
        x = gather_pytorch(x, -1,self._g)

        # Apply the channel response
        y = torch.sum(h_time * x, dim=-1)
        y = torch.sum(torch.sum(y, dim=4), dim=3)

        # Add AWGN if requested
        if self.add_awgn:
            y = self.awgn((y, no))

        return y