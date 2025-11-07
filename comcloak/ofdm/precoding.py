# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to OFDM transmit precoding"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from comcloak.supplement import gather_pytorch
from comcloak.mimo import zero_forcing_precoder
from comcloak.utils import flatten_dims

class RemoveNulledSubcarriers(nn.Module):
    def __init__(self, resource_grid):
        super().__init__()
        self.resource_grid = resource_grid

    def forward(self, x):
        # Assuming the first subcarrier is nulled as an example
        return x[:, :, :, :, :, :, 1:]

class ZFPrecoder(nn.Module):
    def __init__(self, resource_grid, stream_management, return_effective_channel=False, dtype=torch.complex64):
        super().__init__()
        self.resource_grid = resource_grid
        self.stream_management = stream_management
        self.return_effective_channel = return_effective_channel
        self.dtype = dtype
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.resource_grid)

    def compute_effective_channel(self, h, g):
        h = h.permute(0, 1, 3, 5, 6, 2, 4).to(g.dtype)
        g = g.unsqueeze(1)
        h_eff = torch.matmul(h, g)
        h_eff = h_eff.permute(0, 1, 5, 2, 6, 3, 4)
        h_eff = self.remove_nulled_scs(h_eff)
        return h_eff

    def forward(self, inputs):
        x, h = inputs
        x_precoded = x.permute(0, 1, 3, 4, 2).to(self.dtype)
        h_pc = h.permute(3, 1, 2, 4, 5, 6, 0)

        h_pc_desired = gather_pytorch(h_pc, torch.tensor(self.stream_management.precoding_ind), axis=1, batch_dims=1)
        h_pc_desired = flatten_dims(h_pc_desired, num_dims=2, axis=1)
        h_pc_desired = h_pc_desired.permute(5, 0, 3, 4, 1, 2).to(self.dtype)
        x_precoded, g = zero_forcing_precoder(x_precoded, h_pc_desired, return_precoding_matrix=True)
        x_precoded = x_precoded.permute(0, 1, 4, 2, 3)

        if self.return_effective_channel:
            h_eff = self.compute_effective_channel(h, g)
            return x_precoded, h_eff
        else:
            return x_precoded
