# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to the resource grid"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
sys.path.insert(0, 'D:\sionna-main')
# Assuming these functions are defined elsewhere
from pilot_pattern_z import PilotPattern, EmptyPilotPattern, KroneckerPilotPattern
from sionna.utils import flatten_last_dims, flatten_dims, split_dim

class ResourceGrid():
    """Defines a `ResourceGrid` spanning multiple OFDM symbols and subcarriers."""
    def __init__(self, num_ofdm_symbols, fft_size, subcarrier_spacing, num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=0, num_guard_carriers=(0, 0), dc_null=False, pilot_pattern=None, pilot_ofdm_symbol_indices=None, dtype=torch.complex64):
        super().__init__()
        self._dtype = dtype
        self._num_ofdm_symbols = num_ofdm_symbols
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._cyclic_prefix_length = int(cyclic_prefix_length)
        self._num_tx = num_tx
        self._num_streams_per_tx = num_streams_per_tx
        self._num_guard_carriers = np.array(num_guard_carriers)
        self._dc_null = dc_null
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self.pilot_pattern = pilot_pattern
        self._check_settings()

    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @property
    def num_tx(self):
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self):
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self):
        return self._fft_size * self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self):
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
        return n

    @property
    def effective_subcarrier_ind(self):
        num_gc = self._num_guard_carriers
        sc_ind = np.arange(num_gc[0], self.fft_size - num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind - num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self):
        n = self.num_effective_subcarriers * self._num_ofdm_symbols - self.num_pilot_symbols
        return torch.tensor(n, dtype=torch.int32)

    @property
    def num_pilot_symbols(self):
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        n = (self._fft_size - self.num_effective_subcarriers) * self._num_ofdm_symbols
        return torch.tensor(n, dtype=torch.int32)

    @property
    def num_guard_carriers(self):
        return self._num_guard_carriers

    @property
    def dc_ind(self):
        return int(self._fft_size / 2 - (self._fft_size % 2 == 1) / 2)

    @property
    def fft_size(self):
        return self._fft_size

    @property
    def subcarrier_spacing(self):
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self):
        return (1. + self.cyclic_prefix_length / self.fft_size) / self.subcarrier_spacing

    @property
    def bandwidth(self):
        return self.fft_size * self.subcarrier_spacing

    @property
    def num_time_samples(self):
        return (self.fft_size + self.cyclic_prefix_length) * self._num_ofdm_symbols

    @property
    def dc_null(self):
        return self._dc_null

    @property
    def pilot_pattern(self):
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(self, value):
        if value is None:
            value = EmptyPilotPattern(self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols, self.num_effective_subcarriers, dtype=self._dtype)
        elif isinstance(value, PilotPattern):
            pass
        elif isinstance(value, str):
            assert value in ["kronecker", "empty"], "Unknown pilot pattern"
            if value == "empty":
                value = EmptyPilotPattern(self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols, self.num_effective_subcarriers, dtype=self._dtype)
            elif value == "kronecker":
                assert self._pilot_ofdm_symbol_indices is not None, "You must provide pilot_ofdm_symbol_indices."
                value = KroneckerPilotPattern(self, self._pilot_ofdm_symbol_indices, dtype=self._dtype)
        else:
            raise ValueError("Unsupported pilot_pattern")
        self._pilot_pattern = value

    def _check_settings(self):
        assert self._num_ofdm_symbols > 0, "`num_ofdm_symbols` must be positive`."
        assert self._fft_size > 0, "`fft_size` must be positive`."
        assert self._cyclic_prefix_length >= 0, "`cyclic_prefix_length must be nonnegative."
        assert self._cyclic_prefix_length <= self._fft_size, "`cyclic_prefix_length cannot be longer than `fft_size`."
        assert self._num_tx > 0, "`num_tx` must be positive`."
        assert self._num_streams_per_tx > 0, "`num_streams_per_tx` must be positive`."
        assert len(self._num_guard_carriers) == 2, "`num_guard_carriers` must have two elements."
        assert np.all(np.greater_equal(self._num_guard_carriers, 0)), "`num_guard_carriers` must have nonnegative entries."
        assert np.sum(self._num_guard_carriers) <= self._fft_size - self._dc_null, "Total number of guardcarriers cannot be larger than `fft_size`."
        assert self._dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"
        return True

    def build_type_grid(self):
        shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols]
        gc_l = 2 * torch.ones(shape + [self._num_guard_carriers[0]], dtype=torch.int32)
        gc_r = 2 * torch.ones(shape + [self._num_guard_carriers[1]], dtype=torch.int32)
        dc = 3 * torch.ones(shape + [int(self._dc_null)], dtype=torch.int32)
        mask = self.pilot_pattern.mask
        split_ind = self.dc_ind - self._num_guard_carriers[0]
        rg_type = torch.cat([gc_l, mask[..., :split_ind], dc, mask[..., split_ind:], gc_r], -1)
        return rg_type

    def show(self, tx_ind=0, tx_stream_ind=0):
        fig = plt.figure()
        data = self.build_type_grid()[tx_ind, tx_stream_ind]
        cmap = colors.ListedColormap([[60 / 256, 8 / 256, 72 / 256], [45 / 256, 91 / 256, 128 / 256], [45 / 256, 172 / 256, 111 / 256], [250 / 256, 228 / 256, 62 / 256]])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(np.transpose(data), interpolation="nearest", origin="lower", cmap=cmap, norm=norm, aspect="auto")
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5], orientation="vertical", shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("Subcarrier Index")
        plt.xlabel("OFDM Symbol")
        plt.xticks(range(0, data.shape[0]))
        return fig

class ResourceGridMapper(nn.Module):
    """Maps a tensor of modulated data symbols to a ResourceGrid."""
    def __init__(self, resource_grid, dtype=torch.complex64):
        super().__init__()
        self._resource_grid = resource_grid
        self._dtype = dtype
        self._rg_type = self._resource_grid.build_type_grid()
        self._pilot_ind = torch.where(self._rg_type == 1)
        self._data_ind = torch.where(self._rg_type == 0)

    def forward(self, inputs):
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3)
        template = torch.zeros(self._rg_type.shape, dtype=self._dtype)
        template[self._pilot_ind] = pilots
        template = template.unsqueeze(-1)

        batch_size = inputs.shape[0]
        re = inputs.shape[1]
        n = inputs.shape[2]
        assert re == self._resource_grid.num_data_symbols, "Mismatch between number of data symbols in the ResourceGrid and the provided data."
        inputs = inputs.view(batch_size, self._resource_grid.num_tx, self._resource_grid.num_streams_per_tx, re, n)
        rg = torch.tile(template, (batch_size, 1, 1, 1, 1))
        rg[..., self._data_ind[2], :] = inputs
        return rg

class ResourceGridDemapper(nn.Module):
    """Extracts the data symbols from a resource grid."""
    def __init__(self, resource_grid, dtype=torch.complex64):
        super().__init__()
        self._resource_grid = resource_grid
        self._dtype = dtype
        self._rg_type = self._resource_grid.build_type_grid()
        self._pilot_ind = torch.where(self._rg_type == 1)
        self._data_ind = torch.where(self._rg_type == 0)

    def forward(self, inputs):
        assert inputs.shape[1:-1] == self._rg_type.shape, "The input shape must be the same as the resource grid shape."
        batch_size = inputs.shape[0]
        n = inputs.shape[-1]
        pilots = torch.view(inputs[..., self._pilot_ind[2], :], (batch_size, self._resource_grid.num_tx, self._resource_grid.num_streams_per_tx, -1, n))
        inputs = torch.view(inputs[..., self._data_ind[2], :], (batch_size, self._resource_grid.num_tx, self._resource_grid.num_streams_per_tx, -1, n))
        pilots = split_dim(pilots, 3, pilots.shape[3] // n)
        return inputs, pilots


class RemoveNulledSubcarriers(nn.Module):
    """
    RemoveNulledSubcarriers(resource_grid, **kwargs)

    Removes nulled guard and/or DC subcarriers from a resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of ResourceGrid.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], torch.complex64
        Full resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex64
        Resource grid without nulled subcarriers.
    """
    def __init__(self, resource_grid, **kwargs):
        super(RemoveNulledSubcarriers, self).__init__()
        self._sc_ind = resource_grid.effective_subcarrier_ind

    def forward(self, inputs):
        return torch.index_select(inputs, dim=-1, index=torch.tensor(self._sc_ind).to(inputs.device))
