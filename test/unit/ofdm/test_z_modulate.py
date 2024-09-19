# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import torch
# import pytest
import unittest
import numpy as np
from torch import nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'D:\sionna-main')
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("../")
# from sionna.mimo import StreamManagement
# from .ofdm_test_module import *

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')

from comcloak.ofdm import OFDMModulator, OFDMDemodulator,\
    count_errors, count_block_errors, pam, pam_gray, qam, QAMSource, SymbolSource, BinarySource, Mapper, Constellation

class TestOFDMModulator(unittest.TestCase):
    def test_cyclic_prefixes(self):
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(1, fft_size + 1):
            modulator = OFDMModulator(cp_length)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = torch.reshape(x_time, [batch_size, num_ofdm_symbols, -1])
            self.assertTrue(torch.equal(x_time[..., :cp_length], x_time[..., -cp_length:]))

        cp_length = fft_size + 1
        modulator = OFDMModulator(cp_length)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        with self.assertRaises(AssertionError):
            x_time = modulator(x)

    def test_higher_dimensions(self):
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(1, fft_size + 1):
            modulator = OFDMModulator(cp_length)
            x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = torch.reshape(x_time, batch_size + [num_ofdm_symbols, -1])
            self.assertTrue(torch.equal(x_time[..., :cp_length], x_time[..., -cp_length:]))

class TestOFDMDemodulator(unittest.TestCase):
    def test_cyclic_prefixes(self):
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(0, fft_size + 1):
        # for cp_length in range(0, 1):
            modulator = OFDMModulator(cp_length)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)
            # print("x:",x)
            # print("x_time:",x_time)
            # print("x_hat:",x_hat)
            # print(torch.abs(x - x_hat))
            self.assertTrue(torch.max(torch.abs(x - x_hat)).item() < 1e-5)

    def test_higher_dimensions(self):
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(1, fft_size + 1):
            modulator = OFDMModulator(cp_length)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length)
            x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)
            self.assertTrue(torch.max(torch.abs(x - x_hat)).item() < 1e-5)

    def test_overlapping_input(self):
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in [0, 12]:
            modulator = OFDMModulator(cp_length)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = torch.cat([x_time, x_time[..., :10]], dim=-1)
            x_hat = demodulator(x_time)
            self.assertTrue(torch.max(torch.abs(x - x_hat)).item() < 1e-5)

class TestOFDMModDemod(unittest.TestCase):
    def test_end_to_end(self):
        """E2E test verying that all shapes can be properly inferred (see Issue #7)"""
        class E2ESystem(nn.Module):
            def __init__(self, cp_length, padding):
                super(E2ESystem, self).__init__()
                self.cp_length = cp_length
                self.padding = padding
                self.fft_size = 72
                self.num_ofdm_symbols = 14
                self.qam = QAMSource(4)
                self.mod = OFDMModulator(self.cp_length)
                self.demod = OFDMDemodulator(self.fft_size, 0, self.cp_length)

            def forward(self, batch_size):
                x_rg = self.qam([batch_size, 1, 1, self.num_ofdm_symbols, self.fft_size])
                x_time = self.mod(x_rg)
                pad = torch.zeros_like(x_time)[..., :self.padding]
                x_time = torch.cat([x_time, pad], dim=-1)
                x_f = self.demod(x_time)
                return x_f

        for cp_length in [0, 1, 5, 12]:
            for padding in [0, 1, 5, 71]:
                e2e = E2ESystem(cp_length, padding)
                self.assertEqual(e2e(128).shape, (128, 1, 1, e2e.num_ofdm_symbols, e2e.fft_size))

if __name__ == '__main__':
    unittest.main()
