import os
print("Current directory:", os.getcwd())
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("./")
# import pytest
import unittest
import numpy as np
import torch
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')

import comcloak
from comcloak.utils import QAMSource, matrix_sqrt, complex_normal
from comcloak.channel import FlatFadingChannel
from comcloak.mimo.equalization_z import lmmse_equalizer, zf_equalizer, mf_equalizer
from comcloak.channel.utils import exp_corr_mat

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, 
                 equalizer,
                 num_tx_ant,
                 num_rx_ant,
                 num_bits_per_symbol,
                 colored_noise=False,
                 rho=None):
        super(Model, self).__init__()
        self.qam_source = QAMSource(num_bits_per_symbol)
        self.channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=not colored_noise, return_channel=True)
        self.equalizer = equalizer
        self.colored_noise = colored_noise
        if self.colored_noise:
            self.s = exp_corr_mat(rho, self.channel._num_rx_ant)
        else:
            self.s = torch.eye(self.channel._num_rx_ant, dtype=torch.complex64)

    def forward(self, batch_size, no):
        x = self.qam_source([batch_size, self.channel._num_tx_ant])
        if self.colored_noise:
            y, h = self.channel(x)
            s = no.to(y.dtype) * torch.eye(self.channel._num_rx_ant, dtype=torch.complex64) + self.s
            s_12 = matrix_sqrt(s)
            w = complex_normal([batch_size, self.channel._num_rx_ant, 1])
            w = torch.squeeze(torch.matmul(s_12, w), -1)
            y += w
        else:
            y, h = self.channel([x, no])
            s = no.to(y.dtype) * self.s

        x_hat, no_eff = self.equalizer(y, h, s)
        err = x - x_hat
        err_mean = torch.mean(err)
        err_var = torch.mean(torch.abs(err) ** 2)
        no_eff_mean = torch.mean(no_eff)

        return err_mean, err_var, no_eff_mean

class TestMIMOEqualizers(unittest.TestCase):

    def test_error_statistics_awgn(self):
        torch.manual_seed(1)
        num_tx_ant = 4
        num_rx_ant = 8
        num_bits_per_symbol = 4
        batch_size = 1000000
        num_batches = 10
        nos = torch.tensor([0.01, 0.1, 1, 3, 10], dtype=torch.float32)
        equalizers = [lmmse_equalizer, zf_equalizer, mf_equalizer]
        for eq in equalizers:
            model = Model(eq, num_tx_ant, num_rx_ant, num_bits_per_symbol)
            for no in nos:
                for i in range(num_batches):
                    if i == 0:
                        err_mean, err_var, no_eff_mean = [e/num_batches for e in model(batch_size, no)]
                    else:
                        a, b, c = [e/num_batches for e in model(batch_size, no)]
                        err_mean += a
                        err_var += b
                        no_eff_mean += c

                # Check that the measured error has zero mean
                self.assertTrue(np.abs(err_mean.item()) < 1e-3)
                # Check that the estimated error variance matches the measured error variance
                self.assertTrue(np.abs(err_var.item() - no_eff_mean.item()) / no_eff_mean.item() < 1e-3)

    def test_error_statistics_colored(self):
        torch.manual_seed(1)
        num_tx_ant = 4
        num_rx_ant = 8
        num_bits_per_symbol = 4
        batch_size = 1000000
        num_batches = 10
        nos = torch.tensor([0.01, 0.1, 1, 3, 10], dtype=torch.float32)
        equalizers = [lmmse_equalizer, zf_equalizer, mf_equalizer]
        for eq in equalizers:
            model = Model(eq, num_tx_ant, num_rx_ant, num_bits_per_symbol, colored_noise=True, rho=0.95)
            for no in nos:
                for i in range(num_batches):
                    if i == 0:
                        err_mean, err_var, no_eff_mean = [e/num_batches for e in model(batch_size, no)]
                    else:
                        a, b, c = [e/num_batches for e in model(batch_size, no)]
                        err_mean += a
                        err_var += b
                        no_eff_mean += c

                # Check that the measured error has zero mean
                self.assertTrue(np.abs(err_mean.item()) < 1e-3)
                # Check that the estimated error variance matches the measured error variance
                self.assertTrue(np.abs(err_var.item() - no_eff_mean.item()) / no_eff_mean.item() < 1e-3)

if __name__ == "__main__":
    unittest.main()