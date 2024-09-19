import sys
sys.path.insert(0, 'D:\sionna-main')
import torch
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("../")
# import pytest
import unittest
import numpy as np
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')

from comcloak.mimo.utils_z import complex2real_vector, real2complex_vector
from comcloak.mimo.utils_z import complex2real_matrix, real2complex_matrix
from comcloak.mimo.utils_z import complex2real_covariance, real2complex_covariance
from comcloak.mimo.utils_z import complex2real_channel, real2complex_channel, whiten_channel

from comcloak.utils.tensors import matrix_inv
from comcloak.utils import matrix_pinv
from comcloak.utils import matrix_sqrt, complex_normal, matrix_inv
from comcloak.channel.utils import exp_corr_mat
from comcloak.utils import QAMSource

class TestComplexTransformations(unittest.TestCase):
    def test_whiten_channel_noise_covariance(self):
        num_rx = 16
        num_tx = 4
        batch_size = 1000000

        qam_source = QAMSource(8, dtype=torch.complex128)
        r = exp_corr_mat(0.8, num_rx, dtype=torch.complex128).to(device)
        r_12 = matrix_sqrt(r)
        s = (exp_corr_mat(0.5, num_rx, dtype=torch.complex128) + torch.eye(num_rx, dtype=torch.complex128)).to(device)
        s_12 = matrix_sqrt(s)

        def fun():
            x = qam_source([batch_size, num_tx, 1]).to(device)
            h = torch.matmul(r_12.unsqueeze(0), torch.randn(batch_size, num_rx, num_tx, dtype=torch.complex128, device=device))
            w = torch.squeeze(torch.matmul(s_12.unsqueeze(0), torch.randn(batch_size, num_rx, 1, dtype=torch.complex128, device=device)), -1)
            hx = torch.squeeze(torch.matmul(h, x), -1)
            y = hx + w

            # Compute noise error after whitening the complex channel
            yw, hw, sw = whiten_channel(y, h, s)
            hwx = torch.squeeze(torch.matmul(hw, x), -1)
            ww = yw - hwx
            err_w = torch.matmul(ww.conj().transpose(-2, -1), ww) / torch.tensor(batch_size, dtype=ww.dtype) - sw

            # Compute noise error after whitening the real valued channel
            yr, hr, sr = complex2real_channel(y, h, s)
            yrw, hrw, srw = whiten_channel(yr, hr, sr)
            xr = torch.unsqueeze(complex2real_vector(x[..., 0]), -1)
            hrwxr = torch.squeeze(torch.matmul(hrw, xr), -1)
            wrw = yrw - hrwxr
            err_rw = torch.matmul(wrw.transpose(-2, -1), wrw) / torch.tensor(batch_size, dtype=ww.dtype) - srw

            # Compute noise covariance after transforming the complex whitened channel to real
            ywr, hwr, swr = complex2real_channel(yw, hw, sw)
            hwrxr = torch.squeeze(torch.matmul(hwr, xr), -1)
            wwr = ywr - hwrxr
            err_wr = torch.matmul(wwr.transpose(-2, -1), wwr) / torch.tensor(batch_size, dtype=ww.dtype) - swr
            return err_w, err_rw, err_wr

        num_iterations = 1
        for i in range(num_iterations):
            if i==0:
                err_w, err_rw, err_wr = [e/num_iterations for e in fun()]
            else:
                a, b, c = fun()
                err_w += a/num_iterations
                err_rw += b/num_iterations
                err_wr += c/num_iterations

        self.assertTrue(np.max(np.abs(err_w.numpy())) < 1e-3)
        self.assertTrue(np.max(np.abs(err_rw.numpy())) < 1e-3)
        self.assertTrue(np.max(np.abs(err_wr.numpy())) < 1e-3)



if __name__ == "__main__":
    unittest.main()