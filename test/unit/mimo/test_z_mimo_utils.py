import torch
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("./")
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

    def test_vector(self):
        shapes = [
            [1],
            [20, 1],
            [30, 20],
            [30, 20, 40]
        ]
        for shape in shapes:
            z = torch.rand(shape, dtype=torch.complex128)
            x = z.real
            y = z.imag

            # complex2real transformation
            zr = complex2real_vector(z)
            x_, y_ = torch.split(zr, zr.shape[-1] // 2, dim=-1)
            self.assertTrue(np.array_equal(x.numpy(), x_.numpy()))
            self.assertTrue(np.array_equal(y.numpy(), y_.numpy()))

            # real2complex transformation
            zc = real2complex_vector(zr)
            self.assertTrue(np.array_equal(z.numpy(), zc.numpy()))

    def test_matrix(self):
        shapes = [
            [1, 1],
            [20, 1],
            [1, 20],
            [30, 20],
            [30, 20, 40],
            [12, 45, 64, 42]
        ]
        for shape in shapes:
            h = torch.rand(shape, dtype=torch.complex128)
            h_r = h.real
            h_i = h.imag

            # complex2real transformation
            hr = complex2real_matrix(h)
            self.assertTrue(np.array_equal(h_r.numpy(), hr[..., :shape[-2], :shape[-1]].numpy()))
            self.assertTrue(np.array_equal(h_r.numpy(), hr[..., shape[-2]:, shape[-1]:].numpy()))
            self.assertTrue(np.array_equal(h_i.numpy(), hr[..., shape[-2]:, :shape[-1]].numpy()))
            self.assertTrue(np.array_equal(-h_i.numpy(), hr[..., :shape[-2], shape[-1]:].numpy()))

            # real2complex transformation
            hc = real2complex_matrix(hr)
            self.assertTrue(np.array_equal(h.numpy(), hc.numpy()))

    def test_covariance(self):
        ns = [1, 2, 5, 13]
        batch_dims = [
            [1],
            [5, 30],
            [4, 5, 10]
        ]
        for shape in batch_dims:
            for n in ns:
                a = torch.rand(shape)
                r = exp_corr_mat(a, n)
                r_r = r.real / 2
                r_i = r.imag / 2

                # complex2real transformation
                rr = complex2real_covariance(r)
                self.assertTrue(np.allclose(r_r.numpy(), rr[..., :n, :n].numpy()))
                self.assertTrue(np.allclose(r_r.numpy(), rr[..., n:, n:].numpy()))
                self.assertTrue(np.allclose(r_i.numpy(), rr[..., n:, :n].numpy()))
                self.assertTrue(np.allclose(-r_i.numpy(), rr[..., :n, n:].numpy()))

                # real2complex transformation
                rc = real2complex_covariance(rr)
                self.assertTrue(np.allclose(r.numpy(), rc.numpy()))

    def test_covariance_statistics(self):
        """Test that the statistics of the real-valued equivalent random vector match the target statistics"""
        batch_size = 1000000
        num_batches = 100
        n = 8
        r = exp_corr_mat(0.8, 8)
        rr = complex2real_covariance(r)
        r_12 = torch.linalg.cholesky(r).to(torch.complex128)
        print(r_12.dtype)
        def fun():
            w = torch.matmul(r_12, torch.randn(n, batch_size, dtype=torch.complex128))
            w = w.transpose(-2, -1)
            wr = complex2real_vector(w)
            r_hat = torch.matmul(wr.transpose(-2, -1), wr) / batch_size
            return r_hat

        r_hat = torch.zeros_like(rr)
        for _ in range(num_batches):
            r_hat += fun() / num_batches

        self.assertTrue(np.max(np.abs(rr.numpy() - r_hat.numpy())) < 1e-3)

    def test_whiten_channel_noise_covariance(self):
        num_rx = 16
        num_tx = 4
        batch_size = 1000000

        qam_source = QAMSource(8, dtype=torch.complex128)
        r = exp_corr_mat(0.8, num_rx, dtype=torch.complex128)
        r_12 = matrix_sqrt(r)
        s = exp_corr_mat(0.5, num_rx, dtype=torch.complex128) + torch.eye(num_rx, dtype=torch.complex128)
        s_12 = matrix_sqrt(s)

        def fun():
            x = qam_source([batch_size, num_tx, 1])
            h = torch.matmul(r_12.unsqueeze(0), torch.randn(batch_size, num_rx, num_tx, dtype=torch.complex128))
            w = torch.squeeze(torch.matmul(s_12.unsqueeze(0), torch.randn(batch_size, num_rx, 1, dtype=torch.complex128)), -1)
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

        num_iterations = 100
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

    def test_whiten_channel_symbol_recovery(self):
        """Check that the whitened channel can be used to recover the symbols"""
        num_rx = 16
        num_tx = 4
        batch_size = 1000000

        qam_source = QAMSource(8, dtype=torch.complex128)
        s = exp_corr_mat(0.5, num_rx, dtype=torch.complex128) + torch.eye(num_rx, dtype=torch.complex128)
        s_12 = matrix_sqrt(s)
        r = exp_corr_mat(0.8, num_rx, dtype=torch.complex128)
        r_12 = matrix_sqrt(r)

        def fun():
            # Noise free transmission
            x = qam_source([batch_size, num_tx, 1])
            h = torch.matmul(r_12.unsqueeze(0), torch.randn(batch_size, num_rx, num_tx, dtype=torch.complex128))
            hx = torch.squeeze(torch.matmul(h, x), -1)
            y = hx

            # Compute symbol error on detection on the complex whitened channel
            yw, hw, sw = whiten_channel(y, h, s)
            xw = torch.matmul(torch.linalg.pinv(hw), yw.unsqueeze(-1))

            err_w = torch.mean(x - xw, dim=0)
            return err_w

        err_w = fun()
        self.assertTrue(np.max(np.abs(err_w.numpy())) < 1e-6)

if __name__ == "__main__":
    unittest.main()