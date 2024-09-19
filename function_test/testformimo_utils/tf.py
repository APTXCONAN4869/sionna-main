#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import sys
sys.path.insert(0, 'D:\sionna-main')
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
# import pytest
import unittest
import numpy as np
import tensorflow as tf

from sionna.utils.tensors import matrix_inv
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
import sionna
from sionna.mimo.utils import complex2real_vector, real2complex_vector
from sionna.mimo.utils import complex2real_matrix, real2complex_matrix
from sionna.mimo.utils import complex2real_covariance, real2complex_covariance
from sionna.mimo.utils import complex2real_channel, real2complex_channel, whiten_channel
from sionna.utils import matrix_pinv
from sionna.utils import matrix_sqrt, complex_normal, matrix_inv
from sionna.channel.utils import exp_corr_mat
from sionna.utils import QAMSource

class Complex2Real(unittest.TestCase):
    def test_whiten_channel_noise_covariance(self):
        # Generate channel outputs
        num_rx = 16
        num_tx = 4
        batch_size = 1000000
        qam_source = QAMSource(8, dtype=tf.complex128)

        r = exp_corr_mat(0.8, num_rx, dtype=tf.complex128)
        r_12 = matrix_sqrt(r)
        s = exp_corr_mat(0.5, num_rx, dtype=tf.complex128) + tf.eye(num_rx, dtype=tf.complex128)
        s_12 = matrix_sqrt(s)

        sionna.config.xla_compat = True
        @tf.function(jit_compile=True)
        def fun():
            x = qam_source([batch_size, num_tx, 1])
            h = tf.matmul(tf.expand_dims(r_12,0), complex_normal([batch_size, num_rx, num_tx], dtype=tf.complex128))
            w = tf.squeeze(tf.matmul(tf.expand_dims(s_12, 0), complex_normal([batch_size, num_rx, 1], dtype=tf.complex128)), -1)
            hx = tf.squeeze(tf.matmul(h, x), -1)
            y = hx+w

            # Compute noise error after whitening the complex channel
            yw, hw, sw = whiten_channel(y, h, s)
            hwx = tf.squeeze(tf.matmul(hw, x), -1)
            ww = yw - hwx
            err_w = tf.matmul(ww, ww, adjoint_a=True)/tf.cast(batch_size, ww.dtype) - sw

            # Compute noise error after whitening the real valued channel
            yr, hr, sr = complex2real_channel(y, h, s)
            yrw, hrw, srw = whiten_channel(yr, hr, sr)
            xr = tf.expand_dims(complex2real_vector(x[...,0]), -1)
            hrwxr = tf.squeeze(tf.matmul(hrw, xr), -1)
            wrw = yrw - hrwxr
            err_rw = tf.matmul(wrw, wrw, transpose_a=True)/tf.cast(batch_size, wrw.dtype) - srw

            # Compute noise covariance after transforming the complex whitened channel to real
            ywr, hwr, swr = complex2real_channel(yw, hw, sw)
            hwrxr = tf.squeeze(tf.matmul(hwr, xr), -1)
            wwr = ywr - hwrxr
            err_wr = tf.matmul(wwr, wwr, transpose_a=True)/tf.cast(batch_size, wwr.dtype) - swr
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
        self.assertTrue(np.max(np.abs(err_w))<1e-3)
        self.assertTrue(np.max(np.abs(err_rw))<1e-3)
        self.assertTrue(np.max(np.abs(err_wr))<1e-3)

if __name__ == '__main__':
    unittest.main()