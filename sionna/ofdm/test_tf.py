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

from sionna.ofdm import OFDMModulator, OFDMDemodulator, ResourceGrid, ResourceGridMapper, ResourceGridDemapper
from sionna.mimo import StreamManagement# only numpy no tf
from sionna.utils import QAMSource
from tensorflow.keras import Model

# import pytest
import unittest
import numpy as np
import tensorflow as tf

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


class TestResourceGridDemapper(unittest.TestCase):

    def test_various_params(self):
        """data_dim dimension is omitted"""
        fft_size = 72

        def func(cp_length, num_tx, num_streams_per_tx):
            rg = ResourceGrid(num_ofdm_symbols=14,
                              fft_size=fft_size,
                              subcarrier_spacing=30e3,
                              num_tx=num_tx,
                              num_streams_per_tx=num_streams_per_tx,
                              cyclic_prefix_length=cp_length)
            sm = StreamManagement(np.ones([1, rg.num_tx]), rg.num_streams_per_tx)
            rg_mapper = ResourceGridMapper(rg)
            rg_demapper = ResourceGridDemapper(rg, sm)
            modulator = OFDMModulator(rg.cyclic_prefix_length)
            demodulator = OFDMDemodulator(rg.fft_size, 0, rg.cyclic_prefix_length)
            qam_source = QAMSource(4)
            x = qam_source([128, rg.num_tx, rg.num_streams_per_tx, rg.num_data_symbols])
            x_rg = rg_mapper(x)
            x_time = modulator(x_rg)
            y = demodulator(x_time)
            x_hat = rg_demapper(y)
            # print("x:",x)
            # print("x_rg:",x_rg)
            # print("x_time:",x_time)
            # print("y:",y)
            # print("x_hat:",x_hat)
            return np.max(np.abs(x-x_hat))

        for cp_length in [0,1]:
            for num_tx in [2]:
                for num_streams_per_tx in [2]:
                    err = func(cp_length, num_tx, num_streams_per_tx)
                    self.assertTrue(err<1e-5)

            
if __name__ == '__main__':
    unittest.main()