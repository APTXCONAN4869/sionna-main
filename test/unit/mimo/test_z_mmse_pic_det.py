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

from comcloak.mimo import LinearDetector, MMSEPICDetector
from comcloak.channel import FlatFadingChannel, exp_corr_mat, PerColumnModel
from comcloak.utils import BinarySource, sim_ber, ebnodb2no
from comcloak.mapping import Mapper

class TestMMSEPICDetector(unittest.TestCase):
    NUM_BITS_PER_SYMBOL = 4
    CHANNEL_CORR_A = 0.8
    MAX_ERR = 5e-2

    def run_e2e(self, det, batch_dims, num_rx_ant, num_tx_ant, ebno_dbs, exec_mode, dtype):
        torch.manual_seed(42)

        num_bits_per_symbol = TestMMSEPICDetector.NUM_BITS_PER_SYMBOL
        batch_dims = torch.tensor(batch_dims, dtype=torch.int32)

        # Transmitter
        binary_source = BinarySource(dtype=dtype)
        mapper = Mapper("qam", num_bits_per_symbol, dtype=dtype)

        # Channel
        spatial_corr_mat = exp_corr_mat(TestMMSEPICDetector.CHANNEL_CORR_A, num_rx_ant, dtype)
        spatial_corr = PerColumnModel(spatial_corr_mat)
        channel = FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=spatial_corr, return_channel=True, dtype=dtype)

        # Detector
        if det == 'mmse-pic':
            detector = MMSEPICDetector(demapping_method="maxlog", num_iter=1, output="bit",
                                       constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, dtype=dtype)
        elif det == 'lmmse':
            detector = LinearDetector(equalizer="lmmse", output="bit", demapping_method="maxlog",
                                      constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, dtype=dtype)

        # Bits shape and parameters
        bits_shape = torch.cat([batch_dims, torch.tensor([num_tx_ant, num_bits_per_symbol])])
        prior = torch.zeros(bits_shape, dtype=dtype.real_dtype)
        s = torch.eye(num_rx_ant, dtype=dtype)

        def _run(batch_size, ebno_db):
            # Set noise power
            no = ebnodb2no(ebno_db, num_bits_per_symbol, 1.0)

            # Transmitter
            bits = binary_source(bits_shape)
            x = mapper(bits)
            x = x.squeeze(dim=-1)

            # Channel
            y, h = channel((x, no))

            # Detector
            s_ = no.to(dtype) * s
            if det == 'mmse-pic':
                llrs = detector((y, h, prior, s_))
            elif det == 'lmmse':
                llrs = detector((y, h, s_))

            return bits, llrs

        # Compile for execution mode
        if exec_mode == 'eager':
            _run_c = _run
        elif exec_mode == 'graph':
            _run_c = torch.jit.script(_run)
        elif exec_mode == 'xla':
            _run_c = torch.jit.script(_run)

        # Run over N0s and simulate BER
        ber, _ = sim_ber(_run_c, ebno_dbs, 1, max_mc_iter=100, num_target_bit_errors=1000,
                         soft_estimates=True, early_stop=False, dtype=dtype)

        return ber

    def run_test(self, batch_dims, num_rx_ant, num_tx_ant, ebno_dbs):
        # Test eager - simple precision
        ber_lmmse = self.run_e2e('lmmse', batch_dims, num_rx_ant, num_tx_ant, ebno_dbs, 'eager', torch.complex64)
        ber_mmse_pic = self.run_e2e('mmse-pic', batch_dims, num_rx_ant, num_tx_ant, ebno_dbs, 'eager', torch.complex64)
        max_err = np.max(np.abs(ber_lmmse - ber_mmse_pic) / np.abs(ber_lmmse))
        self.assertTrue(max_err < TestMMSEPICDetector.MAX_ERR)

        # Test eager - double precision
        ber_lmmse = self.run_e2e('lmmse', batch_dims, num_rx_ant, num_tx_ant, ebno_dbs, 'eager', torch.complex128)
        ber_mmse_pic = self.run_e2e('mmse-pic', batch_dims, num_rx_ant, num_tx_ant, ebno_dbs, 'eager', torch.complex128)
        max_err = np.max(np.abs(ber_lmmse - ber_mmse_pic) / np.abs(ber_lmmse))
        self.assertTrue(max_err < TestMMSEPICDetector.MAX_ERR)

    def test_one_time_one(self):
        self.run_test([64], 1, 1, [20.0])

    def test_one_time_n(self):
        self.run_test([64], 16, 1, [-5.0])

    def test_m_time_n(self):
        self.run_test([64], 16, 4, [0.0])

    def test_batch_dims(self):
        detector = MMSEPICDetector(demapping_method="maxlog", num_iter=1, output="bit",
                                   constellation_type="qam", num_bits_per_symbol=2, dtype=torch.complex64)

        # Generate random tensors
        y = torch.randn([8, 4, 3, 16, 2])
        y = torch.complex(y[..., 0], y[..., 1])
        h = torch.randn([8, 4, 3, 16, 2, 2])
        h = torch.complex(h[..., 0], h[..., 1])
        s = torch.eye(16, dtype=torch.complex64)
        prior = torch.zeros([8, 4, 3, 2, 2])

        # Run the detector
        llrs = detector((y, h, prior, s))
        self.assertEqual(llrs.shape, torch.Size([8, 4, 3, 2, 2]))

    def test_prior_symbols(self):
        detector = MMSEPICDetector(demapping_method="maxlog", num_iter=1, output="symbol",
                                   constellation_type="qam", num_bits_per_symbol=2, dtype=torch.complex64)

        y = torch.randn([64, 16, 2])
        y = torch.complex(y[..., 0], y[..., 1])
        h = torch.randn([64, 16, 2, 2])
        h = torch.complex(h[..., 0], h[..., 1])
        s = torch.eye(16, dtype=torch.complex64)
        prior = torch.randn([64, 2, 4])

        logits = detector((y, h, prior, s))
        self.assertEqual(logits.shape, torch.Size([64, 2, 4]))

    def test_multiple_iterations(self):
        detector = MMSEPICDetector(demapping_method="maxlog", num_iter=3, output="bit",
                                   constellation_type="qam", num_bits_per_symbol=2, dtype=torch.complex64)

        y = torch.randn([64, 16, 2])
        y = torch.complex(y[..., 0], y[..., 1])
        h = torch.randn([64, 16, 2, 2])
        h = torch.complex(h[..., 0], h[..., 1])
        s = torch.eye(16, dtype=torch.complex64)
        prior = torch.randn([64, 2, 2])

        logits = detector((y, h, prior, s))
        self.assertEqual(logits.shape, torch.Size([64, 2, 2]))


if __name__ == '__main__':
    unittest.main()