try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("./")

import unittest
import numpy as np
import torch
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')

from comcloak.nr import TBEncoder, TBDecoder, calculate_tb_size
from comcloak.utils import BinarySource

class TestTBDecoder(unittest.TestCase):
    """Test TBDecoder"""

    def test_identity(self):
        """Test that receiver can recover info bits."""

        source = BinarySource()

        # define test parameters
        # the tests cover the following scenarios
        # 1.) Single CB segmentation
        # 2.) Long CB / multiple CWs
        # 3.) Deactivated scrambler
        # 4.) N-dimensional inputs
        # 5.) zero padding

        bs = [[10], [10], [10], [10, 13, 14], [2]]
        tb_sizes = [6656, 60456, 984, 984, 50000]
        num_coded_bits = [13440, 100800, 2880, 2880, 100000]
        num_bits_per_symbols = [4, 8, 2, 2, 4]
        num_layers = [1, 1, 2, 4, 2]
        n_rntis = [1337, 45678, 1337, 1337, 1337]
        sc_ids = [1, 1023, 2, 42, 42]
        use_scramblers = [True, True, False, True, True]

        for i, _ in enumerate(tb_sizes):
            encoder = TBEncoder(
                        target_tb_size=tb_sizes[i],
                        num_coded_bits=num_coded_bits[i],
                        target_coderate=tb_sizes[i]/num_coded_bits[i],
                        num_bits_per_symbol=num_bits_per_symbols[i],
                        num_layers=num_layers[i],
                        n_rnti=n_rntis[i], # used for scrambling
                        n_id=sc_ids[i], # used for scrambling
                        channel_type="PUSCH",
                        codeword_index=0,
                        use_scrambler=use_scramblers[i],
                        verbose=False,
                        output_dtype=torch.float32,
                        )

            decoder = TBDecoder(encoder=encoder,
                                num_bp_iter=10,
                                cn_type="minsum")

            u = source(bs[i] + [encoder.k])
            c = encoder(u)
            llr_ch = 2 * c - 1  # apply BPSK
            u_hat, crc_status = decoder(llr_ch)

            # all info bits can be recovered
            self.assertTrue(np.array_equal(u.numpy(), u_hat.numpy()))
            # all crc checks are valid
            self.assertTrue(np.array_equal(crc_status.numpy(),
                                           np.ones_like(crc_status.numpy())))

    # def test_scrambling(self):
    #     """Test that (de-)scrambling works as expected."""

    #     source = BinarySource()
    #     bs = 10

    #     n_rnti_ref = 1337
    #     sc_id_ref = 42

    #     # add offset to both scrambling indices
    #     n_rnti_offset = [0, 1, 0]
    #     sc_id_offset = [0, 0, 1]

    #     init = True
    #     for i, _ in enumerate(n_rnti_offset):
    #         encoder = TBEncoder(
    #             target_tb_size=60456,
    #             num_coded_bits=100800,
    #             target_coderate=60456 / 100800,
    #             num_bits_per_symbol=4,
    #             n_rnti=n_rnti_ref + n_rnti_offset[i],
    #             n_id=sc_id_ref + sc_id_offset[i],
    #             use_scrambler=True,
    #             verbose=False,
    #         )

    #         if init:
    #             decoder = TBDecoder(encoder=encoder, num_bp_iter=20, cn_type="minsum")

    #         if not init:
    #             u = source([bs, encoder.k])
    #             c = encoder(u)
    #             llr_ch = 2 * c - 1  # apply BPSK
    #             u_hat, crc_status = decoder(llr_ch)

    #             self.assertFalse(torch.equal(u, u_hat))
    #             self.assertTrue(torch.equal(crc_status, torch.zeros_like(crc_status)))

    #         init = False

    # def test_crc(self):
    #     """Test that CRC detects the correct erroneous positions."""

    #     source = BinarySource()
    #     bs = 10

    #     encoder = TBEncoder(
    #         target_tb_size=60456,
    #         num_coded_bits=100800,
    #         target_coderate=60456 / 100800,
    #         num_bits_per_symbol=4,
    #         n_rnti=12367,
    #         n_id=312,
    #         use_scrambler=True,
    #     )

    #     decoder = TBDecoder(encoder=encoder, num_bp_iter=20, cn_type="minsum")

    #     u = source([bs, encoder.k])
    #     c = encoder(u)
    #     llr_ch = 2 * c - 1  # apply BPSK

    #     err_pos = 7
    #     llr_ch = llr_ch.clone()
    #     llr_ch[err_pos, 500:590] = -10  # overwrite some LLR positions

    #     u_hat, crc_status = decoder(llr_ch)

    #     crc_status_ref = torch.ones_like(crc_status)
    #     crc_status_ref[err_pos] = 0

    #     self.assertTrue(torch.equal(crc_status, crc_status_ref))

if __name__ == '__main__':
    unittest.main()