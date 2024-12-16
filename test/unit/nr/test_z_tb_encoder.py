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

from os import walk # to load generator matrices from files
import os
from comcloak.nr import TBEncoder, TBDecoder, calculate_tb_size
from comcloak.utils import BinarySource



class TestTBEncoder(unittest.TestCase):
    """Test TBEncoder"""

    # def test_reference(self):
    #     """Test against reference implementation"""
    #     # load matlab cases
    #     ref_path = '../test/unit/nr/tb_refs/'
    #     f = []
    #     for (_, _, filenames) in os.walk(ref_path):
    #         # filter only npz files
    #         files = [fi for fi in filenames if fi.endswith(".npz")]
    #         f.extend(files)

    #     # load test data
    #     for fn in f:
    #         data = np.load(os.path.join(ref_path, fn))
    #         # restore data
    #         u_ref = torch.tensor(data["u_ref"], dtype=torch.float32)
    #         c_ref = torch.tensor(data["c_ref"], dtype=torch.float32)
    #         n_id = data["n_id"]
    #         n_rnti = data["n_rnti"]
    #         target_coderate = data["coderate"]
    #         num_bits_per_symbol = data["num_bits_per_symbol"]
    #         num_layers = data["num_layers"]
    #         num_coded_bits = c_ref.shape[1]
    #         tb_size = u_ref.shape[1]

    #         # run tests
    #         encoder = TBEncoder(
    #             num_coded_bits=num_coded_bits,
    #             target_tb_size=tb_size,
    #             target_coderate=target_coderate,
    #             num_bits_per_symbol=num_bits_per_symbol,
    #             num_layers=num_layers,
    #             n_rnti=n_rnti,
    #             n_id=n_id,
    #             channel_type="PUSCH",
    #             codeword_index=0,
    #             use_scrambler=True,
    #             verbose=False,
    #             output_dtype=torch.float32
    #         )

    #         decoder = TBDecoder(encoder, cn_type="minsum")

    #         c = encoder(u_ref)
    #         u, _ = decoder(2 * c - 1)

    #         self.assertTrue(torch.equal(c, c_ref))
    #         self.assertTrue(torch.equal(u, u_ref))

    def test_multi_stream(self):
        """Test that n_rnti and n_id can be provided as list"""

        n_rnti = [224, 42, 1, 1337, 45666, 2333, 2133]
        n_id = [42, 123, 0, 3, 32, 456, 875]

        bs = 10
        tb_size = 50000
        num_coded_bits = 100000
        target_coderate = tb_size / num_coded_bits
        num_bits_per_symbol = 4
        num_layers = 2

        encoder = TBEncoder(
            target_tb_size=tb_size,
            num_coded_bits=num_coded_bits,
            target_coderate=target_coderate,
            num_bits_per_symbol=num_bits_per_symbol,
            num_layers=num_layers,
            n_rnti=n_rnti,
            n_id=n_id,
            channel_type="PUSCH",
            codeword_index=0,
            use_scrambler=True,
            verbose=False,
            output_dtype=torch.float32
        )

        decoder = TBDecoder(encoder)

        source = BinarySource()

        u = source([bs, len(n_rnti), encoder.k])
        c = encoder(u)
        u_hat, _ = decoder(2 * c - 1)

        self.assertTrue(torch.equal(u, u_hat))

        # individual encoders
        c_ref = torch.zeros_like(c)
        for idx, (nr, ni) in enumerate(zip(n_rnti, n_id)):
            encoder = TBEncoder(
                target_tb_size=tb_size,
                num_coded_bits=num_coded_bits,
                target_coderate=target_coderate,
                num_bits_per_symbol=num_bits_per_symbol,
                num_layers=num_layers,
                n_rnti=nr,
                n_id=ni,
                channel_type="PUSCH",
                codeword_index=0,
                use_scrambler=True,
                verbose=False,
                output_dtype=torch.float32
            )
            c_ref[:, idx, :] = encoder(u[:, idx, :])

        self.assertTrue(torch.equal(c, c_ref))

if __name__ == '__main__':
    unittest.main()