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

import json
import scipy
import numpy as np
from comcloak.utils import compute_ber
from comcloak.channel import OFDMChannel, RayleighBlockFading, TimeChannel
from comcloak.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from comcloak.mimo import StreamManagement

import unittest
import torch
import numpy as np

def run_test(pusch_configs, channel_estimator="perfect", domain="freq", num_rx=1, num_rx_ant=8, graph_mode=False, jit_compile=False, batch_size=128, dtype=torch.complex64):
    """Configurable function for various test cases"""
    # Assume sionna.config.xla_compat is handled elsewhere in PyTorch
    num_tx = len(pusch_configs)
    num_tx_ant = pusch_configs[0].num_antenna_ports
    l_min, l_max = -1, 3

    pusch_transmitter = PUSCHTransmitter(pusch_configs, output_domain=domain, dtype=dtype)

    stream_management = None
    if num_rx == 2:
        rx_tx_association = np.eye(2, dtype=bool)
        stream_management = StreamManagement(rx_tx_association, pusch_configs[0].num_layers)

    pusch_receiver = PUSCHReceiver(
        pusch_transmitter,
        stream_management=stream_management,
        input_domain=domain,
        l_min=l_min,
        channel_estimator=channel_estimator,
        dtype=dtype
    )

    rayleigh = RayleighBlockFading(
        num_rx=num_rx,
        num_rx_ant=num_rx_ant,
        num_tx=num_tx,
        num_tx_ant=num_tx_ant,
        dtype=dtype
    )

    if domain == "freq":
        channel = OFDMChannel(
            rayleigh,
            pusch_transmitter.resource_grid,
            add_awgn=False,
            normalize_channel=True,
            return_channel=True,
            dtype=dtype
        )
    else:
        channel = TimeChannel(
            rayleigh,
            pusch_transmitter.resource_grid.bandwidth,
            pusch_transmitter.resource_grid.num_time_samples,
            l_min=l_min,
            l_max=l_max,
            add_awgn=False,
            normalize_channel=True,
            return_channel=True,
            dtype=dtype
        )

    def run():
        x, b = pusch_transmitter(batch_size)
        y, h = channel(x)
        if channel_estimator == "perfect":
            b_hat = pusch_receiver([y, h, torch.tensor(0.001, dtype=dtype.real_dtype)])
        else:
            b_hat = pusch_receiver([y, torch.tensor(0.001, dtype=dtype.real_dtype)])
        return compute_ber(b, b_hat)

    if graph_mode:
        if jit_compile:
            run_graph = torch.jit.script(run)
            res = run_graph()
        else:
            res = run()
    else:
        res = run()
    
    return res

class TestPUSCHReceiver(unittest.TestCase):
    """Tests for PUSCHReceiver"""

    def test_01(self):
        """Test perfect and imperfect CSI in all execution modes"""
        torch.manual_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 1
        pusch_config.dmrs.dmrs_port_set = [0, 1]
        pusch_configs = [pusch_config]
        for jit_compile in [False, True]:
            for graph_mode in [False, True]:
                if jit_compile and not graph_mode:
                    continue
                ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq", graph_mode=graph_mode, jit_compile=jit_compile)
                if jit_compile:
                    self.assertFalse(torch.isnan(ber).any())
                else:
                    self.assertEqual(ber, 0.0)
                ber = run_test(pusch_configs, channel_estimator=None, domain="freq", graph_mode=graph_mode, jit_compile=jit_compile)
                if jit_compile:
                    self.assertFalse(torch.isnan(ber).any())
                else:
                    self.assertEqual(ber, 0.0)
                ber = run_test(pusch_configs, channel_estimator="perfect", domain="time", graph_mode=graph_mode, jit_compile=jit_compile)
                if jit_compile:
                    self.assertFalse(torch.isnan(ber).any())
                else:
                    self.assertEqual(ber, 0.0)
                ber = run_test(pusch_configs, channel_estimator=None, domain="time", graph_mode=graph_mode, jit_compile=jit_compile)
                if jit_compile:
                    self.assertFalse(torch.isnan(ber).any())
                else:
                    self.assertEqual(ber, 0.0)

    def test_02(self):
        """Multi transmitter, multi stream test"""
        torch.manual_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0, 2]
        pusch_config.dmrs.additional_position = 1

        pusch_config2 = pusch_config.clone()
        pusch_config2.dmrs.dmrs_port_set = [1, 3]

        pusch_configs = [pusch_config, pusch_config2]
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="freq")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="time")
        self.assertEqual(ber, 0.0)

    def test_03(self):
        """Multi transmitter, multi stream, no precoding"""
        torch.manual_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 2
        pusch_config.precoding = "non-codebook"
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0, 2]
        pusch_config.dmrs.additional_position = 1

        pusch_config2 = pusch_config.clone()
        pusch_config2.dmrs.dmrs_port_set = [1, 3]

        pusch_configs = [pusch_config, pusch_config2]
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="freq")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="time")
        self.assertEqual(ber, 0.0)

    def test_04(self):
        """Very large transport block"""
        torch.manual_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 273
        pusch_config.tb.mcs_index = 26
        pusch_config.tb.mcs_table = 2
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.num_layers = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 1
        pusch_config.dmrs.dmrs_port_set = [0, 1, 6, 7]
        pusch_config.dmrs.additional_position = 0
        pusch_configs = [pusch_config]
        ber = run_test(pusch_configs, channel_estimator=None, batch_size=2)
        self.assertEqual(ber, 0.0)

    def test_05(self):
        """Very short transport block"""
        torch.manual_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 1
        pusch_config.num_antenna_ports = 1
        pusch_config.precoding = "non-codebook"
        pusch_config.num_layers = 1
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.additional_position = 0
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0]
        pusch_config.mapping_type = "B"
        pusch_config.symbol_allocation = [5, 2]
        pusch_config.tb.mcs_index = 10
        pusch_configs = [pusch_config]
        ber = run_test(pusch_configs, channel_estimator=None, batch_size=128)
        self.assertEqual(ber, 0.0)
    
    def test_06(self):
        """Multi transmitter, multi stream, multi receiver test"""
        torch.random.set_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports=4
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length=2
        pusch_config.dmrs.additional_position=1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0,1]
        pusch_config.tb.mcs_index = 10

        pusch_config2 = pusch_config.clone()
        pusch_config.dmrs.dmrs_port_set = [2,3]

        pusch_configs = [pusch_config, pusch_config2]

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="freq")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time")
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="time")
        self.assertEqual(ber, 0.0)

    def test_07(self):
        """Multi transmitter, multi stream, multi receiver test in torch.complex128"""
        torch.random.set_seed(1)
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports=4
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length=2
        pusch_config.dmrs.additional_position=1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0,1]
        pusch_config.tb.mcs_index = 10

        pusch_config2 = pusch_config.clone()
        pusch_config.dmrs.dmrs_port_set = [2,3]

        pusch_configs = [pusch_config, pusch_config2]
        
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq", dtype=torch.complex128)
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="freq", dtype=torch.complex128)
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time", dtype=torch.complex128)
        self.assertEqual(ber, 0.0)
        ber = run_test(pusch_configs, channel_estimator=None, domain="time", dtype=torch.complex128)
        self.assertEqual(ber, 0.0)

