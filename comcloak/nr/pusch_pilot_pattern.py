import torch
import numpy as np
import warnings
from collections.abc import Sequence
import numpy as np
from comcloak.ofdm import PilotPattern
from .pusch_config import PUSCHConfig

class PUSCHPilotPattern:
    """
    Class defining a pilot pattern for NR PUSCH in PyTorch.

    Parameters
    ----------
    pusch_configs : instance or list of PUSCHConfig
        PUSCH Configurations according to which the pilot pattern
        will be created. One configuration is needed for each transmitter.

    dtype : torch.dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `torch.complex64`.
    """

    def __init__(self, pusch_configs, dtype=torch.complex64):
        # Check correct type of pusch_configs
        if isinstance(pusch_configs, PUSCHConfig):
            pusch_configs = [pusch_configs]
        elif isinstance(pusch_configs, Sequence):
            for c in pusch_configs:
                assert isinstance(c, PUSCHConfig), \
                    "Each element of pusch_configs must be a valid PUSCHConfig"
        else:
            raise ValueError("Invalid value for pusch_configs")

        # Check validity of provided pusch_configs
        num_tx = len(pusch_configs)
        num_streams_per_tx = pusch_configs[0].num_layers
        dmrs_grid = pusch_configs[0].dmrs_grid
        num_subcarriers = dmrs_grid[0].shape[0]
        num_ofdm_symbols = pusch_configs[0].l_d
        precoding = pusch_configs[0].precoding
        dmrs_ports = []
        num_pilots = np.sum(pusch_configs[0].dmrs_mask)

        for pusch_config in pusch_configs:
            assert pusch_config.num_layers == num_streams_per_tx, \
                "All pusch_configs must have the same number of layers"
            assert pusch_config.dmrs_grid[0].shape[0] == num_subcarriers, \
                "All pusch_configs must have the same number of subcarriers"
            assert pusch_config.l_d == num_ofdm_symbols, \
                "All pusch_configs must have the same number of OFDM symbols"
            assert pusch_config.precoding == precoding, \
                "All pusch_configs must have the same precoding method"
            assert np.sum(pusch_config.dmrs_mask) == num_pilots, \
                "All pusch_configs must have the same number of masked REs"
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                for port in pusch_config.dmrs.dmrs_port_set:
                    if port in dmrs_ports:
                        msg = f"DMRS port {port} used by multiple transmitters"
                        warnings.warn(msg)
            dmrs_ports += pusch_config.dmrs.dmrs_port_set

        # Create mask and pilots tensors
        mask = torch.zeros((num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers), dtype=torch.bool)
        pilots = torch.zeros((num_tx, num_streams_per_tx, num_pilots), dtype=dtype)
        
        for i, pusch_config in enumerate(pusch_configs):
            for j in range(num_streams_per_tx):
                ind0, ind1 = pusch_config.symbol_allocation
                mask[i, j] = torch.transpose(torch.tensor(pusch_config.dmrs_mask[:, ind0:ind0+ind1], dtype=torch.bool), 0, 1)
                dmrs_grid = torch.transpose(torch.tensor(pusch_config.dmrs_grid[j, :, ind0:ind0+ind1], dtype=dtype), 0, 1)
                pilots[i, j] = dmrs_grid[mask[i, j]]

        # Store the mask and pilots as attributes
        self.mask = mask
        self.pilots = pilots
        self.trainable = False
        self.normalize = False
        self.dtype = dtype
