import numpy as np
import torch
import torch.nn as nn
import comcloak
from comcloak.mimo import StreamManagement
from comcloak.ofdm import OFDMDemodulator, LinearDetector
from comcloak.utils import insert_dims
from comcloak.channel import time_to_ofdm_channel

import torch
import torch.nn as nn

class PUSCHReceiver(nn.Module):
    """
    PUSCHReceiver(pusch_transmitter, channel_estimator=None, mimo_detector=None, tb_decoder=None, 
                  return_tb_crc_status=False, stream_management=None, input_domain="freq", 
                  l_min=None, dtype=torch.complex64)

    实现一个完整的5G NR PUSCH接收机，支持批量处理多个发射机发送的信号。支持时域或频域输入。
    """
    
    def __init__(self, 
                 pusch_transmitter,
                 channel_estimator=None,
                 mimo_detector=None,
                 tb_decoder=None,
                 return_tb_crc_status=False,
                 stream_management=None,
                 input_domain="freq",
                 l_min=None,
                 dtype=torch.complex64):
        super().__init__()
        
        assert dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"
        self.dtype = dtype

        assert input_domain in ["time", "freq"], "input_domain must be 'time' or 'freq'"
        self.input_domain = input_domain

        self.return_tb_crc_status = return_tb_crc_status
        self.resource_grid = pusch_transmitter.resource_grid

        # (Optional) Initialize OFDMDemodulator for time-domain input
        if self.input_domain == "time":
            assert l_min is not None, "l_min must be provided for input_domain=='time'"
            self.l_min = l_min
            self.ofdm_demodulator = OFDMDemodulator(
                fft_size=pusch_transmitter._num_subcarriers,
                l_min=self.l_min,
                cyclic_prefix_length=pusch_transmitter._cyclic_prefix_length
            )

        # Initialize ChannelEstimator
        self.perfect_csi = False
        self.w = None

        if channel_estimator is None:
            # Default Channel Estimator
            self.channel_estimator = PUSCHLSChannelEstimator(
                self.resource_grid,
                pusch_transmitter._dmrs_length,
                pusch_transmitter._dmrs_additional_position,
                pusch_transmitter._num_cdm_groups_without_data,
                interpolation_type='lin',
                dtype=dtype
            )
        elif channel_estimator == "perfect":
            # Perfect CSI
            self.perfect_csi = True
            if pusch_transmitter._precoding == "codebook":
                self.w = pusch_transmitter._precoder._w.unsqueeze(2)
        else:
            # User-defined channel estimator
            self.channel_estimator = channel_estimator

        # Initialize StreamManagement
        if stream_management is None:
            # Default StreamManagement
            rx_tx_association = torch.ones((1, pusch_transmitter._num_tx), dtype=torch.bool)
            self.stream_management = StreamManagement(rx_tx_association, pusch_transmitter._num_layers)
        else:
            self.stream_management = stream_management

        # Initialize MIMO Detector
        if mimo_detector is None:
            self.mimo_detector = LinearDetector(
                detection_type="lmmse",
                output_type="bit",
                soft_detection="maxlog",
                resource_grid=pusch_transmitter.resource_grid,
                stream_management=self.stream_management,
                modulation_type="qam",
                num_bits_per_symbol=pusch_transmitter._num_bits_per_symbol,
                dtype=dtype
            )
        else:
            self.mimo_detector = mimo_detector

        # Initialize LayerDemapper
        self.layer_demapper = LayerDemapper(
            layer_mapper=pusch_transmitter._layer_mapper,
            num_bits_per_symbol=pusch_transmitter._num_bits_per_symbol
        )

        # Initialize TBDecoder
        if tb_decoder is None:
            self.tb_decoder = TBDecoder(
                tb_encoder=pusch_transmitter._tb_encoder,
                output_dtype=torch.float32 if dtype == torch.complex64 else torch.float64
            )
        else:
            self.tb_decoder = tb_decoder

    def forward(self, inputs):
        """
        Forward propagation for PUSCHReceiver.

        Inputs
        ------
        inputs : tuple
            If perfect CSI is used, inputs = (y, h, no).
            Otherwise, inputs = (y, no).

        Returns
        -------
        b_hat : torch.Tensor
            Decoded information bits.

        tb_crc_status : torch.Tensor (optional)
            Transport block CRC status.
        """
        if self.perfect_csi:
            y, h, no = inputs
        else:
            y, no = inputs

        # Optional OFDM Demodulation
        if self.input_domain == "time":
            y = self.ofdm_demodulator(y)

        # Channel Estimation
        if self.perfect_csi:
            # Transform time-domain to frequency-domain channel if needed
            if self.input_domain == "time":
                h = time_to_ofdm_channel(h, self.resource_grid, self.l_min)

            if self.w is not None:
                # Reshape and multiply with precoding matrices
                h = h.permute(0, 1, 3, 5, 6, 2, 4)
                h = torch.matmul(h, self.w)
                h = h.permute(0, 1, 5, 2, 6, 3, 4)
            
            h_hat = h
            err_var = torch.zeros_like(h_hat.real)
        else:
            h_hat, err_var = self.channel_estimator((y, no))

        # MIMO Detection
        llr = self.mimo_detector((y, h_hat, err_var, no))

        # Layer Demapping
        llr = self.layer_demapper(llr)

        # TB Decoding
        b_hat, tb_crc_status = self.tb_decoder(llr)

        if self.return_tb_crc_status:
            return b_hat, tb_crc_status
        else:
            return b_hat
