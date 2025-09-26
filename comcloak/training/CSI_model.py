import os
gpu_num = 1 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
print("Current directory:", os.getcwd())
import sys
sys.path.append("./")
import comcloak
# Load the required comcloak components
from comcloak.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from comcloak.channel import AWGN, RayleighBlockFading, OFDMChannel, TimeChannel, time_lag_discrete_time_channel
from comcloak.channel.tr38901 import AntennaArray, UMi, UMa, RMa
from comcloak.channel import gen_single_sector_topology as gen_topology
from comcloak.utils import compute_ber, ebnodb2no, BinarySource, ebnodb2no, expand_to_rank, insert_dims
from comcloak.supplement import get_real_dtype
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import combinations
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')

from comcloak.ofdm import LMMSEInterpolator, KBestDetector, LinearDetector, LSChannelEstimator
from comcloak.nr import PUSCHReceiver, TBDecoder, PUSCHTransmitter, PUSCHLSChannelEstimator
from comcloak.utils import flatten_last_dims, split_dim, flatten_dims
class BaselineEstimater(nn.Module):
    """BaselineReceiver class implementing a Sionna baseline receiver for
    different receiver architectures.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    dtype : tf.complex64, optional
        The datatype of the layer, by default tf.complex64.

    return_tb_status : bool, optional
        Whether to return transport block status, by default False.

    Input
    -----
    inputs : list
        [y, no] or [y, h, no] (only for 'baseline_perf_csi')

        y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex64
            The received OFDM resource grid after cyclic prefix removal and FFT.

        no : tf.float32
            Noise variance. Must have broadcastable shape to ``y``.

        h : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex64
            Channel frequency responses. Only required for for
            'baseline_perf_csi'.

    Output
    ------
    b_hat : [batch_size, num_tx, tb_size], tf.float32
        The reconstructed payload bits of each transport block.

    tb_crc_status : [batch_size, num_tx], tf.bool
        Transport block CRC status. Only returned if `return_tb_status`
        is `True`.
    """

    def __init__(self,
                 sys_parameters,
                 dtype=torch.complex64,
                 return_tb_status=False,
                 mcs_arr_eval_idx=0,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)
        self._sys_parameters = sys_parameters
        self._return_tb_status = return_tb_status


        self._transmitters = sys_parameters.transmitters
        if self._transmitters._precoding=="codebook":
            self._w = self._transmitters._precoder._w
            self._w = insert_dims(self._w, 2, 1)
        ###################################
        # Channel Estimation
        ###################################
        if sys_parameters.system in ('baseline_lmmse_kbest',
                                     'baseline_lmmse_lmmse'):
            # Setup channel estimator for non-perfect CSI

            # Use low-complexity LMMSE interpolator for large bandwidth parts
            # to keep computational complexity feasible.
            # Remark: dimensions are hard-coded in config. Needs to be adjusted
            # for different PRB dimensions.
            
            # Use standard Sionna LMMSE interpolator over all PRBs
            interpolator = LMMSEInterpolator(
                sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid.pilot_pattern,
                cov_mat_time=sys_parameters.time_cov_mat,
                cov_mat_freq=sys_parameters.freq_cov_mat,
                cov_mat_space=sys_parameters.space_cov_mat,
                order="s-f-t"
            )
            pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][0]
            self._est = PUSCHLSChannelEstimator(
                resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=\
                    pc.dmrs.num_cdm_groups_without_data,
                interpolator=interpolator
            )
        elif sys_parameters.system in ('baseline_lsnn_lmmse'):
            pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][0]
            self._est = PUSCHLSChannelEstimator(
                resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn"
            )
        elif sys_parameters.system in ('baseline_lslin_lmmse'):
            pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][0]
            self._est = PUSCHLSChannelEstimator(
                resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="lin"
            )
            #self._est = LSChannelEstimator(
            #            resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
            #            interpolation_type="lin")
        elif sys_parameters.system in ('baseline_perf_csi_lmmse',
                                       'baseline_perf_csi_kbest'):
            self._est = "perfect"


    def call(self, inputs):
        if self._sys_parameters.system in ("baseline_perf_csi_kbest",
                                           "baseline_perf_csi_lmmse"):
            y, h, no = inputs
            if self._w is not None:
                # Reshape h to put channel matrix dimensions last
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_tx_ant]
                h = h.permute(0, 1, 3, 5, 6, 2, 4)

                # Multiply by precoding matrices to compute effective channels
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_streams]
                h = torch.matmul(h, self._w)

                # Reshape
                # [batch size, num_rx, num_rx_ant, num_tx, num_streams,...
                #  ...num_ofdm_symbols, fft_size]
                h = h.permute(0, 1, 5, 2, 6, 3, 4)
            
            h_hat = h
            err_var = torch.tensor(0, dtype=get_real_dtype(h_hat.dtype))
        else:
            y, no = inputs
            h_hat, err_var = self._est([y, no])
        return h_hat
        
class CSI_model(nn.Module):
    def __init__(self, 
                 sys_parameters, 
                 training=False, 
                 return_tb_status=False,
                 mcs_arr_eval_idx=0):
        super().__init__()
        self._sys_parameters = sys_parameters
        self._training = training
        self._return_tb_status = return_tb_status
        self._mcs_arr_eval_idx = mcs_arr_eval_idx

        ###################################
        # Transmitter
        ###################################
        self._source = BinarySource()
        self._transmitters = sys_parameters.transmitters

        ###################################
        # Channel
        ###################################
        self._channel = sys_parameters.channel

        ###################################
        # Estimater
        ###################################

        self._sys_name = f"Baseline - LS/lin+LMMSE"
        self._estimater = BaselineEstimater(
                            self._sys_parameters,
                            return_tb_status=return_tb_status,
                            mcs_arr_eval_idx=mcs_arr_eval_idx)
        
    def forward(self, batch_size, ebno_db, num_tx=None, mcs_arr_eval_idx=None):
        # randomly sample num_tx active dmrs ports
        if num_tx is None:
            num_tx = self._sys_parameters.max_num_tx
        # if nothing is specified, select one pre-specified MCS
        if mcs_arr_eval_idx is None:
            mcs_arr_eval_idx = self._mcs_arr_eval_idx