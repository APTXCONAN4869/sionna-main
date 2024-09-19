# import sys
# sys.path.insert(0, 'D:\sionna-main')
# please ensure your current directory is '$:\sionna-main'
import os
print("Current directory:", os.getcwd())
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("../")
    
from comcloak.mimo import StreamManagement
from comcloak.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, PilotPattern, KroneckerPilotPattern, LMMSEInterpolator, tdl_freq_cov_mat, tdl_time_cov_mat
from comcloak.channel.tr38901 import Antenna, AntennaArray, UMi
from comcloak.channel import gen_single_sector_topology as gen_topology
from comcloak.channel import subcarrier_frequencies, cir_to_ofdm_channel
from comcloak.channel import ApplyOFDMChannel, exp_corr_mat
from comcloak.utils import QAMSource, ebnodb2no
from comcloak.mapping import Mapper
from comcloak.channel.tr38901 import TDL

# import pytest
import unittest
import numpy as np
import itertools
import torch
import torch.nn.functional as F

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')
def freq_int(h, i, j):
    """Linear interpolation along the second axis on a 2D resource grid
    - h is [num_ofdm_symbols, num_subcarriers]
    - i, j are arrays indicating the indices of nonzero pilots
    """
    h_int = np.zeros_like(h)
    h_int[i, j] = h[i, j]

    x_0 = np.zeros_like(h)
    x_1 = np.zeros_like(h)
    y_0 = np.zeros_like(h)
    y_1 = np.zeros_like(h)
    x = np.zeros_like(h)
    for a in range(h_int.shape[0]):
        x[a] = np.arange(0, h_int.shape[1])
        pilot_ind = np.where(h_int[a])[0]
        if len(pilot_ind)==1:
            x_0[a] = x_1[a] = pilot_ind[0]
            y_0[a] = y_1[a] = h_int[a, pilot_ind[0]]
        elif len(pilot_ind)>1:
            x0 = 0
            x1 = 1
            for b in range(h_int.shape[1]):
                x_0[a, b] = pilot_ind[x0]
                x_1[a, b] = pilot_ind[x1]
                y_0[a, b] = h_int[a, pilot_ind[x0]]
                y_1[a, b] = h_int[a, pilot_ind[x1]]
                if b==pilot_ind[x1] and x1<len(pilot_ind)-1:
                    x0 = x1
                    x1 += 1
    h_int = (x-x_0)*np.divide(y_1-y_0, x_1-x_0, out=np.zeros_like(h), where=x_1-x_0!=0) + y_0
    return h_int

def time_int(h, time_avg=False):
    """Linear interpolation along the first axis on a 2D resource grid
    - h is [num_ofdm_symbols, num_subcarriers]
    """
    x_0 = np.zeros_like(h)
    x_1 = np.zeros_like(h)
    y_0 = np.zeros_like(h)
    y_1 = np.zeros_like(h)
    x = np.repeat(np.expand_dims(np.arange(0, h.shape[0]), 1), [h.shape[1]], axis=1)

    pilot_ind = np.where(np.sum(np.abs(h), axis=-1))[0]

    if time_avg:
        hh = np.sum(h, axis=0)/len(pilot_ind)
        h[pilot_ind] = hh

    if len(pilot_ind)==1:
        h_int = np.repeat(h[pilot_ind], [h.shape[0]], axis=0)
        return h_int
    elif len(pilot_ind)>1:
        x0 = 0
        x1 = 1
        for a in range(h.shape[0]):
            x_0[a] = pilot_ind[x0]
            x_1[a] = pilot_ind[x1]
            y_0[a] = h[pilot_ind[x0]]
            y_1[a] = h[pilot_ind[x1]]
            if a==pilot_ind[x1] and x1<len(pilot_ind)-1:
                x0 = x1
                x1 += 1
    h_int = (x-x_0)*np.divide(y_1-y_0, x_1-x_0, out=np.zeros_like(h), where=x_1-x_0!=0) + y_0
    return h_int

def linear_int(h, i, j, time_avg=False):
    """Linear interpolation on a 2D resource grid
    - h is [num_ofdm_symbols, num_subcarriers]
    - i, j are arrays indicating the indices of nonzero pilots
    """
    h_int = freq_int(h, i, j)
    return time_int(h_int, time_avg)

def check_linear_interpolation(self, pilot_pattern, time_avg=False, mode="eager"):
    "Simulate channel estimation with linear interpolation for a 3GPP UMi channel model"
    scenario = "umi"
    carrier_frequency = 3.5e9
    direction = "uplink"
    num_ut = pilot_pattern.num_tx
    num_streams_per_tx = pilot_pattern.num_streams_per_tx
    num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
    fft_size = pilot_pattern.num_effective_subcarriers
    batch_size = 1
    torch.manual_seed(1)

    ut_array = Antenna(polarization="single",
                       polarization_type="V",
                       antenna_pattern="omni",
                       carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    channel_model = UMi(carrier_frequency=carrier_frequency,
                        o2i_model="low",
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction=direction,
                        enable_pathloss=False,
                        enable_shadow_fading=False)

    topology = gen_topology(batch_size, num_ut, scenario, min_ut_velocity=0, max_ut_velocity=30)
    # print('topology:',topology)# OK
    channel_model.set_topology(*topology)

    rx_tx_association = np.zeros([1, num_ut])
    rx_tx_association[0, :] = 1
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                      fft_size=fft_size,
                      subcarrier_spacing=30e3,
                      num_tx=num_ut,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=0,
                      pilot_pattern=pilot_pattern)

    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    # print('frequencies: -------------',frequencies)# OK
    channel_freq = ApplyOFDMChannel(add_awgn=False)
    rg_mapper = ResourceGridMapper(rg)
    
    if time_avg:
        ls_est = LSChannelEstimator(rg, interpolation_type="lin_time_avg")
    else:
        ls_est = LSChannelEstimator(rg, interpolation_type="lin")

    def fun():
        x = torch.zeros([batch_size, num_ut, rg.num_streams_per_tx, rg.num_data_symbols], dtype=torch.complex64)
        x_rg = rg_mapper(x)
        a, tau = channel_model(num_time_samples=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
        # print('a:---------',a)
        # print('tau:---------',tau)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
        y = channel_freq([x_rg, h_freq])  # noiseless channel
        h_hat_lin, _n = ls_est([y, 0.0])
        # print('x_rg:----------', x_rg)
        # print('h_freq:----------', h_freq)
        # print('h_hat_lin:----------', h_hat_lin)
        return x_rg, h_freq, h_hat_lin

    def fun_graph():
        return fun()

    def fun_xla():
        return fun()

    if mode == "eager":
        x_rg, h_freq, h_hat_lin = fun()
    elif mode == "graph":
        x_rg, h_freq, h_hat_lin = fun_graph()
    elif mode == "xla":
        x_rg, h_freq, h_hat_lin = fun_xla()

    for tx in range(num_ut):
        # Get non-zero pilot indices
        i, j = np.where(np.abs(x_rg[0, tx, 0].numpy()))
        h = h_freq[0, 0, 0, tx, 0].numpy()
        h_hat_lin_numpy = linear_int(h, i, j, time_avg)
        # print('test1:',h_hat_lin_numpy)
        # print('test2:',h_hat_lin[0, 0, 0, tx, 0].numpy())
        self.assertTrue(np.allclose(h_hat_lin_numpy, h_hat_lin[0, 0, 0, tx, 0].numpy()))

class TestLinearInterpolator(unittest.TestCase):

    def test_sparse_pilot_pattern(self):
        "One UT has two pilots, three others have just one"
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        mask = np.zeros([num_ut, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
        mask[...,[2,3,10,11],:] = True
        num_pilots = np.sum(mask[0,0])
        pilots = np.zeros([num_ut, num_streams_per_tx, num_pilots])
        pilots[0,0,10] = 1
        pilots[0,0,234] = 1
        pilots[1,0,20] = 1
        pilots[2,0,70] = 1
        pilots[3,0,120] = 1
        pilot_pattern = PilotPattern(mask, pilots)
        check_linear_interpolation(self, pilot_pattern, mode="eager")
        # check_linear_interpolation(self, pilot_pattern, mode="graph")
        # check_linear_interpolation(self, pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_01(self):
    #     num_ut = 4
    #     num_streams_per_tx = 1
    #     num_ofdm_symbols = 14
    #     fft_size = 64
    #     pilot_ofdm_symbol_indices = [2, 11]
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_02(self):
    #     "Only a single pilot symbol"
    #     num_ut = 4
    #     num_streams_per_tx = 1
    #     num_ofdm_symbols = 14
    #     fft_size = 64
    #     pilot_ofdm_symbol_indices = [2]
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_03(self):
    #     "Only one pilot per UT"
    #     num_ut = 16
    #     num_streams_per_tx = 1
    #     num_ofdm_symbols = 14
    #     fft_size = 16
    #     pilot_ofdm_symbol_indices = [2]
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_04(self):
    #     "Multi UT, multi stream"
    #     num_ut = 4
    #     num_streams_per_tx = 2
    #     num_ofdm_symbols = 14
    #     fft_size = 64
    #     pilot_ofdm_symbol_indices = [2, 5, 8]
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_05(self):
    #     "Single UT, only pilots"
    #     num_ut = 1
    #     num_streams_per_tx = 1
    #     num_ofdm_symbols = 5
    #     fft_size = 64
    #     pilot_ofdm_symbol_indices = np.arange(0, num_ofdm_symbols)
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_06(self):
    #     num_ut = 4
    #     num_streams_per_tx = 1
    #     num_ofdm_symbols = 14
    #     fft_size = 64
    #     pilot_ofdm_symbol_indices = [2,3,8, 11]
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    # def test_kronecker_pilot_patterns_with_time_averaging(self):
    #     num_ut = 4
    #     num_streams_per_tx = 1
    #     num_ofdm_symbols = 14
    #     fft_size = 64
    #     pilot_ofdm_symbol_indices = [2,11]
    #     rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
    #               fft_size=fft_size,
    #               subcarrier_spacing=30e3,
    #               num_tx=num_ut,
    #               num_streams_per_tx=num_streams_per_tx,
    #               cyclic_prefix_length=0,
    #               pilot_pattern="kronecker",
    #               pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
    #     check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

#######################################################
# Test LMMSE interpolation
#######################################################

# class TestLMMSEInterpolator(unittest.TestCase):

#     BATCH_SIZE = 1
#     EBN0DBs = [0.0]
#     ATOL_LOW_PREC = 1e-3
#     ATOL_HIGH_PREC = 1e-10

#     def pilot_pattern_2_pilot_mask(self, pilot_pattern):
#         data_mask = pilot_pattern.mask
#         pilots = pilot_pattern.pilots

#         num_tx = data_mask.shape[0]
#         num_streams_per_tx = data_mask.shape[1]
#         num_ofdm_symbols = data_mask.shape[2]
#         num_effective_subcarriers = data_mask.shape[3]
#         pilot_mask = torch.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], dtype=torch.bool)
        
#         for tx in range(num_tx):
#             for st in range(num_streams_per_tx):
#                 pil_ind = 0
#                 for sb in range(num_ofdm_symbols):
#                     for sc in range(num_effective_subcarriers):
#                         if data_mask[tx, st, sb, sc]:
#                             if torch.abs(pilots[tx, st, pil_ind]) > 0.:
#                                 pilot_mask[tx, st, sb, sc] = True
#                             pil_ind += 1
#         return pilot_mask

#     def map_estimates_to_rg(self, h_hat, err_var, pilot_pattern):
#         data_mask = pilot_pattern.mask
#         pilots = pilot_pattern.pilots

#         batch_size = h_hat.shape[0]
#         num_rx = h_hat.shape[1]
#         num_rx_ant = h_hat.shape[2]
#         num_tx = h_hat.shape[3]
#         num_streams_per_tx = h_hat.shape[4]
#         num_ofdm_symbols = data_mask.shape[2]
#         num_effective_subcarriers = data_mask.shape[3]

#         h_hat_rg = torch.zeros([batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], dtype=torch.complex64)
#         err_var_rg = torch.zeros([batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], dtype=torch.float32)
        
#         for bs in range(batch_size):
#             for rx in range(num_rx):
#                 for ra in range(num_rx_ant):
#                     for tx in range(num_tx):
#                         for st in range(num_streams_per_tx):
#                             pil_ind = 0
#                             for sb in range(num_ofdm_symbols):
#                                 for sc in range(num_effective_subcarriers):
#                                     if data_mask[tx, st, sb, sc]:
#                                         if torch.abs(pilots[tx, st, pil_ind]) > 0.:
#                                             h_hat_rg[bs, rx, ra, tx, st, sb, sc] = h_hat[bs, rx, ra, tx, st, pil_ind]
#                                             err_var_rg[bs, rx, ra, tx, st, sb, sc] = err_var[bs, rx, ra, tx, st, pil_ind]
#                                         pil_ind += 1
#         return h_hat_rg, err_var_rg

#     def reference_lmmse_interpolation_1d_one_axis(self, cov_mat, h_hat, err_var, pattern, last_step):
#         err_var_old = err_var

#         dim_size = pattern.shape[0]
#         pil_ind = torch.where(pattern)[0]
#         num_pil = pil_ind.shape[0]

#         pi_mat = torch.zeros([dim_size, num_pil])
#         k = 0
#         for i in range(dim_size):
#             if pattern[i]:
#                 pi_mat[i, k] = 1.0
#                 k += 1

#         int_mat = torch.matmul(torch.matmul(pi_mat.T, cov_mat), pi_mat)
#         err_var = err_var[pil_ind]
#         int_mat = int_mat + torch.diag(err_var)
#         int_mat = torch.linalg.inv(int_mat)
#         int_mat = torch.matmul(pi_mat, torch.matmul(int_mat, pi_mat.T))
#         int_mat = torch.matmul(cov_mat, int_mat)

#         h_hat = torch.matmul(int_mat, h_hat)

#         mask_mat = torch.zeros([dim_size, dim_size])
#         for i in range(dim_size):
#             if pattern[i]:
#                 mask_mat[i, i] = 1.0
#         err_var = cov_mat - torch.matmul(int_mat, torch.matmul(mask_mat, cov_mat))
#         err_var = torch.diag(err_var).real

#         if not last_step:
#             int_mat_h = torch.conj(int_mat.T)
#             h_hat_var = torch.matmul(int_mat, torch.matmul(cov_mat + torch.diag(err_var_old), int_mat_h))
#             h_hat_var = torch.diag(h_hat_var).real
#             s = 2. / (1. + h_hat_var - err_var)
#             h_hat = s * h_hat
#             err_var = s * (s - 1) * h_hat_var + (1. - s) + s * err_var

#         return h_hat, err_var

#     def reference_spatial_smoothing_one_re(self, cov_mat, h_hat, err_var, last_step):
#         A = cov_mat + torch.diag(err_var)
#         A = torch.linalg.inv(A)
#         A = torch.matmul(cov_mat, A)

#         h_hat = h_hat.unsqueeze(-1)
#         h_hat = torch.matmul(A, h_hat)
#         h_hat = h_hat.squeeze(-1)

#         err_var_out = cov_mat - torch.matmul(A, cov_mat)
#         err_var_out = torch.diag(err_var_out).real

#         if not last_step:
#             Ah = torch.conj(A.T)
#             h_hat_var = torch.matmul(A, torch.matmul(cov_mat + torch.diag(err_var), Ah))
#             h_hat_var = torch.diag(h_hat_var).real
#             s = 2. / (1. + h_hat_var - err_var_out)
#             h_hat = s * h_hat
#             err_var_out = s * (s - 1) * h_hat_var + (1. - s) + s * err_var_out

#         return h_hat, err_var_out

#     def reference_spatial_smoothing(self, cov_mat, h_hat, err_var, last_step):
#         h_hat = h_hat.permute(0, 1, 3, 4, 5, 6, 2)
#         err_var = err_var.permute(0, 1, 3, 4, 5, 6, 2)

#         h_hat_shape = h_hat.shape
#         num_rx_ant = h_hat.shape[-1]
#         h_hat = h_hat.reshape(-1, num_rx_ant)
#         err_var = err_var.reshape(-1, num_rx_ant)

#         i = 0
#         for h_hat_, err_var_ in zip(h_hat, err_var):
#             h_hat_new, err_var_new = self.reference_spatial_smoothing_one_re(cov_mat, h_hat_, err_var_, last_step)
#             h_hat[i] = h_hat_new
#             err_var[i] = err_var_new
#             i += 1

#         h_hat = h_hat.reshape(h_hat_shape)
#         err_var = err_var.reshape(h_hat_shape)

#         h_hat = h_hat.permute(0, 1, 6, 2, 3, 4, 5)
#         err_var = err_var.permute(0, 1, 6, 2, 3, 4, 5)

#         return h_hat, err_var

#     def reference_lmmse_interpolation_1d(self, cov_mat, h_hat, err_var, pattern, last_step):
#         batch_size = h_hat.shape[0]
#         num_rx = h_hat.shape[1]
#         num_rx_ant = h_hat.shape[2]
#         num_tx = h_hat.shape[3]
#         num_tx_streams = h_hat.shape[4]
#         outer_dim_size = h_hat.shape[5]
#         inner_dim_size = h_hat.shape[6]

#         for b, rx, ra, tx, ts, od in itertools.product(range(batch_size),
#                                                        range(num_rx),
#                                                        range(num_rx_ant),
#                                                        range(num_tx),
#                                                        range(num_tx_streams),
#                                                        range(outer_dim_size)):
#             h_hat_ = h_hat[b, rx, ra, tx, ts, od]
#             err_var_ = err_var[b, rx, ra, tx, ts, od]
#             pattern_ = pattern[tx, ts, od]
#             if torch.any(pattern_):
#                 h_hat_, err_var_ = self.reference_lmmse_interpolation_1d_one_axis(cov_mat, h_hat_, err_var_, pattern_, last_step)
#                 h_hat[b, rx, ra, tx, ts, od] = h_hat_
#                 err_var[b, rx, ra, tx, ts, od] = err_var_

#         pattern_update_mask = torch.any(pattern, dim=-1, keepdim=True)
#         pattern = torch.logical_or(pattern, pattern_update_mask)

#         return h_hat, err_var, pattern

#     def reference_lmmse_interpolation(self, cov_mat_time, cov_mat_freq, cov_mat_space, h_hat, err_var, pattern, order):
#         pilot_mask = pattern
#         pilot_mask_update = torch.zeros_like(pattern, dtype=torch.bool)

#         i = 0
#         for c, l, t in zip(order['channel'], order['order'], order['last_step']):
#             if c == "t":
#                 h_hat, err_var, pilot_mask = self.reference_lmmse_interpolation_1d(cov_mat_time, h_hat, err_var, pilot_mask, t)
#                 pilot_mask_update = torch.logical_or(pilot_mask_update, torch.logical_not(pilot_mask))
#             elif c == "f":
#                 h_hat, err_var, pilot_mask = self.reference_lmmse_interpolation_1d(cov_mat_freq, h_hat, err_var, pilot_mask, t)
#                 pilot_mask_update = torch.logical_or(pilot_mask_update, torch.logical_not(pilot_mask))
#             elif c == "s":
#                 h_hat, err_var = self.reference_spatial_smoothing(cov_mat_space, h_hat, err_var, t)
#             i += 1

#         return h_hat, err_var, pilot_mask_update
#     def run_e2e_link(self, batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
#             num_ofdm_symbols, fft_size, pilot_pattern, ebno_db, exec_mode, dtype):

#         tdl_model = 'A'
#         subcarrier_spacing = 30e3  # Hz
#         num_bits_per_symbol = 2
#         delay_spread = 300e-9  # s
#         carrier_frequency = 3.5e9  # Hz
#         speed = 5.  # m/s

#         sm = StreamManagement(torch.ones([num_rx, num_tx]), num_streams_per_tx)
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                         fft_size=fft_size,
#                         subcarrier_spacing=subcarrier_spacing,
#                         num_tx=num_tx,
#                         num_streams_per_tx=num_streams_per_tx,
#                         cyclic_prefix_length=0,
#                         pilot_pattern=pilot_pattern,
#                         dtype=dtype)

#         # Transmitter
#         qam_source = QAMSource(num_bits_per_symbol, dtype=dtype)
#         mapper = Mapper("qam", num_bits_per_symbol, dtype=dtype)
#         rg_mapper = ResourceGridMapper(rg, dtype=dtype)

#         # OFDM Channel
#         los_angle_of_arrival = np.pi / 4.
#         channel_model = TDL(tdl_model, delay_spread, carrier_frequency, min_speed=speed, max_speed=speed,
#                             los_angle_of_arrival=los_angle_of_arrival, dtype=dtype)
#         channel_freq = ApplyOFDMChannel(add_awgn=True, dtype=dtype)
#         frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing, dtype=dtype)

#         # The LS channel estimator will provide channel estimates and error variances
#         cov_mat_freq = tdl_freq_cov_mat(tdl_model, subcarrier_spacing, fft_size, delay_spread, dtype)
#         cov_mat_time = tdl_time_cov_mat(tdl_model, speed, carrier_frequency, rg.ofdm_symbol_duration,
#                                         num_ofdm_symbols, los_angle_of_arrival, dtype)
#         cov_mat_space = exp_corr_mat(0.9, num_rx_ant, dtype)
#         lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t")
#         ls_est_lmmse_ft = LSChannelEstimator(rg, interpolator=lmmse_inter_ft, dtype=dtype)
#         lmmse_inter_tf = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="t-f")
#         ls_est_lmmse_tf = LSChannelEstimator(rg, interpolator=lmmse_inter_tf, dtype=dtype)
#         lmmse_inter_tsf = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="t-s-f")
#         ls_est_lmmse_tsf = LSChannelEstimator(rg, interpolator=lmmse_inter_tsf, dtype=dtype)

#         # For computing the reference interpolation
#         ls_no_interp = LSChannelEstimator(rg, interpolation_type=None, dtype=dtype)

#         def _run():
#             no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1.0)
#             x = qam_source([batch_size, num_tx, num_streams_per_tx, rg.num_data_symbols])
#             x_rg = rg_mapper(x)

#             a, tau = channel_model(batch_size, num_ofdm_symbols, sampling_frequency=1. / rg.ofdm_symbol_duration)
#             h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
#             y = channel_freq([x_rg, h_freq, no])

#             h_hat_lmmse_ft, err_var_lmmse_ft = ls_est_lmmse_ft([y, no])
#             h_hat_lmmse_tf, err_var_lmmse_tf = ls_est_lmmse_tf([y, no])
#             h_hat_lmmse_tsf, err_var_lmmse_tsf = ls_est_lmmse_tsf([y, no])
#             h_hat_no_int, err_var_no_int = ls_no_interp([y, no])

#             return h_hat_no_int, err_var_no_int, h_hat_lmmse_ft, err_var_lmmse_ft, h_hat_lmmse_tf, err_var_lmmse_tf, h_hat_lmmse_tsf, err_var_lmmse_tsf, h_freq

#         if exec_mode == 'eager':
#             _run_compiled = _run
#         elif exec_mode == 'graph':
#             _run_compiled = torch.jit.script(_run)
#         elif exec_mode == 'xla':
#             _run_compiled = torch.jit.script(_run)

#         run_output = _run_compiled()
#         h_hat_no_int = run_output[0].numpy()
#         err_var_no_int = run_output[1].numpy()
#         err_var_no_int = np.broadcast_to(err_var_no_int, h_hat_no_int.shape)
#         h_hat_lmmse_ft = run_output[2].numpy()
#         err_var_lmmse_ft = run_output[3].numpy()
#         h_hat_lmmse_tf = run_output[4].numpy()
#         err_var_lmmse_tf = run_output[5].numpy()
#         h_hat_lmmse_tsf = run_output[6].numpy()
#         err_var_lmmse_tsf = run_output[7].numpy()
#         h_freq = run_output[8].numpy()

#         # Reference estimate
#         h_hat_lmmse_ft_ref, err_var_lmmse_ft_ref = self.reference_lmmse_interpolation(cov_mat_time.numpy(),
#                                                                                     cov_mat_freq.numpy(),
#                                                                                     cov_mat_space.numpy(),
#                                                                                     h_hat_no_int, err_var_no_int,
#                                                                                     pilot_pattern, "f-t")
#         h_hat_lmmse_tf_ref, err_var_lmmse_tf_ref = self.reference_lmmse_interpolation(cov_mat_time.numpy(),
#                                                                                     cov_mat_freq.numpy(),
#                                                                                     cov_mat_space.numpy(),
#                                                                                     h_hat_no_int, err_var_no_int,
#                                                                                     pilot_pattern, "t-f")
#         h_hat_lmmse_tsf_ref, err_var_lmmse_tsf_ref = self.reference_lmmse_interpolation(cov_mat_time.numpy(),
#                                                                                     cov_mat_freq.numpy(),
#                                                                                     cov_mat_space.numpy(),
#                                                                                     h_hat_no_int, err_var_no_int,
#                                                                                     pilot_pattern, "t-s-f")

#         # Compute errors
#         max_err_h_hat_ft = np.max(np.abs(h_hat_lmmse_ft_ref - h_hat_lmmse_ft))
#         max_err_err_var_lmmse_ft = np.max(np.abs(err_var_lmmse_ft_ref - err_var_lmmse_ft))
#         max_err_h_hat_tf = np.max(np.abs(h_hat_lmmse_tf_ref - h_hat_lmmse_tf))
#         max_err_err_var_lmmse_tf = np.max(np.abs(err_var_lmmse_tf_ref - err_var_lmmse_tf))
#         max_err_h_hat_tsf = np.max(np.abs(h_hat_lmmse_tsf_ref - h_hat_lmmse_tsf))
#         max_err_err_var_lmmse_tsf = np.max(np.abs(err_var_lmmse_tsf_ref - err_var_lmmse_tsf))

#         return max_err_h_hat_ft, max_err_err_var_lmmse_ft, max_err_h_hat_tf, max_err_err_var_lmmse_tf, max_err_h_hat_tsf, max_err_err_var_lmmse_tsf

#     def run_test(self, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#                     fft_size, mask, pilots):

#         torch.random.set_seed(42)
#         def _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#                     fft_size, pilot_pattern, ebno_db, exec_mode, dtype):
#             # if exec_mode == 'xla':
#             #     sionna.Config.xla_compat = True
#             outputs = self.run_e2e_link(TestLMMSEInterpolator.BATCH_SIZE, num_rx, num_rx_ant, num_tx,
#                 num_streams_per_tx, num_ofdm_symbols, fft_size, pilot_pattern, ebno_db, exec_mode, dtype)
#             # if exec_mode == 'xla':
#             #     sionna.Config.xla_compat = False

#             if dtype == torch.complex64 or exec_mode == "xla":
#                 atol = TestLMMSEInterpolator.ATOL_LOW_PREC
#             else:
#                 atol = TestLMMSEInterpolator.ATOL_HIGH_PREC

#             max_err_h_hat_ft = outputs[0]
#             self.assertTrue(np.allclose(max_err_h_hat_ft, 0.0, atol=atol))

#             max_err_err_var_lmmse_ft = outputs[1]
#             self.assertTrue(np.allclose(max_err_err_var_lmmse_ft, 0.0, atol=atol))

#             max_err_h_hat_tf = outputs[2]
#             self.assertTrue(np.allclose(max_err_h_hat_tf, 0.0, atol=atol))

#             max_err_err_var_lmmse_tf = outputs[3]
#             self.assertTrue(np.allclose(max_err_err_var_lmmse_tf, 0.0, atol=atol))

#             max_err_h_hat_tsf = outputs[4]
#             self.assertTrue(np.allclose(max_err_h_hat_tsf, 0.0, atol=atol))

#             max_err_err_var_lmmse_tsf = outputs[5]
#             self.assertTrue(np.allclose(max_err_err_var_lmmse_tsf, 0.0, atol=atol))

#         for ebno_db in TestLMMSEInterpolator.EBN0DBs:
#             # 32bit precision
#             pilot_pattern = PilotPattern(mask, pilots, dtype=torch.complex64)
#             ebno_db_sp = torch.tensor(ebno_db, torch.float32)
#             _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#                     fft_size, pilot_pattern, ebno_db_sp, "eager", torch.complex64)
#             _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#                     fft_size, pilot_pattern, ebno_db_sp, "graph", torch.complex64)
#             # XLA is not supported
#             # _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#             #         fft_size, pilot_pattern, ebno_db_sp, "xla", torch.complex64)
#             # 64bit precision
#             pilot_pattern = PilotPattern(mask, pilots, dtype=torch.complex128)
#             ebno_db_dp = torch.tensor(ebno_db, torch.float64)
#             _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#                     fft_size, pilot_pattern, ebno_db_dp, "eager", torch.complex128)
#             _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#                     fft_size, pilot_pattern, ebno_db_dp, "graph", torch.complex128)
#             # XLA is not supported
#             # _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
#             #         fft_size, pilot_pattern, ebno_db_dp, "xla", torch.complex128)


#     def test_sparse_pilot_pattern(self):
#         "One UT has two pilots, three others have just one"
#         num_tx = 4
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 14
#         fft_size = 12
#         mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
#         mask[...,5,:] = True
#         num_pilots = np.sum(mask[0,0])
#         pilots = np.zeros([num_tx, num_streams_per_tx, num_pilots])
#         pilots[0,0,[0,11]] = 1
#         pilots[1,0,1] = 1
#         pilots[2,0,5] = 1
#         pilots[3,0,10] = 1
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size, mask, pilots)

#     def test_kronecker_pilot_patterns_01(self):
#         num_tx = 1
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 14
#         fft_size = 64
#         pilot_ofdm_symbol_indices = [2, 11]
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=30e3,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         pilot_pattern = KroneckerPilotPattern(rg, pilot_ofdm_symbol_indices)
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#                         pilot_pattern.mask, pilot_pattern.pilots)

#     def test_kronecker_pilot_patterns_02(self):
#         "Only a single pilot symbol"
#         num_tx = 4
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 14
#         fft_size = 64
#         pilot_ofdm_symbol_indices = [2]
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=30e3,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#                         pilot_pattern.mask, pilot_pattern.pilots)

#     def test_kronecker_pilot_patterns_03(self):
#         "Only one pilot per UT"
#         num_tx = 16
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 14
#         fft_size = 16
#         pilot_ofdm_symbol_indices = [2]
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=30e3,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#                         pilot_pattern.mask, pilot_pattern.pilots)

#     def test_kronecker_pilot_patterns_04(self):
#         "Multi UT, multi stream"
#         num_tx = 4
#         num_streams_per_tx = 2
#         num_ofdm_symbols = 14
#         fft_size = 64
#         pilot_ofdm_symbol_indices = [2, 5, 8]
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=30e3,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#                         pilot_pattern.mask, pilot_pattern.pilots)

#     def test_kronecker_pilot_patterns_05(self):
#         "Single UT, only pilots"
#         num_tx = 1
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 5
#         fft_size = 64
#         pilot_ofdm_symbol_indices = np.arange(0, num_ofdm_symbols)
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=30e3,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#                         pilot_pattern.mask, pilot_pattern.pilots)

#     def test_kronecker_pilot_patterns_06(self):
#         num_tx = 4
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 14
#         fft_size = 64
#         pilot_ofdm_symbol_indices = [2,3,8, 11]
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=30e3,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#                         pilot_pattern.mask, pilot_pattern.pilots)

#     def test_order_error(self):

#         tdl_model = 'A'
#         subcarrier_spacing = 30e3 # Hz
#         num_bits_per_symbol = 2
#         delay_spread = 300e-9 # s
#         carrier_frequency = 3.5e9 # Hz
#         speed = 5. # m/s
#         los_angle_of_arrival=np.pi/4.
#         fft_size = 12
#         num_rx_ant = 16
#         num_tx = 4
#         num_streams_per_tx = 1
#         num_ofdm_symbols = 14
#         pilot_ofdm_symbol_indices = [2,3,8, 11]
#         rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                 fft_size=fft_size,
#                 subcarrier_spacing=subcarrier_spacing,
#                 num_tx=num_tx,
#                 num_streams_per_tx=num_streams_per_tx,
#                 cyclic_prefix_length=0,
#                 pilot_pattern="kronecker",
#                 pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
#         pilot_pattern = rg.pilot_pattern
#         cov_mat_freq = tdl_freq_cov_mat(tdl_model, subcarrier_spacing, fft_size, delay_spread)
#         cov_mat_time = tdl_time_cov_mat(tdl_model, speed, carrier_frequency, rg.ofdm_symbol_duration,
#                                         num_ofdm_symbols, los_angle_of_arrival)
#         cov_mat_space = exp_corr_mat(0.9, num_rx_ant)

#         # Testing random input order
#         with self.assertRaises(AssertionError):
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="hello")

#         # Test multiple --
#         with self.assertRaises(AssertionError):
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f--t")

#         # Test multiple s,f, or t
#         with self.assertRaises(AssertionError):
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-f-t")
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t-t")
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s-s-t")

#         # Test multiple s,f, or t
#         with self.assertRaises(AssertionError):
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-f-t")
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t-t")
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s-s-t")

#         # Test no t or no f
#         with self.assertRaises(AssertionError):
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s")
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="s-t")

#         # Test s but no spatial covariance matrix
#         with self.assertRaises(AssertionError):
#             lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t-s")

if __name__ == '__main__':
    unittest.main()