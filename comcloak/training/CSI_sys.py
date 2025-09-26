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
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from comcloak.channel.tr38901 import Antenna, AntennaArray, CDL
from comcloak.channel import OFDMChannel
from comcloak.mimo import StreamManagement
from comcloak.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from comcloak.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, expand_to_rank
from comcloak.fec.ldpc.encoding import LDPC5GEncoder
from comcloak.fec.ldpc.decoding import LDPC5GDecoder
from comcloak.mapping import Mapper, Demapper
from comcloak.utils.metrics import compute_ber
from comcloak.utils import sim_ber

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import ceil


############################################
## Channel configuration
NUM_BS_ANT = 2
carrier_frequency = 3.5e9 # Hz
delay_spread = 100e-9 # s
cdl_model = "C" # CDL model to use
speed = 10.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = -5.0
ebno_db_max = 10.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 30e3 # Hz
fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
num_bits_per_symbol = 2 # QPSK
coderate = 0.5 # Coderate for LDPC code

############################################
## Neural receiver configuration
num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver
compressed_bits = 1024
num_lstm_layers = 2
lstm_hidden_dim_factor = 8
############################################
## Training configuration
num_training_iterations = 10000 # Number of training iterations
training_batch_size = 128 # Training batch size
model_weights_path = "./comcloak/training/CSI_sys_weights" # Location to save the neural receiver weights once training is done
train_log_path = "./comcloak/training/train_log_CSI3.txt"
############################################
## Evaluation configuration
results_filename = "CSI_sys_results" # Location to save the results
############################################

stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                  1)               # One stream per transmitter
resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = 1,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)
# Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate)
ut_antenna = Antenna(polarization="single",
                     polarization_type="V",
                     antenna_pattern="38.901",
                     carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        num_cols=int(NUM_BS_ANT/2),
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
class ResidualBlock(nn.Module):
    def __init__(self, num_ofdm_symbols, fft_size, num_conv_channels):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm([fft_size, num_ofdm_symbols, num_conv_channels])  # LayerNorm for spatial dims
        self.conv1 = nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([fft_size, num_ofdm_symbols, num_conv_channels])
        self.conv2 = nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        # LayerNorm expects [B, C, H, W], normalize over C
        z = self.norm1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.norm2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + residual
    
class CompressBlock(nn.Module):
    def __init__(self, num_ofdm_symbols, fft_size, num_conv_channels, compressed_bits):
        super(CompressBlock, self).__init__()
        self.conv = nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=4, stride=2, padding=1)
        self.compressed_layer = nn.Linear(num_conv_channels * ceil(num_ofdm_symbols/2) * ceil(fft_size/2), compressed_bits)
    def forward(self, x):
        # LayerNorm expects [B, C, H, W], normalize over C
        z = self.conv(x)
        z = F.relu(z)
        # z = self.conv(z)
        # z = F.relu(z)
        # Flatten the output to feed it to the compressed layer
        z = z.view(z.size(0), -1)
        z_compressed = self.compressed_layer(z)
        return z_compressed

class ExpandBlock(nn.Module):
    def __init__(self, num_ofdm_symbols, fft_size, num_conv_channels, compressed_bits):
        super(ExpandBlock, self).__init__()
        self.expand_layer = nn.Linear(compressed_bits, num_conv_channels * ceil(num_ofdm_symbols/2) * ceil(fft_size/2))
        # The output of the expand layer is reshaped to match the input of the deconvolution layer
        # The deconvolution layer will upsample the feature map to the original size
        self.deconv = nn.ConvTranspose2d(num_conv_channels, num_conv_channels, kernel_size=4, stride=2, padding=1, output_padding=0)

    def forward(self, z_compressed):
        z = self.expand_layer(z_compressed)
        z = z.view(z.size(0), -1, ceil(num_ofdm_symbols/2), ceil(fft_size/2))
        z = F.relu(z)
        z = self.deconv(z)
        return z
       
class CSI_Network(nn.Module):
    def __init__(self):
        super(CSI_Network, self).__init__()
        self.lstm_layer = nn.LSTM(fft_size*(2*NUM_BS_ANT+1), lstm_hidden_dim_factor*fft_size, num_lstm_layers, batch_first=True)
        self.input_conv = nn.Conv2d(lstm_hidden_dim_factor, num_conv_channels, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock(num_ofdm_symbols, fft_size, num_conv_channels)
        self.res_block2 = ResidualBlock(num_ofdm_symbols, fft_size, num_conv_channels)
        self.res_block3 = ResidualBlock(num_ofdm_symbols, fft_size, num_conv_channels)
        self.compressed_layer = nn.Linear(num_conv_channels * num_ofdm_symbols * fft_size, compressed_bits)
        self.expand_layer = nn.Linear(compressed_bits, num_conv_channels * num_ofdm_symbols * fft_size)
        # self.compressed_layer = CompressBlock(num_ofdm_symbols, fft_size, num_conv_channels, compressed_bits)
        # self.expand_layer = ExpandBlock(num_ofdm_symbols, fft_size, num_conv_channels, compressed_bits)
        self.res_block4 = ResidualBlock(num_ofdm_symbols, fft_size, num_conv_channels)
        self.res_block5 = ResidualBlock(num_ofdm_symbols, fft_size, num_conv_channels)
        self.res_block6 = ResidualBlock(num_ofdm_symbols, fft_size, num_conv_channels)
        self.output_conv = nn.Conv2d(num_conv_channels, 2*NUM_BS_ANT, kernel_size=3, padding=1)

    def forward(self, inputs):
        # y: [B, Nr_ant, S, F], complex
        # no: [B], float
        y, no = inputs
        y = y.permute(0, 2, 3, 1)    
        # [B, S, F, Nr_ant] → real/imag: [B, S, F, Nr_ant]
        y_real = y.real
        y_imag = y.imag


        # noise scalar to [B, S, F, 1]
        B, S, F, _ = y_real.shape
        # Feeding the noise power in log10 scale helps with the performance
        no = insert_dims(torch.log10(no),3, 1) # [B,1,1,1]
        no = no.expand(-1, S, F, 1)

        # concat along last axis → [B, S, F, 2*Nr_ant + 1]
        z = torch.cat([y_real, y_imag, no], dim=-1)
        # z = z.permute(0, 3, 1, 2)# [B, 2*Nr_ant + 1, S, F]

        z = z.reshape(B, S, F * z.shape[-1])
        # Input conv z:[B, C, H, W]


        # h = inputs # [B, Nr, Nr_ant, str_per_tx, S, F]
        # h = h.squeeze()? # [B, Nr_ant, S, F] not here, just analyse
        z, _ = self.lstm_layer(z)
        z = z.reshape(B, S, F, lstm_hidden_dim_factor)
        z = z.permute(0, 3, 1, 2)# [B, 2*Nr_ant + 1, S, F]
        z = self.input_conv(z)
        # Residual blocks
        z = self.res_block1(z)
        z = self.res_block2(z)
        z = self.res_block3(z)

        z = z.view(z.size(0), -1)

        z_compressed = self.compressed_layer(z)
        z = self.expand_layer(z_compressed)

        z = z.view(z.size(0), -1, num_ofdm_symbols, fft_size)

        # Residual blocks
        z = self.res_block4(z)
        z = self.res_block5(z)
        z = self.res_block6(z)
        # Output conv
        z = self.output_conv(z)

        z = z.permute(0, 2, 3, 1)
        return z


# ## Transmitter
# binary_source = BinarySource()
# mapper = Mapper("qam", num_bits_per_symbol)
# rg_mapper = ResourceGridMapper(resource_grid)

# ## Channel
# cdl = CDL(cdl_model, delay_spread, carrier_frequency,
#           ut_antenna, bs_array, "uplink", min_speed=speed)
# channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

# ## Receiver
# neural_receiver = CSI_Network()
# rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) 

# batch_size = 64
# ebno_db = torch.full([batch_size], 5.0)
# no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
# ## Transmitter
# # Generate codewords
# c = binary_source([batch_size, 1, 1, n])
# print("c shape: ", c.shape)
# # Map bits to QAM symbols
# x = mapper(c)
# print("x shape: ", x.shape)
# # Map the QAM symbols to a resource grid
# x_rg = rg_mapper(x)
# print("x_rg shape: ", x_rg.shape)

# ######################################
# ## Channel
# # A batch of new channel realizations is sampled and applied at every inference
# no_ = expand_to_rank(no, x_rg.dim())
# y,_ = channel([x_rg, no_])
# print("y shape: ", y.shape)

# ######################################
# ## Receiver       
# # The neural receiver computes LLRs from the frequency domain received symbols and N0
# y = y.squeeze(1)
# llr = neural_receiver([y, no])
# print("llr shape: ", llr.shape)
# # Reshape the input to fit what the resource grid demapper is expected
# llr = insert_dims(llr, 2, 1)
# print("llr shape: ", llr.shape)
# # Extract data-carrying resource elements. The other LLRs are discarded
# llr = rg_demapper(llr)
# llr = torch.reshape(llr, [batch_size, 1, 1, n])
# print("Post RG-demapper LLRs: ", llr.shape)

class E2ESystem(nn.Module):
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)

        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_antenna, bs_array, "uplink", min_speed=speed)
        self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "CSI-sys": # Neural receiver
            self._CSI_sys = CSI_Network()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def forward(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = torch.full((batch_size,), ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, 1, 1, n])
        else:
            b = self._binary_source([batch_size, 1, 1, k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, x_rg.dim())
        y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, x_hat.dim())
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "CSI-sys":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = y.squeeze(1)
            y = y.to(device)
            no = no.to(device)
            h_hat = self._CSI_sys([y, no])
            # llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            # llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            # llr = torch.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            # c = c.to(device)
            h = h.squeeze()
            h = h.permute(0, 2, 3, 1)
            h_real = h.real
            h_imag = h.imag
            h_ri = torch.cat([h_real, h_imag], dim=-1)
            h_ri = h_ri.to(device)
            #complex?
            # bce = F.binary_cross_entropy_with_logits(llr, h_ri, reduction='none')
            # bce = torch.mean(bce)
            # rate = torch.tensor(1.0, dtype=torch.float32) - bce/torch.math.log(2.)
            # return rate
            # mean = h_ri.mean(dim=[0,1,2,3], keepdim=True)
            # std = h_ri.std(dim=[0,1,2,3], keepdim=True)
            # H_norm = (h_ri - mean) / (std + 1e-6)
            # H_hat_norm = (h_hat - mean) / (std + 1e-6)
            denom = torch.mean(torch.abs(h_ri)**2) + 1e-8
            loss = torch.mean(torch.abs(h_ri - h_hat)**2) / denom
            # loss = F.mse_loss(H_hat_norm, H_norm)
            # loss = torch.nn.functional.smooth_l1_loss(h_hat, h_ri)
            # loss = torch.mean(torch.abs(h_ri - h_hat)**2)

            if loss > 1e4:
                torch.save(h_hat.detach().cpu(), f"./comcloak/training/save_tensors/h_hat_step{i}.pt")
                torch.save(h_ri.detach().cpu(), f"./comcloak/training/save_tensors/h_ri_step{i}.pt")

            return loss
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
        

# # The end-to-end system equipped with the neural receiver is instantiated for training.
# # When called, it therefore returns the estimated BMD rate
model = E2ESystem('CSI-sys', training=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(num_training_iterations):
    # Sample Eb/No
    ebno_db = random.uniform(ebno_db_min, ebno_db_max)
    ebno_db_tensor = torch.tensor(ebno_db, dtype=torch.float32).to(device)

    # Forward and backward
    optimizer.zero_grad()
    loss = model(training_batch_size, ebno_db_tensor)  
    # loss = -rate  

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Iteration {i}/{num_training_iterations}  Loss: {loss.item():.4f}", end="\r")
        with open(train_log_path, "a") as f:
            f.write(f"{i},{ebno_db:.2f},{loss.item():.6f}\n")
# Save weights using pickle
weights = {k: v.cpu() for k, v in model.state_dict().items()}
with open(model_weights_path, 'wb') as f:
    pickle.dump(weights, f)


