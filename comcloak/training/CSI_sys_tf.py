import os
gpu_num = 1 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
print("Current directory:", os.getcwd())
import sys
sys.path.append("./")
import sionna
# Load the required sionna components
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

import tensorflow as tf
from keras import Model
from keras.layers import Layer, LayerNormalization, Conv2D, Conv2DTranspose, Dense, LSTM, Bidirectional

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
train_log_path = "./comcloak/training/training_log/train_log_CSI3.txt"
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
class ResidualBlock(Layer):
    def __init__(self, num_ofdm_symbols, fft_size, num_conv_channels):
        super(ResidualBlock, self).__init__()
        self.norm1 = LayerNormalization([fft_size, num_ofdm_symbols, num_conv_channels])  # LayerNorm for spatial dims
        self.conv1 = Conv2D(num_conv_channels, num_conv_channels, kernel_size=3, padding=1)
        self.norm2 = LayerNormalization([fft_size, num_ofdm_symbols, num_conv_channels])
        self.conv2 = Conv2D(num_conv_channels, num_conv_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        # LayerNorm expects [B, C, H, W], normalize over C
        z = self.norm1(x)
        z = tf.nn.relu(z)
        z = self.conv1(z)
        z = self.norm2(z)
        z = tf.nn.relu(z)
        z = self.conv2(z)
        return z + residual
    
class CompressBlock(Model):
    def __init__(self, num_ofdm_symbols, fft_size, num_conv_channels, compressed_bits):
        super().__init__()
        self.H = ceil(num_ofdm_symbols / 2)
        self.W = ceil(fft_size / 2)
        self.C = num_conv_channels
        self.conv = Conv2D(self.C, kernel_size=4, stride=2, padding=1)
        self.compressed_layer = Dense(compressed_bits)
    def forward(self, x):
        # LayerNorm expects [B, C, H, W], normalize over C
        z = self.conv(x)
        z = tf.nn.relu(z)
        # z = self.conv(z)
        # z = F.relu(z)
        # Flatten the output to feed it to the compressed layer
        z = tf.reshape(z,[z.shape[0], -1])
        z_compressed = self.compressed_layer(z)
        return z_compressed

class ExpandBlock(Model):
    def __init__(self, num_ofdm_symbols, fft_size, num_conv_channels):
        super().__init__()
        self.H = ceil(num_ofdm_symbols / 2)
        self.W = ceil(fft_size / 2)
        self.C = num_conv_channels
        self.dense = Dense(self.H * self.W * self.C)
        self.deconv = Conv2DTranspose(self.C, kernel_size=4, strides=2, padding='same', activation=None)

    def call(self, x):
        x = self.dense(x)  # [B, H*W*C]
        x = tf.reshape(x, [-1, self.H, self.W, self.C])  # NHWC
        return self.deconv(x)  # [B, H*2, W*2, C]       

class CSI_Network(Model):
    def __init__(self):
        super().__init__()
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.num_conv_channels = num_conv_channels
        self.lstm = Bidirectional(
            LSTM(units=lstm_hidden_dim_factor * fft_size, return_sequences=True),
            merge_mode='concat'
        )
        self.input_conv = Conv2D(num_conv_channels, 3, padding='same', activation='relu')

        self.res_blocks_pre = [ResidualBlock(num_conv_channels) for _ in range(3)]
        self.compressed_layer = Dense(compressed_bits)
        self.expand_layer = Dense(num_conv_channels * num_ofdm_symbols * fft_size)
        self.res_blocks_post = [ResidualBlock(num_conv_channels) for _ in range(3)]

        self.output_conv = Conv2D(2 * NUM_BS_ANT, 3, padding='same')

    def call(self, inputs):  # y, no
        y, no = inputs  # y: [B, S, F, Nr_ant], no: [B]

        y_real = tf.math.real(y)
        y_imag = tf.math.imag(y)
        no = tf.math.log(no + 1e-9) / tf.math.log(tf.constant(10.))
        no = tf.reshape(no, [-1, 1, 1, 1])
        no = tf.tile(no, [1, self.num_ofdm_symbols, self.fft_size, 1])
        z = tf.concat([y_real, y_imag, no], axis=-1)  # [B, S, F, 2*Nr_ant+1]

        z = tf.reshape(z, [-1, self.num_ofdm_symbols, self.fft_size * z.shape[-1]])
        z = self.lstm(z)
        z = tf.reshape(z, [-1, self.num_ofdm_symbols, self.fft_size, -1])
        z = self.input_conv(z)

        for block in self.res_blocks_pre:
            z = block(z)

        z = tf.reshape(z, [z.shape[0], -1])  # flatten
        z_compressed = self.compressed_layer(z)
        z = self.expand_layer(z_compressed)
        z = tf.reshape(z, [-1, self.num_ofdm_symbols, self.fft_size, self.num_conv_channels])

        for block in self.res_blocks_post:
            z = block(z)

        out = self.output_conv(z)  # [B, S, F, 2*Nr_ant]
        return out

class E2ESystem(Model):
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
            ebno_db = tf.fill([batch_size], ebno_db)

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
        no_ = expand_to_rank(no, tf.rank(x_rg))
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
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "CSI-sys":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
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
            h = tf.squeeze(h, axis=1)
            h = tf.transpose(h, perm=[0, 2, 3, 1])
            h_real = tf.math.real(h)
            h_imag = tf.math.imag(h)
            h_ri = tf.concat([h_real, h_imag], axis=-1)
            #complex?
            # bce = F.binary_cross_entropy_with_logits(llr, h_ri, reduction='none')
            # bce = torch.mean(bce)
            # rate = torch.tensor(1.0, dtype=torch.float32) - bce/torch.math.log(2.)
            # return rate
            # mean = h_ri.mean(dim=[0,1,2,3], keepdim=True)
            # std = h_ri.std(dim=[0,1,2,3], keepdim=True)
            # H_norm = (h_ri - mean) / (std + 1e-6)
            # H_hat_norm = (h_hat - mean) / (std + 1e-6)
            denom = tf.reduce_mean(tf.abs(h_ri)**2) + 1e-8
            loss = tf.reduce_mean(tf.abs(h_ri - h_hat)**2) / denom
            # loss = F.mse_loss(H_hat_norm, H_norm)
            # loss = torch.nn.functional.smooth_l1_loss(h_hat, h_ri)
            # loss = torch.mean(torch.abs(h_ri - h_hat)**2)

            if loss > 1e4:
                np.save(h_hat.detach().cpu().numpy(), f"./comcloak/training/save_tensors/h_hat_step{i}.pt")
                np.save(h_ri.detach().cpu().numpy(), f"./comcloak/training/save_tensors/h_ri_step{i}.pt")

            return loss
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
        

# # The end-to-end system equipped with the neural receiver is instantiated for training.
# # When called, it therefore returns the estimated BMD rate
model = E2ESystem('CSI-sys', training=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for i in range(num_training_iterations):
    # Sampling a batch of SNRs
    ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model(training_batch_size, ebno_db)
        # Tensorflow optimizers only know how to minimize loss function.
        # Therefore, a loss function is defined as the additive inverse of the BMD rate
    # Computing and applying gradients
    weights = model.trainable_weights
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    # Periodically printing the progress
    if i % 10 == 0:
        print('Iteration {}/{}  Loss: {:.4f} bit'.format(i, num_training_iterations, loss.numpy()), end='\r')
        with open(train_log_path, "a") as f:
            f.write(f"{i},{ebno_db:.2f},{loss.item():.6f}\n")
# Save the weights in a file
weights = model.get_weights()
with open(model_weights_path, 'wb') as f:
    pickle.dump(weights, f)



