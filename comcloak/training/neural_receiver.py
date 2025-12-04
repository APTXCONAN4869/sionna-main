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
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, TDL
from sionna.channel import OFDMChannel, ApplyOFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
    LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper, ZFPrecoder
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, expand_to_rank, log10
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

import tensorflow as tf
from keras import Model
from keras.layers import Layer, LayerNormalization, Conv2D, Conv2DTranspose, Dense, LSTM, Bidirectional, Embedding
from tensorflow import nn
from math import ceil

from comcloak.training.time_consuming_channel import ChannelMatrix
############################################
## Channel configuration
num_ut = 1
num_bs = 1
num_ut_ant = 8
num_bs_ant = 16
num_streams_per_tx = 4
rx_tx_association = np.array([[1]])

carrier_frequency = 2.6e9 # Hz
delay_spread = 300e-9 # s
cdl_model = "B" # CDL model to use
tdl_model = "B" # TDL model to use
speed = 1.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = -15.0
ebno_db_max = 25.0

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
max_num_bits_per_symbol = 6 # Maximum number of bits per symbol among all MCSs
mcs_list = ["QPSK", "16QAM", "64QAM"] # List of MCS
coderate = 0.5 # Coderate for LDPC code
mcs_dict_order = {
    "QPSK": 2, 
    "16QAM": 4,
    "64QAM": 6
}
mcs_dict = {
    "QPSK": 0, 
    "16QAM": 1,
    "64QAM": 2
}
############################################
## Neural receiver configuration
num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver
compressed_bits = 1024
num_lstm_layers = 2
lstm_hidden_dim_factor = 8
num_experts = 4 # Number of convolutional kernel experts for the CondConv2D layers
############################################
## Training configuration
num_training_iterations = 10000 # Number of training iterations
training_batch_size = 300 # Training batch size
batch_size_mcs = training_batch_size // len(mcs_list)
mcs_id = []
for m in mcs_list:
    mcs_id.extend([mcs_dict[m]] * batch_size_mcs)
mcs_id = tf.constant(mcs_id, dtype=tf.int32)

model_weights_path = "./comcloak/training/Receiver_sys_weights" # Location to save the neural receiver weights once training is done
train_log_path = "./comcloak/training/train_log/nrx_log/Receiver_log_v0.txt"
############################################
## Evaluation configuration
results_filename = "Receiver_sys_results" # Location to save the results
############################################


stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                  1)               # One stream per transmitter
resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = num_streams_per_tx,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)

ut_array = AntennaArray(num_rows=1,
                        num_cols=int(num_ut_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
bs_array = AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

class FiLM(Layer):
    def __init__(self, feature_dim, cond_dim):
        super(FiLM, self).__init__()
        # Conditional network：input condition -> gamma, beta
        self.fc_gamma = Dense(feature_dim)
        self.fc_beta = Dense(feature_dim)
    
    def call(self, x, cond):
        # cond: [batch, cond_dim]
        gamma = self.fc_gamma(cond)  # [batch, feature_dim]
        beta = self.fc_beta(cond)    # [batch, feature_dim]
        
        # broadcast to x shape
        gamma = tf.expand_dims(gamma, 1)  
        beta = tf.expand_dims(beta, 1)
        
        return gamma * x + beta

class CondConv2D(Layer):
    def __init__(self, filters, kernel_size, num_experts, cond_dim, strides=1, padding="same"):
        super(CondConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else (kernel_size, kernel_size)
        self.num_experts = num_experts
        self.cond_dim = cond_dim
        self.strides = strides
        self.padding = padding.upper()  # "SAME" or "VALID"

        # K kernel experts every single one trainable
        # self.expert_kernels = self.add_weight(
        #     shape=(num_experts, self.kernel_size[0], self.kernel_size[1], None, self.filters),
        #     initializer="glorot_uniform",
        #     trainable=True,
        #     name="expert_kernels"
        # )
        # self.expert_bias = self.add_weight(
        #     shape=(num_experts, self.filters),
        #     initializer="zeros",
        #     trainable=True,
        #     name="expert_bias"
        # )

        # Conditional network -> α (softmax to ensure sum to 1)
        self.alpha_layer = Dense(num_experts, activation="softmax")
    
    def build(self, input_shape):
        in_channels = input_shape[-1]
        # Now we know the number of input channels, we can fix the shape of the expert convolution kernels
        self.expert_kernels = self.add_weight(
            shape=(self.num_experts, self.kernel_size[0], self.kernel_size[1], in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="expert_kernels"
        )
        self.expert_bias = self.add_weight(
            shape=(self.num_experts, self.filters),
            initializer="zeros",
            trainable=True,
            name="expert_bias"
        )
    
    def call(self, inputs, cond):
        """
        inputs: [batch, H, W, C_in]
        cond:   [batch, cond_dim]
        """
        batch_size = tf.shape(inputs)[0]

        # α coefficients (batch, K)
        alphas = self.alpha_layer(cond)

        # Mixture of convolution kernels (batch, kh, kw, Cin, Cout)
        kernels = tf.einsum("bk,k...->b...", alphas, self.expert_kernels)
        bias = tf.einsum("bk,kf->bf", alphas, self.expert_bias)

        # Convolution for each sample in the batch
        outputs = []
        for i in range(batch_size):
            out = tf.nn.conv2d(
                inputs[i:i+1],
                kernels[i],
                strides=[1, self.strides, self.strides, 1],
                padding=self.padding
            )
            out = tf.nn.bias_add(out, bias[i])
            outputs.append(out)
        
        return tf.concat(outputs, axis=0)

class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """

    def build(self, input_shape):

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        # self._conv_1 = CondConv2D(filters=num_conv_channels, kernel_size=3, num_experts=4, cond_dim=16)
        self._conv_1 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = nn.relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = nn.relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z

class Receiver_Network(Model):
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.

    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.

    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.

    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """

    def build(self, input_shape):

        # Input convolution
        self._input_conv = Conv2D(filters=num_conv_channels,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = CondConv2D(filters=max_num_bits_per_symbol,
                                    kernel_size=[3,3],
                                    num_experts=num_experts,
                                    cond_dim=4)

    def call(self, inputs):
        y, no = inputs

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])# Feeding the noise power helps with the performance
        # z : [batch size, num ofdm symbols, num subcarriers, 2*rx_ant + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Input conv
        embedding = Embedding(input_dim=3, output_dim=8)
        # [batch size, cond_dim]
        cond_vec = embedding(mcs_id)
        z = self._input_conv(z)
        # [batch size, num ofdm symbols, num subcarriers, num conv channels]
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        # [batch_size, num_ofdm_symbols, fft_size, num_bits_per_symbol]
        z = self._output_conv(z, cond_vec)

        return z

class E2ESystem(Model):
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training

        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        # cdl = CDL(cdl_model, delay_spread, carrier_frequency,
        #           ut_array, bs_array, "downlink", min_speed=speed)
        self._channel_model = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_array, bs_array, "downlink", min_speed=speed)
        self._channel_model = TDL(tdl_model, delay_spread, carrier_frequency,
                  num_rx_ant=num_ut_ant, num_tx_ant=num_bs_ant, min_speed=speed)
        
        # self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True, precoder=True)
        self._channel = ChannelMatrix(resource_grid, training_batch_size, num_ut, num_bs)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()

        if training:
            self._num_bits_per_symbol = [mcs_dict_order[mcs] for mcs in mcs_list]
            self._n, self._k, self._encoder, self._mapper = [], [], [], []
            self._num_mcs_supported = len(mcs_list)
            # for mcs_name, num_bits_per_symbol in zip(mcs_list, self._num_bits_per_symbol):
            for num_bits_per_symbol in self._num_bits_per_symbol:
                n = int(resource_grid.num_data_symbols * num_bits_per_symbol)
                k = int(n * coderate)
                self._n.append(n)
                self._k.append(k)
                self._encoder.append(LDPC5GEncoder(k, n))
                self._mapper.append(Mapper("qam", num_bits_per_symbol))

        self._rg_mapper = ResourceGridMapper(resource_grid)
        self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager)
        self._zf_precoder = ZFPrecoder(resource_grid, stream_manager, return_effective_channel=True)
        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)
            self._demapper = Demapper("app", "qam", self._num_bits_per_symbol) #problem here
        elif system == "neural-receiver": # Neural receiver
            self._receiver_sys = Receiver_Network()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        
        self._decoder = []
        for idx in range(len(mcs_list)):
            self._decoder.append(LDPC5GDecoder(self._encoder[idx], hard_out=True))

    def __call__(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size_mcs], ebno_db)

        ######################################
        ## Transmitter
        x_mcs, no_mcs, b_mcs, c_mcs = [], [], [], []
        for idx in range(len(mcs_list)):
            no_mcs.append(ebnodb2no(ebno_db, self._num_bits_per_symbol[idx], coderate))
            #     c = self._binary_source([batch_size, 1, 1, n])
            b_mcs.append(self._binary_source([batch_size_mcs, 1, num_streams_per_tx, self._k[idx]]))
            c_mcs.append(self._encoder[idx](b_mcs[idx]))# [batch_size_mcs, 1, num_streams_per_tx, self._n[idx]]
            # Modulation
            x_mcs.append(self._mapper[idx](c_mcs[idx])) # [batch_size_mcs, 1, num_streams_per_tx, n/Constellation.num_bits_per_symbol]
        # b = tf.concat(b_mcs, axis=0) # last dim is different for different mcs, so cannot concat
        # c = tf.concat(c_mcs, axis=0) # same here
        no = tf.concat(no_mcs, axis=0) # [batch size]    
        x = tf.concat(x_mcs, axis=0) # [batch size, 1, num_streams_per_tx, n/Constellation.num_bits_per_symbol]
        
        # Mapping to the resource grid
        x_rg = self._rg_mapper(x)
        
        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))

        h_freq = self._channel(self._channel_model)
        x_rg, g = self._zf_precoder([x_rg, h_freq])
        y = self._channel_freq([x_rg, h_freq, no_])
        # y shape: [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        # y, h = self._channel([x_rg, no_])

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
        elif self._system == "neural-receiver":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            # y shape: [batch_size, rx_ant, num_ofdm_symbols, fft_size]
            llrs = self._receiver_sys([y, no])# Feeding the noise power in log10 scale helps with the performance
            # llrs shape: [batch_size, num_ofdm_symbols, fft_size, num_bits_per_symbol]
            # For traditional equalizer we use demmaper because it returns datasymbols only but not here
            # So we need to use resource grid demapper to extract data-carrying REs
            llr = insert_dims(llr, 2, 1)# [batch_size, 1, 1, num_ofdm_symbols, fft_size, num_bits_per_symbol]
            # expect input: [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]
            # output: [batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]
            llr = self._rg_demapper(llrs)

            llrs_ = []
            for idx in range(self._num_mcs_supported):
                llrs_mcs = tf.gather(
                                llrs,
                                indices=tf.range(self._num_bits_per_symbol[idx]),
                                axis=-1)
                llrs_.append(llrs_mcs)
            # llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            # llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            # llr = torch.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c_mcs, llr)
            bce = tf.reduce_mean(bce)
            loss = -tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            # loss = F.mse_loss(H_hat_norm, H_norm)
            # loss = torch.nn.functional.smooth_l1_loss(h_hat, h_ri)
            # loss = torch.mean(torch.abs(h_ri - h_hat)**2)

            return loss
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b_mcs,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
      
# # The end-to-end system equipped with the neural receiver is instantiated for training.
# # When called, it therefore returns the estimated BMD rate
model = E2ESystem('neural-receiver', training=True)
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