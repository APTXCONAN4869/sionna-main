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
from sionna.utils.metrics import compute_ber, BitwiseMutualInformation
from sionna.utils import sim_ber
import tensorflow as tf
from keras import Model
from keras.layers import Layer, Conv2D, LayerNormalization, Dense, Activation

from comcloak.training.CSI_sys import CSI_Network

import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

############################################
## Training configuration
num_training_iterations = 30000 # Number of training iterations
training_batch_size = 128 # Training batch size
model_weights_path = "./comcloak/training/feedback_test/weights" # Location to save the neural receiver weights once training is done
train_log_path = "./comcloak/training/feedback_test/train_log.txt"
############################################
## Evaluation configuration
results_filename = "Evaluation_results" # Location to save the results
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
num_bits_per_symbol = [2, 4, 6] # QPSK
coderate = 0.5 # Coderate for LDPC code
modulation_scheme = ["4QAM", "16QAM", "64QAM"] # Modulation scheme
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
class E2ESystem(Model):
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        
        # self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)
        self._feedback_handler = FeedbackHandler(system, training)
        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_antenna, bs_array, "uplink", min_speed=speed)
        self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        
        if system == 'baseline-perfect-csi': # Perfect CSI
            self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
        elif system == 'baseline-ls-estimation': # LS estimation
            self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
        # Components required by both baselines
        self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
        self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        # self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        # CSI network
        self._csi_net = CSI_Network()

        # Bitwise Mutual Information metric
        self._bmi_metric = BitwiseMutualInformation()

    def mutual_info_loss(b, llr):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(b, llr) / tf.math.log(2.0)

    def __call__(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill((batch_size,), ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
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
        if self._system == 'baseline-perfect-csi':
            h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
            err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
        elif self._system == 'baseline-ls-estimation':
            h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
        CSI_feedback_tensor = self._csi_net([y, no]) # CSI feedback network
        coderate_hat, modulation_idx = self._feedback_handler(CSI_feedback_tensor)
        # Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
        n = int(resource_grid.num_data_symbols*num_bits_per_symbol[modulation_idx])
        # Number of information bits per codeword
        k = int(n*coderate_hat)
        self._encoder = LDPC5GEncoder(k, n)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
        llr = self._demapper([x_hat, no_eff_]) # Demapping
        b_hat = self._decoder(llr)
        # Compute the Bit Error Rate
        ber = compute_ber(b, b_hat)
        # Compute the Bitwise Mutual Information
        self._bmi_metric.update_state(b, llr)
        loss = -self._bmi_metric.result()
        return loss, ber, coderate_hat, modulation_idx

class FeedbackHandler(Layer):
    """FeedbackHandler(system, training=False)
    This class handles the feedback mechanism for the end-to-end system.
    """
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.coderate_hat = Dense(1, activation='sigmoid')
        self.modulation_logits = Dense(3)
        self.modulation_softmax = Activation('softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        coderate_hat = 0.25 + 0.65 * self.coderate_hat(x) # Coderate between 0.25 and 0.9
        # Modulation index is a softmax over three classes: QPSK, 16-QAM, and 64-QAM
        # The index is used to select the modulation scheme
        # from a predefined list of modulation schemes
        # The logits are passed through a softmax layer to obtain the probabilities
        modulation_idx = self.modulation_softmax(self.modulation_logits(x))
        return coderate_hat, modulation_idx


# train
model = E2ESystem('CSI-sys', training=True)

optimizer = tf.keras.optimizers.Adam()

for i in range(num_training_iterations):
    # Sampling a batch of SNRs
    ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
    # Forward pass
    with tf.GradientTape() as tape:
        loss, ber, coderate_hat, modulation_idx= model(training_batch_size, ebno_db)

    # Computing and applying gradients
    weights = model.trainable_weights
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    # Periodically printing the progress
    if i % 10 == 0:
        print('Iteration {}/{}  Loss: {:.4f}  BER: {:.4f}  Coderate: {:.4f}  Modulation_idx: {} '.format(
            i, num_training_iterations, loss.numpy(), ber.numpy(), coderate_hat.numpy(), modulation_idx.numpy()), end='\r')
        with open(train_log_path, "a") as f:
            f.write(f"{i},{ebno_db:.2f},{loss.item():.6f},{ber.item():.6f},{coderate_hat.item():.6f},{modulation_idx.numpy().tolist()}\n")

# Save the weights in a file
weights = model.get_weights()
with open(model_weights_path, 'wb') as f:
    pickle.dump(weights, f)

##############################################################################################
    
    
# # Range of SNRs over which the systems are evaluated
# ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
#                      ebno_db_max, # Max SNR for evaluation
#                      0.5) # Step

# # Dictionary storing the evaluation results
# BLER = {}

# model = E2ESystem('baseline-perfect-csi')
# _,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=5)
# BLER['baseline-perfect-csi'] = bler.numpy()

# model = E2ESystem('baseline-ls-estimation')
# _,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=5)
# BLER['baseline-ls-estimation'] = bler.numpy()

# plt.figure(figsize=(10,6))
# # Baseline - Perfect CSI
# plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# # Baseline - LS Estimation
# plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# #
# plt.xlabel(r"$E_b/N_0$ (dB)")
# plt.ylabel("BLER")
# plt.grid(which="both")
# plt.ylim((1e-4, 1.0))
# plt.legend()
# plt.tight_layout()
# plt.show() 