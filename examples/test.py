import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 生成示例数据
snr_db = np.linspace(0, 10, 100)  # SNR 从 0 到 10 dB
ber = np.exp(-snr_db / 2)  # 假设的 BER 曲线

# 设置画布
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=18)
ax.set_ylabel("BER", fontsize=18)
ax.set_title("Dynamic BER Curve", fontsize=20)
ax.set_yscale("log")  # 以对数形式显示 BER
ax.set_xlim(0, 10)
ax.set_ylim(1e-5, 1)  # 设置 Y 轴范围
ax.grid(which="both")

# 初始化空曲线
line, = ax.plot([], [], "b-", linewidth=2, label="BER")  # 蓝色折线
ax.legend(fontsize=15)

# 初始化函数
def init():
    line.set_data([], [])
    return line,

# 更新函数：每帧绘制前 i 个点
def update(i):
    line.set_data(snr_db[:i], ber[:i])  # 逐步增加数据点
    return line,

# 创建动画，每 100ms 更新一次
ani = animation.FuncAnimation(fig, update, frames=len(snr_db), init_func=init, interval=100, blit=True, repeat=False)

plt.show()

import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Current directory:", os.getcwd())
try:
    import comcloak
except ImportError as e:
    import sys
    sys.path.append("../")

# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    # os.system("pip install sionna")
    import sionna as sn

# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np
 
# also try %matplotlib widget

import matplotlib.pyplot as plt

# for performance measurements 
import time

# For the implementation of the Keras models
from tensorflow.keras import Model

NUM_BITS_PER_SYMBOL = 2 # QPSK
class UncodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, block_length):
        """
        A keras model of an uncoded transmission over the AWGN channel.

        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.

        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).

        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.

        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.

        Output
        ------
        (bits, llr):
            Tuple:

        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.

        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """

        super().__init__() # Must call the Keras model initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()

    # @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)

EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel

# ber_plots = sn.utils.PlotBER("AWGN")
