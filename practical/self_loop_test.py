# test_realtime.py
import numpy as np
import socket
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from driver import *

# ==========================
# 参数
# ==========================
Fs = 10e6
TxPointN = 40000
RxPointN = 5000
amp_target = 32767

# ==========================
# 加载波形
# ==========================
mat = sio.loadmat("Re_Signal_Fs.mat")
base = mat["Re_Signal_Fs"].flatten()
waveform16 = np.tile(base, (16,1))

# ==========================
# ZERO PAD
# ==========================
num_ch, num_samples = waveform16.shape
waveform_pad = np.zeros((TxPointN,16))

copy_len = min(num_samples, TxPointN)
waveform_pad[:copy_len, :] = waveform16[:, :copy_len].T

# ==========================
# 归一化 ±32767
# ==========================
max_val = np.max(np.abs(waveform_pad))
scale = amp_target / max_val
waveform_pad = waveform_pad * scale

# ==========================
# 建 TCP
# ==========================
def connect(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    return s

tcp_tx = connect("192.168.200.11", 2333)
tcp_rx = connect("192.168.200.11", 2334)

# ==========================
# DAC 发送
# ==========================
tx_set_dac_length(tcp_tx, TxPointN)
tx_write_dac_data(tcp_tx, waveform_pad, TxPointN)
tx_set_dac_start(tcp_tx)

time.sleep(0.1)

# ==========================
# 配置 ADC
# ==========================
rx_set_adc_length(tcp_tx, RxPointN)
rx_set_adc_trigger(tcp_tx)

time.sleep(0.1)

# ==========================
# 初始化实时图
# ==========================
plt.ion()
fig = plt.figure(figsize=(8,9))

ax1 = fig.add_subplot(211)
time_line, = ax1.plot(np.zeros(RxPointN))
ax1.set_title("实时 Rx 波形(CH1)")

ax2 = fig.add_subplot(212)
freq_axis = np.linspace(-Fs/2, Fs/2, RxPointN)
freq_line, = ax2.plot(freq_axis, np.zeros(RxPointN))
ax2.set_title("实时 Rx 频谱(CH1)")

plt.tight_layout()

print("开始实时采集 Ctrl+C 停止…")

# ==========================
# 实时采集
# ==========================
while True:
    data = rx_read_adc_data(tcp_rx, RxPointN)

    ch1 = np.real(data[:,0])

    time_line.set_ydata(ch1)

    X = np.fft.fftshift(np.fft.fft(ch1))
    freq_line.set_ydata(np.abs(X))

    plt.pause(0.01)
