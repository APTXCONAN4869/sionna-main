# test_realtime.py
import numpy as np
import socket
import time
import scipy.io as sio
from scipy.io import savemat
import matplotlib.pyplot as plt
from driver import *
import os
import torch
print("Current directory:", os.getcwd())
# ==========================
# 参数
# ==========================
Fs = 10e6
TxPointN = 40000
RxPointN = 10000
amp_target = 32767

# ==========================
# 加载波形
# ==========================
mat = sio.loadmat("d:/sionna-main/practical/precursor.mat")
precursor = mat["precursor"].flatten()
wave_data = torch.load('d:/sionna-main/practical/wave_data64QAM.pt')
base = wave_data.flatten().numpy()

# --- RMS 归一化对齐 ---
def rms_norm_align(signal, target_rms):
    """将 signal 缩放为指定的 RMS 值"""
    current_rms = np.sqrt(np.mean(np.abs(signal)**2))
    if current_rms == 0:
        return signal
    scale = target_rms / current_rms
    return signal * scale

# 以前导信号为参考标准
target_rms = np.sqrt(np.mean(np.abs(precursor)**2))

# 缩放 base 到与 precursor 相同的 RMS
base_scaled = rms_norm_align(base, target_rms)

T_RI = 4
precursor[-T_RI:] += base_scaled[:T_RI]
base = np.concatenate((precursor, base_scaled))
waveform16 = np.tile(base, (16,1))

# data_dict = {
#         'base': base
# }
# savemat('D:\\sionna-main\\practical\\base64QAM.mat', data_dict)

# ==========================
# ZERO PAD
# ==========================
num_ch, num_samples = waveform16.shape
waveform_pad = np.zeros((TxPointN,16), dtype=complex)

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
feedback1=tx_set_dac_length(tcp_tx, TxPointN)
feedback2=tx_write_dac_data(tcp_tx, waveform_pad, TxPointN)
feedback3=tx_set_dac_start(tcp_tx)

time.sleep(0.1)

# ==========================
# 配置 ADC
# ==========================
feedback4=rx_set_adc_length(tcp_tx, RxPointN)
feedback5=rx_set_adc_trigger(tcp_tx)

time.sleep(0.1)

# ==========================
# 初始化实时图
# # ==========================
# plt.ion()
# fig = plt.figure(figsize=(8,9))

# ax1 = fig.add_subplot(211)
# time_line, = ax1.plot(np.zeros(RxPointN))
# ax1.set_title("Realtime Rx waveform (CH1)")

# ax2 = fig.add_subplot(212)
# freq_axis = np.linspace(-Fs/2, Fs/2, RxPointN)
# freq_line, = ax2.plot(freq_axis, np.zeros(RxPointN))
# ax2.set_title("Realtime Rx spectrum (CH1)")

# plt.tight_layout()

# print("开始实时采集 Ctrl+C 停止…")

# ==========================
# 实时采集
# ==========================
count = 0
while True:
    data = rx_read_adc_data(tcp_rx, RxPointN)
    count += 1
    print(f"Capture count: {count}")
    if count ==10:
        np.save("self_loop_data64QAM.npy", data)

    ch1 = data[:,0]
    np.save("64QAMch1.npy", ch1)
    # time_line.set_ydata(ch1)

    X = np.fft.fftshift(np.fft.fft(ch1))
    # freq_line.set_ydata(np.abs(X))
    plt.pause(0.01)
 