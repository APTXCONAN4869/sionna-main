import socket
import numpy as np
import struct
def rx_read_adc_data(sock, adc_length):
    # 1. 接收 adc_length * 64 字节
    total_bytes = adc_length * 64
    buf = b''
    while len(buf) < total_bytes:
        buf += sock.recv(total_bytes - len(buf))

    # 转成 numpy array
    data = np.frombuffer(buf, dtype=np.uint8)

    # 2. reshape 成 (4, adc_length*16)
    data = data.reshape(4, adc_length * 16)

    # 3. 取 I 和 Q
    I_bytes = data[0:2, :].reshape(-1)
    Q_bytes = data[2:4, :].reshape(-1)

    # 4. 小端解码成 int16
    I = np.frombuffer(I_bytes.tobytes(), dtype='<i2')  # <i2: little-endian int16
    Q = np.frombuffer(Q_bytes.tobytes(), dtype='<i2')

    # 5. 转成复数
    adc_complex = I.astype(np.float64) + 1j * Q.astype(np.float64)

    # 6. reshape 成 (adc_length, 16)
    adc_complex = adc_complex.reshape(adc_length, 16)

    return adc_complex


def tx_write_dac_data(sock, dac_iq_data, dac_length):
    # ========== 1. Send header ==========
    frame_head = b'\x12\x34'
    pkg_type = b'\x05'
    
    addr = struct.pack('<I', 0)
    length_bytes = struct.pack('<I', dac_length)

    header = frame_head + pkg_type + addr + length_bytes
    sock.sendall(header)

    # FPGA 会返回 4 字节确认
    sock.recv(4)

    # ========== 2. 构造波形数据（核心）==========
    # dac_iq_data shape = (dac_length, 16)
    # 每个元素为 complex

    # Allocate buffer
    out = bytearray(64 * dac_length)

    idx = 0
    for i in range(dac_length):
        for ch in range(16):
            I = np.int16(np.real(dac_iq_data[i, ch]))
            Q = np.int16(np.imag(dac_iq_data[i, ch]))

            out[idx:idx+2] = struct.pack('<h', I)
            out[idx+2:idx+4] = struct.pack('<h', Q)
            idx += 4

    # ========== 3. 发送波形 ==========
    sock.sendall(out)

    # 再次读回 FPGA 对波形写入的 ACK
    sock.recv(4)
       

def send_and_recv(sock, data, resp_len=4):
    sock.sendall(data)
    return sock.recv(resp_len)


# ========== 1. 设置 ADC 采样长度 ==========
def rx_set_adc_length(sock, adc_length):
    head = b'\x12\x34'
    pkg_type = b'\x01'              # ADC length
    length_bytes = struct.pack('<I', adc_length)  # uint32, little-endian
    zero_pad = b'\x00' * 4

    data = head + pkg_type + length_bytes + zero_pad
    return send_and_recv(sock, data)


# ========== 2. 触发 ADC 采集 ==========
def rx_set_adc_trigger(sock):
    head = b'\x12\x34'
    pkg_type = b'\x03'
    zero_pad = b'\x00' * 8

    data = head + pkg_type + zero_pad
    return send_and_recv(sock, data)


# ========== 3. 设置 DAC 波形长度 ==========
def tx_set_dac_length(sock, dac_length):
    head = b'\x12\x34'
    pkg_type = b'\x02'   # DAC length
    length_bytes = struct.pack('<I', dac_length)
    zero_pad = b'\x00' * 4

    data = head + pkg_type + length_bytes + zero_pad
    return send_and_recv(sock, data)


# ========== 4. 启动 DAC 输出 ==========
def tx_set_dac_start(sock):
    head = b'\x12\x34'
    pkg_type = b'\x04'
    zero_pad = b'\x00' * 8

    data = head + pkg_type + zero_pad
    return send_and_recv(sock, data)
