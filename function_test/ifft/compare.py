import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 PyTorch 的输出
output_ifft_pt = np.load(os.path.join(current_dir, 'ifft_pt.npy'))
#output_functionB_pt = np.load(os.path.join(current_dir, 'functionB_pt.npy'))

# 读取 TensorFlow 的输出
output_ifft_tf = np.load(os.path.join(current_dir, 'ifft_tf.npy'))
#output_functionB_tf = np.load(os.path.join(current_dir, 'functionB_tf.npy'))

# 比较输出
ifft_equal = np.array_equal(output_ifft_pt, output_ifft_tf)
#functionB_equal = np.array_equal(output_functionB_pt, output_functionB_tf)

print("ifft Test Passed:", ifft_equal)
#print("functionB Test Passed:", functionB_equal)

if ifft_equal != True:
    print("pytorch output:\n", output_ifft_pt)
    print("tensorflow output:\n", output_ifft_tf)
#    print("pytorch output:\n", output_functionB_pt)
#    print("tensorflow output:\n", output_functionB_tf)