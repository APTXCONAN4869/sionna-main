import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 PyTorch 的输出
output_matrix_sqrt_inv_pt = np.load(os.path.join(current_dir, 'matrix_sqrt_inv_pt.npy'))
# 读取 TensorFlow 的输出
output_matrix_sqrt_inv_tf = np.load(os.path.join(current_dir, 'matrix_sqrt_inv_tf.npy'))
# 比较输出
matrix_sqrt_inv_equal = np.array_equal(output_matrix_sqrt_inv_pt, output_matrix_sqrt_inv_tf)

print("matrix_sqrt_inv Test Passed:", matrix_sqrt_inv_equal)

if matrix_sqrt_inv_equal != True:
    print("pytorch output:\n", output_matrix_sqrt_inv_pt)
    print("tensorflow output:\n", output_matrix_sqrt_inv_tf)