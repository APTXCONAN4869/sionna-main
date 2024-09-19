import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 PyTorch 的输出
output_matrix_inv_pt = np.load(os.path.join(current_dir, 'matrix_inv_pt.npy'))
# output_functionB_pt = np.load(os.path.join(current_dir, 'functionB_pt.npy'))

# 读取 TensorFlow 的输出
output_matrix_inv_tf = np.load(os.path.join(current_dir, 'matrix_inv_tf.npy'))
# output_functionB_tf = np.load(os.path.join(current_dir, 'functionB_tf.npy'))

# 比较输出

matrix_inv_equal = np.array_equal(output_matrix_inv_pt, output_matrix_inv_tf)
# functionB_equal = np.array_equal(output_functionB_pt, output_functionB_tf)

print("matrix_inv Test Passed:", matrix_inv_equal)
# print("functionB Test Passed:", functionB_equal)

if matrix_inv_equal != True:
    print("pytorch output:\n", output_matrix_inv_pt)
    print("tensorflow output:\n", output_matrix_inv_tf)