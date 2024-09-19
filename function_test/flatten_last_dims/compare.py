import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 PyTorch 的输出
output_flatten_last_dims_pt = np.load(os.path.join(current_dir, 'flatten_last_dims_pt.npy'))
# output_functionB_pt = np.load(os.path.join(current_dir, 'functionB_pt.npy'))

# 读取 TensorFlow 的输出
output_flatten_last_dims_tf = np.load(os.path.join(current_dir, 'flatten_last_dims_tf.npy'))
# output_functionB_tf = np.load(os.path.join(current_dir, 'functionB_tf.npy'))

# 比较输出
flatten_last_dims_equal = np.array_equal(output_flatten_last_dims_pt, output_flatten_last_dims_tf)
# functionB_equal = np.array_equal(output_functionB_pt, output_functionB_tf)

print("flatten_last_dims Test Passed:", flatten_last_dims_equal)
# print("functionB Test Passed:", functionB_equal)
