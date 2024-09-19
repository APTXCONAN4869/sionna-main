import numpy as np
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 PyTorch 的输出
output_functionA_pt = np.load(os.path.join(current_dir, 'functionA_pt.npy'))
output_functionB_pt = np.load(os.path.join(current_dir, 'functionB_pt.npy'))

# 读取 TensorFlow 的输出
output_functionA_tf = np.load(os.path.join(current_dir, 'functionA_tf.npy'))
output_functionB_tf = np.load(os.path.join(current_dir, 'functionB_tf.npy'))

# 比较输出
functionA_equal = np.array_equal(output_functionA_pt, output_functionA_tf)
functionB_equal = np.array_equal(output_functionB_pt, output_functionB_tf)

print("functionA Test Passed:", functionA_equal)
print("functionB Test Passed:", functionB_equal)

if functionB_equal != True:
    print("pytorch output:\n", output_functionA_pt)
    print("tensorflow output:\n", output_functionA_tf)
    print("pytorch output:\n", output_functionB_pt)
    print("tensorflow output:\n", output_functionB_tf)