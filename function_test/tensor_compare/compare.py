import numpy as np
import os
import torch
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 PyTorch 的输出
output_pttensor = np.load(os.path.join(current_dir, 'pttensor.npy'))
# output_functionB_pt = np.load(os.path.join(current_dir, 'functionB_pt.npy'))

# 读取 TensorFlow 的输出
output_tftensor = np.load(os.path.join(current_dir, 'tftensor.npy'))
# output_functionB_tf = np.load(os.path.join(current_dir, 'functionB_tf.npy'))

# 比较输出
# functionA_equal = np.array_equal(output_pttensor, output_tftensor)
functionA_equal = np.allclose(output_pttensor,output_tftensor)
# functionB_equal = np.array_equal(output_functionB_pt, output_functionB_tf)

print("functionA Test Passed:", functionA_equal)
# print("functionB Test Passed:", functionB_equal)

if functionA_equal != True:
    print("output:\n", np.any(np.abs(output_pttensor - output_tftensor) > 1))
    output = output_pttensor - output_tftensor
    print("output:\n", output[output != 0])
    # non_zero_elements = output_pttensor[output_pttensor != 0 + 0j]
    # print("pytorch output:\n", output_pttensor)
    # non_zero_elements = output_tftensor[output_tftensor != 0 + 0j]
    # print("tensorflow output:\n", output_tftensor)
    # output_pttensor = torch.tensor(output_pttensor)
    # output_tftensor = torch.tensor(output_tftensor)
    # # result = torch.where(output_pttensor != 0, output_tftensor / output_pttensor, torch.zeros_like(output_tftensor))
    # # result = torch.nan_to_num(torch.tensor(output_pttensor) / torch.tensor(output_tftensor), nan=0.0, posinf=0.0, neginf=0.0)
    # non_zero_indices = torch.nonzero(output_pttensor)
    # print('pt:\n',non_zero_indices)
    # non_zero_indices = torch.nonzero(output_tftensor)
    # print('tf:\n',non_zero_indices)

    # print("pytorch output:\n", output_functionB_pt) 
    # print("tensorflow output:\n", output_functionB_tf)