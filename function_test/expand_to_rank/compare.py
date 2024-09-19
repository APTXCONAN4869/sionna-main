import numpy as np

# 读取 PyTorch 的输出
output_torch_insert_dims = np.load('output_torch_insert_dims.npy')
output_torch_expand_to_rank = np.load('output_torch_expand_to_rank.npy')

# 读取 TensorFlow 的输出
output_tf_insert_dims = np.load('output_tf_insert_dims.npy')
output_tf_expand_to_rank = np.load('output_tf_expand_to_rank.npy')

# 比较输出
insert_dims_equal = np.array_equal(output_torch_insert_dims, output_tf_insert_dims)
expand_to_rank_equal = np.array_equal(output_torch_expand_to_rank, output_tf_expand_to_rank)

print("Insert Dims Test Passed:", insert_dims_equal)
print("Expand to Rank Test Passed:", expand_to_rank_equal)
