import numpy as np
import tensorflow as tf

# 创建一个随机数组作为测试输入
input_array = np.random.randn(10, 10)

# 使用 numpy 的 ifftshift
numpy_output = np.fft.fftshift(input_array, axes=-1)

# 使用 TensorFlow 的 ifftshift
# 需要将输入转换为 TensorFlow 张量
tf_input_array = tf.convert_to_tensor(input_array, dtype=tf.float32)
tf_output = tf.signal.fftshift(tf_input_array, axes=-1)

# 将 TensorFlow 的输出转换为 numpy 数组
tf_output_array = tf_output.numpy()

# 比较两个输出是否相同
are_equal = np.allclose(numpy_output, tf_output_array)

print(f"Outputs are equal: {are_equal}")
