import tensorflow as tf
import numpy as np
import os

'''
input =[ [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]],
 
         [[[7, 7, 7], [8, 8, 8]],
         [[9, 9, 9], [10, 10, 10]],
         [[11, 11, 11], [12, 12, 12]]],
 
        [[[13, 13, 13], [14, 14, 14]],
         [[15, 15, 15], [16, 16, 16]],
         [[17, 17, 17], [18, 18, 18]]]
         ]
tf.convert_to_tensor(input)
print(tf.shape(input))
output=tf.gather(input, tf.convert_to_tensor([[0,2],[0,1]]), axis=0)
print(output)
'''


def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last ``num_dims`` dimensions of a ``tensor``.
    It is a simplified version of the function ``flatten_dims``.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of ``tensor``.

    Returns:
        A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
        dimensions, but the same number of elements.
    """
    msg = "`num_dims` must be >= 2"
    tf.debugging.assert_greater_equal(num_dims, 2, msg)

    msg = "`num_dims` must <= rank(`tensor`)"
    tf.debugging.assert_less_equal(num_dims, tf.rank(tensor), msg)

    if num_dims==len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tf.shape(tensor)
        last_dim = tf.reduce_prod(tensor.shape[-num_dims:])
        new_shape = tf.concat([shape[:-num_dims], [last_dim]], 0)

    return tf.reshape(tensor, new_shape)

'''
num_ut = 4
num_streams_per_tx = 1
num_ofdm_symbols = 14
fft_size = 64
mask = np.zeros([num_ut, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
mask[...,[2,3,10,11],:] = True
mask = tf.cast(mask, tf.int32)
print("mask:\n",mask)
num_pilots = np.sum(mask[0,0])
print("num_pilots:\n",num_pilots)
pilots = np.zeros([num_ut, num_streams_per_tx, num_pilots])
pilots[0,0,10] = 1
pilots[0,0,234] = 1
pilots[1,0,20] = 1
pilots[2,0,70] = 1
pilots[3,0,120] = 1
print("pilots:\n",pilots)
# Precompute indices to gather received pilot signals
num_pilot_symbols = tf.shape(pilots)[-1]
print("num_pilot_symbols:\n",num_pilot_symbols)
mask = flatten_last_dims(mask)
print("mask:\n",mask)
pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
print("idex:\n",pilot_ind)
pilot_ind = pilot_ind[...,:num_pilot_symbols]
print("pilot_ind:\n",pilot_ind) 
# y_pilots = tf.gather(y_eff_flat, pilot_ind, axis=-1)
'''
