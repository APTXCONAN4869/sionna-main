# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the OFDM Modulator"""

import torch
import torch.nn as nn
import torch.fft
from numpy.fft import ifftshift
import torch

def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last `num_dims` dimensions of a `tensor`.
    It is a simplified version of the function `flatten_dims`.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of `tensor`.

    Returns:
        A tensor of the same type as `tensor` with `num_dims`-1 lesser
        dimensions, but the same number of elements.
    """
    assert num_dims >= 2, "`num_dims` must be >= 2"
    assert num_dims <= len(tensor.shape), "`num_dims` must <= rank(`tensor`)"

    if num_dims == len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tensor.shape
        last_dim = torch.prod(torch.tensor(shape[-num_dims:]))
        new_shape = list(shape[:-num_dims]) + [last_dim.item()]

    return tensor.reshape(new_shape)

def ifft(tensor, axis=-1):
    r"""Computes the normalized IDFT along a specified axis.

    This operation computes the normalized one-dimensional discrete inverse
    Fourier transform (IDFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{X}\in\mathbb{C}^N`, the IDFT
    :math:`\mathbf{x}\in\mathbb{C}^N` is computed as

    .. math::
        x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
            j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.

    Input
    -----
    tensor : torch.Tensor
        Tensor of arbitrary shape.

    axis : int
        Indicates the dimension along which the IDFT is taken.

    Output
    ------
    : torch.Tensor
        Tensor of the same dtype and shape as ``tensor``.
    """
    fft_size = torch.tensor(tensor.shape[axis], dtype=tensor.dtype)
    scale = torch.sqrt(fft_size)

    if axis not in [-1, tensor.ndim]:
        output = torch.fft.ifft(torch.swapaxes(tensor, axis, -1))
        output = torch.swapaxes(output, axis, -1)
    else:
        output = torch.fft.ifft(tensor)

    return scale * output


class OFDMModulator(nn.Module):
    """
    OFDMModulator(cyclic_prefix_length, **kwargs)

    Computes the time-domain representation of an OFDM resource grid
    with (optional) cyclic prefix.

    Parameters
    ----------
    cyclic_prefix_length : int
        Integer indicating the length of the
        cyclic prefix that it prepended to each OFDM symbol. It cannot
        be longer than the FFT size.

    Input
    -----
    : [...,num_ofdm_symbols,fft_size], torch.complex64
        A resource grid in the frequency domain.

    Output
    ------
    : [...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)], torch.complex64
        Time-domain OFDM signal.
    """

    def __init__(self, cyclic_prefix_length=0):
        super(OFDMModulator, self).__init__()
        self.cyclic_prefix_length = cyclic_prefix_length
        
    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value):
        assert value >= 0, "`cyclic_prefix_length` must be nonnegative."
        self._cyclic_prefix_length = value

    def forward(self, inputs):
        # Verify that cyclic prefix is not longer than the FFT size.
        fft_size = inputs.shape[-1]
        # print("cyclic_prefix_length:",self.cyclic_prefix_length)
        # print("fft_size:",fft_size)
        assert self.cyclic_prefix_length <= fft_size, \
            "shape(inputs)[-1] must not be smaller than `cylic_prefix_length`"
        # Shift DC subcarrier to first position
        # print(inputs)# tensor
        # print("inputs:",inputs)
        inputs = ifftshift(inputs, axes=-1)
        # Compute IFFT along the last dimension
        x = ifft(torch.tensor(inputs))
        # Obtain cyclic prefix
        cp = x[..., inputs.shape[-1]-self._cyclic_prefix_length:]
        # print("inputs after ifftshift:",inputs)# array
        # print("x after ifft:",x)
        # print("cp:", cp)
        # Prepend cyclic prefix
        x = torch.cat([cp, x], dim=-1)

        # Serialize last two dimensions
        # print(x.dim())
        x = flatten_last_dims(x, 2)
        # print("x after flatten:",x)

        return x
