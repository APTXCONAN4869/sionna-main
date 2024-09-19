#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Miscellaneous utility functions of the Sionna package."""
import torch
import torch.nn as nn
import numpy as np
# from comcloak.utils.metrics import count_errors, count_block_errors
from .metrics import count_errors, count_block_errors
from comcloak.supplememt import get_real_dtype
from comcloak.mapping import Mapper, Constellation
import time
# from sionna import signal

class BinarySource(nn.Module):

    def __init__(self, dtype=torch.float32, seed=None, **kwargs):
        super().__init__()
        self._dtype = dtype
        self._seed = seed
        self._rng = None
        if self._seed is not None:
            self._rng = torch.Generator().manual_seed(self._seed)

    def forward(self, inputs):
        if self._seed is not None:
            return torch.randint(0, 2, size = inputs.tolist(), generator=self._rng, dtype=torch.int32).to(self._dtype)
        else:
            # return torch.randint(0, 2, size = inputs.tolist(), dtype=torch.int32).to(self._dtype)
            # 设置随机数生成器
            rng = np.random.default_rng(seed=12345)  # 你可以根据需要设置种子

            # 使用 randint 生成随机整数
            random_integers = rng.integers(low=0, high=2, size=inputs.tolist(), dtype=np.int32)

            # 转换数据类型
            result = random_integers.astype(np.float32)  # self._dtype 在此示例中假设为 float32
            return torch.tensor(result,  dtype=self._dtype)
        
class SymbolSource(nn.Module):
    r"""SymbolSource(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, return_bits=False, seed=None, dtype=torch.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random constellation symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [torch.complex64, torch.complex128], torch.DType
        The output dtype. Defaults to torch.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, torch.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], torch.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """

    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=torch.complex64,
                 **kwargs
                ):
        super().__init__()
        constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype)
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(seed=seed, dtype=get_real_dtype(dtype))  # Changed from dtype.real to torch.float32
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              dtype=dtype)

    def forward(self, inputs):
        shape =  torch.cat((torch.tensor(inputs), torch.tensor([self._num_bits_per_symbol])))
        b = self._binary_source(shape.to(torch.int32))
        # print(b.shape)
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)
        # print(x.shape)
        result = torch.squeeze(x, -1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            result.append(torch.squeeze(ind, -1))
        if self._return_bits:
            result.append(b)

        return result

class QAMSource(SymbolSource):
    r"""QAMSource(num_bits_per_symbol=None, return_indices=False, return_bits=False, seed=None, dtype=torch.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random QAM symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [torch.complex64, torch.complex128], torch.DType
        The output dtype. Defaults to torch.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random QAM symbols.

    symbol_indices : ``shape``, torch.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], torch.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=torch.complex64,
                 **kwargs
                ):
        super().__init__(constellation_type="qam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         dtype=dtype,
                         **kwargs)

##############################
def complex_normal(shape, var=1.0, dtype=torch.complex64):
    r"""Generates a tensor of complex normal random variables.

    Input
    -----
    shape : tuple or list
        The desired shape.

    var : float
        The total variance, i.e., each complex dimension has
        variance ``var/2``.

    dtype: torch.dtype
        The desired dtype. Defaults to `torch.complex64`.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor of complex normal random variables.
    """
    # Half the variance for each dimension
    var_dim = var / 2
    stddev = torch.sqrt(torch.tensor(var_dim, dtype=get_real_dtype(dtype)))

    # Generate complex Gaussian noise with the right variance
    xr = torch.normal(mean=0, std=stddev, size=shape, dtype=get_real_dtype(dtype))
    xi = torch.normal(mean=0, std=stddev, size=shape, dtype=get_real_dtype(dtype))
    x = torch.complex(xr, xi)

    return x

def ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid=None):
    r"""Compute the noise variance `No` for a given `Eb/No` in dB.

    The function takes into account the number of coded bits per constellation
    symbol, the coderate, as well as possible additional overheads related to
    OFDM transmissions, such as the cyclic prefix and pilots.

    The value of `No` is computed according to the following expression

    .. math::
        N_o = \left(\frac{E_b}{N_o} \frac{r M}{E_s}\right)^{-1}

    where :math:`2^M` is the constellation size, i.e., :math:`M` is the
    average number of coded bits per constellation symbol,
    :math:`E_s=1` is the average energy per constellation per symbol,
    :math:`r\in(0,1]` is the coderate,
    :math:`E_b` is the energy per information bit,
    and :math:`N_o` is the noise power spectral density.
    For OFDM transmissions, :math:`E_s` is scaled
    according to the ratio between the total number of resource elements in
    a resource grid with non-zero energy and the number
    of resource elements used for data transmission. Also the additionally
    transmitted energy during the cyclic prefix is taken into account, as
    well as the number of transmitted streams per transmitter.

    Input
    -----
    ebno_db : float
        The `Eb/No` value in dB.

    num_bits_per_symbol : int
        The number of bits per symbol.

    coderate : float
        The coderate used.

    resource_grid : ResourceGrid
        An (optional) instance of :class:`~sionna.ofdm.ResourceGrid`
        for OFDM transmissions.

    Output
    ------
    : float
        The value of :math:`N_o` in linear scale.
    """

    if torch.is_tensor(ebno_db):
        dtype = ebno_db.dtype
    else:
        dtype = torch.float32

    ebno = torch.pow(torch.tensor(10.0, dtype=dtype), ebno_db / 10.0)

    energy_per_symbol = 1.0
    if resource_grid is not None:
        # Divide energy per symbol by the number of transmitted streams
        energy_per_symbol /= resource_grid.num_streams_per_tx

        # Number of nonzero energy symbols.
        # We do not account for the nulled DC and guard carriers.
        cp_overhead = resource_grid.cyclic_prefix_length / resource_grid.fft_size
        num_syms = resource_grid.num_ofdm_symbols * (1 + cp_overhead) \
                    * resource_grid.num_effective_subcarriers
        energy_per_symbol *= num_syms / resource_grid.num_data_symbols

    no = 1.0 / (ebno * coderate * num_bits_per_symbol / energy_per_symbol)

    return no

import torch

def hard_decisions(llr):
    """Transforms LLRs into hard decisions.

    Positive values are mapped to `1`.
    Nonpositive values are mapped to `0`.

    Input
    -----
    llr : any non-complex torch.dtype
        Tensor of LLRs.

    Output
    ------
    : Same shape and dtype as ``llr``
        The hard decisions.
    """
    zero = torch.tensor(0, dtype=llr.dtype, device=llr.device)

    return (llr > zero).to(dtype=llr.dtype)
