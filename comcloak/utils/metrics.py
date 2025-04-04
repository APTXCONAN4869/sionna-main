#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Userful metrics for the Sionna library."""


import torch

def count_errors(b, b_hat):
    """
    Counts the number of bit errors between two binary tensors.

    Input
    -----
        b : torch.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : torch.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : torch.int64
            A scalar, the number of bit errors.
    """
    errors = torch.not_equal(b, b_hat)
    errors = errors.to(torch.int64)
    return errors.sum().item()

def count_block_errors(b, b_hat):
    """
    Counts the number of block errors between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i.e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : torch.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : torch.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : torch.int64
            A scalar, the number of block errors.
    """
    errors = torch.any(torch.not_equal(b, b_hat), dim=-1)
    errors = errors.to(torch.int64)
    return errors.sum().item()

def compute_ser(s, s_hat):
    """Computes the symbol error rate (SER) between two integer tensors.

    Input
    -----
        s : tf.int
            A tensor of arbitrary shape filled with integers indicating
            the symbol indices.

        s_hat : tf.int
            A tensor of the same shape as ``s`` filled with integers indicating
            the estimated symbol indices.

    Output
    ------
        : tf.float64
            A scalar, the SER.
    """
    ser = torch.not_equal(s, s_hat)
    ser = torch.tensor(ser, dtype=torch.float64) # torch.float64 to suport large batch-sizes
    return torch.mean(ser)

def compute_ber(b, b_hat):
    """Computes the bit error rate (BER) between two binary tensors.

    Input
    -----
        b : torch.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : torch.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : torch.float64
            A scalar, the BER.
    """
    ber = torch.not_equal(b, b_hat)
    ber = torch.tensor(ber, dtype=torch.float64) # torch.float64 to suport large batch-sizes
    return torch.mean(ber)

