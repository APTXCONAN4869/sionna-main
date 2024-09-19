# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class for creating a CIR sampler, usable as a channel model, from a CIR generator"""

import torch
from torch.utils.data import Dataset, DataLoader

class CIRDataset(Dataset):
    """CIRDataset(cir_generator, batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps, dtype=torch.complex64)

    Creates a channel model from a dataset that can be used with classes such as
    :class:`~sionna.channel.TimeChannel` and :class:`~sionna.channel.OFDMChannel`.
    The dataset is defined by a `generator <https://wiki.python.org/moin/Generators>`_.

    The batch size is configured when instantiating the dataset or through the :attr:`~sionna.channel.CIRDataset.batch_size` property.
    The number of time steps (`num_time_steps`) and sampling frequency (`sampling_frequency`) can only be set when instantiating the dataset.
    The specified values must be in accordance with the data.

    Example
    --------

    The following code snippet shows how to use this class as a channel model.

    >>> my_generator = MyGenerator(...)
    >>> channel_model = CIRDataset(my_generator,
    ...                            batch_size,
    ...                            num_rx,
    ...                            num_rx_ant,
    ...                            num_tx,
    ...                            num_tx_ant,
    ...                            num_paths,
    ...                            num_time_steps)
    >>> dataloader = DataLoader(channel_model, batch_size=batch_size, shuffle=True)
    >>> for a, tau in dataloader:
    ...     # process a and tau

    where ``MyGenerator`` is a generator

    >>> class MyGenerator:
    ...
    ...     def __call__(self):
    ...         ...
    ...         yield a, tau

    that returns complex-valued path coefficients ``a`` with shape
    `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
    and real-valued path delays ``tau`` (in seconds)
    `[num_rx, num_tx, num_paths]`.

    Parameters
    ----------
    cir_generator : `generator <https://wiki.python.org/moin/Generators>`_
        Generator that returns channel impulse responses ``(a, tau)`` where
        ``a`` is the tensor of channel coefficients of shape
        `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
        and dtype ``dtype``, and ``tau`` the tensor of path delays
        of shape  `[num_rx, num_tx, num_paths]` and dtype ``dtype.real_dtype``.

    batch_size : int
        Batch size

    num_rx : int
        Number of receivers (:math:`N_R`)

    num_rx_ant : int
        Number of antennas per receiver (:math:`N_{RA}`)

    num_tx : int
        Number of transmitters (:math:`N_T`)

    num_tx_ant : int
        Number of antennas per transmitter (:math:`N_{TA}`)

    num_paths : int
        Number of paths (:math:`M`)

    num_time_steps : int
        Number of time steps

    dtype : torch.dtype
        Complex datatype to use for internal processing and output.
        Defaults to `torch.complex64`.

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], torch.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], torch.float
        Path delays [s]
    """

    def __init__(self, cir_generator, batch_size, num_rx, num_rx_ant, num_tx,
                 num_tx_ant, num_paths, num_time_steps, dtype=torch.complex64):

        self.cir_generator = cir_generator
        self.batch_size = batch_size
        self.num_time_steps = num_time_steps
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_paths = num_paths
        self.dtype = dtype

        # Convert generator output to PyTorch tensor
        self.data = []
        for a, tau in self.cir_generator():
            self.data.append((torch.tensor(a, dtype=self.dtype),
                              torch.tensor(tau, dtype=self.dtype.real_dtype)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def batch_size(self):
        """Batch size"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set the batch size"""
        self._batch_size = value

