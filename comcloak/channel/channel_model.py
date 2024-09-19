#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Abstract class proving an interface for channel models"""

from abc import ABC, abstractmethod

class ChannelModel(ABC):
    # pylint: disable=line-too-long
    r"""ChannelModel()

    Abstract class that defines an interface for channel models.

    Any channel model which generates channel impulse responses must implement this interface.
    All the channel models available in Sionna, such as :class:`~sionna.channel.RayleighBlockFading` or :class:`~sionna.channel.tr38901.TDL`, implement this interface.

    *Remark:* Some channel models only require a subset of the input parameters.

    Input
    -----

    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    @abstractmethod
    def __call__(self,  batch_size, num_time_steps, sampling_frequency):

        return NotImplemented


"""
这段代码定义了一个名为`ChannelModel`的抽象类，它提供了一个用于通道模型的接口。下面是对代码中各部分的详细解释：

1. `class ChannelModel(ABC):`
   这行代码定义了一个名为`ChannelModel`的类，并且指定它是一个抽象类（`ABC`表示抽象基类），意味着它不能直接实例化，而是用来作为其他类的基类或接口。

2. `ChannelModel()`
   这是类的文档字符串(docstring),用于描述类的作用和用法。在这里描述了`ChannelModel`类的抽象特性和接口定义。

3. `@abstractmethod`
   这是一个装饰器，用于标记一个抽象方法，即`__call__`方法。抽象方法需要在子类中被实现，否则会抛出`NotImplementedError`异常。

4. `def __call__(self,  batch_size, num_time_steps, sampling_frequency):`
   这是一个抽象方法，定义了通道模型的调用方式。它接受`batch_size`（批大小）、`num_time_steps`（时间步数）和`sampling_frequency`（采样频率）等参数，并且返回通道路径的系数和延迟。

总体来说，这段代码定义了一个通道模型的抽象接口，任何实现了这个接口的通道模型都需要提供`__call__`方法来生成通道脉冲响应。

"""