# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import torch
import pytest
import unittest
import numpy as np
from torch import nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'D:\\sionna-main\\sionna-main')
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
# from sionna.mimo import StreamManagement
# from .ofdm_test_module import *

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Number of GPUs available :', torch.cuda.device_count())
if torch.cuda.is_available():
    gpu_num = 0  # Number of the GPU to be used
    print('Only GPU number', gpu_num, 'used.')


