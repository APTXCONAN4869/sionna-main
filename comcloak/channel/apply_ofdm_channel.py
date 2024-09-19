import torch
import torch.nn as nn
import torch.nn.functional as F
from comcloak.channel.awgn import AWGN
from comcloak.channel.utils import expand_to_rank
class ApplyOFDMChannel(nn.Module):
    def __init__(self, add_awgn=True, dtype=torch.complex64):
        super(ApplyOFDMChannel, self).__init__()
        self.add_awgn = add_awgn
        self.dtype = dtype
        if self.add_awgn:
            self.awgn = AWGN(dtype=self.dtype)

    def forward(self, inputs):
        if self.add_awgn:
            x, h_freq, no = inputs
        else:
            x, h_freq = inputs

        # Apply the channel response
        x = expand_to_rank(x, h_freq.dim(), axis=1)
        y = torch.sum(torch.sum(h_freq * x, dim=4), dim=3)

        # Add AWGN if requested
        if self.add_awgn:
            y = self.awgn((y, no))

        return y