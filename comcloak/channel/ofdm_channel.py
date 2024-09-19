import torch 
import torch.nn as nn

from . import GenerateOFDMChannel, ApplyOFDMChannel

class OFDMChannel(nn.Module):
    def __init__(self, channel_model, resource_grid, add_awgn=True,
                 normalize_channel=False, return_channel=False,
                 dtype=torch.complex64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cir_sampler = channel_model
        self._rg = resource_grid
        self._add_awgn = add_awgn
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        self._dtype = dtype

        self._generate_channel = GenerateOFDMChannel(self._cir_sampler,
                                                     self._rg,
                                                     self._normalize_channel,
                                                     self._dtype)
        
        self._apply_channel = ApplyOFDMChannel( self._add_awgn,
                                               self._dtype)
        
    def forward(self, inputs):

        if self._add_awgn:
            x,no = inputs
        else:
            x = inputs
        
        h_freq = self._generate_channel(x.shape[0])
        if self._add_awgn:
            y = self._apply_channel([x, h_freq, no])
        else:
            y = self._apply_channel([x,h_freq])
        
        if self._return_channel:
            return y, h_freq
        else:
            return y
