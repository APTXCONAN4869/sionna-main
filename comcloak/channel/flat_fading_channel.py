import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from comcloak.channel.awgn import AWGN
from comcloak.utils import complex_normal




class GenerateFlatFadingChannel(nn.Module):

    def __init__(self, num_tx_ant, num_rx_ant, spatial_corr=None, dtype=torch.complex64, **kwargs):
        super().__init__(**kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._dtype = dtype
        self.spatial_corr = spatial_corr

    @property
    def spatial_corr(self):
        """The :class:`~sionna.channel.SpatialCorrelation` to be used."""
        return self._spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._spatial_corr = value

    def __call__(self, batch_size):
        # Generate standard complex Gaussian matrices
        shape = [batch_size, self._num_rx_ant, self._num_tx_ant]
        h = complex_normal(shape, dtype=self._dtype)

        # Apply spatial correlation
        if self.spatial_corr is not None:
            h = self.spatial_corr(h)

        return h


class ApplyFlatFadingChannel(nn.Module):
    """
    Applies given channel matrices to a vector input and adds AWGN.

    This class applies a given tensor of flat-fading channel matrices
    to an input tensor. AWGN noise can be optionally added.

    Parameters
    ----------
    add_awgn: bool
        Indicates if AWGN noise should be added to the output.
        Defaults to `True`.

    dtype : torch.complex64, torch.complex128
        The dtype of the output. Defaults to `torch.complex64`.

    Input
    -----
    (x, h, no) :
        Tuple:

    x : [batch_size, num_tx_ant], torch.complex
        Tensor of transmit vectors.

    h : [batch_size, num_rx_ant, num_tx_ant], torch.complex
        Tensor of channel realizations. Will be broadcast to the
        dimensions of ``x`` if needed.

    no : Scalar or Tensor, torch.float
        The noise power ``no`` is per complex dimension.
        Only required if ``add_awgn==True``.
        Will be broadcast to the shape of ``y``.

    Output
    ------
    y : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel output.
        """
    def __init__(self, add_awgn=True, dtype=torch.complex64, **kwargs):
        # super().__init__(requires_grad=False, dtype=dtype, **kwargs)
        super().__init__()
        self._add_awgn = add_awgn
        if self._add_awgn:
            self._awgn = AWGN(dtype=dtype)
    

    def forward(self,inputs):
        if self._add_awgn:
            x, h, no = inputs
        else:
            x, h = inputs

        x = x.to(h.device)  # 确保x在同一个设备上
        x = x.unsqueeze(-1)
        y = torch.matmul(h, x)
        y = y.squeeze(-1)
        no = torch.tensor(no, dtype=torch.float32)
        if self._add_awgn:
            no = no.to(y.device)  # 确保no在同一个设备上
            y = self._awgn((y,no))
        
        return y
        

class FlatFadingChannel(nn.Module):
    def __init__(self,
                 num_tx_ant,
                 num_rx_ant,
                 spatial_corr=None,
                 add_awgn=True,
                 return_channel=False,
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__()
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._add_awgn = add_awgn
        self._return_channel = return_channel
        self._gen_chn = GenerateFlatFadingChannel(self._num_tx_ant,
                                                  self._num_rx_ant,
                                                  spatial_corr,
                                                  dtype=dtype)
        self._app_chn = ApplyFlatFadingChannel(add_awgn=add_awgn, dtype=dtype)

    @property
    def spatial_corr(self):
        """The :class:`~sionna.channel.SpatialCorrelation` to be used."""
        return self._gen_chn.spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._gen_chn.spatial_corr = value

    @property
    def generate(self):
        """Calls the internal :class:`GenerateFlatFadingChannel`."""
        return self._gen_chn

    @property
    def apply(self):
        """Calls the internal :class:`ApplyFlatFadingChannel`."""
        return self._app_chn

    def forward(self, inputs):
        if self._add_awgn:
            x, no =inputs
        else:
            x = inputs
        
        # Generate a batch of channel realizations
        batch_size = x.shape[0]
        h = self._gen_chn(batch_size)

        # Apply the channel to the input
        if self._add_awgn:
            y = self._app_chn([x, h, no])
        else:
            y = self._app_chn([x, h])

        if self._return_channel:
            return y, h
        else:
            return y




     