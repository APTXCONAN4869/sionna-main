import torch
import torch.nn as nn
from comcloak.ofdm import LSChannelEstimator
from comcloak.utils import expand_to_rank, split_dim

import torch
import torch.nn as nn

class PUSCHLSChannelEstimator(nn.Module):
    """
    LSChannelEstimator for NR PUSCH Transmissions using PyTorch.
    """

    def __init__(self,
                 resource_grid,
                 dmrs_length,
                 dmrs_additional_position,
                 num_cdm_groups_without_data,
                 interpolation_type="nn",
                 interpolator=None,
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__()
        self.resource_grid = resource_grid
        self.interpolation_type = interpolation_type
        self.interpolator = interpolator
        self.dtype = dtype

        self._dmrs_length = dmrs_length
        self._dmrs_additional_position = dmrs_additional_position
        self._num_cdm_groups_without_data = num_cdm_groups_without_data

        # Number of DMRS OFDM symbols
        self._num_dmrs_syms = self._dmrs_length * (self._dmrs_additional_position + 1)

        # Number of pilot symbols per DMRS OFDM symbol
        # Some pilot symbols can be zero (for masking)
        self._num_pilots_per_dmrs_sym = int(
            self._pilot_pattern.pilots.shape[-1] / self._num_dmrs_syms
        )

    def estimate_at_pilot_locations(self, y_pilots, no):
        """
        Estimate the channel at pilot locations.

        Args:
            y_pilots (torch.Tensor): Observed signals for the pilot-carrying resource elements.
            no (torch.Tensor): Variance of the AWGN.

        Returns:
            h_hat (torch.Tensor): LS channel estimates.
            err_var (torch.Tensor): Channel estimation error variance.
        """
        # Compute LS channel estimates
        h_ls = torch.nan_to_num(y_pilots / self._pilot_pattern.pilots, nan=0.0)
        h_ls_shape = h_ls.shape

        # Compute error variance and broadcast to the shape of h_ls
        no = self.expand_to_rank(no, h_ls.ndimension(), -1)
        pilots = self.expand_to_rank(self._pilot_pattern.pilots, h_ls.ndimension(), 0)
        err_var = torch.nan_to_num(no / (torch.abs(pilots) ** 2), nan=0.0)

        # Optional time and frequency averaging for CDM
        h_hat = h_ls.clone()

        # Time-averaging across adjacent DMRS OFDM symbols
        if self._dmrs_length == 2:
            h_hat = h_hat.view(*h_hat.shape[:-1], self._num_dmrs_syms, self._num_pilots_per_dmrs_sym)
            h_hat = (h_hat[..., 0::2, :] + h_hat[..., 1::2, :]) / 2
            h_hat = h_hat.repeat_interleave(2, dim=-2).reshape(h_ls_shape)
            err_var /= 2

        # Frequency-averaging between adjacent channel estimates
        n = 2 * self._num_cdm_groups_without_data
        k = h_hat.shape[-1] // n

        h_hat = h_hat.view(*h_hat.shape[:-1], k, n)
        mask = torch.abs(h_hat) > 0
        h_hat = h_hat.sum(dim=-1, keepdim=True) / 2
        h_hat = h_hat.repeat_interleave(n, dim=-1).reshape(h_ls_shape)
        h_hat = torch.where(mask, h_hat, torch.tensor(0, dtype=h_hat.dtype, device=h_hat.device))
        err_var /= 2

        return h_hat, err_var

    @staticmethod
    def expand_to_rank(tensor, rank, dim):
        """
        Expands the tensor to the desired rank by inserting dimensions.

        Args:
            tensor (torch.Tensor): Input tensor.
            rank (int): Desired rank.
            dim (int): Dimension to insert.

        Returns:
            torch.Tensor: Expanded tensor.
        """
        while tensor.ndimension() < rank:
            tensor = tensor.unsqueeze(dim)
        return tensor
