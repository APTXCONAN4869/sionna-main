import torch
import torch.nn as nn
from comcloak.ofdm import LSChannelEstimator
from comcloak.utils import expand_to_rank, split_dim

class PUSCHLSChannelEstimator(LSChannelEstimator, nn.Module):
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, dmrs_length, dmrs_additional_position, num_cdm_groups_without_data, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for NR PUSCH Transmissions.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    The implementation is similar to that of :class:`~sionna.ofdm.LSChannelEstimator`.
    However, it additional takes into account the separation of streams in the same CDM group
    as defined in :class:`~sionna.nr.PUSCHDMRSConfig`. This is done through
    frequency and time averaging of adjacent LS channel estimates.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`

    dmrs_length : int, [1,2]
        Length of DMRS symbols. See :class:`~sionna.nr.PUSCHDMRSConfig`.

    dmrs_additional_position : int, [0,1,2,3]
        Number of additional DMRS symbols.
        See :class:`~sionna.nr.PUSCHDMRSConfig`.

    num_cdm_groups_without_data : int, [1,2,3]
        Number of CDM groups masked for data transmissions.
        See :class:`~sionna.nr.PUSCHDMRSConfig`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specified
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams
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
        super().__init__(resource_grid=resource_grid,
                         )
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
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_ls = torch.nan_to_num(y_pilots / self._pilot_pattern.pilots, nan=0.0, posinf=0.0, neginf=0.0)
        h_ls_shape = h_ls.shape

        # Compute error variance and broadcast to the shape of h_ls
        no = expand_to_rank(no, h_ls.dim(), -1)
        pilots = expand_to_rank(self._pilot_pattern.pilots, h_ls.dim(), 0)
        err_var = torch.nan_to_num(no / (torch.abs(pilots) ** 2), nan=0.0, posinf=0.0, neginf=0.0)

        # Optional time and frequency averaging for CDM
        h_hat = h_ls.clone()

        # Time-averaging across adjacent DMRS OFDM symbols
        if self._dmrs_length == 2:
            # Reshape last dim to [num_dmrs_syms, num_pilots_per_dmrs_sym]
            h_hat = split_dim(h_hat, [self._num_dmrs_syms,
                                      self._num_pilots_per_dmrs_sym], 5)
            # h_hat = h_hat.view(*h_hat.shape[:-1], self._num_dmrs_syms, self._num_pilots_per_dmrs_sym)
            h_hat = (h_hat[..., 0::2, :] + h_hat[..., 1::2, :]) / 2
            h_hat = h_hat.repeat_interleave(2, dim=-2).reshape(h_ls_shape)
            err_var /= 2

        # Frequency-averaging between adjacent channel estimates

        # Compute number of elements across which frequency averaging should
        # be done. This includes the zeroed elements.
        n = 2*self._num_cdm_groups_without_data
        k = int(h_hat.shape[-1]/n) # Second dimension
        # Reshape last dimension to [k, n]
        h_hat = split_dim(h_hat, [k, n], 5)
        # h_hat = h_hat.view(*h_hat.shape[:-1], k, n)
        cond = torch.abs(h_hat) > 0
        # has_nan = torch.isnan(h_hat).any()
        h_hat = torch.nan_to_num(h_hat.sum(dim=-1, keepdim=True) / 2, nan=0.0, posinf=0.0, neginf=0.0)
        # h_hat = h_hat.sum(dim=-1, keepdim=True)
        # has_nan = torch.isnan(h_hat).any()
        h_hat = h_hat.repeat_interleave(n, dim=-1)
        h_hat = torch.where(cond, h_hat, torch.tensor(0, dtype=h_hat.dtype, device=h_hat.device))
        h_hat = h_hat.reshape(h_ls_shape)
        err_var /= 2

        return h_hat, err_var
