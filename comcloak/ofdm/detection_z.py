import torch
import torch.nn as nn
import torch.nn.functional as F
from comcloak.ofdm import RemoveNulledSubcarriers
from ofdm_test_module_z import Constellation, flatten_dims, split_dim, flatten_last_dims, expand_to_rank
from sionna.mimo import MaximumLikelihoodDetectorWithPrior as MaximumLikelihoodDetectorWithPrior_
from sionna.mimo import MaximumLikelihoodDetector as MaximumLikelihoodDetector_
from sionna.mimo import LinearDetector as LinearDetector_
from sionna.mimo import KBestDetector as KBestDetector_
from sionna.mimo import EPDetector as EPDetector_
from sionna.mimo import MMSEPICDetector as MMSEPICDetector_


class OFDMDetector(nn.Module):
    def __init__(self, detector, output, resource_grid, stream_management, dtype=torch.complex64, **kwargs):
        super().__init__()
        self._detector = detector
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)
        self._output = output
        self._dtype = dtype

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = torch.argsort(flatten_last_dims(mask), descending=False)
        self._data_ind = data_ind[..., :num_data_symbols]

    def _preprocess_inputs(self, y, h_hat, err_var, no):
        """Pre-process the received signal and compute the
        noise-plus-interference covariance matrix"""

        # Remove nulled subcarriers from y (guards, dc). New shape:
        y_eff = self._removed_nulled_scs(y)

        ####################################################
        ### Prepare the observation y for MIMO detection ###
        ####################################################
        # Transpose y_eff to put num_rx_ant last. New shape:
        y_dt = y_eff.permute(0, 1, 3, 4, 2).contiguous()
        y_dt = y_dt.to(self._dtype)

        ##############################################
        ### Prepare the err_var for MIMO detection ###
        ##############################################
        # New shape is:
        err_var_dt = err_var.expand_as(h_hat)
        err_var_dt = err_var_dt.permute(0, 1, 5, 6, 2, 3, 4).contiguous()
        err_var_dt = flatten_last_dims(err_var_dt, 2)
        err_var_dt = err_var_dt.to(self._dtype)

        ###############################
        ### Construct MIMO channels ###
        ###############################

        # Reshape h_hat for the construction of desired/interfering channels:
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_dt = h_hat.permute(perm).contiguous()

        # Flatten first three dimensions:
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = torch.gather(h_dt, 0, ind_desired)
        h_dt_undesired = torch.gather(h_dt, 0, ind_undesired)

        # Split first dimension to separate RX and TX:
        h_dt_desired = split_dim(h_dt_desired, [self._stream_management.num_rx, self._stream_management.num_streams_per_rx], 0)
        h_dt_undesired = split_dim(h_dt_undesired, [self._stream_management.num_rx, -1], 0)

        # Permute dims to
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = h_dt_desired.permute(perm).contiguous()
        h_dt_desired = h_dt_desired.to(self._dtype)
        h_dt_undesired = h_dt_undesired.permute(perm).contiguous()

        ##################################
        ### Prepare the noise variance ###
        ##################################
        # no is first broadcast to [batch_size, num_rx, num_rx_ant]
        no_dt = no.unsqueeze(-1).expand_as(y[:, :, :, 0, :])
        no_dt = no_dt.unsqueeze(-1).expand_as(y)
        no_dt = no_dt.permute(0, 1, 3, 4, 2).contiguous()
        no_dt = no_dt.to(self._dtype)

        ##################################################
        ### Compute the interference covariance matrix ###
        ##################################################
        # Covariance of undesired transmitters
        s_inf = torch.matmul(h_dt_undesired, h_dt_undesired.transpose(-1, -2))

        # Thermal noise
        s_no = torch.diag_embed(no_dt)

        # Channel estimation errors
        s_csi = torch.diag_embed(err_var_dt.sum(dim=-1))

        # Final covariance matrix
        s = s_inf + s_no + s_csi
        s = s.to(self._dtype)

        return y_dt, h_dt_desired, s

    def _extract_datasymbols(self, z):
        """Extract data symbols for all detected TX"""

        rank_extended = len(z.shape) < 6
        z = z.unsqueeze(-1) if rank_extended else z

        z = z.permute(1, 4, 2, 3, 5, 0).contiguous()
        z = flatten_dims(z, 2, 0)

        stream_ind = self._stream_management.stream_ind
        z = torch.gather(z, 0, stream_ind)

        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        z = split_dim(z, [num_tx, num_streams], 0)

        z = flatten_dims(z, 2, 2)

        z = torch.gather(z, 2, self._data_ind.expand(-1, -1, z.size(2), -1, -1))

        z = z.permute(5, 0, 1, 2, 3).contiguous()

        if self._output == 'bit':
            z = flatten_dims(z, 2, 3)
        if rank_extended:
            z = z.squeeze(-1)

        return z

    def forward(self, inputs):
        y, h_hat, err_var, no = inputs

        y_dt, h_dt_desired, s = self._preprocess_inputs(y, h_hat, err_var, no)

        z = self._detector((y_dt, h_dt_desired, s))

        z = self._extract_datasymbols(z)

        return z

class OFDMDetectorWithPrior(OFDMDetector):
    def __init__(self, detector, output, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, dtype=torch.complex64, **kwargs):
        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype,
                         **kwargs)

        # Constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)

        # Precompute indices to map priors to a resource grid
        rg_type = resource_grid.build_type_grid()
        remove_nulled_sc = RemoveNulledSubcarriers(resource_grid)
        self._data_ind_scatter = torch.where(remove_nulled_sc(rg_type) == 0)

    def preprocess_inputs(self, y, h_hat, prior, err_var, no):
        y_dt, h_dt_desired, s = super().preprocess_inputs(y, h_hat, err_var, no)

        if self.output == 'bit':
            prior = prior.view(prior.shape[0], prior.shape[1], prior.shape[2], -1, self._constellation.num_bits_per_symbol)
        else:
            prior = prior.view(prior.shape[0], prior.shape[1], prior.shape[2], -1, self._constellation.num_points)

        template = torch.zeros(
            (self.resource_grid.num_tx,
             self.resource_grid.num_streams_per_tx,
             self.resource_grid.num_ofdm_symbols,
             self.resource_grid.num_effective_subcarriers,
             prior.shape[-1],
             prior.shape[0]),
            dtype=self.dtype.real_dtype
        )

        prior = prior.permute(1, 2, 3, 4, 0)
        prior = prior.view(-1, prior.shape[-2], prior.shape[-1])

        indices = self._data_ind_scatter.expand_as(prior)
        template.scatter_(1, indices, prior)

        prior = template.permute(5, 2, 3, 0, 1, 4)
        prior = prior.view(prior.shape[0], prior.shape[1], -1, prior.shape[-2], prior.shape[-1])
        prior = prior.permute(0, 1, 2, 3, 4).contiguous()
        prior = prior.unsqueeze(1).expand(y.shape[0], y.shape[1], -1, -1, -1, -1)

        return y_dt, h_dt_desired, prior, s

    def forward(self, inputs):
        y, h_hat, prior, err_var, no = inputs
        y_dt, h_dt_desired, prior, s = self.preprocess_inputs(y, h_hat, prior, err_var, no)
        z = self.detector([y_dt, h_dt_desired, prior, s])
        z = self.extract_datasymbols(z)
        return z

class MaximumLikelihoodDetector(OFDMDetector):
    # pylint: disable=line-too-long
    r"""MaximumLikelihoodDetector(output, demapping_method, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=torch.complex64, **kwargs)

    Maximum-likelihood (ML) detection for OFDM MIMO transmissions.

    This layer implements maximum-likelihood (ML) detection
    for OFDM MIMO transmissions. Both ML detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.MaximumLikelihoodDetector`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        Type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    demapping_method : One of ["app", "maxlog"], str
        Demapping method used

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        Instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    ------
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], torch.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], torch.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), torch.float
        Variance of the AWGN noise

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float or [batch_size, num_tx, num_streams, num_data_symbols], torch.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 demapping_method,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):

        # Instantiate the maximum-likelihood detector
        detector = MaximumLikelihoodDetector_(output=output,
                            demapping_method=demapping_method,
                            num_streams = stream_management.num_streams_per_rx,
                            constellation_type=constellation_type,
                            num_bits_per_symbol=num_bits_per_symbol,
                            constellation=constellation,
                            hard_out=hard_out,
                            dtype=dtype,
                            **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype,
                         **kwargs)

class MaximumLikelihoodDetectorWithPrior(OFDMDetectorWithPrior):
    # pylint: disable=line-too-long
    r"""MaximumLikelihoodDetectorWithPrior(output, demapping_method, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=torch.complex64, **kwargs)

    Maximum-likelihood (ML) detection for OFDM MIMO transmissions, assuming prior
    knowledge of the bits or constellation points is available.

    This layer implements maximum-likelihood (ML) detection
    for OFDM MIMO transmissions assuming prior knowledge on the transmitted data is available.
    Both ML detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.MaximumLikelihoodDetectorWithPrior`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        Type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    demapping_method : One of ["app", "maxlog"], str
        Demapping method used

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        Instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    ------
    (y, h_hat, prior, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], torch.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex
        Channel estimates for all streams from all transmitters

    prior : [batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", logits of the transmitted constellation points are expected.

    err_var : [Broadcastable to shape of ``h_hat``], torch.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), torch.float
        Variance of the AWGN noise

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float or [batch_size, num_tx, num_streams, num_data_symbols], torch.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 demapping_method,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):

        # Instantiate the maximum-likelihood detector
        detector = MaximumLikelihoodDetectorWithPrior_(output=output,
                            demapping_method=demapping_method,
                            num_streams = stream_management.num_streams_per_rx,
                            constellation_type=constellation_type,
                            num_bits_per_symbol=num_bits_per_symbol,
                            constellation=constellation,
                            hard_out=hard_out,
                            dtype=dtype,
                            **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         constellation_type=constellation_type,
                         num_bits_per_symbol=num_bits_per_symbol,
                         constellation=constellation,
                         dtype=dtype,
                         **kwargs)

class LinearDetector(OFDMDetector):
    # pylint: disable=line-too-long
    r"""LinearDetector(equalizer, output, demapping_method, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=torch.complex64, **kwargs)

    This layer wraps a MIMO linear equalizer and a :class:`~sionna.mapping.Demapper`
    for use with the OFDM waveform.

    Both detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.LinearDetector`.

    Parameters
    ----------
    equalizer : str, one of ["lmmse", "zf", "mf"], or an equalizer function
        Equalizer to be used. Either one of the existing equalizers, e.g.,
        :func:`~sionna.mimo.lmmse_equalizer`, :func:`~sionna.mimo.zf_equalizer`, or
        :func:`~sionna.mimo.mf_equalizer` can be used, or a custom equalizer
        function provided that has the same input/output specification.

    output : One of ["bit", "symbol"], str
        Type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    demapping_method : One of ["app", "maxlog"], str
        Demapping method used

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        Instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    ------
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], torch.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], torch.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), torch.float
        Variance of the AWGN

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float or [batch_size, num_tx, num_streams, num_data_symbols], torch.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 equalizer,
                 output,
                 demapping_method,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):

        # Instantiate the linear detector
        detector = LinearDetector_(equalizer=equalizer,
                                   output=output,
                                   demapping_method=demapping_method,
                                   constellation_type=constellation_type,
                                   num_bits_per_symbol=num_bits_per_symbol,
                                   constellation=constellation,
                                   hard_out=hard_out,
                                   dtype=dtype,
                                   **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype,
                         **kwargs)

class KBestDetector(OFDMDetector):
    # pylint: disable=line-too-long
    r"""KBestDetector(output, num_streams, k, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, use_real_rep=False, list2llr=None, dtype=torch.complex64, **kwargs)

    This layer wraps the MIMO K-Best detector for use with the OFDM waveform.

    Both detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.KBestDetector`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        Type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    num_streams : torch.int
        Number of transmitted streams

    k : torch.int
        Number of paths to keep. Cannot be larger than the
        number of constellation points to the power of the number of
        streams.

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        Instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    use_real_rep : bool
        If `True`, the detector use the real-valued equivalent representation
        of the channel. Note that this only works with a QAM constellation.
        Defaults to `False`.

    list2llr: `None` or instance of :class:`~sionna.mimo.List2LLR`
        The function to be used to compute LLRs from a list of candidate solutions.
        If `None`, the default solution :class:`~sionna.mimo.List2LLRSimple`
        is used.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    ------
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], torch.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], torch.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), torch.float
        Variance of the AWGN

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float or [batch_size, num_tx, num_streams, num_data_symbols], torch.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 num_streams,
                 k,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 use_real_rep=False,
                 list2llr="default",
                 dtype=torch.complex64,
                 **kwargs):

        # Instantiate the K-Best detector
        detector = KBestDetector_(output=output,
                                  num_streams=num_streams,
                                  k=k,
                                  constellation_type=constellation_type,
                                  num_bits_per_symbol=num_bits_per_symbol,
                                  constellation=constellation,
                                  hard_out=hard_out,
                                  use_real_rep=use_real_rep,
                                  list2llr=list2llr,
                                  dtype=dtype,
                                  **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype,
                         **kwargs)

class EPDetector(OFDMDetector):
    # pylint: disable=line-too-long
    r"""EPDetector(output, resource_grid, stream_management, num_bits_per_symbol, hard_out=False, l=10, beta=0.9, dtype=torch.complex64, **kwargs)

    This layer wraps the MIMO EP detector for use with the OFDM waveform.

    Both detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.EPDetector`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        Type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    num_bits_per_symbol : int
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    l : int
        Number of iterations. Defaults to 10.

    beta : float
        Parameter :math:`\beta\in[0,1]` for update smoothing.
        Defaults to 0.9.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        Precision used for internal computations. Defaults to ``torch.complex64``.
        Especially for large MIMO setups, the precision can make a significant
        performance difference.

    Input
    ------
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], torch.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], torch.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), torch.float
        Variance of the AWGN

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float or [batch_size, num_tx, num_streams, num_data_symbols], torch.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    For numerical stability, we do not recommend to use this function in Graph
    mode with XLA, i.e., within a function that is decorated with
    ``@torch.function(jit_compile=True)``.
    However, it is possible to do so by setting
    ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 output,
                 resource_grid,
                 stream_management,
                 num_bits_per_symbol=None,
                 hard_out=False,
                 l=10,
                 beta=0.9,
                 dtype=torch.complex64,
                 **kwargs):

        # Instantiate the EP detector
        detector = EPDetector_(output=output,
                               num_bits_per_symbol=num_bits_per_symbol,
                               hard_out=hard_out,
                               l=l,
                               beta=beta,
                               dtype=dtype,
                               **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype,
                         **kwargs)

class MMSEPICDetector(OFDMDetectorWithPrior):
    # pylint: disable=line-too-long
    r"""MMSEPICDetector(output, resource_grid, stream_management, demapping_method="maxlog", num_iter=1, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=torch.complex64, **kwargs)

    This layer wraps the MIMO MMSE PIC detector for use with the OFDM waveform.

    Both detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.MMSEPICDetector`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        Type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.
        Defaults to "maxlog".

    num_iter : int
        Number of MMSE PIC iterations.
        Defaults to 1.

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        An instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        Precision used for internal computations. Defaults to ``torch.complex64``.
        Especially for large MIMO setups, the precision can make a significant
        performance difference.

    Input
    ------
    (y, h_hat, prior, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], torch.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], torch.complex
        Channel estimates for all streams from all transmitters

    prior : [batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", logits of the transmitted constellation points are expected.

    err_var : [Broadcastable to shape of ``h_hat``], torch.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), torch.float
        Variance of the AWGN

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], torch.float or [batch_size, num_tx, num_streams, num_data_symbols], torch.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    For numerical stability, we do not recommend to use this function in Graph
    mode with XLA, i.e., within a function that is decorated with
    ``@torch.function(jit_compile=True)``.
    However, it is possible to do so by setting
    ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 output,
                 resource_grid,
                 stream_management,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):

        # Instantiate the EP detector
        detector = MMSEPICDetector_(output=output,
                                    demapping_method=demapping_method,
                                    num_iter=num_iter,
                                    constellation_type=constellation_type,
                                    num_bits_per_symbol=num_bits_per_symbol,
                                    constellation=constellation,
                                    hard_out=hard_out,
                                    dtype=dtype,
                                    **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         constellation_type=constellation_type,
                         num_bits_per_symbol=num_bits_per_symbol,
                         constellation=constellation,
                         dtype=dtype,
                         **kwargs)
