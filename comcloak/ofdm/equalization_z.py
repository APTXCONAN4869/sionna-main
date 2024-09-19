import torch
import torch.nn as nn
import comcloak
from ofdm_test_module_z import flatten_dims, split_dim, flatten_last_dims, expand_to_rank
from comcloak.ofdm import RemoveNulledSubcarriers

from sionna.mimo import lmmse_equalizer, zf_equalizer, mf_equalizer

def gather_pytorch(input_data, indices=None, batch_dims=0, axis=0):
    input_data = torch.tensor(input_data)
    indices = torch.tensor(indices)
    if batch_dims == 0:
        if axis < 0:
            axis = len(input_data.shape) + axis
        data = torch.index_select(input_data, axis, indices.flatten())
        shape_input = list(input_data.shape)
        # shape_ = delete(shape_input, axis)
        # 连接列表
        shape_output = shape_input[:axis] + \
            list(indices.shape) + shape_input[axis + 1:]
        data_output = data.reshape(shape_output)
        return data_output
    else:
        data_output = []
        for data,ind in zip(input_data, indices):
            r = gather_pytorch(data, ind, batch_dims=batch_dims-1)
            data_output.append(r)
        return torch.stack(data_output)

class OFDMEqualizer(nn.Module):
    r"""OFDMEqualizer(equalizer, resource_grid, stream_management, dtype=torch.complex64, **kwargs)

    Layer that wraps a MIMO equalizer for use with the OFDM waveform.

    The parameter ``equalizer`` is a callable (e.g., a function) that
    implements a MIMO equalization algorithm for arbitrary batch dimensions.

    This class pre-processes the received resource grid ``y`` and channel
    estimate ``h_hat``, and computes for each receiver the
    noise-plus-interference covariance matrix according to the OFDM and stream
    configuration provided by the ``resource_grid`` and
    ``stream_management``, which also accounts for the channel
    estimation error variance ``err_var``. These quantities serve as input
    to the equalization algorithm that is implemented by the callable ``equalizer``.
    This layer computes soft-symbol estimates together with effective noise
    variances for all streams which can, e.g., be used by a
    :class:`~sionna.mapping.Demapper` to obtain LLRs.

    Note
    -----
    The callable ``equalizer`` must take three inputs:

    * **y** ([...,num_rx_ant], torch.complex) -- 1+D tensor containing the received signals.
    * **h** ([...,num_rx_ant,num_streams_per_rx], torch.complex) -- 2+D tensor containing the channel matrices.
    * **s** ([...,num_rx_ant,num_rx_ant], torch.complex) -- 2+D tensor containing the noise-plus-interference covariance matrices.

    It must generate two outputs:

    * **x_hat** ([...,num_streams_per_rx], torch.complex) -- 1+D tensor representing the estimated symbol vectors.
    * **no_eff** (torch.float) -- Tensor of the same shape as ``x_hat`` containing the effective noise variance estimates.

    Parameters
    ----------
    equalizer : Callable
        Callable object (e.g., a function) that implements a MIMO equalization
        algorithm for arbitrary batch dimensions

    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    dtype : torch.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `torch.complex64`.

    Input
    -----
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
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], torch.complex
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], torch.float
        Effective noise variance for each estimated symbol
    """
    
    def __init__(self,
                 equalizer,
                 resource_grid,
                 stream_management,
                 dtype=torch.complex64):
        super().__init__()
        assert callable(equalizer)
        assert isinstance(resource_grid, comcloak.ofdm.ResourceGrid)
        assert isinstance(stream_management, comcloak.mimo.StreamManagement)
        self._equalizer = equalizer
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._dtype = dtype
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = torch.argsort(mask.flatten(), descending=False)
        self._data_ind = data_ind[:num_data_symbols]

    def forward(self, inputs):
        y, h_hat, err_var, no = inputs
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # err_var has a shape that is broadcastable to h_hat

        # no has shape [batch_size, num_rx, num_rx_ant]
        # or just the first n dimensions of this

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # Remove nulled subcarriers from y (guards, dc)
        y_eff = self._removed_nulled_scs(y)

        # Prepare the observation y for MIMO detection
        # Transpose y_eff to put num_rx_ant last. New shape:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        y_dt = y_eff.permute(0, 1, 3, 4, 2).to(self._dtype)

        # Prepare the err_var for MIMO detection
        # New shape is:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
        err_var_dt = err_var.expand_as(h_hat).permute(0, 1, 5, 6, 2, 3, 4).flatten(2, 3).to(self._dtype)

        # Construct MIMO channels
        # Reshape h_hat for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = h_hat.permute(1, 3, 4, 0, 2, 5, 6)
        # Flatten first tthree dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = gather_pytorch(h_dt, ind_desired, axis=0)
        h_dt_undesired = gather_pytorch(h_dt, ind_undesired, axis=0)

        # Split first dimension to separate RX and TX
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt_desired = split_dim(h_dt_desired,
                                 [self._stream_management.num_rx,
                                  self._stream_management.num_streams_per_rx],
                                 0)
        h_dt_undesired = split_dim(h_dt_undesired,
                                   [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = h_dt_desired.permute(*perm).to(self._dtype)
        h_dt_undesired = h_dt_undesired.permute(*perm)

        # Prepare the noise variance
        # no is first broadcast to [batch_size, num_rx, num_rx_ant]
        # then the rank is expanded to that of y
        # then it is transposed like y to the final shape
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        no_dt = expand_to_rank(no, 3, -1)
        no_dt = torch.broadcast_to(no_dt, y.shape[:3])
        no_dt = expand_to_rank(no_dt, y.dim(), -1)
        no_dt = no_dt.permute(0,1,3,4,2).to(self._dtype)


        # Compute the interference covariance matrix
        # Covariance of undesired transmitters
        s_inf = torch.matmul(h_dt_undesired, h_dt_undesired.transpose(-1, -2))
        #Thermal noise
        s_no = torch.diag_embed(no_dt)
        # Channel estimation errors
        # As we have only error variance information for each element,
        # we simply sum them across transmitters and build a
        # diagonal covariance matrix from this
        s_csi = torch.diag_embed(err_var_dt.sum(dim=-1))
        # Final covariance matrix
        s = s_inf + s_no + s_csi

        # Compute symbol estimate and effective noise variance
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., num_stream_per_rx]
        x_hat, no_eff = self._equalizer(y_dt, h_dt_desired, s)

        # Extract data symbols for all detected TX
        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, batch_size]
        x_hat = x_hat.permute(1, 4, 2, 3, 0)
        no_eff = no_eff.permute(1, 4, 2, 3, 0)
        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,...
        #  ...,num_effective_subcarriers, batch_size]
        x_hat = x_hat.flatten(2, 0)
        no_eff = no_eff.flatten(2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        x_hat = gather_pytorch(x_hat, stream_ind, axis=0)
        no_eff = gather_pytorch(no_eff, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compared to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        x_hat = split_dim(x_hat, [num_tx, num_streams], 0)
        no_eff = split_dim(no_eff, [num_tx, num_streams], 0)
        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,...
        #  ..., batch_size]
        x_hat = flatten_dims(x_hat, 2, 2)
        no_eff = flatten_dims(no_eff, 2, 2)

        # Broadcast no_eff to the shape of x_hat
        no_eff = torch.broadcast_to(no_eff, x_hat.shape)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, batch_size]
        x_hat = gather_pytorch(x_hat, self._data_ind, batch_dims=2, axis=2)
        no_eff = gather_pytorch(no_eff, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        x_hat = x_hat.permute(3, 0, 1, 2)
        no_eff = no_eff.permute(3, 0, 1, 2)

        return x_hat, no_eff


#
class LMMSEEqualizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """LMMSEEqualizer(resource_grid, stream_management, whiten_interference=True, dtype=torch.complex64, **kwargs)

    LMMSE equalization for OFDM MIMO transmissions.

    This layer computes linear minimum mean squared error (LMMSE) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.mimo.lmmse_equalizer`. The layer
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.mapping.Demapper` to obtain LLRs.

    Parameters
    ----------
    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    whiten_interference : bool
        If `True` (default), the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used which
        can be numerically more stable.

    dtype : torch.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `torch.complex64`.

    Input
    -----
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
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], torch.complex
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], torch.float
        Effective noise variance for each estimated symbol

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 whiten_interference=True,
                 dtype=torch.complex64,
                 **kwargs):

        def equalizer(y, h, s):
            return lmmse_equalizer(y, h, s, whiten_interference)

        super().__init__(equalizer=equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype, **kwargs)

#
class ZFEqualizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """ZFEqualizer(resource_grid, stream_management, dtype=torch.complex64, **kwargs)

    ZF equalization for OFDM MIMO transmissions.

    This layer computes zero-forcing (ZF) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.mimo.zf_equalizer`. The layer
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.mapping.Demapper` to obtain LLRs.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    dtype : torch.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `torch.complex64`.

    Input
    -----
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
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], torch.complex
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], torch.float
        Effective noise variance for each estimated symbol

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__(equalizer=zf_equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype, **kwargs)


class MFEqualizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """MFEqualizer(resource_grid, stream_management, dtype=torch.complex64, **kwargs)

    MF equalization for OFDM MIMO transmissions.

    This layer computes matched filter (MF) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.mimo.mf_equalizer`. The layer
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.mapping.Demapper` to obtain LLRs.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    dtype : torch.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `torch.complex64`.

    Input
    -----
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
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], torch.complex
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], torch.float
        Effective noise variance for each estimated symbol

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__(equalizer=mf_equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype, **kwargs)
