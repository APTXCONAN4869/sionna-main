import tensorflow as tf
import numpy as np
import os
import sys
sys.path.insert(0, 'D:\sionna-main')
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义 TensorFlow 代码
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.channel.tr38901 import models
from sionna.utils import flatten_last_dims, expand_to_rank, matrix_inv
from sionna.ofdm import ResourceGrid, RemoveNulledSubcarriers
from sionna import PI, SPEED_OF_LIGHT
import numpy as np
from scipy.special import jv
import itertools
from abc import ABC, abstractmethod
import json
from importlib_resources import files
#
class BaseChannelEstimator(ABC, Layer):
    # pylint: disable=line-too-long
    r"""BaseChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Abstract layer for implementing an OFDM channel estimator.

    Any layer that implements an OFDM channel estimator must implement this
    class and its
    :meth:`~sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    abstract method.

    This class extracts the pilots from the received resource grid ``y``, calls
    the :meth:`~sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    method to estimate the channel for the pilot-carrying resource elements,
    and then interpolates the channel to compute channel estimates for the
    data-carrying resouce elements using the interpolation method specified by
    ``interpolation_type`` or the ``interpolator`` object.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

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
        or `None`. In the latter case, the interpolator specfied
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
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(resource_grid, ResourceGrid),\
            "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern
        self._removed_nulled_scs = RemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in ["nn","lin","lin_time_avg",None], \
            "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        if interpolator is not None:
            assert isinstance(interpolator, BaseChannelInterpolator), \
        "`interpolator` must implement the BaseChannelInterpolator interface"
            self._interpol = interpolator
        elif self._interpolation_type == "nn":
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern,
                                                time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        mask = flatten_last_dims(self._pilot_pattern.mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

    @abstractmethod
    def estimate_at_pilot_locations(self, y_pilots, no):
        r"""
        Estimates the channel for the pilot-carrying resource elements.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implement this class.

        Input
        -----
        y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex
            Observed signals for the pilot-carrying resource elements

        no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
            Variance of the AWGN

        Output
        ------
        h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex
            Channel estimates for the pilot-carrying resource elements

        err_var : Same shape as ``h_hat``, tf.float
            Channel estimation error variance for the pilot-carrying
            resource elements
        """
        pass

    def call(self, inputs):

        y, no = inputs

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        y_eff = self._removed_nulled_scs(y)

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)

        # Gather pilots along the last dimensions
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)

        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(h_hat, err_var)
            err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))

        return h_hat, err_var

class BaseChannelInterpolator(ABC):
    # pylint: disable=line-too-long
    r"""BaseChannelInterpolator()

    Abstract layer for implementing an OFDM channel interpolator.

    Any layer that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator is used by an OFDM channel estimator
    (:class:`~sionna.ofdm.BaseChannelEstimator`) to compute channel estimates
    for the data-carrying resource elements from the channel estimates for the
    pilot-carrying resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    @abstractmethod
    def __call__(self, h_hat, err_var):
        pass

class NearestNeighborInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""NearestNeighborInterpolator(pilot_pattern)

    Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    accross a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../figures/nearest_neighbor_interpolation.png

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1))
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int32)
        for a in range(gather_ind.shape[0]): # For each pilot pattern...
            i_p, j_p = np.where(mask[a]) # ...determine the pilot indices

            for i in range(mask_shape[-2]): # Iterate over...
                for j in range(mask_shape[-1]): # ... all resource elements

                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i-i_p) + np.abs(j-j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a])==0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance...
                    ind = np.argmin(d)

                    # ... and store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask, i.e.:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers]
        self._gather_ind = tf.reshape(gather_ind, mask_shape)

    def _interpolate(self, inputs):
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_pilots, k, l, m]
        perm = tf.roll(tf.range(tf.rank(inputs)), -3, 0)
        inputs = tf.transpose(inputs, perm)

        # Interpolate through gather. Shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  ..., num_effective_subcarriers, k, l, m]
        outputs = tf.gather(inputs, self._gather_ind, 2, batch_dims=2)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        outputs = tf.transpose(outputs, perm)

        return outputs

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var


# 创建测试张量（从文件中读取 PyTorch 的输入）
tensor_torch = np.load(os.path.join(current_dir, 'tensor_torch.npy'))
tensor_tf = tf.constant(tensor_torch)

# 测试 TensorFlow 的函数
num_dims = 2
axis = 1
output_functionA = functionA(tensor_tf, num_dims, axis)

target_rank = 5
output_functionB = functionB(tensor_tf, target_rank, axis)

# 保存输出到文件
np.save(os.path.join(current_dir, 'functionA_tf.npy'), output_functionA.numpy())
np.save(os.path.join(current_dir, 'functionB_tf.npy'), output_functionB.numpy())
