
from comcloak.channel.utils import cir_to_time_channel
# from . import cir_to_time_channel

class GenerateTimeChannel:

    def __init__(self, channel_model, bandwidth, num_time_samples, l_min, l_max,
                 normalize_channel=False):

        # Callable used to sample channel input responses
        self._cir_sampler = channel_model

        self._l_min = l_min
        self._l_max = l_max
        self._l_tot = l_max - l_min + 1
        self._bandwidth = bandwidth
        self._num_time_steps = num_time_samples
        self._normalize_channel = normalize_channel
    def __call__(self, batch_size=None):

        # Sample channel impulse responses
        # pylint: disable=unbalanced-tuple-unpacking
        h, tau = self._cir_sampler( batch_size,
                                    self._num_time_steps + self._l_tot - 1,
                                    self._bandwidth)

        hm = cir_to_time_channel(self._bandwidth, h, tau, self._l_min,
                                 self._l_max, self._normalize_channel)

        return hm