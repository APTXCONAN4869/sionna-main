"""Class for generating channel frequency responses"""
import torch

from .utils import subcarrier_frequencies, cir_to_ofdm_channel

class GenerateOFDMChannel:

    def __init__(self, channel_model, resource_grid, normalize_channel=False,
                 dtype=torch.complex64):
        

        # Callable used to sample channel input responses
        self._cir_sampler = channel_model

        # We need those in call()
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._subcarrier_spacing = resource_grid.subcarrier_spacing
        self._num_subcarriers = resource_grid.fft_size
        self._normalize_channel = normalize_channel
        self._sampling_frequency = 1./resource_grid.ofdm_symbol_duration

        # Frequencies of the subcarriers
        self._frequencies = subcarrier_frequencies(self._num_subcarriers,
                                                   self._subcarrier_spacing,
                                                   dtype)
    def __call__(self, batch_size=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler( batch_size,
                                    self._num_ofdm_symbols,
                                    self._sampling_frequency)

        h_freq = cir_to_ofdm_channel(self._frequencies, h, tau,
                                     self._normalize_channel)

        return h_freq                