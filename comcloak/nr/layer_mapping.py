import torch
import torch.nn as nn
import numpy as np
from comcloak.utils import flatten_last_dims, split_dim
class  LayerMapper(nn.Module):
    r"""
    LayerMapper(num_layers=1, verbose=False, **kwargs)
    Performs MIMO layer mapping of modulated symbols to layers as defined in
    [3GPP38211]_.

    The LayerMapper supports PUSCH and PDSCH channels and follows the procedure
    as defined in Sec. 6.3.1.3 and Sec. 7.3.1.3 in [3GPP38211]_, respectively.

    As specified in Tab. 7.3.1.3.-1 [3GPP38211]_, the LayerMapper expects two
    input streams for multiplexing if more than 4 layers are active (only
    relevant for PDSCH).

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        num_layers: int, 1 (default) | [1,...,8]
            Number of MIMO layers. If ``num_layers`` >=4, a list of two inputs is expected.
        verbose: bool, False (default)
            If True, additional parameters are printed.

    Input
    -----
        inputs: [batch_size, n], or [[batch_size, n1], [batch_size, n2]], torch.complex
            2+D tensor containing the sequence of symbols to be mapped. If ``num_layers`` >=4,
            a list of two inputs is expected and `n1`/`n2` must match 3GPP requirements.

    Output
    ------
        : [batch_size, num_layers, n/num_layers], torch.complex
            2+D tensor containing the sequence of symbols mapped to the MIMO layers.
    """

    def __init__(self, num_layers=1, verbose=False):
        super().__init__()
        self.shape = None
        assert isinstance(verbose, bool), "verbose must be bool"
        self._verbose = verbose

        assert num_layers in range(1,9), \
                            'num_layers must be between 1 and 8.'
        self._num_layers = num_layers

        # follow Tab. 7.3.1.3-1 from 38.211 for CW multiplexing
        if self._num_layers<5:
            self._num_codewords=1
        elif self._num_layers==5:
            self._num_codewords=2
            self._num_layers0 = 2
            self._num_layers1 = 3
        elif self._num_layers==6:
            self._num_codewords=2
            self._num_layers0 = 3
            self._num_layers1 = 3
        elif self._num_layers==7:
            self._num_codewords=2
            self._num_layers0 = 3
            self._num_layers1 = 4
        elif self._num_layers==8:
            self._num_codewords=2
            self._num_layers0 = 4
            self._num_layers1 = 4
        else:
            raise ValueError("Invalid number of layers.")

        if self._verbose: # provide information about layer configuration
            print("Number of layers: ", self._num_layers)
            if self._num_codewords==2:
                print("Dual codeword mode active and cw multiplexing as " \
                      "defined in Tab. 7.3.1.3-1 from 38.211 applied.")
                print(f"Length of cw1/cw2: {self._num_layers0}/"\
                      f"{self._num_layers1} ")
    #########################################
    # Public methods and properties
    #########################################

    @property
    def num_codewords(self):
        """Number of input codewords for layer mapping. Can be either 1 or 2."""
        return self._num_codewords

    @property
    def num_layers(self):
        """ Number of MIMO layers"""
        return self._num_layers

    @property
    def num_layers0(self):
        r"""Number of layers for first codeword (only relevant for
        `num_codewords` =2)"""
        if self._num_codewords==1:
            return self._num_layers
        return self._num_layers0

    @property
    def num_layers1(self):
        r"""Number of layers for second codeword (only relevant for
        `num_codewords` =2)"""
        if self._num_codewords==1:
            return 0 # no second stream
        return self._num_layers1
    def forward(self, inputs):
        """Test input shapes for consistency."""
        if isinstance(inputs, (list, tuple)):
            self.shape = [x.shape for x in inputs]
        else:
            self.shape = inputs.shape
        if self._num_codewords==1: # single cw mode
            assert not isinstance(self.shape[0], torch.Size), \
                        "Only single input codeword expected."
            assert self.shape[-1]%self._num_layers==0,\
                    "Invalid input dimensions: last dimension must be a " \
                    "multiple of num_layers."
        else: # dual cw mode
            # inputs must be a list of two streams
            s0 = list(self.shape[0])
            s1 = list(self.shape[1])
            

            assert s0[-1]%self._num_layers0==0,\
                    "Invalid input dimensions: last dimension of first input "\
                    "must be a multiple of num_layers0."
            assert s1[-1]%self._num_layers1==0,\
                    "Invalid input dimensions: last dimension of second input "\
                    "must be a multiple of num_layers1."

            # verify that length of tb1 and tb2 fit together
            assert s0[-1]/self._num_layers0 == s1[-1]/self._num_layers1, \
                    f"Invalid input dimensions: length of first input must be "\
                    f"{self._num_layers0/self._num_layers1:.2f} of the length "\
                    f"of the second input."
        """Applies MIMO Layer mapping as defined in Sec. 6.3.1.3 and Sec.
        7.3.1.3 38.211."""
        
        if self._num_codewords==1:
            s = inputs.shape[-1]
            # inputs = inputs
            y = split_dim(inputs,(int(s/self._num_layers), self._num_layers),
                          axis=len(inputs.shape)-1)
        else:
            # for PDSCH only: support dual stream multiplexing
            x0 = inputs[0]
            x1 = inputs[1]
            s0 = x0.shape[-1]
            s1 = x1.shape[-1]

            y0 = split_dim(x0,(int(s0/self._num_layers0), self._num_layers0),
                           axis=len(x0.shape)-1)
            y1 = split_dim(x1,(int(s1/self._num_layers1), self._num_layers1),
                           axis=len(x1.shape)-1)

            y = y = np.concatenate((y0, y1), axis=-1)

        # swap last two dimensions
        # print(y.shape)
        # y = y.transpose(-1, -2)
        y = np.swapaxes(y, -1, -2)
        
        
        return torch.tensor(y)


class LayerDemapper(nn.Module):
    """
    LayerDemapper(layer_mapper, num_bits_per_symbol=1)

    Demaps MIMO layers to coded transport block(s) by following Sec. 6.3.1.3
    and Sec. 7.3.1.3 in [3GPP38211]_.

    This layer must be associated to a :class:`~sionna.nr.LayerMapper` and
    performs the inverse operation.

    It is assumed that ``num_bits_per_symbol`` consecutive LLRs belong to
    a single symbol position. This allows to apply the LayerDemapper after
    demapping symbols to LLR values.

    If the layer mapper is configured for dual codeword transmission, a list of
    both transport block streams is returned.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        layer_mapper: LayerMapper
            Associated LayerMapper.
        num_bits_per_symbol: int, 1 (default)
            Modulation order. Defines how many consecutive LLRs are associated to the same symbol position.

    Input
    -----
        inputs : [batch_size, num_layers, n/num_layers], torch.float
            2+D tensor containing MIMO layer data sequences.

    Output
    ------
        : [batch_size, n], or [[batch_size, n1], [batch_size, n2]], torch.float
            2+D tensor containing the sequence of bits after layer demapping.
     Note
    ----
    As it is more convenient to apply the layer demapper after demapping
    symbols to LLRs, this layer groups the input sequence into groups of
    ``num_bits_per_symbol`` LLRs before restoring the original symbol sequence.
    This behavior can be deactivated by setting ``num_bits_per_symbol`` =1.
    
    """

    def __init__(self, layer_mapper, num_bits_per_symbol=1):
        super().__init__()
        self.shape = None
        assert isinstance(layer_mapper, LayerMapper), "layer_mapper must be LayerMapper."
        self._mapper = layer_mapper

        assert num_bits_per_symbol%1==0, \
                    "num_bits_per_symbol must be int."
        self._num_bits_per_symbol = num_bits_per_symbol

    def forward(self, inputs):
        """Test input shapes for consistency."""
        if isinstance(inputs, (list, tuple)):
            self.shape = [x.shape for x in inputs]
        else:
            self.shape = inputs.shape
        # check that second last dimension equals number of expected streams
        num_layers = self._mapper.num_layers
        assert list(self.shape)[-2]==num_layers, \
            "Invalid input dimension: input shape must be [...,num_layers,n]."

        assert list(self.shape)[-1]%self._num_bits_per_symbol==0, \
            "Invalid input dimension: last dimension must be a multiple of " \
            "num_bits_per_symbol."
        """Demaps multiple layers back to transport block stream(s)."""

        # group llrs into blocks of num_bits_per_symbol values
        s = inputs.shape[-1]
        x = split_dim(inputs,
                     (int(s/self._num_bits_per_symbol),
                      self._num_bits_per_symbol),
                     axis=len(inputs.shape)-1)
        x = np.swapaxes(x, -2, -3)

        if self._mapper.num_codewords==1:
            y = flatten_last_dims(x, num_dims=3)
            return y
        else:
            # multiplex into two codewords/streams
            # only relevant for PDSCH with dual codeword transmission

            y0 = flatten_last_dims(x[...,:self._mapper.num_layers0,:],
                                   num_dims=3)
            y1 = flatten_last_dims(x[...,self._mapper.num_layers0:,:],
                                   num_dims=3)
            return [y0, y1]
