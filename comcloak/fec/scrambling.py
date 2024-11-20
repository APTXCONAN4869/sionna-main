"""Layers for scrambling, descrambling and utility functions."""
import numpy as np
import torch
import torch.nn as nn
from comcloak.utils import expand_to_rank
from comcloak.nr.utils import generate_prng_seq


def assert_type(tensor, expected_type):
    assert tensor.dtype == expected_type, f"Expected type {expected_type}, but got {tensor.dtype}"

class Scrambler(nn.Module):
    r"""
    Scrambler(seed=None, keep_batch_constant=False, sequence=None, binary=True, keep_state=True, dtype=torch.float32)

    Randomly flips the state/sign of a sequence of bits or LLRs, respectively.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
    seed : int
        Defaults to None. Defines the initial state of the
        pseudo-random generator to generate the scrambling sequence.
        If None, a random integer will be generated. Only used when
        keep_state is True.

    keep_batch_constant : bool
        Defaults to False. If True, all samples in the batch are scrambled
        with the same scrambling sequence. Otherwise, per sample a random
        sequence is generated.

    sequence : torch.Tensor of 0s and 1s or None
        If provided, the seed will be ignored and the explicit scrambling
        sequence is used. Shape must be broadcastable to `x`.

    binary : bool
        Defaults to True. Indicates whether bit-sequence should be flipped
        or the signs should be flipped (soft-value/LLR domain-based).

    keep_state : bool
        Defaults to True. Indicates whether the scrambling sequence should
        be kept constant.

    dtype : torch.dtype
        Defaults to `torch.float32`. Defines the datatype for internal
        calculations and the output dtype.
     Input
    -----
        (x, seed, binary):
            Either Tuple ``(x, seed, binary)`` or  ``(x, seed)`` or ``x`` only
            (no tuple) if the internal  seed should be used:

        x: torch.float
            1+D tensor of arbitrary shape.
        seed: int
            An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (call with same random seed).
        binary: bool
            Overrules the init parameter `binary` iff explicitly given.
            Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

    Output
    ------
        : torch.float
            1+D tensor of same shape as ``x``.

    Note
    ----
        For inverse scrambling, the same scrambler can be re-used (as the values
        are flipped again, i.e., result in the original state). However,
        ``keep_state`` must be set to True as a new sequence would be generated
        otherwise.

        The scrambler layer is stateless, i.e., the seed is either random
        during each call or must be explicitly provided during init/call.
        This simplifies XLA/graph execution.
        If the seed is provided in the init() function, this fixed seed is used
        for all calls. However, an explicit seed can be provided during
        the call function to realize `true` random states.

        Scrambling is typically used to ensure equal likely `0`  and `1` for
        sources with unequal bit probabilities. As we have a perfect source in
        the simulations, this is not required. However, for all-zero codeword
        simulations and higher-order modulation, so-called "channel-adaptation"
        [Pfister03]_ is required.

    Raises
    ------
        AssertionError
            If ``seed`` is not int.

        AssertionError
            If ``keep_batch_constant`` is not bool.

        AssertionError
            If ``binary`` is not bool.

        AssertionError
            If ``keep_state`` is not bool.

        AssertionError
            If ``seed`` is provided to list of inputs but not an
            int.

        TypeError
            If `dtype` of ``x`` is not as expected.
    """

    def __init__(self,
                 seed=None,
                 keep_batch_constant=False,
                 binary=True,
                 sequence=None,
                 keep_state=True,
                 dtype=torch.float32):

        super(Scrambler, self).__init__()

        if dtype not in (torch.float16, torch.float32, torch.float64,
                         torch.int8, torch.int32, torch.int64,
                         torch.uint8, torch.uint16, torch.uint32):
            raise TypeError("Unsupported dtype.")

        self.dtype = dtype
        assert isinstance(keep_batch_constant, bool), \
            "keep_batch_constant must be bool."
        self._keep_batch_constant = keep_batch_constant
        if seed is not None:
            if sequence is not None:
                print("Note: explicit scrambling sequence provided. " \
                      "Seed will be ignored.")
            assert isinstance(seed, int), "seed must be int."
        else:
            seed = int(np.random.uniform(0, 2**31-1))

        assert isinstance(binary, bool), "binary must be bool."
        assert isinstance(keep_state, bool), "keep_state must be bool."
        self._binary = binary
        self._keep_state = keep_state

        self._check_input = True
        # if keep_state==True this seed is used to generate scrambling sequences
        self._seed = self._seed = (1337, seed)
        # if an explicit sequence is provided the above parameters will be
        # ignored
        self._sequence = None

        if sequence is not None:
            sequence = torch.tensor(sequence, dtype=self.dtype)

            # check that sequence is binary
            is_binary = torch.all((sequence == 0) | (sequence == 1))
            assert is_binary.item(), "Scrambling sequence must be binary."
            self._sequence = sequence
    #########################################
    # Public methods and properties
    #########################################

    @property
    def seed(self):
        """Seed used to generate random sequence."""
        return self._seed[1] # only return the non-fixed seed

    @property
    def keep_state(self):
        """Indicates if new random sequences are used per call."""
        return self._keep_state

    @property
    def sequence(self):
        """Explicit scrambling sequence if provided."""
        return self._sequence

    #########################
    # Utility methods
    #########################

    def _generate_scrambling(self, input_shape, seed):
        r"""Generates a random sequence of `0`s and `1`s that can be used
        to initialize a scrambler and updates the internal attributes.
        """
        torch.manual_seed(seed[0] ^ seed[1])
        if self._keep_batch_constant:
            input_shape_no_bs = input_shape[1:]
            seq = torch.randint(0, 2, input_shape_no_bs, dtype=torch.int32)
            seq = seq.unsqueeze(0)  # expand batch dim for broadcasting
        else:
            seq = torch.randint(0, 2, input_shape, dtype=torch.int32)

        return seq.to(self.dtype)

    def forward(self, inputs):
        r"""scrambling function.

        This function returns the scrambled version of ``inputs``.

        ``inputs`` can be either a list ``[x, seed]`` or single tensor ``x``.

        Args:
            inputs (List): ``[x, seed]``, where
            ``x`` (torch.float32): Tensor of arbitrary shape.
            ``seed`` (int): An integer defining the state of the random number
                generator. If explicitly given, the global internal seed is
                replaced by this seed. Can be used the realize random
                scrambler/descrambler pairs (call with same random seed).

        Returns:
            `torch.float32`: Tensor of same shape as the input.

        Raises:
            AssertionError
                If ``seed`` is not None or int.

            TypeError
                If `dtype` of ``x`` is not as expected.
        """
        is_binary = self._binary # can be overwritten if explicitly provided
        if isinstance(inputs, (tuple, list)):
            if len(inputs)==1: # if user wants to call with call([x])
                seed = None
                x = inputs
            elif len(inputs)==2:
                x, seed = inputs
            elif len(inputs)==3:
            # allow that is_binary flag can be explicitly provided (descrambler)
                x, seed, is_binary = inputs
                # is binary can be either a tensor or bool
                if isinstance(is_binary, torch.Tensor):
                    if is_binary.dtype != torch.bool:
                        raise TypeError("binary must be bool.")
                else: # is boolean
                    assert isinstance(is_binary, bool), \
                    "binary must be bool."
            else:
                raise TypeError("inputs cannot have more than 3 entries.")
        else:
            seed = None
            x = inputs
        expected_type = self.dtype
        x = torch.tensor(x, dtype=self.dtype)
        assert_type(x, expected_type)
        input_shape = x.shape

        # generate random sequence on-the-fly (due to unknown shapes during
        # compile/build time)
        # use seed if explicit seed is provided
        if seed is not None:
            #assert seed.dtype.is_integer, "seed must be int."
            seed = (1337, seed)
        # only generate a new random sequence if keep_state==False
        elif self._keep_state:
            # use sequence as defined by seed
            seed = self._seed
        else:
            # generate new seed for each call
            # Note: not necessarily random if XLA is active
            seed = np.random.uniform(size=[2],
                                     low=0,
                                     high=2**31-1,
                                     )
            seed = torch.tensor(seed, dtype=torch.int32)
        # Apply explicit sequence if provided
        if self._sequence is not None:
            rand_seq = self._sequence
        else:
            rand_seq = self._generate_scrambling(input_shape, seed)

        if is_binary:
            x_out = torch.abs(x - rand_seq)
        else:
            rand_seq_bipol = -2 * rand_seq + 1
            x_out = x * rand_seq_bipol

        return x_out

class TB5GScrambler(nn.Module):
    r"""TB5GScrambler(n_rnti=1, n_id=1, binary=True, channel_type="PUSCH", codeword_index=0,  dtype=torch.float32, **kwargs)

    Implements the pseudo-random bit scrambling as defined in
    [3GPPTS38211_scr]_ Sec. 6.3.1.1 for the "PUSCH" channel and in Sec. 7.3.1.1
    for the "PDSCH" channel.

    Only for the "PDSCH" channel, the scrambler can be configured for two
    codeword transmission mode. Hereby, ``codeword_index`` corresponds to the
    index of the codeword to be scrambled.

    If ``n_rnti`` are a list of ints, the scrambler assumes that the second
    last axis contains `len(` ``n_rnti`` `)` elements. This allows independent
    scrambling for multiple independent streams.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        n_rnti: int or list of ints
            RNTI identifier provided by higher layer. Defaults to 1 and must be
            in range `[0, 65335]`. If a list is provided, every list element
            defines a scrambling sequence for multiple independent streams.

        n_id: int or list of ints
            Scrambling ID related to cell id and provided by higher layer.
            Defaults to 1 and must be in range `[0, 1023]`. If a list is
            provided, every list element defines a scrambling sequence for
            multiple independent streams.

        binary: bool
            Defaults to True. Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

        channel_type: str
            Can be either "PUSCH" or "PDSCH".

        codeword_index: int
            Scrambler can be configured for two codeword transmission.
            ``codeword_index`` can be either 0 or 1.

        dtype: torch.dtype
            Defaults to `torch.float32`. Defines the datatype for internal
            calculations and the output dtype.
    Input
    -----
        (x, binary):
            Either Tuple ``(x, binary)`` or  ``x`` only

        x: torch.float
            1+D tensor of arbitrary shape. If ``n_rnti`` and ``n_id`` are a
            list, it is assumed that ``x`` has shape
            `[...,num_streams, n]` where `num_streams=len(` ``n_rnti`` `)`.

        binary: bool
            Overrules the init parameter `binary` iff explicitly given.
            Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

    Output
    ------
        : torch.float
            1+D tensor of same shape as ``x``.

    Note
    ----
        The parameters radio network temporary identifier (RNTI) ``n_rnti`` and
        the datascrambling ID ``n_id`` are usually provided be the higher layer protocols.

        For inverse scrambling, the same scrambler can be re-used (as the values
        are flipped again, i.e., result in the original state).
    """

    def __init__(self, n_rnti=1, n_id=1, binary=True, channel_type="PUSCH",
                 codeword_index=0, dtype=torch.float32):
        super().__init__()

        if dtype not in (torch.float16, torch.float32, torch.float64, 
                         torch.int8, torch.int32, torch.int64, 
                         torch.uint8):
            raise TypeError("Unsupported dtype.")
        self.dtype = dtype
        assert isinstance(binary, bool), "binary must be a bool."
        assert channel_type in ("PDSCH", "PUSCH"), "Unsupported channel_type."
        assert codeword_index in (0, 1), "codeword_index must be 0 or 1."

        self._binary = binary
        self._check_input = True
        self._input_shape = None

        # allow list input for independent multi-stream scrambling
        if isinstance(n_rnti, (list, tuple)):
            assert isinstance(n_id, (list, tuple)), \
                                "n_id must be a list of same length as n_rnti."

            assert len(n_rnti)==len(n_id), \
                        "n_rnti and n_id must be of same length."

            self._multi_stream = True
        else:
            n_rnti = [n_rnti]
            n_id = [n_id]
            self._multi_stream = False

        # check all entries for consistency
        for idx, (nr, ni) in enumerate(zip(n_rnti, n_id)):
            # allow floating inputs, but verify that it represent an int value
            assert(nr%1==0), "n_rnti must be integer."
            assert nr in range(2**16), "n_rnti must be in [0, 65535]."
            n_rnti[idx] = int(nr)
            assert(ni%1==0), "n_rnti must be integer."
            assert ni in range(2**10), "n_id must be in [0, 1023]."
            n_id[idx] = int(ni)

        self._c_init = []
        if channel_type=="PUSCH":
            # defined in 6.3.1.1 in 38.211
            for nr, ni in zip(n_rnti, n_id):
                self._c_init += [nr * 2**15 + ni]
        elif channel_type =="PDSCH":
            # defined in 7.3.1.1 in 38.211
            for nr, ni in zip(n_rnti, n_id):
                self._c_init += [nr * 2**15 + codeword_index * 2**14 + ni]
    #########################################
    # Public methods and properties
    #########################################

    @property
    def keep_state(self):
        """Required for descrambler, is always `True` for the TB5GScrambler."""
        return True

    #########################
    # Utility methods
    #########################
    def _generate_scrambling(self, input_shape):
        r"""Returns random sequence of `0`s and `1`s following
        [3GPPTS38211_scr]_ ."""

        seq = generate_prng_seq(input_shape[-1], self._c_init[0])
        seq = torch.tensor(seq, dtype=self.dtype)
        seq = expand_to_rank(seq, len(input_shape), axis=0)

        if self._multi_stream:
            for c in self._c_init[1:]:
                s = generate_prng_seq(input_shape[-1], c)
                s = torch.tensor(s, dtype=self.dtype)
                s = expand_to_rank(s, len(input_shape), axis=0)
                seq = torch.cat([seq, s], dim=-2)

        return seq

    def forward(self, inputs):
        """Applies scrambling to the input tensor."""
        
        is_binary = self._binary  # This can be overridden by explicit input
        
        # Determine the input tensor and optional binary flag
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 1:
                x, = inputs
            elif len(inputs) == 2:
                # allow that binary flag is explicitly provided (descrambler)
                x, is_binary = inputs
                # is_binary can be either a tensor or bool
                if isinstance(is_binary, torch.Tensor):
                    if not is_binary.dtype.is_bool:
                        raise TypeError("binary must be bool.")
                else: # is boolean
                    assert isinstance(is_binary, bool), \
                    "binary must be bool."
            else:
                raise TypeError("inputs cannot have more than 3 entries.")
        else:
            x = inputs
        x = torch.tensor(x, dtype=self.dtype)
        # Set the shape of input if not already set
        if not x.shape[-1]==self._input_shape:
            input_shape = x.shape
            # input can be also a list, we are only interested in the shape of x
            
            self._input_shape = input_shape

            # in multi-stream mode, the axis=-2 must have dimension=len(c_init)
            if self._multi_stream:
                assert input_shape[-2]==len(self._c_init), \
                    "Dimension of axis=-2 must be equal to len(n_rnti)."

            self._sequence = self._generate_scrambling(input_shape)

        

        # Apply binary or sign flipping based on is_binary flag
        if is_binary:
            # flip the bits by subtraction and map -1 to 1 via abs(.) operator
            x_out = torch.abs(x - self._sequence)
        else:
            rand_seq_bipol = -2 * self._sequence + 1
            x_out = x * rand_seq_bipol

        return x_out

class Descrambler(nn.Module):
    r"""Descrambler(scrambler, binary=True, dtype=None, **kwargs)

    Descrambler for a given scrambler.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        scrambler: Scrambler, TB5GScrambler
            Associated `Scrambler` or `TB5GScrambler` instance which
            should be descrambled.

        binary: bool
            Defaults to True. Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

        dtype: torch.dtype
            Defaults to `None`. Defines the datatype for internal calculations
            and the output dtype. If no explicit dtype is provided, the dtype
            from the associated scrambler is used.
    Input
    -----
        (x, seed):
            Either Tuple ``(x, seed)`` or ``x`` only (no tuple) if the internal
            seed should be used:

        x: torch.float
            1+D tensor of arbitrary shape.

        seed: int
            An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (call with same random seed).

    Output
    ------
        : torch.float
            1+D tensor of same shape as ``x``.

    Raise
    -----
        AssertionError
            If ``scrambler`` is not an instance of `Scrambler`.

        AssertionError
            If ``seed`` is provided to list of inputs but not an
            int.

        TypeError
            If `dtype` of ``x`` is not as expected.
    """

    def __init__(self, scrambler, binary=True, dtype=None):
        super().__init__()
        self.dtype = dtype
        assert isinstance(scrambler, (Scrambler, TB5GScrambler)), \
            "scrambler must be an instance of Scrambler or TB5GScrambler."
        self._scrambler = scrambler

        assert isinstance(binary, bool), "binary must be bool."
        self._binary = binary

        # if dtype is None, use the same dtype as associated scrambler
        if dtype is None:
            dtype = self._scrambler.dtype

        # Warning if scrambler's state cannot be retained
        if self._scrambler.keep_state is False:
            print("Warning: scrambler uses random sequences that cannot be " \
                  "access by descrambler. Please use keep_state=True and " \
                  "provide explicit random seed as input to call function.")

        if self._scrambler.dtype != self.dtype:
            print("Scrambler and descrambler are using different " \
                "dtypes. This will cause an internal implicit cast.")


    #########################################
    # Public methods and properties
    #########################################
    @property
    def scrambler(self):
        """Returns the associated scrambler instance."""
        return self._scrambler

    #########################
    # Utility methods
    #########################
    def forward(self, inputs):
        r"""Descrambling function.

        This function returns the descrambled version of ``inputs``.

        ``inputs`` can be either a list ``[x, seed]`` or single tensor ``x``.

        Args:
            inputs (List): ``[x, seed]``, where
            ``x`` (torch.float32): Tensor of arbitrary shape.
            ``seed`` (int): An integer defining the state of the random number
                generator. If not explicitly given, the global internal seed is
                replaced by this seed. Can be used the realize random
                scrambler/descrambler pairs (must be called with same random
                seed).

        Returns:
            `torch.float32`: Tensor of same shape as the input.

        Raises:
            AssertionError: If ``seed`` is not `None` or `int`.
        """
        if isinstance(self._scrambler, Scrambler):
            if isinstance(inputs, (tuple, list)):
                if len(inputs) > 2:
                    raise TypeError("inputs cannot have more than 2 entries.")
                else:
                    inputs.append(self._binary)
            else:
                s = self._scrambler.seed
                inputs = (inputs, s, self._binary)

        elif isinstance(self._scrambler, TB5GScrambler):
            if isinstance(inputs, (tuple, list)):
                if len(inputs) > 1:
                    raise TypeError("inputs cannot have more than 1 entry.")
                else:
                    inputs.append(self._binary)
            else:
                inputs = (inputs, self._binary)

        else:
            raise TypeError("Unknown Scrambler type.")

        x_out = self._scrambler(inputs)
        
        # Ensure output dtype matches descrambler dtypes
        return x_out.to(self.dtype)
