import warnings
import numpy as np
import torch
from comcloak.utils import expand_to_rank, matrix_sqrt_inv, flatten_last_dims, flatten_dims, split_dim, insert_dims, hard_decisions
from comcloak.mapping import Constellation, SymbolLogits2LLRs, LLRs2SymbolLogits, PAM2QAM, Demapper, SymbolDemapper, SymbolInds2Bits, DemapperWithPrior, SymbolLogits2Moments
from comcloak.mimo.utils_z import complex2real_channel, whiten_channel, List2LLR, List2LLRSimple, complex2real_matrix, complex2real_vector, real2complex_vector
from comcloak.mimo.equalization_z import lmmse_equalizer, zf_equalizer, mf_equalizer
from comcloak.supplement import gather_pytorch, gather_nd_pytorch, get_real_dtype
import torch.nn as nn
import torch.nn.functional as F


class LinearDetector(nn.Module):
    # pylint: disable=line-too-long
    r"""LinearDetector(equalizer, output, demapping_method, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    Convenience class that combines an equalizer,
    such as :func:`~sionna.mimo.lmmse_equalizer`, and a :class:`~sionna.mapping.Demapper`.

    Parameters
    ----------
    equalizer : str, one of ["lmmse", "zf", "mf"], or an equalizer function
        The equalizer to be used. Either one of the existing equalizers
        :func:`~sionna.mimo.lmmse_equalizer`, :func:`~sionna.mimo.zf_equalizer`, or
        :func:`~sionna.mimo.mf_equalizer` can be used, or a custom equalizer
        callable provided that has the same input/output specification.

    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

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

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices

    Output
    ------
    One of:

    : [..., num_streams, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`

    : [..., num_streams, num_points], tf.float or [..., num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`
       Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you might need to set ``sionna.Config.xla_compat=true``. This depends on the
    chosen equalizer function. See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 equalizer,
                 output,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):
        super(LinearDetector, self).__init__()

        self._output = output
        self._hard_out = hard_out

        # Determine the equalizer to use
        if isinstance(equalizer, str):
            assert equalizer in ["lmmse", "zf", "mf"], "Unknown equalizer."
            if equalizer == "lmmse":
                self._equalizer = lmmse_equalizer
            elif equalizer == "zf":
                self._equalizer = zf_equalizer
            else:
                self._equalizer = mf_equalizer
        else:
            self._equalizer = equalizer

        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        constellation = Constellation.create_or_check_constellation(
                                                            constellation_type,
                                                            num_bits_per_symbol,
                                                            constellation,
                                                            dtype=dtype)
        self._constellation = constellation

        # Determine the demapper to use
        if output == "bit":
            self._demapper = Demapper(demapping_method, 
                                      constellation=self._constellation,
                                      hard_out=hard_out,
                                      dtype=dtype)
        else:
            self._demapper = SymbolDemapper(constellation=self._constellation,
                                            hard_out=hard_out,
                                            dtype=dtype)

    def forward(self, inputs):
        x_hat, no_eff = self._equalizer(*inputs)
        z = self._demapper([x_hat, no_eff])

        # Reshape to the expected output shape
        num_streams = inputs[1].shape[-1]
        if self._output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            z = split_dim(z, [num_streams, num_bits_per_symbol], z.dim()-1)

        return z

class MaximumLikelihoodDetector(nn.Module):
    # pylint: disable=line-too-long
    r"""
    MaximumLikelihoodDetector(output, demapping_method, num_streams, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, with_prior=False, dtype=tf.complex64, **kwargs)

    MIMO maximum-likelihood (ML) detector.
    If the ``with_prior`` flag is set, prior knowledge on the bits or constellation points is assumed to be available.

    This layer implements MIMO maximum-likelihood (ML) detection assuming the
    following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^K` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.
    If the ``with_prior`` flag is set, it is assumed that prior information of the transmitted signal :math:`\mathbf{x}` is available,
    provided either as LLRs on the bits mapped onto :math:`\mathbf{x}` or as logits on the individual
    constellation points forming :math:`\mathbf{x}`.

    Prior to demapping, the received signal is whitened:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}

    The layer can compute ML detection of symbols or bits with either
    soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
    and not jointly for the entire vector :math:`\textbf{x}` (or the underlying vector
    of bits).

    **\ML detection of bits:**

    Soft-decisions on bits are called log-likelihood ratios (LLR).
    With the “app” demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is then computed according to

    .. math::
        \begin{align}
            LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
                    &=\ln\left(\frac{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }\right)
        \end{align}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the
    sets of vectors of constellation points for which the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is equal to 1 and 0, respectively.
    :math:`\Pr\left( \mathbf{x} \right)` is the prior distribution of the vector of
    constellation points :math:`\mathbf{x}`. Assuming that the constellation points and
    bit levels are independent, it is computed from the prior of the bits according to

    .. math::
        \Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)

    where :math:`LLR_p(k,i)` is the prior knowledge of the :math:`i\text{th}` bit of the
    :math:`k\text{th}` user given as an LLR and which is set to :math:`0` if no prior knowledge is assumed to be available,
    and :math:`\sigma\left(\cdot\right)` is the sigmoid function.
    The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)`.

    With the "maxlog" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is approximated like

    .. math::
        \begin{align}
            LLR(k,i) \approx&\ln\left(\frac{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }\right)\\
                = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
                    \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
            \end{align}

    **ML detection of symbols:**

    Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

    With the “app” demapping method, the logit for the
    constellation point :math:`c \in \mathcal{C}` of the :math:`k\text{th}` user  is computed according to

    .. math::
        \begin{align}
            \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right)\right).
        \end{align}

    With the "maxlog" demapping method, the logit for the constellation point :math:`c \in \mathcal{C}`
    of the :math:`k\text{th}` user  is approximated like

    .. math::
        \text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
                \right).

    When hard decisions are requested, this layer returns for the :math:`k` th stream

    .. math::
        \hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right) \right)

    where :math:`\mathcal{C}` is the set of constellation points.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    num_streams : tf.int
        Number of transmitted streams

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

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the bits or constellation points is available.
        This prior information is given as LLRs (for bits) or log-probabilities (for constellation points) as an
        additional input to the layer.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h, s) or (y, h, prior, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices.

    prior : [...,num_streams,num_bits_per_symbol] or [...,num_streams,num_points], tf.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", then LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", then logits of the transmitted constellation points are expected.
        Only required if the ``with_prior`` flag is set.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    One of:

    : [..., num_streams, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [..., num_streams, num_points], tf.float or [..., num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
       Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 demapping_method,
                 num_streams,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 with_prior=False,
                 dtype=torch.complex64,
                 ):
        super().__init__()

        assert dtype in [torch.complex64, torch.complex128],\
            "dtype must be torch.complex64 or torch.complex128"

        assert output in ("bit", "symbol"), "Unknown output"

        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        self._output = output
        self._demapping_method = demapping_method
        self._hard_out = hard_out
        self._with_prior = with_prior

        # Determine the reduce function for LLR computation
        if self._demapping_method == "app":
            self._reduce = torch.logsumexp
        else:
            self._reduce = torch.max

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)

        # Utility function to compute
        # vecs : [num_vecs, num_streams] The list of all possible transmitted vectors.
        # vecs_ind : [num_vecs, num_streams] The list of all possible transmitted vectors
        #   constellation indices
        # c : [num_vecs/num_points, num_streams, num_points] Which is such that `c[:,k,s]`
        #   gives the symbol indices in the first dimension of `vecs` for which
        #   the `k`th stream transmitted the `s`th constellation point.
        vecs, vecs_ind, c = self._build_vecs(num_streams)
        self._vecs = torch.tensor(vecs, dtype=dtype)
        self._vecs_ind = torch.tensor(vecs_ind, dtype=torch.int32)
        self._c = torch.tensor(c, dtype=torch.int32)

        if output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            self._logits2llr = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=num_bits_per_symbol,
                hard_out=hard_out,
                dtype=dtype.real_dtype)
            self._llrs2logits = LLRs2SymbolLogits(
                num_bits_per_symbol=num_bits_per_symbol,
                hard_out=False,
                dtype=dtype.real_dtype)
    @property
    def constellation(self):
        return self._constellation

    def _build_vecs(self, num_streams):
        """
        Utility function for building the list of all possible transmitted
        vectors of constellation points and the symbol indices corresponding to
        all possibly transmitted constellation points for every stream.

        Input
        ------
        num_streams : int
            Number of transmitted streams

        Output
        -------
        vecs : [num_vecs, K], tf.complex
            List of all possible transmitted vectors.

        c : [num_vecs/num_points, num_streams, num_points], int
            `c[:,k,s]` gives the symbol indices in the first dimension of `vecs`
            for which the `k`th stream transmitted the `s`th symbol.
        """
        points = self._constellation.points
        num_points = points.shape[0]

        def _build_vecs_(n):
            if n == 1:
                # If there is a single stream, then the list of possibly
                # transmitted vectors corresponds to the constellation points.
                # No recusrion is needed.
                vecs = np.expand_dims(points, axis=1)
                vecs_ind = np.expand_dims(np.arange(num_points), axis=1)
            else:
                # If the number of streams is `n >= 2` streams, then the list
                # of possibly transmitted vectors is
                #
                # [c_1 v , c_2 v, ..., c_N v]
                #
                # where `[c_1, ..., c_N]` is the constellation of size N, and
                # `v` is the list of possible vectors for `n-1` streams.
                # This list has therefore length `N x len(v)`.
                #
                # Building the list for `n-1` streams, recursively.
                v, vi = _build_vecs_(n-1)
                # Building the list of `n` streams by appending the
                # constellation points.
                vecs = []
                vecs_ind = []
                for i,p in enumerate(points):
                    vecs.append(np.concatenate([np.full([v.shape[0], 1], p),
                                                v], axis=1))
                    vecs_ind.append(np.concatenate([np.full([v.shape[0], 1], i),
                                                vi], axis=1))
                vecs = np.concatenate(vecs, axis=0)
                vecs_ind = np.concatenate(vecs_ind, axis=0)
            return vecs, vecs_ind

        # Building the list of possible vectors for the `k` streams.
        # [num_vecs, K]
        vecs, vecs_ind = _build_vecs_(num_streams)

        tx_ind = np.arange(num_streams)
        tx_ind = np.expand_dims(tx_ind, axis=0)
        tx_ind = np.tile(tx_ind, [vecs_ind.shape[0], 1])
        vecs_ind = np.stack([tx_ind, vecs_ind], axis=-1)

        # Compute symbol indices for every stream.
        # For every constellation point `p` and for every stream `j`, we gather
        # the list of vector indices from `vecs` corresponding the vectors for
        # which the `jth` stream transmitted `p`.
        # [num_vecs/num_points, num_streams, num_points]
        c = []
        for p in points:
            c_ = []
            for j in range(num_streams):
                c_.append(np.where(vecs[:,j]==p)[0])
            c_ = np.stack(c_, axis=-1)
            c.append(c_)
        c = np.stack(c, axis=-1)

        return vecs, vecs_ind, c

    def forward(self, inputs):
        if self._with_prior:
            y, h, prior, s = inputs

            # If operating on bits, computes prior on symbols from the prior
            # on bits
            if self._output == 'bit':
                # [..., K, num_points]
                prior = self._llrs2logits(prior)
        else:
            y, h, s = inputs

        s_inv = matrix_sqrt_inv(s)

        # Whiten the observation
        y = y.unsqueeze(-1)
        y = (s_inv @ y).squeeze(-1)
        # Compute channel after whitening
        h = s_inv @ h
        # Add extra dims for broadcasting with the dimensions corresponding
        # to all possible transmimtted vectors
        # Shape: [..., 1, M, K]
        h = h.unsqueeze(-3)
        # Add extra dims for broadcasting with the dimensions corresponding
        # to all possible transmimtted vectors
        # Shape: [..., 1, M]
        y = y.unsqueeze(-2)

        # Reshape list of all possible vectors from
        # [num_vecs, K]
        # to
        # [1,...,1, num_vecs, K, 1]
        vecs = self._vecs.unsqueeze(-1)
        vecs = expand_to_rank(vecs, h.dim(), 0)
        # Compute exponents
        # [..., num_vecs]
        diff = y - (h @ vecs).squeeze(-1)
        exponents = -torch.sum(diff.abs() ** 2, dim=-1)

        # Add prior
        if self._with_prior:
            # [..., num_vecs, K]
            prior = expand_to_rank(prior, exponents.dim(), axis=0)
            prior_rank = prior.dim()
            transpose_ind = torch.cat([
            torch.tensor([prior_rank - 2, prior_rank - 1]),  # Last two dimensions
            torch.arange(prior_rank - 2)  # The rest of the dimensions
            ])
            prior = prior.permute(transpose_ind.tolist())
            prior = gather_nd_pytorch(prior, self._vecs_ind)
            transpose_ind = torch.cat([
            torch.arange(2, prior_rank),  
            torch.tensor([0, 1])  
            ])
            prior = prior.permute(transpose_ind.tolist())
            # [..., num_vecs]
            prior = torch.sum(prior, dim=-1)
            exponents = exponents + prior

        # Gather exponents for all symbols
        # [..., num_vecs/num_points, K, num_points]
        exp = gather_pytorch(exponents, self._c, axis=-1)

        # Compute logits on constellation points
        # [..., K, num_points]
        logits = self._reduce(exp, dim=-3)

        if self._output == 'bit':
            # Compute LLRs or hard decisions
            return self._logits2llr(logits)
        else:
            if self._hard_out:
                return torch.argmax(logits, dim=-1)
            else:
                return logits

class MaximumLikelihoodDetectorWithPrior(MaximumLikelihoodDetector):
    r"""
    MaximumLikelihoodDetectorWithPrior(output, demapping_method, num_streams, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    MIMO maximum-likelihood (ML) detector, assuming prior
    knowledge on the bits or constellation points is available.

    This class is deprecated as the functionality has been integrated
    into :class:`~sionna.mimo.MaximumLikelihoodDetector`.

    This layer implements MIMO maximum-likelihood (ML) detection assuming the
    following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^K` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.
    It is assumed that prior information of the transmitted signal :math:`\mathbf{x}` is available,
    provided either as LLRs on the bits modulated onto :math:`\mathbf{x}` or as logits on the individual
    constellation points forming :math:`\mathbf{x}`.

    Prior to demapping, the received signal is whitened:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}

    The layer can compute ML detection of symbols or bits with either
    soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
    and not jointly for the entire vector :math:`\textbf{x}` (or the underlying vector
    of bits).

    **\ML detection of bits:**

    Soft-decisions on bits are called log-likelihood ratios (LLR).
    With the “app” demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is then computed according to

    .. math::
        \begin{align}
            LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
                    &=\ln\left(\frac{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }\right)
        \end{align}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the
    sets of vectors of constellation points for which the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is equal to 1 and 0, respectively.
    :math:`\Pr\left( \mathbf{x} \right)` is the prior distribution of the vector of
    constellation points :math:`\mathbf{x}`. Assuming that the constellation points and
    bit levels are independent, it is computed from the prior of the bits according to

    .. math::
        \Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)

    where :math:`LLR_p(k,i)` is the prior knowledge of the :math:`i\text{th}` bit of the
    :math:`k\text{th}` user given as an LLR, and :math:`\sigma\left(\cdot\right)` is the sigmoid function.
    The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)`.

    With the "maxlog" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is approximated like

    .. math::
        \begin{align}
            LLR(k,i) \approx&\ln\left(\frac{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }\right)\\
                = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
                    \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
            \end{align}

    **ML detection of symbols:**

    Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

    With the “app” demapping method, the logit for the
    constellation point :math:`c \in \mathcal{C}` of the :math:`k\text{th}` user  is computed according to

    .. math::
        \begin{align}
            \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right)\right).
        \end{align}

    With the "maxlog" demapping method, the logit for the constellation point :math:`c \in \mathcal{C}`
    of the :math:`k\text{th}` user  is approximated like

    .. math::
        \text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
                \right).

    When hard decisions are requested, this layer returns for the :math:`k` th stream

    .. math::
        \hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right) \right)

    where :math:`\mathcal{C}` is the set of constellation points.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    num_streams : tf.int
        Number of transmitted streams

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

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h, prior, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices.

    prior : [...,num_streams,num_bits_per_symbol] or [...,num_streams,num_points], tf.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", then LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", then logits of the transmitted constellation points are expected.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    One of:

    : [..., num_streams, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [..., num_streams, num_points], tf.float or [..., num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
       Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                output,
                demapping_method,
                num_streams,
                constellation_type=None,
                num_bits_per_symbol=None,
                constellation=None,
                hard_out=False,
                dtype=torch.complex64,
                **kwargs):
        super().__init__(output=output,
                        demapping_method=demapping_method,
                        num_streams=num_streams,
                        constellation_type=constellation_type,
                        num_bits_per_symbol=num_bits_per_symbol,
                        constellation=constellation,
                        hard_out=hard_out,
                        with_prior=True,
                        dtype=dtype,
                        **kwargs)

class KBestDetector(nn.Module):
    r"""KBestDetector(output, num_streams, k, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, use_real_rep=False, list2llr=None, dtype=tf.complex64)

    MIMO K-Best detector

    This layer implements K-Best MIMO detection as described
    in (Eq. 4-5) [FT2015]_. It can either generate hard decisions (for symbols
    or bits) or compute LLRs.

    The algorithm operates in either the complex or real-valued domain.
    Although both options produce identical results, the former has the advantage
    that it can be applied to arbitrary non-QAM constellations. It also reduces
    the number of streams (or depth) by a factor of two.

    The way soft-outputs (i.e., LLRs) are computed is determined by the
    ``list2llr`` function. The default solution
    :class:`~sionna.mimo.List2LLRSimple` assigns a predetermined
    value to all LLRs without counter-hypothesis.

    This layer assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    In a first optional step, the channel model is converted to its real-valued equivalent,
    see :func:`~sionna.mimo.complex2real_channel`. We assume in the sequel the complex-valued
    representation. Then, the channel is whitened using :func:`~sionna.mimo.whiten_channel`:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}.

    Next, the columns of :math:`\tilde{\mathbf{H}}` are sorted according
    to their norm in descending order. Then, the QR decomposition of the
    resulting channel matrix is computed:

    .. math::
        \tilde{\mathbf{H}} = \mathbf{Q}\mathbf{R}

    where :math:`\mathbf{Q}\in\mathbb{C}^{M\times S}` is unitary and
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is upper-triangular.
    The channel outputs are then pre-multiplied by :math:`\mathbf{Q}^{\mathsf{H}}`.
    This leads to the final channel model on which the K-Best detection algorithm operates:

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S`,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S`, and :math:`\bar{\mathbf{n}}\in\mathbb{C}^S`
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    **LLR Computation**

    The K-Best algorithm produces :math:`K` candidate solutions :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. If the real-valued channel representation is used, the distance
    metrics are scaled by 0.5 to account for the reduced noise power in each complex dimension.
    A hard-decision is simply the candidate with the shortest distance.
    Various ways to compute LLRs from this list (and possibly
    additional side-information) are possible. The (sub-optimal) default solution
    is :class:`~sionna.mimo.List2LLRSimple`. Custom solutions can be provided.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    num_streams : tf.int
        Number of transmitted streams

    k : tf.int
        The number of paths to keep. Cannot be larger than the
        number of constellation points to the power of the number of
        streams.

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
        Defaults to `False`. The detector cannot compute soft-symbols.

    use_real_rep : bool
        If `True`, the detector use the real-valued equivalent representation
        of the channel. Note that this only works with a QAM constellation.
        Defaults to `False`.

    list2llr: `None` or instance of :class:`~sionna.mimo.List2LLR`
        The function to be used to compute LLRs from a list of candidate solutions.
        If `None`, the default solution :class:`~sionna.mimo.List2LLRSimple`
        is used.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    -----
    (y, h, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices

    Output
    ------
    One of:

    : [...,num_streams,num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`

    : [...,num_streams,2**num_points], tf.float or [...,num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`
       Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    
    def __init__(self,
                 output,
                 num_streams,
                 k,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 use_real_rep=False,
                 list2llr="default",
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__()
        assert dtype in [torch.complex64, torch.complex128],\
            "dtype must be torch.complex64 or torch.complex128."

        assert output in ("bit", "symbol"), "Unknown output"

        if constellation is None:
            assert constellation_type is not None and \
                   num_bits_per_symbol is not None, \
                   "You must provide either constellation or constellation_type and num_bits_per_symbol."
        else:
            assert constellation_type is None and \
                   num_bits_per_symbol is None, \
                   "You must provide either constellation or constellation_type and num_bits_per_symbol."

        if constellation is not None:
            assert constellation.dtype == dtype, \
                "Constellation has wrong dtype."

        self._output = output
        self._hard_out = hard_out
        self._use_real_rep = use_real_rep

        if self._use_real_rep:
            # Real-valued representation is used
            err_msg = "Only QAM can be used for the real-valued representation"
            if constellation_type is not None:
                assert constellation_type == "qam", err_msg
            else:
                assert constellation._constellation_type == "qam", err_msg

            # Double the number of streams to dectect
            self._num_streams = 2 * num_streams

            # Half the number of bits for the PAM constellation
            if num_bits_per_symbol is None:
                n = constellation.num_bits_per_symbol // 2
                self._num_bits_per_symbol = n
            else:
                self._num_bits_per_symbol = num_bits_per_symbol // 2

             # Geerate a PAM constellation with 0.5 energy
            c = Constellation("pam",
                                self._num_bits_per_symbol,
                                normalize=False,
                                dtype=dtype)            
            c._points /= torch.std(c.points) * torch.sqrt(torch.tensor(2.0, dtype=dtype))
            self._constellation = c.points

            self._pam2qam = PAM2QAM(2 * self._num_bits_per_symbol)

        else:
            # Complex-valued representation is used
            # Number of streams is equal to number of transmitters
            self._num_streams = num_streams

            # Create constellation or take the one provided
            c = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)
            self._constellation = c.points
            self._num_bits_per_symbol = c.num_bits_per_symbol

        # Number of constellation symbols
        self._num_symbols = self._constellation.shape[0]

        # Number of best paths to keep
        self._k = np.minimum(k, self._num_symbols**self._num_streams)
        if self._k < k:
            warnings.warn(f"KBestDetector: The provided value of k={k} is larger than the possible maximum number of paths. It has been set to k={self._k}.")

        num_paths = [1]
        for l in range(1, self._num_streams + 1):
            num_paths.append(min(self._k, self._num_symbols ** l))
        self._num_paths = torch.tensor(num_paths, dtype=torch.int32)

        indices = torch.zeros([self._num_streams, self._k * self._num_streams, 2], dtype=torch.int32)
        for l in range(self._num_streams):
            ind = torch.zeros([self._num_paths[l + 1], self._num_streams], dtype=torch.int32)
            ind[:, :l + 1] = 1
            ind = torch.stack(torch.where(ind), dim=-1)
            indices[l, :ind.shape[0], :ind.shape[1]] = ind
        self._indices = indices

        if self._output == "bit":
            if not self._hard_out:
                if list2llr == "default":
                    self.list2llr = List2LLRSimple(self._num_bits_per_symbol)
                else:
                    self.list2llr = list2llr
            else:
                n = 2 * self._num_bits_per_symbol if self._use_real_rep else self._num_bits_per_symbol
                self._symbolinds2bits = SymbolInds2Bits(n, dtype=dtype.real_dtype)
        else:
            assert self._hard_out, "Soft-symbols are not supported for this detector."

    @property
    def list2llr(self):
        return self._list2llr

    @list2llr.setter
    def list2llr(self, value):
        assert isinstance(value, List2LLR)
        self._list2llr = value

    def _preprocessing(self, inputs):

        y, h, s = inputs
        # np.save('/home/wzs/project/sionna-main/function_test/tensor_compare/pttensor.npy', h.numpy())
        # Convert to real-valued representation if desired
        if self._use_real_rep:
            y, h, s = complex2real_channel(y, h, s)

        y, h = whiten_channel(y, h, s, return_s=False)

        h_norm = torch.sum(h.abs() ** 2, dim=1)
        # column_order = torch.argsort(h_norm, dim=-1, descending=True)
        flattened_tensor = h_norm.view(-1, h_norm.shape[-1])
        column_order = torch.stack([
            torch.tensor(sorted(range(flattened_tensor.shape[-1]), key=lambda x: (-flattened_tensor[i, x], x)))
            for i in range(flattened_tensor.shape[0])
        ]).reshape(h_norm.shape)
        
        h = gather_pytorch(h, column_order, axis=-1, batch_dims=1)

        q, r = torch.qr(h)

        y = torch.squeeze(torch.matmul(q.conj().transpose(-1, -2), y.unsqueeze(-1)), -1)

        return y, r, column_order

    def _select_best_paths(self, dists, path_syms, path_inds):
        num_paths = path_syms.shape[1]
        k = min(num_paths, self._k)
        dists, ind = torch.topk(-dists, k=k, sorted=True)
        dists = -dists

        path_syms = gather_pytorch(path_syms, ind, axis=1, batch_dims=1)
        path_inds = gather_pytorch(path_inds, ind, axis=1, batch_dims=1)

        return dists, path_syms, path_inds

    def _next_layer(self, y, r, dists, path_syms, path_inds, stream):

        batch_size = y.shape[0]

        # Streams are processed in reverse order
        stream_ind = self._num_streams-1-stream

        # Current number of considered paths
        num_paths = gather_pytorch(self._num_paths, stream)

        dists_o = dists.clone()
        path_syms_o = path_syms.clone()
        path_inds_o = path_inds.clone()

        dists = dists[..., :num_paths]
        path_syms = path_syms[..., :num_paths, :stream]
        path_inds = path_inds[..., :num_paths, :stream]

        dists = dists.repeat(1, self._num_symbols)
        path_syms = path_syms.repeat(1, self._num_symbols, 1)
        path_inds = path_inds.repeat(1, self._num_symbols, 1)

        syms = self._constellation.view(1, -1)
        syms = syms.repeat(self._k, 1)
        syms = syms.view(1, -1, 1)
        syms = syms.repeat(batch_size, 1, 1)
        syms = syms[:, :num_paths * self._num_symbols]
        path_syms = torch.cat([path_syms, syms], dim=-1)

        inds = torch.arange(0, self._num_symbols).view(1, -1)
        inds = inds.repeat(self._k, 1)
        inds = inds.view(1, -1, 1)
        inds = inds.repeat(batch_size, 1, 1)
        inds = inds[:, :num_paths * self._num_symbols]
        path_inds = torch.cat([path_inds, inds], dim=-1)

        y = y[:, stream_ind].unsqueeze(-1)
        r = torch.flip(r[:, stream_ind, stream_ind:], [-1]).unsqueeze(1)
        delta = (y - torch.sum(r * path_syms, dim=-1)).abs().pow(2)

        dists += delta

        dists, path_syms, path_inds = self._select_best_paths(dists, path_syms, path_inds)

        tensor = dists_o.permute(1, 0)
        updates = dists.permute(1, 0)
        indices = torch.arange(updates.shape[0], dtype=torch.int32).unsqueeze(-1)
        dists = tensor[tuple(indices.t())] + updates
        dists = dists.permute(1, 0)
        tensor = path_syms_o.permute(1, 2, 0)
        updates = path_syms.permute(1, 2, 0).contiguous().view(-1, batch_size)
        indices = self._indices[stream, :self._num_paths[stream + 1] * (stream + 1)]
        path_syms = tensor[tuple(indices.t())] + updates
        path_syms = path_syms.permute(2, 0, 1)
        tensor = path_inds_o.permute(1, 2, 0)
        updates = path_inds.permute(1, 2, 0).contiguous().view(-1, batch_size)
        path_inds = tensor[tuple(indices.t())] + updates
        path_inds = path_inds.permute(2, 0, 1)

        return dists, path_syms, path_inds

    def _unsort(self, column_order, tensor, transpose=True):
        # Undo the column sorting
        # If transpose=True, the unsorting is done along the last dimension
        # Otherwise, sorting is done along the second-last index
        unsort_inds = torch.argsort(column_order, axis=-1)
        if transpose:
            tensor = tensor.permute(0, 2, 1)
        tensor = gather_pytorch(tensor, unsort_inds, axis=-2, batch_dims=1)
        if transpose:
            tensor = tensor.permute(0, 2, 1)
        return tensor

    def _logits2llrs(self, logits, path_inds):
        # Implementation details depend on List2LLR class
        llrs = self.list2llr(logits, path_inds)
        return llrs

    def forward(self, inputs):
        for tensor in inputs:
            input_shape = tensor.shape 
            assert input_shape[-2] >= input_shape[-1], \
                "The number of receive antennas cannot be smaller than the number of streams"
        # Flatten the batch dimensions
        y, h, s = inputs
        batch_shape = y.shape[:-1]
        num_batch_dims = len(batch_shape)
        if num_batch_dims > 1:
            y = flatten_dims(y, num_batch_dims, 0)
            h = flatten_dims(h, num_batch_dims, 0)
            s = flatten_dims(s, num_batch_dims, 0)
            inputs = (y,h,s)

        # Initialization
        # (i) (optional) Convert to real-valued representation
        # (ii) Whiten channel
        # (iii) Sort columns of H by decreasing column norm
        # (iv) QR Decomposition of H
        # (v) Project y onto Q'
        y, r, column_order = self._preprocessing(inputs)
        batch_size = y.shape[0]
        # Tensor to keep track of the aggregate distances of all paths
        dists = torch.zeros([batch_size, self._k], dtype=get_real_dtype(y.dtype))
        # Tensor to store constellation symbols of all paths
        path_syms = torch.zeros([batch_size, self._k, self._num_streams], dtype=y.dtype)
        # Tensor to store constellation symbol indices of all paths
        path_inds = torch.zeros([batch_size, self._k, self._num_streams], dtype=torch.int32)

        # Sequential K-Best algorithm
        for stream in range(0, self._num_streams):
            dists, path_syms, path_inds = self._next_layer(y,
                                                           r,
                                                           dists,
                                                           path_syms,
                                                           path_inds,
                                                           stream)

        # Reverse order as detection started with the last symbol first
        path_syms = torch.flip(path_syms, dims=[-1])
        path_inds = torch.flip(path_syms, dims=[-1])

        # Processing for hard-decisions
        if self._hard_out:
            path_inds = self._unsort(column_order, path_inds)
            hard_dec = path_inds[:,0,:]

            # Real-valued representation
            if self._use_real_rep:
                hard_dec = \
                    self._pam2qam(hard_dec[...,:self._num_streams//2],
                                  hard_dec[...,self._num_streams//2:])

            # Hard decisions on bits
            if self._output=="bit":
                hard_dec = self._symbolinds2bits(hard_dec)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                hard_dec = split_dim(hard_dec, batch_shape, 0)

            return hard_dec

        # Processing for soft-decisions
        else:
            # Real-valued representation
            if self._use_real_rep:
                llr = self.list2llr([y, r, dists, path_inds, path_syms])
                llr = self._unsort(column_order, llr, transpose=False)

                # Combine LLRs from PAM symbols in the correct order
                llr1 = llr[:,:self._num_streams//2]
                llr2 = llr[:,self._num_streams//2:]
                llr1 = llr1.unsqueeze(-1)
                llr2 = llr2.unsqueeze(-1)
                llr = torch.cat([llr1, llr2], -1)
                llr = torch.reshape(llr, [-1, self._num_streams//2,
                                   2*self._num_bits_per_symbol])

            # Complex-valued representation
            else:
                llr = self.list2llr([y, r, dists, path_inds, path_syms])
                llr = self._unsort(column_order, llr, transpose=False)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                llr = split_dim(llr, batch_shape, 0)

            return llr

class EPDetector(nn.Module):
    def __init__(self, output, num_bits_per_symbol, hard_out=False, l=10, beta=0.9, dtype=torch.complex64):
        super().__init__()
        assert dtype in [torch.complex64, torch.complex128], "Invalid dtype"
        self._cdtype = dtype
        self._rdtype = torch.float32 if dtype == torch.complex64 else torch.float64

        # Numerical stability threshold
        self._prec = 1e-6 if dtype == torch.complex64 else 1e-12

        assert output in ("bit", "symbol"), "Unknown output"
        self._output = output
        self._hard_out = hard_out

        if self._output == "symbol":
            self._pam2qam = PAM2QAM(num_bits_per_symbol, hard_out)
        else:
            self._symbollogits2llrs = SymbolLogits2LLRs("maxlog", num_bits_per_symbol // 2, hard_out=hard_out)
            self._demapper = Demapper("maxlog", "pam", num_bits_per_symbol // 2)

        assert l >= 1, "l must be a positive integer"
        self._l = l
        assert 0.0 <= beta <= 1.0, "beta must be in [0,1]"
        self._beta = beta

        # Create PAM constellations for real-valued detection
        self._num_bits_per_symbol = num_bits_per_symbol // 2
        points = Constellation("pam", self._num_bits_per_symbol).points

        # Scale constellation points to half the energy
        self._points = torch.tensor(points / np.sqrt(2.0), dtype=self._rdtype)

        # Average symbol energy
        self._es = torch.tensor(np.var(self._points), dtype=self._rdtype)

    def compute_sigma_mu(self, h_t_h, h_t_y, no, lam, gam):
        """Equations (28) and (29)"""
        lam = torch.diag_embed(lam)
        gam = gam.unsqueeze(-1)
        
        sigma = torch.linalg.inv(h_t_h + no * lam)
        mu = torch.squeeze(torch.matmul(sigma, h_t_y + no * gam), dim=-1)
        sigma *= no
        sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)
        
        return sigma, mu

    def compute_v_x_obs(self, sigma, mu, lam, gam):
        """Equations (31) and (32)"""
        v_obs = torch.clamp(1 / (1 / sigma - lam), min=self._prec)
        x_obs = v_obs * (mu / sigma - gam)
        return v_obs, x_obs

    def compute_v_x(self, v_obs, x_obs):
        """Equation (33)"""
        x_obs = x_obs.unsqueeze(-1)
        v_obs = v_obs.unsqueeze(-1)

        points = self._points.unsqueeze(0).expand_as(x_obs)
        logits = -torch.pow(x_obs - points, 2) / (2 * v_obs)
        pmf = F.softmax(logits, dim=-1)

        x = torch.sum(points * pmf, dim=-1, keepdim=True)
        v = torch.sum((points - x)**2 * pmf, dim=-1)
        v = torch.clamp(v, min=self._prec)
        x = torch.squeeze(x, dim=-1)

        return v, x, logits

    def update_lam_gam(self, v, v_obs, x, x_obs, lam, gam):
        """Equations (35), (36), (37), (38)"""
        lam_old = lam
        gam_old = gam

        lam = 1 / v - 1 / v_obs
        gam = x / v - x_obs / v_obs

        lam_new = torch.where(lam < 0, lam_old, lam)
        gam_new = torch.where(lam < 0, gam_old, gam)

        lam_damp = (1 - self._beta) * lam_new + self._beta * lam_old
        gam_damp = (1 - self._beta) * gam_new + self._beta * gam_old

        return lam_damp, gam_damp

    def forward(self, y, h, s):
        # Flatten the batch dimensions
        batch_shape = y.shape[:-1]
        num_batch_dims = len(batch_shape)
        if num_batch_dims > 1:
            y = y.view(-1, *y.shape[-2:])
            h = h.view(-1, *h.shape[-3:])
            s = s.view(-1, *s.shape[-3:])
        
        n_t = h.shape[-1]

        # Whiten channel
        y, h, s = whiten_channel(y, h, s)

        # Convert channel to real-valued representation
        y, h, s = complex2real_channel(y, h, s)

        # Convert all inputs to desired dtypes
        y = y.to(self._rdtype)
        h = h.to(self._rdtype)
        no = torch.tensor(0.5, dtype=self._rdtype)

        # Initialize gamma and lambda
        gam = torch.zeros(*y.shape[:-1], h.shape[-1], dtype=y.dtype)
        lam = torch.ones(*y.shape[:-1], h.shape[-1], dtype=y.dtype) / self._es

        # Precompute values
        h_t_h = torch.matmul(h, h.transpose(-2, -1))
        y = y.unsqueeze(-1)
        h_t_y = torch.matmul(h, y)
        no = no.expand_as(h_t_h)

        for _ in range(self._l):
            sigma, mu = self.compute_sigma_mu(h_t_h, h_t_y, no, lam, gam)
            v_obs, x_obs = self.compute_v_x_obs(sigma, mu, lam, gam)
            v, x, logits = self.compute_v_x(v_obs, x_obs)
            lam, gam = self.update_lam_gam(v, v_obs, x, x_obs, lam, gam)

        # Extract the logits for the 2 PAM constellations for each stream
        pam1_logits = logits[..., :n_t, :]
        pam2_logits = logits[..., n_t:, :]

        if self._output == "symbol" and self._hard_out:
            # Take hard decisions on PAM symbols
            pam1_ind = torch.argmax(pam1_logits, dim=-1)
            pam2_ind = torch.argmax(pam2_logits, dim=-1)

            # Transform to QAM indices
            qam_ind = self._pam2qam(pam1_ind, pam2_ind)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                qam_ind = qam_ind.view(*batch_shape, -1)

            return qam_ind

        elif self._output == "symbol" and not self._hard_out:
            qam_logits = self._pam2qam(pam1_logits, pam2_logits)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                qam_logits = qam_logits.view(*batch_shape, -1)

            return qam_logits

        elif self._output == "bit":
            # Compute LLRs for both PAM constellations
            llr1 = self._symbollogits2llrs(pam1_logits)
            llr2 = self._symbollogits2llrs(pam2_logits)

            # Put LLRs in the correct order and shape
            llr = torch.stack([llr1, llr2], dim=-1)
            llr = llr.view(*llr.shape[:-1], -1)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                llr = llr.view(*batch_shape, -1)

            return llr

class MMSEPICDetector(nn.Module):
    def __init__(self,
                 output,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64):
        super(MMSEPICDetector, self).__init__()

        assert isinstance(num_iter, int), "num_iter must be an integer"
        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        assert dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"

        self.num_iter = num_iter
        self.output = output
        self.epsilon = 1e-4
        self.realdtype = get_real_dtype(dtype)
        self.demapping_method = demapping_method
        self.hard_out = hard_out

        # Create constellation object
        self.constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype
        )

        # Soft symbol mapping
        self.llr_2_symbol_logits = LLRs2SymbolLogits(
            self.constellation.num_bits_per_symbol,
            dtype=self.realdtype
        )

        if self.output == "symbol":
            self.llr_2_symbol_logits_output = LLRs2SymbolLogits(
                self.constellation.num_bits_per_symbol,
                dtype=self.realdtype,
                hard_out=hard_out
            )
            self.symbol_logits_2_llrs = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=self.constellation.num_bits_per_symbol
            )
        self.symbol_logits_2_moments = SymbolLogits2Moments(
            constellation=self.constellation,
            dtype=self.realdtype
        )

        # soft output demapping
        self.bit_demapper = DemapperWithPrior(
            demapping_method=demapping_method,
            constellation=self.constellation,
            dtype=dtype
        )


    def whiten_channel(self, y, h, s):
        # Placeholder for whitening channel logic
        # Implement or import the actual whitening logic here
        pass

    def call(self, inputs):
        y, h, prior, s = inputs
        
        # Preprocessing
        y, h = self.whiten_channel(y, h, s)

        # Matched filtering of y
        y_mf = torch.matmul(h, y.unsqueeze(-1)).squeeze(-1)

        # Step 1: compute Gramm matrix
        g = torch.matmul(h.transpose(-2, -1), h)

        # For XLA compatibility, this implementation performs the MIMO equalization in the real-valued domain
        hr = self.complex2real_matrix(h)
        gr = torch.matmul(hr.transpose(-2, -1), hr)

        # Compute a priori LLRs
        if self.output == "symbol":
            llr_a = self.symbol_logits_2_llrs(prior)
        else:
            llr_a = prior
        llr_shape = llr_a.shape

        def mmse_pic_self_iteration(llr_d, llr_a, it):
            # MMSE PIC takes in a priori LLRs
            llr_a = llr_d

            # Step 2: compute soft symbol estimates and variances
            x_logits = self.llr_2_symbol_logits(llr_a)
            x_hat, var_x = self.symbol_logits_2_moments(x_logits)

            # Step 3: perform parallel interference cancellation
            y_mf_pic = y_mf + g.unsqueeze(-1) * x_hat.unsqueeze(-2) - torch.matmul(g, x_hat.unsqueeze(-1)).squeeze(-1)

            # Step 4: compute A^-1 matrix
            var_x = torch.cat([var_x, var_x], dim=-1)
            var_x_row_vec = var_x.unsqueeze(-2)
            a = gr * var_x_row_vec

            a_inv = torch.linalg.inv(a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype))

            # Step 5: compute unbiased MMSE filter and outputs
            mu = torch.sum(a_inv * gr, dim=-1)

            y_mf_pic_trans = self.complex2real_vector(y_mf_pic.transpose(-2, -1))
            y_mf_pic_trans = torch.cat([y_mf_pic_trans, y_mf_pic_trans], dim=-2)

            x_hat = torch.sum(a_inv * y_mf_pic_trans, dim=-1) / mu.unsqueeze(-1)

            var_x = mu / torch.clamp(1 - var_x * mu, min=self.epsilon)
            var_x, _ = torch.split(var_x, 2, dim=-1)

            no_eff = 1. / var_x

            # Step 6: LLR demapping (extrinsic LLRs)
            llr_d = self.bit_demapper([x_hat, llr_a, no_eff]).reshape(llr_shape)

            return llr_d, llr_a, it

        def dec_stop(llr_d, llr_a, it):
            return it < self.num_iter

        it = torch.tensor(0)
        null_prior = torch.zeros_like(llr_a, dtype=self.realdtype)
        llr_d, llr_a, _ = self.iterative_loop(
            dec_stop,
            mmse_pic_self_iteration,
            (llr_a, null_prior, it)
        )
        llr_e = llr_d - llr_a
        if self.output == "symbol":
            out = self.llr_2_symbol_logits_output(llr_e)
        else:
            out = llr_e
            if self.hard_out:
                out = self.hard_decisions(out)

        return out

    def complex2real_matrix(self, x):
        return torch.cat([x.real, x.imag], dim=-1)

    def complex2real_vector(self, x):
        return torch.cat([x.real.unsqueeze(-1), x.imag.unsqueeze(-1)], dim=-1)

    def hard_decisions(self, x):
        return torch.argmax(x, dim=-1)

    def iterative_loop(self, stop_fn, body_fn, init_state):
        state = init_state
        while not stop_fn(*state):
            state = body_fn(*state)
        return state
