import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from comcloak.utils import matrix_sqrt_inv, expand_to_rank, insert_dims
def complex2real_vector(z):
    r"""Transforms a complex-valued vector into its real-valued equivalent.

    Transforms the last dimension of a complex-valued tensor into
    its real-valued equivalent by stacking the real and imaginary
    parts on top of each other.

    For a vector :math:`\mathbf{z}\in \mathbb{C}^M` with real and imaginary
    parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively, this function returns
    the vector :math:`\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`.

    Input
    -----
    : [...,M], torch.complex

    Output
    ------
    : [...,2M], torch.complex.real_dtype
    """
    x = torch.real(z)
    y = torch.imag(z)
    return torch.cat([x, y], dim=-1)

def real2complex_vector(z):
# pylint: disable=line-too-long
    r"""Transforms a real-valued vector into its complex-valued equivalent.

    Transforms the last dimension of a real-valued tensor into
    its complex-valued equivalent by interpreting the first half
    as the real and the second half as the imaginary part.

    For a vector :math:`\mathbf{z}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in \mathbb{R}^{2M}`
    with :math:`\mathbf{x}\in \mathbb{R}^M` and :math:`\mathbf{y}\in \mathbb{R}^M`,
    this function returns
    the vector :math:`\mathbf{x}+j\mathbf{y}\in\mathbb{C}^M`.

    Input
    -----
    : [...,2M], torch.float

    Output
    ------
    : [...,M], torch.complex
    """
    x, y = torch.chunk(z, 2, dim=-1)
    return torch.complex(x, y)

def complex2real_matrix(z):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued matrix into its real-valued equivalent.

    Transforms the last two dimensions of a complex-valued tensor into
    their real-valued matrix equivalent representation.

    For a matrix :math:`\mathbf{Z}\in \mathbb{C}^{M\times K}` with real and imaginary
    parts :math:`\mathbf{X}\in \mathbb{R}^{M\times K}` and
    :math:`\mathbf{Y}\in \mathbb{R}^{M\times K}`, respectively, this function returns
    the matrix :math:`\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}`, given as

    .. math::

        \tilde{\mathbf{Z}} = \begin{pmatrix}
                                \mathbf{X} & -\mathbf{Y}\\
                                \mathbf{Y} & \mathbf{X}
                             \end{pmatrix}.

    Input
    -----
    : [...,M,K], torch.complex

    Output
    ------
    : [...,2M, 2K], torch.complex.real_dtype
    """
    x = z.real
    y = z.imag
    row1 = torch.cat([x, -y], dim=-1)
    row2 = torch.cat([y, x], dim=-1)
    return torch.cat([row1, row2], dim=-2)

def real2complex_matrix(z):
    # pylint: disable=line-too-long
    r"""Transforms a real-valued matrix into its complex-valued equivalent.

    Transforms the last two dimensions of a real-valued tensor into
    their complex-valued matrix equivalent representation.

    For a matrix :math:`\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}`,
    satisfying

    .. math::

        \tilde{\mathbf{Z}} = \begin{pmatrix}
                                \mathbf{X} & -\mathbf{Y}\\
                                \mathbf{Y} & \mathbf{X}
                             \end{pmatrix}

    with :math:`\mathbf{X}\in \mathbb{R}^{M\times K}` and
    :math:`\mathbf{Y}\in \mathbb{R}^{M\times K}`, this function returns
    the matrix :math:`\mathbf{Z}=\mathbf{X}+j\mathbf{Y}\in\mathbb{C}^{M\times K}`.

    Input
    -----
    : [...,2M,2K], torch.float

    Output
    ------
    : [...,M, 2], torch.complex
    """
    m = z.shape[-2] // 2
    k = z.shape[-1] // 2
    x = z[..., :m, :k]
    y = z[..., m:, :k]
    return torch.complex(x, y)

def complex2real_covariance(r):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued covariance matrix to its real-valued equivalent.

    Assume a proper complex random variable :math:`\mathbf{z}\in\mathbb{C}^M` [ProperRV]_
    with covariance matrix :math:`\mathbf{R}= \in\mathbb{C}^{M\times M}`
    and real and imaginary parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively.
    This function transforms the given :math:`\mathbf{R}` into the covariance matrix of the real-valued equivalent
    vector :math:`\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`, which
    is computed as [CovProperRV]_

    .. math::

        \mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
        \begin{pmatrix}
            \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
            \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
        \end{pmatrix}.

    Input
    -----
    : [...,M,M], torch.complex

    Output
    ------
    : [...,2M, 2M], torch.complex.real_dtype
    """
    q = complex2real_matrix(r)
    scale = torch.tensor(2.0, dtype=q.dtype, device=q.device)
    return q / scale

def real2complex_covariance(q):
    # pylint: disable=line-too-long
    r"""Transforms a real-valued covariance matrix to its complex-valued equivalent.

    Assume a proper complex random variable :math:`\mathbf{z}\in\mathbb{C}^M` [ProperRV]_
    with covariance matrix :math:`\mathbf{R}= \in\mathbb{C}^{M\times M}`
    and real and imaginary parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively.
    This function transforms the given covariance matrix of the real-valued equivalent
    vector :math:`\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`, which
    is given as [CovProperRV]_

    .. math::

        \mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
        \begin{pmatrix}
            \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
            \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
        \end{pmatrix},

    into is complex-valued equivalent :math:`\mathbf{R}`.

    Input
    -----
    : [...,2M,2M], torch.float

    Output
    ------
    : [...,M, M], torch.complex
    """
    r = real2complex_matrix(q)
    scale = torch.tensor(2.0, dtype=r.dtype, device=r.device)
    return r * scale

def complex2real_channel(y, h, s):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued MIMO channel into its real-valued equivalent.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}`.

    This function returns the real-valued equivalent representations of
    :math:`\mathbf{y}`, :math:`\mathbf{H}`, and :math:`\mathbf{S}`,
    which are used by a wide variety of MIMO detection algorithms (Section VII) [YH2015]_.
    These are obtained by applying :meth:`~sionna.mimo.complex2real_vector` to :math:`\mathbf{y}`,
    :meth:`~sionna.mimo.complex2real_matrix` to :math:`\mathbf{H}`,
    and :meth:`~sionna.mimo.complex2real_covariance` to :math:`\mathbf{S}`.

    Input
    -----
    y : [...,M], torch.complex
        1+D tensor containing the received signals.

    h : [...,M,K], torch.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], torch.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    : [...,2M], torch.complex.real_dtype
        1+D tensor containing the real-valued equivalent received signals.

    : [...,2M,2K], torch.complex.real_dtype
        2+D tensor containing the real-valued equivalent channel matrices.

    : [...,2M,2M], torch.complex.real_dtype
        2+D tensor containing the real-valued equivalent noise covariance matrices.
    """
    yr = complex2real_vector(y)
    hr = complex2real_matrix(h)
    sr = complex2real_covariance(s)
    return yr, hr, sr

def real2complex_channel(y, h, s):
    # pylint: disable=line-too-long
    r"""Transforms a real-valued MIMO channel into its complex-valued equivalent.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}`.

    This function transforms the real-valued equivalent representations of
    :math:`\mathbf{y}`, :math:`\mathbf{H}`, and :math:`\mathbf{S}`, as, e.g.,
    obtained with the function :meth:`~sionna.mimo.complex2real_channel`,
    back to their complex-valued equivalents (Section VII) [YH2015]_.

    Input
    -----
    y : [...,2M], torch.float
        1+D tensor containing the real-valued received signals.

    h : [...,2M,2K], torch.float
        2+D tensor containing the real-valued channel matrices.

    s : [...,2M,2M], torch.float
        2+D tensor containing the real-valued noise covariance matrices.

    Output
    ------
    : [...,M], torch.complex
        1+D tensor containing the complex-valued equivalent received signals.

    : [...,M,K], torch.complex
        2+D tensor containing the complex-valued equivalent channel matrices.

    : [...,M,M], torch.complex
        2+D tensor containing the complex-valued equivalent noise covariance matrices.
    """
    yc = real2complex_vector(y)
    hc = real2complex_matrix(h)
    sc = real2complex_covariance(s)
    return yc, hc, sc

def whiten_channel(y, h, s, return_s=True):
    # pylint: disable=line-too-long
    r"""Whitens a canonical MIMO channel.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M(\mathbb{R}^M)` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K(\mathbb{R}^K)` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}(\mathbb{R}^{M\times K})` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M(\mathbb{R}^M)` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}(\mathbb{R}^{M\times M})`.

    This function whitens this channel by multiplying :math:`\mathbf{y}` and
    :math:`\mathbf{H}` from the left by :math:`\mathbf{S}^{-\frac{1}{2}}`.
    Optionally, the whitened noise covariance matrix :math:`\mathbf{I}_M`
    can be returned.

    Input
    -----
    y : [...,M], torch.float or torch.complex
        1+D tensor containing the received signals.

    h : [...,M,K], torch.float or torch.complex
        2+D tensor containing the  channel matrices.

    s : [...,M,M], torch.float or complex
        2+D tensor containing the noise covariance matrices.

    return_s : bool
        If `True`, the whitened covariance matrix is returned.
        Defaults to `True`.

    Output
    ------
    : [...,M], torch.float or torch.complex
        1+D tensor containing the whitened received signals.

    : [...,M,K], torch.float or torch.complex
        2+D tensor containing the whitened channel matrices.

    : [...,M,M], torch.float or torch.complex
        2+D tensor containing the whitened noise covariance matrices.
        Only returned if ``return_s`` is `True`.
    """
    # Compute whitening matrix
    s_inv_1_2 = matrix_sqrt_inv(s)
    s_inv_1_2 = expand_to_rank(s_inv_1_2, h.dim(), 0)

    # Whiten observation and channel matrix
    yw = y.unsqueeze(-1)
    yw = torch.matmul(s_inv_1_2, yw).squeeze(-1)
    hw = torch.matmul(s_inv_1_2, h)

    if return_s:
        # Ideal interference covariance matrix after whitening
        sw = torch.eye(s.shape[-2], dtype=s.dtype, device=s.device)
        sw = expand_to_rank(sw, s.dim(), 0)
        return yw, hw, sw
    else:
        return yw, hw
    


class List2LLR(ABC):
    r"""List2LLR()

    Abstract class defining a callable to compute LLRs from a list of
    candidate vectors (or paths) provided by a MIMO detector.

    The following channel model is assumed:

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S` are the channel outputs,
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is an upper-triangular matrix,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S` is the transmitted vector whose entries
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    and :math:`\bar{\mathbf{n}}\in\mathbb{C}^S` is white noise
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    It is assumed that a MIMO detector such as :class:`~sionna.mimo.KBestDetector`
    produces :math:`K` candidate solutions :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. This layer can also be used with the real-valued representation of the channel.

    Input
    -----
    (y, r, dists, path_inds, path_syms) :
        Tuple:

    y : [...,M], torch.complex or torch.float
        Channel outputs of the whitened channel

    r : [...,num_streams, num_streams], same dtype as ``y``
        Upper triangular channel matrix of the whitened channel

    dists : [...,num_paths], torch.float
        Distance metric for each path (or candidate)

    path_inds : [...,num_paths,num_streams], torch.int32
        Symbol indices for every stream of every path (or candidate)

    path_syms : [...,num_path,num_streams], same dtype as ``y``
        Constellation symbol for every stream of every path (or candidate)

    Output
    ------
    llr : [...num_streams,num_bits_per_symbol], torch.float
        LLRs for all bits of every stream

    Note
    ----
    An implementation of this class does not need to make use of all of
    the provided inputs which enable various different implementations.
    """
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

class List2LLRSimple(nn.Module, List2LLR):
    r"""List2LLRSimple(num_bits_per_symbol, llr_clip_val=20.0)

    Computes LLRs from a list of candidate vectors (or paths) provided by a MIMO detector.

    The following channel model is assumed:

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S` are the channel outputs,
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is an upper-triangular matrix,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S` is the transmitted vector whose entries
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    and :math:`\bar{\mathbf{n}}\in\mathbb{C}^S` is white noise
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    It is assumed that a MIMO detector such as :class:`~sionna.mimo.KBestDetector`
    produces :math:`K` candidate solutions :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. This layer can also be used with the real-valued representation of the channel.

    The LLR for the :math:`i\text{th}` bit of the :math:`k\text{th}` stream is computed as

    .. math::
        \begin{align}
            LLR(k,i) &= \log\left(\frac{\Pr(b_{k,i}=1|\bar{\mathbf{y}},\mathbf{R})}{\Pr(b_{k,i}=0|\bar{\mathbf{y}},\mathbf{R})}\right)\\
                &\approx \min_{j \in  \mathcal{C}_{k,i,0}}d_j - \min_{j \in  \mathcal{C}_{k,i,1}}d_j
        \end{align}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the set of indices
    in the list of candidates for which the :math:`i\text{th}` bit of the :math:`k\text{th}`
    stream is equal to 1 and 0, respectively. The LLRs are clipped to :math:`\pm LLR_\text{clip}`
    which can be configured through the parameter ``llr_clip_val``.

    If :math:`\mathcal{C}_{k,i,0}` is empty, :math:`LLR(k,i)=LLR_\text{clip}`;
    if :math:`\mathcal{C}_{k,i,1}` is empty, :math:`LLR(k,i)=-LLR_\text{clip}`.

    Parameters
    ----------
    num_bits_per_symbol : int
        Number of bits per constellation symbol

    llr_clip_val : float
        The absolute values of LLRs are clipped to this value.
        Defaults to 20.0. Can also be a trainable variable.

    Input
    -----
    (y, r, dists, path_inds, path_syms) :
        Tuple:

    y : [...,M], torch.complex or torch.float
        Channel outputs of the whitened channel

    r : [...,num_streams, num_streams], same dtype as ``y``
        Upper triangular channel matrix of the whitened channel

    dists : [...,num_paths], torch.float
        Distance metric for each path (or candidate)

    path_inds : [...,num_paths,num_streams], torch.int32
        Symbol indices for every stream of every path (or candidate)

    path_syms : [...,num_path,num_streams], same dtype as ``y``
        Constellation symbol for every stream of every path (or candidate)

    Output
    ------
    llr : [...num_streams,num_bits_per_symbol], torch.float
        LLRs for all bits of every stream
    """
    def __init__(self, num_bits_per_symbol, llr_clip_val=20.0):
        super().__init__()
        
        # Array composed of binary representations of all symbols indices
        num_points = 2**num_bits_per_symbol
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(num_points):
            a[i, :] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                               dtype=np.int32)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = np.zeros([int(num_points/2), num_bits_per_symbol])
        c1 = np.zeros([int(num_points/2), num_bits_per_symbol])
        for i in range(num_bits_per_symbol):
            c0[:,i] = np.where(a[:,i]==0)[0]
            c1[:,i] = np.where(a[:,i]==1)[0]

        # Convert to tensor and add dummy dimensions needed for broadcasting
        self._c0 = expand_to_rank(torch.tensor(c0, dtype=torch.int32), 5, 0)
        self._c1 = expand_to_rank(torch.tensor(c1, dtype=torch.int32), 5, 0)

        # Assign this absolute value to all LLRs without counter-hypothesis
        self.llr_clip_val = llr_clip_val
    
    @property
    def llr_clip_val(self):
        return self._llr_clip_val

    @llr_clip_val.setter
    def llr_clip_val(self, value):
        self._llr_clip_val = value

    def forward(self, inputs):
        dists, path_inds = inputs[2:4]
        
        # Scaled by 0.5 to account for the reduced noise power in each complex
        # dimension if real channel representation is used.
        if torch.is_floating_point(input[0]):
            dists = dists / 2.0

        # Compute for every symbol in every path which bits are 0 or 1
        # b0/b1: [batch_size, num_path, num_streams, num_bits_per_symbol]

        path_inds = insert_dims(path_inds, 2, axis=-1)
        b0 = (path_inds == self._c0).any(dim=-3)
        b1 = (path_inds == self._c1).any(dim=-3)

        # Compute distances for all bits in all paths, set distance to inf
        # if the bit does not have the correct value
        dists = expand_to_rank(dists, b0.dim(), axis=-1)
        d0 = torch.where(b0, dists, torch.tensor(float('inf')).to(dists.dtype))
        d1 = torch.where(b1, dists, torch.tensor(float('inf')).to(dists.dtype))

        # Compute minimum distance for each bit in each stream
        l0 = d0.min(dim=1).values
        l1 = d1.min(dim=1).values

        # Compute LLRs
        llr = l0 - l1

        # Clip LLRs
        llr = torch.clamp(llr, -self.llr_clip_val, self.llr_clip_val)

        return llr

