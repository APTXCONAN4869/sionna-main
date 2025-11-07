import torch
import torch.linalg as linalg
from comcloak.utils import expand_to_rank, matrix_inv, matrix_pinv
from comcloak.mimo.utils_z import whiten_channel

def lmmse_equalizer(y, h, s, whiten_interference=True):
    """
    MIMO LMMSE Equalizer

    This function implements LMMSE equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Lemma B.19) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of

    .. math::

        \mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
        = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}`
    is important for the :class:`~sionna.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], torch.complex
        1+D tensor containing the received signals.

    h : [...,M,K], torch.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], torch.complex
        2+D tensor containing the noise covariance matrices.

    whiten_interference : bool
        If `True` (default), the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used that
        can be numerically more stable. Defaults to `True`.

    Output
    ------
    x_hat : [...,K], torch.complex
        1+D tensor representing the estimated symbol vectors.

    no_eff : torch.float
        Tensor of the same shape as ``x_hat`` containing the effective noise
        variance estimates.

    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    if not whiten_interference:
        # Compute G
        g = torch.matmul(h, h.conj().transpose(-2, -1)) + s
        g = torch.matmul(h.conj().transpose(-2, -1), linalg.inv(g))
    else:
        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False)

        # Compute G
        i = expand_to_rank(torch.eye(h.shape[-1], dtype=s.dtype, device=h.device),s.dim(), 0)
        g = torch.matmul(h.conj().transpose(-2, -1), h) + i
        g = torch.matmul(linalg.inv(g), h.conj().transpose(-2, -1))

    # Compute Gy
    y = y.unsqueeze(-1)
    gy = torch.matmul(g, y).squeeze(-1)

    # Compute GH
    gh = torch.matmul(g, h)

    # Compute diag(GH)
    d = torch.diagonal(gh, dim1=-2, dim2=-1)

    # Compute x_hat
    x_hat = gy / d

    # Compute residual error variance
    one = torch.tensor(1, dtype=d.dtype, device=d.device)
    no_eff = torch.real(one / d - one)

    return x_hat, no_eff




def zf_equalizer(y, h, s):
    """
    MIMO ZF Equalizer

    This function implements zero-forcing (ZF) equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.10) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

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
    x_hat : [...,K], torch.complex
        1+D tensor representing the estimated symbol vectors.

    no_eff : torch.float
        Tensor of the same shape as ``x_hat`` containing the effective noise
        variance estimates.

    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@torch.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The ZF estimate of x is given as:
    # x_hat = Gy
    # with G=(H'H')^(-1)H'.
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[ee'] = GSG'

    # Compute G
    g = matrix_pinv(h)

    # Compute x_hat
    y = y.unsqueeze(-1)  # Add an extra dimension
    x_hat = torch.matmul(g, y).squeeze(-1)  # Remove the extra dimension

    # Compute residual error variance
    gsg = torch.matmul(g, torch.matmul(s, g.conj().transpose(-2, -1)))
    no_eff = torch.real(torch.diagonal(gsg, dim1=-2, dim2=-1))

    return x_hat, no_eff

def mf_equalizer(y, h, s):
    """
    MIMO MF Equalizer

    This function implements matched filter (MF) equalization for a
    MIMO link, assuming the following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.11) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)\left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)^{\mathsf{H}} + \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}`
    in the definition of :math:`\mathbf{G}`
    is important for the :class:`~sionna.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

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
    x_hat : [...,K], torch.complex
        1+D tensor representing the estimated symbol vectors.

    no_eff : torch.float
        Tensor of the same shape as ``x_hat`` containing the effective noise
        variance estimates.
    """
    # Compute G
    hth = torch.matmul(h.conj().transpose(-2, -1), h)  # H^H * H
    d = torch.diag_embed(1 / torch.diagonal(hth, dim1=-2, dim2=-1))  # Diagonal matrix with 1 / diag(H^H * H)
    g = torch.matmul(d, h.conj().transpose(-2, -1))  # G = diag(H^H * H)^-1 * H^H

    # Compute x_hat
    y = y.unsqueeze(-1)  # Add an extra dimension
    x_hat = torch.matmul(g, y).squeeze(-1)  # Remove the extra dimension

    # Compute residual error variance
    gsg = torch.matmul(torch.matmul(g, s), g.conj().transpose(-2, -1))  # G * S * G^H
    gh = torch.matmul(g, h)  # G * H
    i = expand_to_rank(torch.eye(gsg.shape[-2], dtype=gsg.dtype, device=gsg.device), gsg.dim(), 0) # Identity matrix

    # Compute (I - GH) * (I - GH)^H + GSG^H
    i_gh = i - gh
    no_eff = torch.abs(torch.diagonal(torch.matmul(i_gh, i_gh.conj().transpose(-2, -1)) + gsg, dim1=-2, dim2=-1))

    return x_hat, no_eff

