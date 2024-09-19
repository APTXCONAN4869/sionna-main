import torch
from sionna.utils import matrix_inv

def zero_forcing_precoder(x, h, return_precoding_matrix=False):
    # pylint: disable=line-too-long
    r"""Zero-Forcing (ZF) Precoder

    This function implements ZF precoding for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^K` is the received signal vector,
    :math:`\mathbf{H}\in\mathbb{C}^{K\times M}` is the known channel matrix,
    :math:`\mathbf{G}\in\mathbb{C}^{M\times K}` is the precoding matrix,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the symbol vector to be precoded,
    and :math:`\mathbf{n}\in\mathbb{C}^K` is a noise vector. It is assumed that
    :math:`K\le M`.

    The precoding matrix :math:`\mathbf{G}` is defined as (Eq. 4.37) [BHS2017]_ :

    .. math::

        \mathbf{G} = \mathbf{V}\mathbf{D}

    where

    .. math::

        \mathbf{V} &= \mathbf{H}^{\mathsf{H}}\left(\mathbf{H} \mathbf{H}^{\mathsf{H}}\right)^{-1}\\
        \mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right).

    This ensures that each stream is precoded with a unit-norm vector,
    i.e., :math:`\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K`.
    The function returns the precoded vector :math:`\mathbf{G}\mathbf{x}`.

    Input
    -----
    x : [...,K], torch.complex
        1+D tensor containing the symbol vectors to be precoded.

    h : [...,K,M], torch.complex
        2+D tensor containing the channel matrices

    return_precoding_matrices : bool
        Indicates if the precoding matrices should be returned or not.
        Defaults to False.

    Output
    -------
    x_precoded : [...,M], torch.complex
        Tensor of the same shape and dtype as ``x`` apart from the last
        dimensions that has changed from `K` to `M`. It contains the
        precoded symbol vectors.

    g : [...,M,K], torch.complex
        2+D tensor containing the precoding matrices. It is only returned
        if ``return_precoding_matrices=True``.

    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    # Compute pseudo inverse for precoding
    g = torch.matmul(h, h.conj().transpose(-2, -1))
    g = torch.matmul(h.conj().transpose(-2, -1), matrix_inv(g))

    # Normalize each column to unit power
    norm = torch.sqrt((torch.abs(g)**2).sum(axis=-2, keepdims=True))
    g = g / torch.tensor(norm, g.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = x.unsqueeze(-1)

    # Precode
    x_precoded = torch.matmul(g, x_precoded).squeeze(-1)

    if return_precoding_matrix:
        return (x_precoded, g)
    else:
        return x_precoded