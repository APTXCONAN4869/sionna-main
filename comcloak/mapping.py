#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for (de)mapping, constellation class, and utility functions"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import comcloak
from comcloak.supplement import gather_pytorch, get_real_dtype
# from comcloak.utils import expand_to_rank, hard_decisions, flatten_last_dims
def pam_gray(b):
    # pylint: disable=line-too-long
    r"""Maps a vector of bits to a PAM constellation points with Gray labeling.

    This recursive function maps a binary vector to Gray-labelled PAM
    constellation points. It can be used to generated QAM constellations.
    The constellation is not normalized.

    Input
    -----
    b : [n], NumPy array
        Tensor with with binary entries.

    Output
    ------
    : signed int
        The PAM constellation point taking values in
        :math:`\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}`.

    Note
    ----
    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    if len(b)>1:
        return (1-2*b[0])*(2**len(b[1:]) - pam_gray(b[1:]))
    return 1-2*b[0]

def qam(num_bits_per_symbol, normalize=True):
    r"""Generates a QAM constellation.

    This function generates a complex-valued vector, where each element is
    a constellation point of an M-ary QAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be a multiple of two, e.g., 2, 4, 6, 8, etc.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.complex64
        The QAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a QAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}/2` is the number of bits
    per dimension.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol % 2 == 0 # is even
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be a multiple of 2") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.complex64)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b[0::2]) + 1j*pam_gray(b[1::2]) # PAM in each dimension

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol/2)
        qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(qam_var)
    return c

def pam(num_bits_per_symbol, normalize=True):
    r"""Generates a PAM constellation.

    This function generates a real-valued vector, where each element is
    a constellation point of an M-ary PAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be positive.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.float32
        The PAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a PAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}` is the number of bits
    per symbol.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be positive") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.float32)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b)

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol)
        pam_var = 1/(2**(n-1))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(pam_var)
    return c


class Constellation(nn.Module):
    r"""
    Constellation(constellation_type, num_bits_per_symbol, initial_value=None, normalize=True, center=False, trainable=False, dtype=torch.complex64, **kwargs)

    Constellation that can be used by a (de)mapper.

    This class defines a constellation, i.e., a complex-valued vector of
    constellation points. A constellation can be trainable. The binary
    representation of the index of an element of this vector corresponds
    to the bit label of the constellation point. This implicit bit
    labeling is used by the ``Mapper`` and ``Demapper`` classes.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", the constellation points are randomly initialized
        if no ``initial_value`` is provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    initial_value : :math:`[2^\text{num_bits_per_symbol}]`, NumPy array or Tensor
        Initial values of the constellation points. If ``normalize`` or
        ``center`` are `True`, the initial constellation might be changed.

    normalize : bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    center : bool
        If `True`, the constellation is ensured to have zero mean.
        Defaults to `False`.

    trainable : bool
        If `True`, the constellation points are trainable variables.
        Defaults to `False`.

    dtype : [torch.complex64, torch.complex128], torch.DType
        The dtype of the constellation.

    Output
    ------
    : :math:`[2^\text{num_bits_per_symbol}]`, ``dtype``
        The constellation.

    Note
    ----
    One can create a trainable PAM/QAM constellation. This is
    equivalent to creating a custom trainable constellation which is
    initialized with PAM/QAM constellation points.
    """
    def __init__(self,
                 constellation_type,
                 num_bits_per_symbol,
                 initial_value=None,
                 normalize=True,
                 center=False,
                 trainable=False,
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__()

        assert dtype in [torch.complex64, torch.complex128],\
            "dtype must be torch.complex64 or torch.complex128"
        self._dtype = dtype

        assert constellation_type in ("qam", "pam", "custom"),\
            "Wrong constellation type"
        self._constellation_type = constellation_type

        assert isinstance(normalize, bool), "normalize must be boolean"
        self._normalize = normalize

        assert isinstance(center, bool), "center must be boolean"
        self._center = center

        assert isinstance(trainable, bool), "trainable must be boolean"
        self._trainable = trainable

        # allow float inputs that represent int
        assert isinstance(num_bits_per_symbol, (float,int)),\
            "num_bits_per_symbol must be integer"
        assert (num_bits_per_symbol%1==0),\
            "num_bits_per_symbol must be integer"
        num_bits_per_symbol = int(num_bits_per_symbol)

        if self._constellation_type == "qam":
            assert num_bits_per_symbol % 2 == 0 and num_bits_per_symbol>0,\
                  "num_bits_per_symbol must be a multiple of 2"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            assert initial_value is None, "QAM must not have an initial value"

            points = qam(self._num_bits_per_symbol, normalize=self._normalize)
            points = torch.tensor(points, dtype = self._dtype)
            points = points.to(self._dtype)

        if self._constellation_type == "pam":
            assert num_bits_per_symbol>0,\
                "num_bits_per_symbol must be integer"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            assert initial_value is None, "PAM must not have an initial value"
            points = pam(self._num_bits_per_symbol, normalize=self._normalize)
            points = torch.tensor(points, dtype = self._dtype)
            points = points.to(self._dtype)


        if self._constellation_type == "custom":
            assert num_bits_per_symbol > 0,\
                  "num_bits_per_symbol must be integer"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            # Randomly initialize points if no initial_value is provided
            if initial_value is None:
                
                # 创建随机点
                points = torch.rand((2, 2**self._num_bits_per_symbol), 
                                    dtype=get_real_dtype(self._dtype)) * 0.1 - 0.05
                points = torch.complex(points[0], points[1])
            else:
                assert initial_value.dim() == 1, "initial_value must be 1-dimensional"
                assert initial_value.shape[0] == 2**num_bits_per_symbol, \
                    "initial_value must have shape [2**num_bits_per_symbol]"
                points = initial_value.to(self._dtype)
        self._points = points
    def create_or_check_constellation(  constellation_type=None,
                                        num_bits_per_symbol=None,
                                        constellation=None,
                                        dtype=torch.complex64):
        # pylint: disable=line-too-long
        r"""Static method for conviently creating a constellation object or checking that an existing one
        is consistent with requested settings.

        If ``constellation`` is `None`, then this method creates a :class:`~sionna.mapping.Constellation`
        object of type ``constellation_type`` and with ``num_bits_per_symbol`` bits per symbol.
        Otherwise, this method checks that `constellation` is consistent with ``constellation_type`` and
        ``num_bits_per_symbol``. If it is, ``constellation`` is returned. Otherwise, an assertion is raised.

        Input
        ------
        constellation_type : One of ["qam", "pam", "custom"], str
            For "custom", an instance of :class:`~sionna.mapping.Constellation`
            must be provided.

        num_bits_per_symbol : int
            The number of bits per constellation symbol, e.g., 4 for QAM16.
            Only required for ``constellation_type`` in ["qam", "pam"].

        constellation :  Constellation
            An instance of :class:`~sionna.mapping.Constellation` or
            `None`. In the latter case, ``constellation_type``
            and ``num_bits_per_symbol`` must be provided.

        Output
        -------
        : :class:`~sionna.mapping.Constellation`
            A constellation object.
        """
        constellation_object = None
        if constellation is not None:
            assert constellation_type in [None, "custom"], \
                """`constellation_type` must be "custom"."""
            assert num_bits_per_symbol in \
                     [None, constellation.num_bits_per_symbol], \
                """`Wrong value of `num_bits_per_symbol.`"""
            assert constellation._dtype==dtype, \
                "Constellation has wrong dtype."
            constellation_object = constellation
        else:
            assert constellation_type in ["qam", "pam"], \
                "Wrong constellation type."
            assert num_bits_per_symbol is not None, \
                "`num_bits_per_symbol` must be provided."
            constellation_object = Constellation(   constellation_type,
                                                    num_bits_per_symbol,
                                                    dtype=dtype)
        return constellation_object

    def forward(self, inputs):
        if self._points.dtype in [torch.complex64, torch.complex128]:
            points = self._points
            # print(points)
            points = torch.stack([points.real, points.imag], dim=0)
            # print(points)
            if self._trainable:
                self._points = nn.Parameter(torch.tensor(points, 
                                                        dtype=get_real_dtype(self._dtype)), 
                                                        requires_grad=True)
            else:
                self._points = torch.tensor(points, dtype=get_real_dtype(self._dtype))
        x = self._points
        x = torch.complex(x[0], x[1])
        if self._center:
            x = x - torch.mean(x)
        if self._normalize:
            energy = torch.mean(torch.square(torch.abs(x)))
            energy_sqrt = torch.complex(torch.sqrt(energy),
                                        torch.tensor(0.,
                                        dtype=get_real_dtype(self._dtype)))
            x = x / energy_sqrt
        return x

    @property
    def normalize(self):
        """Indicates if the constellation is normalized or not."""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        assert isinstance(value, bool), "`normalize` must be boolean"
        self._normalize = value

    @property
    def center(self):
        """Indicates if the constellation is centered."""
        return self._center

    @center.setter
    def center(self, value):
        assert isinstance(value, bool), "`center` must be boolean"
        self._center = value

    @property
    def num_bits_per_symbol(self):
        """The number of bits per constellation symbol."""
        return self._num_bits_per_symbol

    @property
    def points(self):
        """The (possibly) centered and normalized constellation points."""
        return self(None)

    def show(self, labels=True, figsize=(7,7)):
        """Generate a scatter-plot of the constellation.

        Input
        -----
        labels : bool
            If `True`, the bit labels will be drawn next to each constellation
            point. Defaults to `True`.

        figsize : Two-element Tuple, float
            Width and height in inches. Defaults to `(7,7)`.

        Output
        ------
        : matplotlib.figure.Figure
            A handle to a matplot figure object.
        """
        maxval = np.max(np.abs(self.points))*1.05
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.xlim(-maxval, maxval)
        plt.ylim(-maxval, maxval)
        plt.scatter(np.real(self.points), np.imag(self.points))
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.grid(True, which="both", axis="both")
        plt.title("Constellation Plot")
        if labels is True:
            for j, p in enumerate(self.points.numpy()):
                plt.annotate(
                    np.binary_repr(j, self.num_bits_per_symbol),
                    (np.real(p), np.imag(p))
                )
        return fig

class Mapper(nn.Module):
    r"""
    Mapper(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, dtype=torch.complex64, **kwargs)

    Maps binary tensors to points of a constellation.

    This class defines a layer that maps a tensor of binary values
    to a tensor of points from a provided constellation.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : bool
        If enabled, symbol indices are additionally returned.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128], torch.DType
        The output dtype. Defaults to torch.complex64.

    Input
    -----
    : [..., n], torch.float or torch.int
        Tensor with with binary entries.

    Output
    ------
    : [...,n/Constellation.num_bits_per_symbol], torch.complex
        The mapped constellation symbols.

    : [...,n/Constellation.num_bits_per_symbol], torch.int32
        The symbol indices corresponding to the constellation symbols.
        Only returned if ``return_indices`` is set to True.


    Note
    ----
    The last input dimension must be an integer multiple of the
    number of bits per constellation symbol.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 dtype=torch.complex64,
                 **kwargs
                 ):
        super().__init__()

        assert dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                                    constellation_type,
                                                                    num_bits_per_symbol,
                                                                    constellation,
                                                                    dtype=dtype)

        self._return_indices = return_indices

        self._binary_base = 2**torch.arange(self._constellation.num_bits_per_symbol-1, -1, -1, dtype=torch.int32)

    @property
    def constellation(self):
        """The Constellation used by the Mapper."""
        return self._constellation

    def forward(self, inputs):
        assert inputs.dim() >= 2, "The input must have at least 2 dimensions"

        # Reshape inputs to the desired format
        new_shape = [-1] + list(inputs.shape[1:-1]) + \
            [int(inputs.shape[-1] / self._constellation.num_bits_per_symbol),
             self._constellation.num_bits_per_symbol]
        # print(new_shape)
        inputs_reshaped = torch.reshape(inputs, new_shape).to(torch.int32)
        # print(inputs_reshaped.shape) #right
        # Convert the last dimension to an integer
        int_rep = torch.sum(inputs_reshaped * self._binary_base, dim=-1)
        # print(int_rep.shape) #right
        # Map integers to constellation symbols
        # int_rep = int_rep.view(-1)
        # print(self._constellation.points.shape)
        # x = torch.gather(self._constellation.points, 0, int_rep)
        x = gather_pytorch(self._constellation.points, int_rep, axis=0)
        # print(x.shape)

        if self._return_indices:
            return x, int_rep
        else:
            return x

###################################################################################


class SymbolLogits2LLRs(nn.Module):
    r"""
    SymbolLogits2LLRs(method, num_bits_per_symbol, hard_out=False, with_prior=False, dtype=torch.float32, **kwargs)

    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.
    If the flag ``with_prior`` is set, prior knowledge on the bits is assumed to be available.

    Parameters
    ----------
    method : One of ["app", "maxlog"], str
        The method used for computing the LLRs.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    hard_out : bool
        If `True`, the layer provides hard-decided bits instead of soft-values.
        Defaults to `False`.

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the bits is available.
        This prior information is given as LLRs as an additional input to the layer.
        Defaults to `False`.

    dtype : One of [torch.float32, torch.float64] torch.DType (dtype)
        The dtype for the input and output.
        Defaults to `torch.float32`.

    Input
    -----
    logits or (logits, prior):
        Tuple:

    logits : [...,n, num_points], torch.float
        Logits on constellation points.

    prior : [num_bits_per_symbol] or [...n, num_bits_per_symbol], torch.float
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]`
        for the entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.
        Only required if the ``with_prior`` flag is set.

    Output
    ------
    : [...,n, num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit.

    Note
    ----
    With the "app" method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of :math:`2^K` constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]` is the vector of logits on the constellation points, :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
        = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }\right)
                .
        \end{align}
    """
    
    def __init__(self,
                 method,
                 num_bits_per_symbol,
                 hard_out=False,
                 with_prior=False,
                 dtype=torch.float32):
        super().__init__()
        assert method in ("app", "maxlog"), "Unknown demapping method"
        self._method = method
        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol
        self._with_prior = with_prior
        num_points = 2 ** num_bits_per_symbol

        # Array composed of binary representations of all symbols indices
        a = torch.zeros((num_points, num_bits_per_symbol), dtype=torch.int16)
        for i in range(0, num_points):
            a[i, :] = torch.tensor([int(x) for x in format(i, f'0{num_bits_per_symbol}b')], dtype=torch.int16)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = torch.zeros((num_points // 2, num_bits_per_symbol), dtype=torch.int32)
        c1 = torch.zeros((num_points // 2, num_bits_per_symbol), dtype=torch.int32)
        for i in range(num_bits_per_symbol):
            c0[:, i] = torch.where(a[:, i] == 0)[0]
            c1[:, i] = torch.where(a[:, i] == 1)[0]
        self._c0 = torch.tensor(c0, dtype=torch.int32) # Symbols with ith bit=0
        self._c1 = torch.tensor(c1, dtype=torch.int32) # Symbols with ith bit=1


        if with_prior:
            # Array of labels from {-1, 1} of all symbols
            # [num_points, num_bits_per_symbol]
            a = 2 * a - 1
            self._a = torch.tensor(a, dtype=dtype)

        # Determine the reduce function for LLR computation
        self._reduce = torch.logsumexp if self._method == "app" else torch.max
    @property
    def num_bits_per_symbol(self):
        return self._num_bits_per_symbol
    def forward(self, inputs):
        if self._with_prior:
            logits, prior = inputs
        else:
            logits = inputs
        # Compute exponents
        exponents = logits

        # Gather exponents for all bits
        # shape [...,n,num_points/2,num_bits_per_symbol]
        exp0 = gather_pytorch(exponents, self._c0, axis=-1, batch_dims=0)
        exp1 = gather_pytorch(exponents, self._c1, axis=-1, batch_dims=0)

        if self._with_prior:
            # Expanding `prior` such that it is broadcastable with
            # shape [..., n or 1, 1, num_bits_per_symbol]
            prior = comcloak.utils.expand_to_rank(prior, logits.dim(), axis=0)
            # Expand the symbol labeling to be broadcastable with prior
            # shape [..., 1, num_points, num_bits_per_symbol]
            a = comcloak.utils.expand_to_rank(self._a, prior.dim(), axis=0)
            # Compute the prior probabilities on symbols exponents
            # shape [..., n or 1, num_points]
            exp_ps = torch.sum(torch.log(torch.sigmoid(a * prior)), dim=-1)
            # Gather prior probability symbol for all bits
            # shape [..., n or 1, num_points/2, num_bits_per_symbol]
            
            exp_ps0 = gather_pytorch(exp_ps, self._c0, axis=-1)
            exp_ps1 = gather_pytorch(exp_ps, self._c1, axis=-1)
        # Compute LLRs using the definition log( Pr(b=1)/Pr(b=0) )
        # shape [..., n, num_bits_per_symbol]
        if self._with_prior:
            max_exp1, _ = self._reduce(exp_ps1 + exp1, axis=-2)
            max_exp0, _ = self._reduce(exp_ps0 + exp0, axis=-2)
            llr = max_exp1 - max_exp0
        else:
            max_exp1, _ = self._reduce(exp1, axis=-2)  
            max_exp0, _ = self._reduce(exp0, axis=-2)  
            llr = max_exp1 - max_exp0

        if self._hard_out:
            return comcloak.utils.hard_decisions(llr)
        else:
            return llr

class SymbolLogits2LLRsWithPrior(SymbolLogits2LLRs):
    r"""
    SymbolLogits2LLRsWithPrior(method, num_bits_per_symbol, hard_out=False, dtype=torch.float32, **kwargs)

    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points,
    assuming that prior knowledge on the bits is available.

    This class is deprecated as the functionality has been integrated
    into :class:`~sionna.mapping.SymbolLogits2LLRs`.

    Parameters
    ----------
    method : One of ["app", "maxlog"], str
        The method used for computing the LLRs.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    hard_out : bool
        If `True`, the layer provides hard-decided bits instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.float32, torch.float64] torch.DType (dtype)
        The dtype for the input and output.
        Defaults to `torch.float32`.

    Input
    -----
    (logits, prior):
        Tuple:

    logits : [...,n, num_points], torch.float
        Logits on constellation points.

    prior : [num_bits_per_symbol] or [...n, num_bits_per_symbol], torch.float
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.

    Output
    ------
    : [...,n, num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit.

    Note
    ----
    With the "app" method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of :math:`2^K` constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]` is the vector of logits on the constellation points, :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
        = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }\right)
                .
        \end{align}
    """
    def __init__(self,
                 method,
                 num_bits_per_symbol,
                 hard_out=False,
                 dtype=torch.float32,
                 **kwargs):
        super().__init__(method=method,
                         num_bits_per_symbol=num_bits_per_symbol,
                         hard_out=False,
                         with_prior=True,
                         dtype=torch.float32,
                         **kwargs)

class Demapper(nn.Module):
    # pylint: disable=line-too-long
    r"""
    Demapper(demapping_method, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, with_prior=False, dtype=torch.complex64, **kwargs)

    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    for a tensor of received symbols.
    If the flag ``with_prior`` is set, prior knowledge on the bits is assumed to be available.

    This class defines a layer implementing different demapping
    functions. All demapping functions are fully differentiable when soft-decisions
    are computed.

    Parameters
    ----------
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
        If `True`, the demapper provides hard-decided bits instead of soft-values.
        Defaults to `False`.

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the bits is available.
        This prior information is given as LLRs as an additional input to the layer.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    -----
    (y,no) or (y, prior, no) :
        Tuple:

    y : [...,n], torch.complex
        The received symbols.

    prior : [num_bits_per_symbol] or [...,num_bits_per_symbol], torch.float
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.
        Only required if the ``with_prior`` flag is set.

    no : Scalar or [...,n], torch.float
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    Output
    ------
    : [...,n*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit.

    Note
    ----
    With the "app" demapping method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" demapping method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)\\
                &= \max_{c\in\mathcal{C}_{i,0}}
                    \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
                 \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
                .
        \end{align}
    """
    
    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 with_prior=False,
                 dtype=torch.complex64,
                 **kwargs):        
        super().__init__()
        self._with_prior = with_prior
        
        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation( 
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)
        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        dtype_real_dtype = get_real_dtype(dtype)
        self._logits2llrs = SymbolLogits2LLRs(demapping_method,
                                              num_bits_per_symbol,
                                              hard_out,
                                              with_prior,
                                              dtype_real_dtype,
                                              **kwargs)

        self._no_threshold = torch.tensor(np.finfo(torch.empty(0, dtype=dtype).numpy().dtype).tiny, dtype=dtype_real_dtype)

    @property
    def constellation(self):
        return self._constellation

    def forward(self, inputs):
        if self._with_prior:
            y, prior, no = inputs
        else:
            y, no = inputs

        # Reshape constellation points to [1,...,1,num_points]
        points_shape = [1] * y.dim() + list(self.constellation.points.shape)
        points = torch.reshape(self.constellation.points, points_shape)

        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        squared_dist = torch.abs(y.unsqueeze(-1) - points) ** 2

        # Add a dummy dimension for broadcasting
        
        no = torch.tensor(no).unsqueeze(-1)
        no = torch.maximum(no, self._no_threshold)

        # Compute exponents
        exponents = -squared_dist / no

        if self._with_prior:
            llr = self._logits2llrs([exponents, prior])
        else:
            llr = self._logits2llrs(exponents)

        # Reshape LLRs to [...,n*num_bits_per_symbol]
        out_shape = list(y.shape[:-1]) + [y.shape[-1] * self.constellation.num_bits_per_symbol]
        llr_reshaped = torch.reshape(llr, out_shape)

        return llr_reshaped

class DemapperWithPrior(Demapper):
    r"""
    DemapperWithPrior(demapping_method, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=torch.complex64, **kwargs)

    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    for a tensor of received symbols, assuming that prior knowledge on the bits is available.

    This class defines a layer implementing different demapping
    functions. All demapping functions are fully differentiable when soft-decisions
    are computed.

    This class is deprecated as the functionality has been integrated
    into :class:`~sionna.mapping.Demapper`.

    Parameters
    ----------
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
        If `True`, the demapper provides hard-decided bits instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    -----
    (y, prior, no) :
        Tuple:

    y : [...,n], torch.complex
        The received symbols.

    prior : [num_bits_per_symbol] or [...,num_bits_per_symbol], torch.float
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.

    no : Scalar or [...,n], torch.float
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    Output
    ------
    : [...,n*num_bits_per_symbol], torch.float
        LLRs or hard-decisions for every bit.

    Note
    ----
    With the "app" demapping method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" demapping method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)\\
                &= \max_{c\in\mathcal{C}_{i,0}}
                    \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
                 \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
                .
        \end{align}
    """
    
    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__(demapping_method=demapping_method, 
                                                constellation_type=constellation_type,
                                                num_bits_per_symbol=num_bits_per_symbol, 
                                                constellation=constellation, 
                                                hard_out=hard_out, 
                                                with_prior=True, 
                                                dtype=dtype,
                                                **kwargs)



class SymbolDemapper(nn.Module):
    r"""
    SymbolDemapper(constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, with_prior=False, dtype=torch.complex64, **kwargs)

    Computes normalized log-probabilities (logits) or hard-decisions on symbols
    for a tensor of received symbols.
    If the ``with_prior`` flag is set, prior knowldge on the transmitted constellation points is assumed to be available.
    The demapping function is fully differentiable when soft-values are
    computed.

    Parameters
    ----------
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
        If `True`, the demapper provides hard-decided symbols instead of soft-values.
        Defaults to `False`.

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the constellation points is available.
        This prior information is given as log-probabilities (logits) as an additional input to the layer.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    -----
    (y, no) or (y, prior, no) :
        Tuple:

    y : [...,n], torch.complex
        The received symbols.

    prior : [num_points] or [...,num_points], torch.float
        Prior for every symbol as log-probabilities (logits).
        It can be provided either as a tensor of shape `[num_points]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_points]`.
        Only required if the ``with_prior`` flag is set.

    no : Scalar or [...,n], torch.float
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    Output
    ------
    : [...,n, num_points] or [...,n], torch.float
        A tensor of shape `[...,n, num_points]` of logits for every constellation
        point if `hard_out` is set to `False`.
        Otherwise, a tensor of shape `[...,n]` of hard-decisions on the symbols.

    Note
    ----
    The normalized log-probability for the constellation point :math:`c` is computed according to

    .. math::
        \ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)

    where :math:`\mathcal{C}` is the set of constellation points used for modulation,
    and :math:`\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}` the prior information on constellation points given as log-probabilities
    and which is set to :math:`\mathbf{0}` if no prior information on the constellation points is assumed to be available.
    """
    
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 with_prior=False,
                 dtype=torch.complex64):
        super().__init__()
        self._hard_out = hard_out
        self._with_prior = with_prior
        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)

    def forward(self, inputs):
        if self._with_prior:
            y, prior, no = inputs
        else:
            y, no = inputs

        points = comcloak.utils.expand_to_rank(self._constellation.points,
                                y.dim()+1, axis=0)
        y = y.unsqueeze(-1)
        d = torch.abs(y - points)

        no = comcloak.utils.expand_to_rank(no, d.dim(), axis=-1)
        exp = -d ** 2 / no

        if self._with_prior:
            prior = comcloak.utils.expand_to_rank(prior, exp.dim(), axis=0)
            exp = exp + prior

        if self._hard_out:
            return torch.argmax(exp, dim=-1).int()
        else:
            return F.log_softmax(exp, dim=-1)

class SymbolDemapperWithPrior(SymbolDemapper):
    r"""
    SymbolDemapperWithPrior(constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=torch.complex64, **kwargs)

    Computes normalized log-probabilities (logits) or hard-decisions on symbols
    for a tensor of received symbols, assuming that prior knowledge on the constellation points is available.
    The demapping function is fully differentiable when soft-values are
    computed.

    This class is deprecated as the functionality has been integrated
    into :class:`~sionna.mapping.SymbolDemapper`.

    Parameters
    ----------
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
        If `True`, the demapper provides hard-decided symbols instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.complex64, torch.complex128] torch.DType (dtype)
        The dtype of `y`. Defaults to torch.complex64.
        The output dtype is the corresponding real dtype (torch.float32 or torch.float64).

    Input
    -----
    (y, prior, no) :
        Tuple:

    y : [...,n], torch.complex
        The received symbols.

    prior : [num_points] or [...,num_points], torch.float
        Prior for every symbol as log-probabilities (logits).
        It can be provided either as a tensor of shape `[num_points]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_points]`.

    no : Scalar or [...,n], torch.float
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    Output
    ------
    : [...,n, num_points] or [...,n], torch.float
        A tensor of shape `[...,n, num_points]` of logits for every constellation
        point if `hard_out` is set to `False`.
        Otherwise, a tensor of shape `[...,n]` of hard-decisions on the symbols.

    Note
    ----
    The normalized log-probability for the constellation point :math:`c` is computed according to

    .. math::
        \ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)

    where :math:`\mathcal{C}` is the set of constellation points used for modulation,
    and :math:`\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}` the prior information on constellation points given as log-probabilities.
    """
    
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64):
        super().__init__(constellation_type=constellation_type,
                        num_bits_per_symbol=num_bits_per_symbol,
                        constellation=constellation,
                        hard_out=hard_out,
                        with_prior=True,
                        dtype=dtype)



class LLRs2SymbolLogits(nn.Module):
    r"""
    LLRs2SymbolLogits(num_bits_per_symbol, hard_out=False, dtype=torch.float32, **kwargs)

    Computes logits (i.e., unnormalized log-probabilities) or hard decisions
    on constellation points from a tensor of log-likelihood ratios (LLRs) on bits.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    hard_out : bool
        If `True`, the layer provides hard-decided constellation points instead of soft-values.
        Defaults to `False`.

    dtype : One of [torch.float32, torch.float64] torch.DType (dtype)
        The dtype for the input and output.
        Defaults to `torch.float32`.

    Input
    -----
    llrs : [..., n, num_bits_per_symbol], torch.float
        LLRs for every bit.

    Output
    ------
    : [...,n, num_points], torch.float or [..., n], torch.int32
        Logits or hard-decisions on constellation points.

    Note
    ----
    The logit for the constellation :math:`c` point
    is computed according to

    .. math::
        \begin{align}
            \log{\left(\Pr\left(c\lvert LLRs \right)\right)}
                &= \log{\left(\prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert LLRs \right)\right)}\\
                &= \log{\left(\prod_{k=0}^{K-1} \text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}\\
                &= \sum_{k=0}^{K-1} \log{\left(\text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}
        \end{align}

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.
    """

    def __init__(self,
                 num_bits_per_symbol,
                 hard_out=False,
                 dtype=torch.float32):
        super().__init__()
        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol
        num_points = int(2 ** num_bits_per_symbol)

        # Array composed of binary representations of all symbols indices
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i,:] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                              dtype=np.int16)
        # Array of labels from {-1, 1} of all symbols
        a = 2 * a - 1
        self._a = torch.tensor(a, dtype=dtype)

    @property
    def num_bits_per_symbol(self):
        return self._num_bits_per_symbol
    def forward(self, llrs):
        # Expand the symbol labeling to be broadcastable with prior
        # shape [1, ..., 1, num_points, num_bits_per_symbol]
        a = comcloak.utils.expand_to_rank(self._a, llrs.dim(), axis=0)

        # Compute the prior probabilities on symbols exponents
        # shape [..., 1, num_points]
        llrs = llrs.unsqueeze(-2)
        logits = torch.sum(F.logsigmoid(a * llrs), dim=-1)

        if self._hard_out:
            return torch.argmax(logits, dim=-1).int()
        else:
            return logits

class SymbolLogits2Moments(nn.Module):
    r"""
    SymbolLogits2Moments(constellation_type=None, num_bits_per_symbol=None, constellation=None, dtype=torch.float32, **kwargs)

    Computes the mean and variance of a constellation from logits (unnormalized log-probabilities) on the
    constellation points.

    More precisely, given a constellation :math:`\mathcal{C} = \left[ c_0,\dots,c_{N-1} \right]` of size :math:`N`, this layer computes the mean and variance
    according to

    .. math::
        \begin{align}
            \mu &= \sum_{n = 0}^{N-1} c_n \Pr \left(c_n \lvert \mathbf{\ell} \right)\\
            \nu &= \sum_{n = 0}^{N-1} \left( c_n - \mu \right)^2 \Pr \left(c_n \lvert \mathbf{\ell} \right)
        \end{align}


    where :math:`\mathbf{\ell} = \left[ \ell_0, \dots, \ell_{N-1} \right]` are the logits, and

    .. math::
        \Pr \left(c_n \lvert \mathbf{\ell} \right) = \frac{\exp \left( \ell_n \right)}{\sum_{i=0}^{N-1} \exp \left( \ell_i \right) }.

    Parameters
    ----------
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

    dtype : One of [torch.float32, torch.float64] torch.DType (dtype)
        The dtype for the input and output.
        Defaults to `torch.float32`.

    Input
    -----
    logits : [...,n, num_points], torch.float
        Logits on constellation points.

    Output
    ------
    mean : [...,n], torch.float
        Mean of the constellation.

    var : [...,n], torch.float
        Variance of the constellation
    """
    
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 dtype=torch.float32):
        super().__init__()

        # Create constellation object
        const_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=const_dtype)

    def forward(self, logits):
        p = F.softmax(logits, dim=-1)
        p_c = torch.complex(p, torch.zeros_like(p, dtype=p.dtype))
        points = self._constellation.points
        points = comcloak.utils.expand_to_rank(points, p.dim(), axis=0)
        mean = torch.sum(p_c * points, dim=-1, keepdim=True)
        var = torch.sum(p * torch.square(torch.abs(points - mean)), dim=-1)
        mean = mean.squeeze(-1)

        return mean, var



class QAM2PAM:
    r"""Transforms QAM symbol indices to PAM symbol indices.

    For indices in a QAM constellation, computes the corresponding indices
    for the two PAM constellations corresponding the real and imaginary
    components of the QAM constellation.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per QAM constellation symbol, e.g., 4 for QAM16.

    Input
    -----
    ind_qam : Tensor, torch.int
        Indices in the QAM constellation

    Output
    -------
    ind_pam1 : Tensor, torch.int
        Indices for the first component of the corresponding PAM modulation

    ind_pam2 : Tensor, torch.int
        Indices for the first component of the corresponding PAM modulation
    """
    
    def __init__(self, num_bits_per_symbol):
        base = [2**i for i in range(num_bits_per_symbol//2-1, -1, -1)]
        base = np.array(base)
        pam1_ind = np.zeros([2**num_bits_per_symbol], dtype=np.int32)
        pam2_ind = np.zeros([2**num_bits_per_symbol], dtype=np.int32)
        for i in range(0, 2**num_bits_per_symbol):
            b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                         dtype=np.int32)
            pam1_ind[i] = np.sum(b[0::2]*base)
            pam2_ind[i] = np.sum(b[1::2]*base)
        self._pam1_ind = torch.tensor(pam1_ind, dtype=torch.int32)
        self._pam2_ind = torch.tensor(pam2_ind, dtype=torch.int32)

    def __call__(self, ind_qam):
        ind_pam1 = gather_pytorch(self._pam1_ind, ind_qam, axis=0)
        ind_pam2 = gather_pytorch(self._pam2_ind, ind_qam, axis=0)
        return ind_pam1, ind_pam2

class PAM2QAM:
    r"""Transforms PAM symbol indices/logits to QAM symbol indices/logits.

    For two PAM constellation symbol indices or logits, corresponding to
    the real and imaginary components of a QAM constellation,
    compute the QAM symbol index or logits.

    Parameters
    ----------
    num_bits_per_symbol : int
        Number of bits per QAM constellation symbol, e.g., 4 for QAM16

    hard_in_out : bool
        Determines if inputs and outputs are indices or logits over
        constellation symbols.
        Defaults to `True`.

    Input
    -----
    pam1 : Tensor, torch.int, or [...,2**(num_bits_per_symbol/2)], torch.float
        Indices or logits for the first PAM constellation

    pam2 : Tensor, torch.int, or [...,2**(num_bits_per_symbol/2)], torch.float
        Indices or logits for the second PAM constellation

    Output
    -------
    qam : Tensor, torch.int, or [...,2**num_bits_per_symbol], torch.float
        Indices or logits for the corresponding QAM constellation
    """
    
    def __init__(self, num_bits_per_symbol, hard_in_out=True):
        num_pam_symbols = 2**(num_bits_per_symbol//2)
        base = np.array([2**i for i in range(num_bits_per_symbol-1, -1, -1)])

        # Create an array of QAM symbol indices, indexed by two PAM indices
        ind = np.zeros([num_pam_symbols, num_pam_symbols], np.int32)
        for i in range(0, num_pam_symbols):
            for j in range(0, num_pam_symbols):
                b1 = np.array(list(np.binary_repr(i,num_bits_per_symbol//2)),
                              dtype=np.int16)
                b2 = np.array(list(np.binary_repr(j,num_bits_per_symbol//2)),
                              dtype=np.int16)
                b = np.zeros([num_bits_per_symbol], np.int32)
                b[0::2] = b1
                b[1::2] = b2
                ind[i, j] = np.sum(b*base)
        self._qam_ind = torch.tensor(ind, dtype=torch.int32)
        self._hard_in_out = hard_in_out

    def __call__(self, pam1, pam2):
        # PAM indices to QAM indices
        if self._hard_in_out:
            shape = pam1.shape
            ind_pam1 = torch.reshape(pam1, [-1, 1])
            ind_pam2 = torch.reshape(pam2, [-1, 1])
            ind_pam = torch.cat([ind_pam1, ind_pam2], dim=-1)
            ind_qam = self._qam_ind[ind_pam[:, 0], ind_pam[:, 1]]
            ind_qam = torch.reshape(ind_qam, shape)
            return ind_qam

        # PAM logits to QAM logits
        else:
            # Compute all combination of sums of logits
            logits_mat = pam1.unsqueeze(-1) + pam2.unsqueeze(-2)

            # Flatten to a vector
            logits = comcloak.utils.flatten_last_dims(logits_mat)

            # Gather symbols in the correct order
            gather_ind = torch.reshape(self._qam_ind, [-1])
            logits = gather_pytorch(logits, gather_ind, axis=-1)
            return logits

class SymbolInds2Bits(nn.Module):
    r"""SymbolInds2Bits(num_bits_per_symbol, dtype=torch.float32, **kwargs)

    Transforms symbol indices to their binary representations.

    Parameters
    ----------
    num_bits_per_symbol : int
        Number of bits per constellation symbol

    dtype: torch.DType
        Output dtype. Defaults to `torch.float32`.

    Input
    -----
    : Tensor, torch.int
        Symbol indices

    Output
    -----
    : input.shape + [num_bits_per_symbol], dtype
        Binary representation of symbol indices
    """
    
    def __init__(self,
                 num_bits_per_symbol,
                 dtype=torch.float32):
        super().__init__()
        num_symbols = 2**num_bits_per_symbol
        b = np.zeros([num_symbols, num_bits_per_symbol])
        for i in range(0, num_symbols):
            b[i,:] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                              dtype=np.int16)
        self._bit_labels = torch.tensor(b, dtype)

    def forward(self, inputs):
        symbol_ind = inputs
        return gather_pytorch(self._bit_labels, symbol_ind)
