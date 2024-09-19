import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def get_real_dtype(dtype):
    """
    Returns the real dtype corresponding to a given complex dtype.
    
    Args:
        dtype: A torch dtype (e.g., torch.complex64, torch.complex128).
        
    Returns:
        The real dtype corresponding to the complex dtype.
    """
    if dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex128:
        return torch.float64
    elif dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        return dtype
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

############################## metrics.py ###############################################################
def count_errors(b, b_hat):
    """
    Counts the number of bit errors between two binary tensors.

    Input
    -----
        b : torch.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : torch.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : torch.int64
            A scalar, the number of bit errors.
    """
    errors = torch.not_equal(b, b_hat)
    errors = errors.to(torch.int64)
    return errors.sum().item()

def count_block_errors(b, b_hat):
    """
    Counts the number of block errors between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i.e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : torch.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : torch.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : torch.int64
            A scalar, the number of block errors.
    """
    errors = torch.any(torch.not_equal(b, b_hat), dim=-1)
    errors = errors.to(torch.int64)
    return errors.sum().item()

############################## metrics.py ###############################################################


############################## mapping.py ###############################################################
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
        points = self._points
        points = torch.stack([points.real, points.imag], dim=0)
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
        inputs_reshaped = torch.reshape(inputs, new_shape).to(torch.int32)

        # Convert the last dimension to an integer
        int_rep = torch.sum(inputs_reshaped * self._binary_base, dim=-1)

        # Map integers to constellation symbols
        # int_rep = int_rep.view(-1)
        x = torch.gather(self._constellation.points, 0, int_rep)

        if self._return_indices:
            return x, int_rep
        else:
            return x

############################# mapping.py #################################################################


############################# misc.py ####################################################################
class BinarySource(nn.Module):
    def __init__(self, dtype=torch.float32, seed=None, **kwargs):
        super().__init__()
        self._dtype = dtype
        self._seed = seed
        self._rng = None
        if self._seed is not None:
            self._rng = torch.Generator().manual_seed(self._seed)

    def forward(self, inputs):
        if self._seed is not None:
            return torch.randint(0, 2, inputs.shape, generator=self._rng, dtype=torch.int32).to(self._dtype)
        else:
            return torch.randint(0, 2, inputs.shape, dtype=torch.int32).to(self._dtype)
        
class SymbolSource(nn.Module):
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=torch.complex64,
                 **kwargs
                ):
        super().__init__()
        constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype)
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(seed=seed, dtype=get_real_dtype(dtype))  # Changed from dtype.real to torch.float32
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              dtype=dtype)

    def forward(self, inputs):
        shape =  torch.cat((torch.tensor(inputs), torch.tensor([self._num_bits_per_symbol])))
        b = self._binary_source(shape.to(torch.int32))
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        result = torch.squeeze(x, -1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            result.append(torch.squeeze(ind, -1))
        if self._return_bits:
            result.append(b)

        return result

class QAMSource(SymbolSource):
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=torch.complex64,
                 **kwargs
                ):
        super().__init__(constellation_type="qam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         dtype=dtype,
                         **kwargs)

############################# misc.py ####################################################################

##################### utils/tensor.py ####################################################################
def flatten_dims(tensor, num_dims, axis):
    """
    Flattens a specified set of dimensions of a tensor.

    This operation flattens `num_dims` dimensions of a `tensor`
    starting at a given `axis`.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be larger than two and less or equal than the
            rank of `tensor`.
        axis (int): The index of the dimension from which to start.

    Returns:
        A tensor of the same type as `tensor` with `num_dims`-1 lesser
        dimensions, but the same number of elements.
    """
    assert num_dims >= 2, "`num_dims` must be >= 2"
    assert num_dims <= len(tensor.shape), "`num_dims` must <= rank(`tensor`)"
    assert 0 <= axis <= len(tensor.shape) - 1, "0<= `axis` <= rank(tensor)-1"
    assert num_dims + axis <= len(tensor.shape), "`num_dims`+`axis` <= rank(`tensor`)"

    if num_dims == len(tensor.shape):
        new_shape = [-1]
    elif axis == 0:
        shape = tensor.shape
        new_shape = [-1] + list(shape[axis + num_dims:])
    else:
        shape = tensor.shape
        flat_dim = torch.prod(torch.tensor(shape[axis:axis + num_dims]))
        new_shape = list(shape[:axis]) + [flat_dim.item()] + list(shape[axis + num_dims:])

    return tensor.reshape(new_shape)

def split_dim(tensor, shape, axis):
    """
    Reshapes a dimension of a tensor into multiple dimensions.

    This operation splits the dimension `axis` of a `tensor` into
    multiple dimensions according to `shape`.

    Args:
        tensor : A tensor.
        shape (list or TensorShape): The shape to which the dimension should
            be reshaped.
        axis (int): The index of the axis to be reshaped.

    Returns:
        A tensor of the same type as `tensor` with len(`shape`)-1
        additional dimensions, but the same number of elements.
    """
    assert 0 <= axis <= len(tensor.shape) - 1, "0<= `axis` <= rank(tensor)-1"

    s = tensor.shape
    new_shape = list(s[:axis]) + list(shape) + list(s[axis + 1:])
    return tensor.reshape(new_shape)

def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last `num_dims` dimensions of a `tensor`.
    It is a simplified version of the function `flatten_dims`.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of `tensor`.

    Returns:
        A tensor of the same type as `tensor` with `num_dims`-1 lesser
        dimensions, but the same number of elements.
    """
    assert num_dims >= 2, "`num_dims` must be >= 2"
    assert num_dims <= len(tensor.shape), "`num_dims` must <= rank(`tensor`)"

    if num_dims == len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tensor.shape
        last_dim = torch.prod(torch.tensor(shape[-num_dims:]))
        new_shape = list(shape[:-num_dims]) + [last_dim.item()]

    return tensor.reshape(new_shape)

def insert_dims(tensor, num_dims, axis=-1):
    """
    Adds multiple length-one dimensions to a tensor.

    This operation is an extension to PyTorch's ``unsqueeze`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Args:
        tensor : A tensor.
        num_dims (int) : The number of dimensions to add.
        axis : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with ``num_dims`` additional
        dimensions inserted at the index specified by ``axis``.
    """
    assert num_dims >= 0, "`num_dims` must be nonnegative."
    
    rank = tensor.dim()
    assert -(rank + 1) <= axis <= rank, "`axis` is out of range `[-(D+1), D]`"

    if axis < 0:
        axis += rank + 1

    for _ in range(num_dims):
        tensor = tensor.unsqueeze(axis)

    return tensor

def expand_to_rank(tensor, target_rank, axis=-1):
    """
    Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a tensor starting at
    axis, so that the rank of the resulting tensor has rank target_rank.
    The dimension index follows Python indexing rules, i.e., zero-based,
    where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If target_rank is smaller than the rank of tensor,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of tensor. Given a tensor of D dimensions,
               axis must be within the range [-(D+1), D] (inclusive).

    Returns:
        A tensor with the same data as tensor, with
        target_rank - rank(tensor) additional dimensions inserted at the
        index specified by axis.
        If target_rank <= rank(tensor), tensor is returned.
    """
    current_rank = tensor.dim()
    num_dims = max(target_rank - current_rank, 0)
    output = insert_dims(tensor, num_dims, axis)

    return output

def matrix_inv(tensor):
    """
    Computes the inverse of a Hermitian matrix.

    Given a batch of Hermitian positive definite matrices
    :math:`\mathbf{A}`, the function
    returns :math:`\mathbf{A}^{-1}`, such that
    :math:`\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as tensor, containing
        the inverse of its last two dimensions.
    """
    if tensor.is_complex():
        s, u = torch.linalg.eigh(tensor)
        
        # Compute inverse of eigenvalues
        s = s.abs()
        assert torch.all(s > 0), "Input must be positive definite."
        s = 1 / s
        s = s.to(u.dtype)
        
        # Matrix multiplication
        s = s.unsqueeze(-2)
        return torch.matmul(u * s, u.conj().transpose(-2, -1))
    else:
        return torch.inverse(tensor)

##################### utils/tensor.py ####################################################################
