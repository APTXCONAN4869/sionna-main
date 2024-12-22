import torch
def flatten_dims(tensor, num_dims, axis):
    r"""
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
    r"""
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
    r"""
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
    r"""
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
    r"""
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
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    current_rank = tensor.dim()
    num_dims = max(target_rank - current_rank, 0)
    output = insert_dims(tensor, num_dims, axis)

    return output

def matrix_inv(tensor):
    r"""
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



##################

def matrix_sqrt(tensor):
    r""" Computes the square root of a matrix.

    Given a batch of Hermitian positive semi-definite matrices
    :math:`\mathbf{A}`, returns matrices :math:`\mathbf{B}`,
    such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor`` containing
        the matrix square root of its last two dimensions.

    Note:
        If you want to use this function in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.config.xla_compat=true``.
        See :py:attr:`~sionna.config.xla_compat`.
    """
    if torch.is_grad_enabled():  # Assuming this corresponds to eager execution in TensorFlow
        # Compute the eigenvalues and eigenvectors
        s, u = torch.linalg.eigh(tensor)
        # Compute sqrt of eigenvalues
        s = torch.abs(s)
        s = torch.sqrt(s)
        s = torch.tensor(s, dtype=u.dtype)

        # Matrix multiplication
        s = s.unsqueeze(-2)
        return torch.matmul(u * s, u.transpose(-2, -1).conj())
    else:
        return torch.linalg.inv(torch.linalg.matrix_power(tensor, 1/2))


def matrix_sqrt_inv(tensor):
    r"""
    Computes the inverse square root of a Hermitian matrix.

    Given a batch of Hermitian positive definite matrices
    :math:`\mathbf{A}`, with square root matrices :math:`\mathbf{B}`,
    such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`, the function
    returns :math:`\mathbf{B}^{-1}`, such that
    :math:`\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor`` containing
        the inverse matrix square root of its last two dimensions.
    """
    if torch.is_grad_enabled():  # Assuming this corresponds to eager execution in TensorFlow
        # Compute the eigenvalues and eigenvectors
        s, u = torch.linalg.eigh(tensor)
        
        # Compute 1/sqrt of eigenvalues
        s = torch.abs(s)
        if torch.any(s <= 0):
            raise ValueError("Input must be positive definite.")
        s = 1.0 / torch.sqrt(s)
        
        # Matrix multiplication
        s = s.unsqueeze(-2)
        return torch.matmul(u * s, u.transpose(-2, -1).conj())
    else:
        return torch.linalg.inv(torch.linalg.matrix_power(tensor, 1/2))


def matrix_pinv(tensor):# torch.linalg.pinv()
    r""" Computes the Moore-Penrose (or pseudo) inverse of a matrix.

    Given a batch of :math:`M \times K` matrices :math:`\mathbf{A}` with rank
    :math:`K` (i.e., linearly independent columns), the function returns
    :math:`\mathbf{A}^+`, such that
    :math:`\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, K]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of shape ([..., K,K]) of the same type as ``tensor``,
        containing the pseudo inverse of its last two dimensions.

    Note:
        If you want to use this function in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.config.xla_compat=true``.
        See :py:attr:`~sionna.config.xla_compat`.
    """
    inv = matrix_inv(torch.matmul(tensor.conj().transpose(-2, -1), tensor))
    return torch.matmul(inv, tensor.conj().transpose(-2, -1))

