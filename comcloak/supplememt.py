import torch
def gather_pytorch(input_data, indices=None, batch_dims=0, axis=0):
    input_data = torch.tensor(input_data)
    indices = torch.tensor(indices)
    if batch_dims == 0:
        if axis < 0:
            axis = len(input_data.shape) + axis
        data = torch.index_select(input_data, axis, indices.flatten())
        shape_input = list(input_data.shape)
        # shape_ = delete(shape_input, axis)
        # 连接列表
        shape_output = shape_input[:axis] + \
            list(indices.shape) + shape_input[axis + 1:]
        data_output = data.reshape(shape_output)
        return data_output
    else:
        data_output = []
        for data,ind in zip(input_data, indices):
            r = gather_pytorch(data, ind, batch_dims=batch_dims-1)
            data_output.append(r)
        return torch.stack(data_output)

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
