import torch
import torch.nn as nn

class PUSCHPrecoder(nn.Module):
    """
    PUSCHPrecoder(precoding_matrices, dtype=torch.complex64)

    Precodes a batch of modulated symbols mapped onto a resource grid
    for PUSCH transmissions. Each transmitter is assumed to have its
    own precoding matrix.

    Parameters
    ----------
    precoding_matrices : list of torch.Tensor, [num_tx, num_antenna_ports, num_layers]
        List of precoding matrices, one for each transmitter.
        All precoding matrices must have the same shape.

    dtype : torch.dtype, one of [torch.complex64, torch.complex128]
        Dtype of inputs and outputs. Defaults to torch.complex64.

    Input
    -----
        : [batch_size, num_tx, num_layers, num_symbols_per_slot, num_subcarriers]
            Batch of resource grids to be precoded

    Output
    ------
        : [batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers]
            Batch of precoded resource grids
    """
    
    def __init__(self, precoding_matrices, dtype=torch.complex64):
        super(PUSCHPrecoder, self).__init__()
        
        assert dtype in [torch.complex64, torch.complex128], \
            "dtype must be torch.complex64 or torch.complex128"
        
        self.dtype = dtype
        self.num_tx = len(precoding_matrices)

        # Ensure all precoding matrices have the same shape
        shape = precoding_matrices[0].shape
        for w in precoding_matrices:
            assert w.shape == shape, "All precoding matrices must have the same shape"

        # Convert list to tensor for efficient operations
        self.w = torch.stack(precoding_matrices).type(dtype)  # Shape: [num_tx, num_antenna_ports, num_layers]

    def forward(self, inputs):
        # inputs shape: [batch_size, num_tx, num_layers, num_symbols_per_slot, num_subcarriers]
        
        # Verify compatibility between inputs and precoding matrices
        _, num_tx, num_layers, _, _ = inputs.shape
        assert num_tx == self.w.shape[0], \
            f"The input shape has {num_tx} transmitters, but {self.w.shape[0]} precoding matrices were provided."
        assert num_layers == self.w.shape[2], \
            f"The input provides {num_layers} layers, but precoding matrices have {self.w.shape[2]} layers."

        # Change ordering of dimensions for multiplication: [batch_size, num_symbols_per_slot, num_subcarriers, num_tx, num_layers]
        inputs = inputs.permute(0, 3, 4, 1, 2)

        # Expand dimensions for matrix multiplication: [batch_size, num_symbols_per_slot, num_subcarriers, num_tx, num_layers, 1]
        inputs = inputs.unsqueeze(-1)

        # Precode: result shape [batch_size, num_symbols_per_slot, num_subcarriers, num_tx, num_antenna_ports]
        z = torch.matmul(self.w.unsqueeze(0).unsqueeze(0).unsqueeze(0), inputs).squeeze(-1)

        # Reorder dimensions to match desired output: [batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers]
        z = z.permute(0, 3, 4, 1, 2)

        return z
