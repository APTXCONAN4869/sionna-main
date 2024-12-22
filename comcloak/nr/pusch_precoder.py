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
        w_list = []
        for w in precoding_matrices:
            assert w.shape[0]==shape[0] and w.shape[1]==shape[1], \
                "All precoding matrices must have the same shape"
            w_list.append(w)

        # w has shape:
        #[num_tx, num_antenna_ports, num_layers]
        self._w = torch.tensor(w_list, dtype=self.dtype)

    def build(self, input_shape):
        _, num_tx, num_layers, _, _ = input_shape
        assert num_tx==len(self._w), \
            f"""The input shape is for {num_tx} transmitters, but you have
                configured precoding matrices for {len(self._w)}."""
        assert num_layers==self._w[0].shape[1], \
            f"""You have configured precoding matrices for
                {self._w[0].shape[1]} layers, but the input
                provides {num_layers} layers."""

    def forward(self, inputs):

        # inputs has shape:
        # [batch_size, num_tx, num_layers, num_symbols_per_slot,...
        #  ..., num_subcarriers]

        # Change ordering of dimensions:
        # [batch_size, num_symbols_per_slot, num_subcarriers, num_tx,...
        #  ..., num_layers]
        inputs = inputs.permute(0, 3, 4, 1, 2)

        # Expand dimensions for matrix multiplication: [batch_size, num_symbols_per_slot, num_subcarriers, num_tx, num_layers, 1]
        inputs = inputs.unsqueeze(-1)

        # Precode: result shape [batch_size, num_symbols_per_slot, num_subcarriers, num_tx, num_antenna_ports]
        z = torch.matmul(self._w.unsqueeze(0).unsqueeze(0).unsqueeze(0), inputs).squeeze(-1)

        # Reorder dimensions to match desired output: [batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers]
        z = z.permute(0, 3, 4, 1, 2)

        return z
