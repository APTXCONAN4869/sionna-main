import torch
from comcloak.channel.channel_model import ChannelModel

class RayleighBlockFading(ChannelModel):
    def __init__(self, 
                 num_rx, 
                 num_rx_ant, 
                 num_tx, 
                 num_tx_ant,
                 dtype= torch.complex64):
        
        assert dtype.is_complex,"'dtype' must be complex type"
        self._dtype = dtype

        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant

    def __call__(self, batch_size, num_time_steps, sampling_frequency=None):
        # Delays
        if self._dtype ==torch.complex32:
            real_dtype = torch.float16
        elif self._dtype == torch.complex64:
            real_dtype = torch.float32
        elif self._dtype == torch.complex128:
            real_dtype = torch.float64
        else: 
            raise TypeError("Not found complex dtype in [complex32,complex64,complex128]")
        
        delays = torch.zeros([batch_size, 
                              self.num_rx, 
                              self.num_tx, 
                              1],
                              dtype=real_dtype)

        # Fading coefficients
        std = torch.tensor(torch.sqrt(torch.tensor(0.5)),dtype=real_dtype)
        h_real = torch.normal(mean=0.0,
                              std=std,
                              size=(batch_size, 
                                    self.num_rx, 
                                    self.num_rx_ant, 
                                    self.num_tx, 
                                    self.num_tx_ant, 
                                    1, 
                                    1),
                            dtype=real_dtype) 
        h_img = torch.normal(mean=0.0,
                             std=std,
                             size=(batch_size,
                                   self.num_rx,
                                   self.num_rx_ant,
                                   self.num_tx,
                                   self.num_tx_ant,
                                   1,
                                   1),
                            dtype=real_dtype)
        h = torch.complex(h_real, h_img)

        # Tile the response over the block
        h = h.repeat(1, 1, 1, 1, 1, 1, num_time_steps)
        return h, delays
    
""" # Example usage
num_rx = 2
num_rx_ant = 2
num_tx = 2
num_tx_ant = 2
dtype = torch.complex64

rayleigh_block_fading = RayleighBlockFading(num_rx, num_rx_ant, num_tx, num_tx_ant, dtype)
batch_size = 4
num_time_steps = 10
h, delays = rayleigh_block_fading(batch_size, num_time_steps)

print(h.shape)  # Expected shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 1, num_time_steps]
print(delays.shape)  # Expected shape: [batch_size, num_rx, num_tx, 1] """