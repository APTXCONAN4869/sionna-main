import numpy as np
import torch
import torch.nn as nn
from comcloak.fec.crc import CRCDecoder
from comcloak.fec.scrambling import  Descrambler
from comcloak.fec.ldpc import LDPC5GDecoder
from comcloak.nr import TBEncoder
from comcloak.supplement import gather_pytorch
import torch
import torch.nn as nn

class TBDecoder(nn.Module):
    r"""TBDecoder(encoder, num_bp_iter=20, cn_type="boxplus-phi", output_dtype=torch.float32, **kwargs)
    5G NR transport block (TB) decoder as defined in TS 38.214
    [3GPP38214]_.

    The transport block decoder takes as input a sequence of noisy channel
    observations and reconstructs the corresponding `transport block` of
    information bits. The detailed procedure is described in TS 38.214
    [3GPP38214]_ and TS 38.211 [3GPP38211]_.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        encoder : :class:`~sionna.nr.TBEncoder`
            Associated transport block encoder used for encoding of the signal.

        num_bp_iter : int, 20 (default)
            Number of BP decoder iterations

        cn_type : str, "boxplus-phi" (default) | "boxplus" | "minsum"
            The check node processing function of the LDPC BP decoder.
            One of {`"boxplus"`, `"boxplus-phi"`, `"minsum"`} where
            '"boxplus"' implements the single-parity-check APP decoding rule.
            '"boxplus-phi"' implements the numerical more stable version of
            boxplus [Ryan]_.
            '"minsum"' implements the min-approximation of the CN update rule
            [Ryan]_.

        output_dtype : torch.float32 (default)
            Defines the datatype for internal calculations and the output dtype.

    Input
    -----
        inputs : [...,num_coded_bits], torch.float
            2+D tensor containing channel logits/llr values of the (noisy)
            channel observations.

    Output
    ------
        b_hat : [...,target_tb_size], torch.float
            2+D tensor containing hard decided bit estimates of all information
            bits of the transport block.

        tb_crc_status : [...], torch.bool
            Transport block CRC status indicating if a transport block was
            (most likely) correctly recovered. Note that false positives are
            possible.
    """

    def __init__(self,
                 encoder,
                 num_bp_iter=20,
                 cn_type="boxplus-phi",
                 output_dtype=torch.float32,
                 **kwargs):

        super().__init__()

        assert output_dtype in (torch.float16, torch.float32, torch.float64), \
                "output_dtype must be (torch.float16, torch.float32, torch.float64)."

        assert isinstance(encoder, TBEncoder), "encoder must be TBEncoder."
        self._tb_encoder = encoder
        self.dtype = output_dtype
        self._num_cbs = encoder.num_cbs

        # init BP decoder
        self._decoder = LDPC5GDecoder(encoder=encoder.ldpc_encoder,
                                      num_iter=num_bp_iter,
                                      cn_type=cn_type,
                                      hard_out=True, # TB operates on bit-level
                                      return_infobits=True,
                                      output_dtype=output_dtype)

        # init descrambler
        if encoder.scrambler is not None:
            self._descrambler = Descrambler(encoder.scrambler,
                                            binary=False)
        else:
            self._descrambler = None

        # init CRC Decoder for CB and TB
        self._tb_crc_decoder = CRCDecoder(encoder.tb_crc_encoder)

        if encoder.cb_crc_encoder is not None:
            self._cb_crc_decoder = CRCDecoder(encoder.cb_crc_encoder)
        else:
            self._cb_crc_decoder = None

    #########################################
    # Public methods and properties
    #########################################

    @property
    def tb_size(self):
        """Number of information bits per TB."""
        return self._tb_encoder.tb_size

    # required for
    @property
    def k(self):
        """Number of input information bits. Equals TB size."""
        return self._tb_encoder.tb_size

    @property
    def n(self):
        """Total number of output codeword bits."""
        return self._tb_encoder.n
    
    def forward(self, inputs):
        """
        Apply transport block decoding.
        
        Parameters
        ----------
        inputs : torch.Tensor
            A tensor of shape [..., num_coded_bits] containing channel logits/LLR values.
        
        Returns
        -------
        u_hat : torch.Tensor
            Tensor containing hard decided bit estimates of all information bits.
        
        tb_crc_status : torch.Tensor
            Tensor indicating if a transport block was correctly recovered.
        """
        input_shape = list(inputs.shape)
        assert input_shape[-1]==self.n, \
            f"Invalid input shape. Expected input length is {self.n}."
        llr_ch = inputs.to(torch.float32)
        
        llr_ch = torch.reshape(llr_ch,
                            (-1, self._tb_encoder.num_tx, self._tb_encoder.n))

        # undo scrambling (only if scrambler was used)
        if self._descrambler is not None:
            llr_scr = self._descrambler(llr_ch)
        else:
            llr_scr = llr_ch
        
        # Undo CB interleaving and puncturing
        num_fillers = self._tb_encoder.ldpc_encoder.n * self._tb_encoder.num_cbs - np.sum(self._tb_encoder.cw_lengths)
        llr_int = torch.cat(
            [llr_scr, torch.zeros(llr_scr.shape[0], self._tb_encoder.num_tx, num_fillers, device=inputs.device)], 
            dim=-1
        )
        llr_int = gather_pytorch(llr_int, self._tb_encoder.output_perm_inv, axis=-1)
        
        # Undo CB concatenation
        llr_cb = torch.reshape(llr_int,
                        (-1, self._tb_encoder.num_tx, self._num_cbs, self._tb_encoder.ldpc_encoder.n))
        # LDPC decoding
        u_hat_cb = self._decoder(llr_cb)

        # CB CRC removal (if relevant)
        if self._cb_crc_decoder is not None:
            # we are ignoring the CB CRC status for the moment
            # Could be combined with the TB CRC for even better estimates
            u_hat_cb_crc, _ = self._cb_crc_decoder(u_hat_cb)
        else:
            u_hat_cb_crc = u_hat_cb

        # undo CB segmentation
        u_hat_tb = torch.reshape(u_hat_cb_crc,
                (-1, self._tb_encoder.num_tx, self.tb_size+self._tb_encoder.tb_crc_encoder.crc_length))

        # TB CRC removal
        u_hat, tb_crc_status = self._tb_crc_decoder(u_hat_tb)
        
        # Restore input shape
        output_shape = input_shape.copy()
        output_shape[0] = -1
        output_shape[-1] = self.tb_size
        u_hat = torch.reshape(u_hat, output_shape)
        
        # Also apply to tb_crc_status
        output_shape[-1] = 1  # Last dim is 1
        tb_crc_status = torch.reshape(tb_crc_status, output_shape)
        
        # Remove zero-padding if applied
        if self._tb_encoder.k_padding > 0:
            u_hat = u_hat[..., : -self._tb_encoder.k_padding]
        
        # Cast to output dtype
        u_hat = u_hat.to(self.dtype)
        tb_crc_status = tb_crc_status.squeeze(-1).bool()
        
        return u_hat, tb_crc_status
