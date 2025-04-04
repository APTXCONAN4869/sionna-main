import torch
import numpy as np
import matplotlib.pyplot as plt
# from comcloak.ofdm import ofdm_test_module_z
from comcloak.utils import QAMSource
from matplotlib import colors

class PilotPattern():
    def __init__(self, mask, pilots, trainable=False, normalize=False, dtype=torch.complex64):
        super().__init__()
        self._dtype = dtype
        if isinstance(mask, np.ndarray):
            self._mask = torch.from_numpy(mask).to(torch.int32)
            self._pilots = torch.nn.Parameter(torch.from_numpy(pilots).to(self._dtype), requires_grad=trainable)
        elif isinstance(mask, torch.Tensor):
            self._mask = mask.to(torch.int32)
            self._pilots = torch.nn.Parameter(pilots.to(self._dtype), requires_grad=trainable)
        self.normalize = normalize
        self._check_settings()

    @property
    def num_tx(self):
        """Number of transmitters"""
        return self._mask.shape[0]

    @property
    def num_streams_per_tx(self):
        return self._mask.shape[1]

    @property
    def num_ofdm_symbols(self):
        return self._mask.shape[2]

    @property
    def num_effective_subcarriers(self):
        return self._mask.shape[3]

    @property
    def num_pilot_symbols(self):
        return self._pilots.shape[-1]

    @property
    def num_data_symbols(self):
        return self._mask.shape[-1] * self._mask.shape[-2] - \
                self.num_pilot_symbols

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        self._normalize = torch.tensor(value, dtype=torch.bool)

    @property
    def mask(self):
        return self._mask

    @property
    def pilots(self):
        if self.normalize:
            scale = torch.abs(self._pilots) ** 2
            scale = 1 / torch.sqrt(torch.mean(scale, dim=-1, keepdim=True))
            scale = scale.to(self._dtype)
            return scale * self._pilots
        else:
            return self._pilots

    @pilots.setter
    def pilots(self, value):
        self._pilots.data = value

    def _check_settings(self):
        assert len(self._mask.shape) == 4, "`mask` must have four dimensions."
        assert len(self._pilots.shape) == 3, "`pilots` must have three dimensions."
        assert np.array_equal(self._mask.shape[:2], self._pilots.shape[:2]), "The first two dimensions of `mask` and `pilots` must be equal."

        num_pilots = torch.sum(self._mask, dim=(-2, -1))
        assert torch.min(num_pilots) == torch.max(num_pilots), "The number of nonzero elements in the masks for all transmitters and streams must be identical."
        assert self.num_pilot_symbols == torch.max(num_pilots), "The shape of the last dimension of `pilots` must equal the number of non-zero entries within the last two dimensions of `mask`."

    @property
    def trainable(self):
        return self._pilots.requires_grad

    def show(self, tx_ind=None, stream_ind=None, show_pilot_ind=False):
        mask = self.mask.detach().cpu().numpy()
        pilots = self.pilots.detach().cpu().numpy()

        if tx_ind is None:
            tx_ind = range(0, self.num_tx)
        elif not isinstance(tx_ind, list):
            tx_ind = [tx_ind]

        if stream_ind is None:
            stream_ind = range(0, self.num_streams_per_tx)
        elif not isinstance(stream_ind, list):
            stream_ind = [stream_ind]

        figs = []
        for i in tx_ind:
            for j in stream_ind:
                q = np.zeros_like(mask[0, 0])
                q[np.where(mask[i, j])] = (np.abs(pilots[i, j]) == 0) + 1
                legend = ["Data", "Pilots", "Masked"]
                fig = plt.figure()
                plt.title(f"TX {i} - Stream {j}")
                plt.xlabel("OFDM Symbol")
                plt.ylabel("Subcarrier Index")
                plt.xticks(range(0, q.shape[1]))
                cmap = plt.cm.tab20c
                b = np.arange(0, 4)
                norm = colors.BoundaryNorm(b, cmap.N)
                im = plt.imshow(np.transpose(q), origin="lower", aspect="auto", norm=norm, cmap=cmap)
                cbar = plt.colorbar(im)
                cbar.set_ticks(b[:-1] + 0.5)
                cbar.set_ticklabels(legend)

                if show_pilot_ind:
                    c = 0
                    for t in range(self.num_ofdm_symbols):
                        for k in range(self.num_effective_subcarriers):
                            if mask[i, j, t, k]:
                                if np.abs(pilots[i, j, c]) > 0:
                                    plt.annotate(c, [t, k])
                                c += 1
                figs.append(fig)

        return figs

class EmptyPilotPattern(PilotPattern):
    def __init__(self,
                 num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 dtype=torch.complex64):
        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        mask = torch.zeros(shape, dtype=torch.bool)
        pilots = torch.zeros(shape[:2] + [0], dtype=dtype)
        super().__init__(mask, pilots, trainable=False, normalize=False, dtype=dtype)

class KroneckerPilotPattern(PilotPattern):
    r"""Simple orthogonal pilot pattern with Kronecker structure.

    This function generates an instance of :class:`~sionna.ofdm.PilotPattern`
    that allocates non-overlapping pilot sequences for all transmitters and
    streams on specified OFDM symbols. As the same pilot sequences are reused
    across those OFDM symbols, the resulting pilot pattern has a frequency-time
    Kronecker structure. This structure enables a very efficient implementation
    of the LMMSE channel estimator. Each pilot sequence is constructed from
    randomly drawn QPSK constellation points.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of a :class:`~sionna.ofdm.ResourceGrid`.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension.
        Defaults to `True`.

    seed : int
        Seed for the generation of the pilot sequence. Different seed values
        lead to different sequences. Defaults to 0.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Note
    ----
    It is required that the ``resource_grid``'s property
    ``num_effective_subcarriers`` is an
    integer multiple of ``num_tx * num_streams_per_tx``. This condition is
    required to ensure that all transmitters and streams get
    non-overlapping pilot sequences. For a large number of streams and/or
    transmitters, the pilot pattern becomes very sparse in the frequency
    domain.

    Examples
    --------
    >>> rg = ResourceGrid(num_ofdm_symbols=14,
    ...                   fft_size=64,
    ...                   subcarrier_spacing = 30e3,
    ...                   num_tx=4,
    ...                   num_streams_per_tx=2,
    ...                   pilot_pattern = "kronecker",
    ...                   pilot_ofdm_symbol_indices = [2, 11])
    >>> rg.pilot_pattern.show();

    .. image:: ../figures/kronecker_pilot_pattern.png

    """
    def __init__(self, resource_grid, pilot_ofdm_symbol_indices, normalize=True, seed=0, dtype=torch.complex64):
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        self._dtype = dtype

        num_pilot_symbols = len(pilot_ofdm_symbol_indices)
        num_seq = num_tx * num_streams_per_tx
        num_pilots = num_pilot_symbols * num_effective_subcarriers / num_seq
        assert num_pilots % 1 == 0, "`num_effective_subcarriers` must be an integer multiple of `num_tx`*`num_streams_per_tx`."

        num_pilots_per_symbol = int(num_pilots / num_pilot_symbols)
        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        mask = torch.zeros(shape, dtype=torch.bool)
        shape[2] = num_pilot_symbols
        pilots = torch.zeros(shape, dtype=torch.complex64)

        mask[..., pilot_ofdm_symbol_indices, :] = True

        qam_source = QAMSource(2, seed=seed, dtype=self._dtype)
        for i in range(num_tx):
            for j in range(num_streams_per_tx):
                # Generate random QPSK symbols
                p = qam_source([1,1,num_pilot_symbols,num_pilots_per_symbol])
                
                pilots[i, j, :, i * num_streams_per_tx + j::num_seq] = p

        pilots = pilots.reshape([num_tx, num_streams_per_tx, -1])
        super().__init__(mask, pilots, trainable=False, normalize=normalize, dtype=self._dtype)
