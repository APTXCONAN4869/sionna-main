import torch
import torch.nn as nn
from comcloak.mapping import Mapper
from comcloak.utils import BinarySource
from comcloak.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator
from .config import Config
from .pusch_config import PUSCHConfig, check_pusch_configs
from .pusch_pilot_pattern import PUSCHPilotPattern
from .pusch_precoder import PUSCHPrecoder
from .tb_encoder import TBEncoder
from .layer_mapping import LayerMapper

import torch
import torch.nn as nn

class PUSCHTransmitter(nn.Module):
    r"""
    PUSCHTransmitter(pusch_configs, return_bits=True, output_domain="freq", dtype=torch.complex64, verbose=False)

    生成批量的 5G NR PUSCH 时隙信号，可以是频域或时域输出。

    参数
    ----------
    pusch_configs : instance or list of PUSCHConfig
        PUSCH 配置。

    return_bits : bool
        如果为 True，生成随机信息比特，并将其作为输出之一返回。默认值为 True。

    output_domain : str, one of ["freq", "time"]
        输出域，默认为 "freq"。

    dtype : One of [torch.complex64, torch.complex128]
        输入和输出的数据类型。默认为 torch.complex64。

    verbose: bool
        如果为 True，则在初始化期间打印额外的参数信息。默认为 False。
    """

    def __init__(self, pusch_configs, return_bits=True, output_domain="freq", dtype=torch.complex64, verbose=False):
        super().__init__()

        # 验证输入
        assert dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"
        self.dtype = dtype

        assert isinstance(return_bits, bool), "return_bits must be bool"
        self.return_bits = return_bits

        assert output_domain in ["time", "freq"], "output_domain must be 'time' or 'freq'"
        self.output_domain = output_domain

        assert isinstance(verbose, bool), "verbose must be bool"
        self.verbose = verbose

        # 如果传入的是单个配置，则转换为列表
        if not isinstance(pusch_configs, list):
            pusch_configs = [pusch_configs]

        # 验证 PUSCH 配置并提取参数
        params = check_pusch_configs(pusch_configs)
        for key, value in params.items():
            setattr(self, f"_{key}", value)

        self.pusch_configs = pusch_configs

        # (可选) 创建 BinarySource
        if self.return_bits:
            self.binary_source = BinarySource(dtype=torch.float32 if dtype == torch.complex64 else torch.float64)

        # 初始化子模块
        self.tb_encoder = TBEncoder(
            target_tb_size=self._tb_size,
            num_coded_bits=self._num_coded_bits,
            target_coderate=self._target_coderate,
            num_bits_per_symbol=self._num_bits_per_symbol,
            num_layers=self._num_layers,
            n_rnti=self._n_rnti,
            n_id=self._n_id,
            channel_type="PUSCH",
            codeword_index=0,
            use_scrambler=True,
            verbose=self.verbose,
            output_dtype=torch.float32 if dtype == torch.complex64 else torch.float64
        )

        self.layer_mapper = LayerMapper(num_layers=self._num_layers, dtype=dtype)
        self.mapper = Mapper("qam", self._num_bits_per_symbol, dtype=dtype)
        self.pilot_pattern = PUSCHPilotPattern(self.pusch_configs, dtype=dtype)

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self._num_ofdm_symbols,
            fft_size=self._num_subcarriers,
            subcarrier_spacing=self._subcarrier_spacing,
            num_tx=self._num_tx,
            num_streams_per_tx=self._num_layers,
            cyclic_prefix_length=self._cyclic_prefix_length,
            pilot_pattern=self.pilot_pattern,
            dtype=dtype
        )

        self.resource_grid_mapper = ResourceGridMapper(self.resource_grid, dtype=dtype)

        if self._precoding == "codebook":
            self.precoder = PUSCHPrecoder(self._precoding_matrices, dtype=dtype)

        if self.output_domain == "time":
            self.ofdm_modulator = OFDMModulator(self._cyclic_prefix_length)

    @property
    def resource_grid(self):
        """OFDM resource grid underlying the PUSCH transmissions"""
        return self.resource_grid

    @property
    def pilot_pattern(self):
        """Aggregate pilot pattern of all transmitters"""
        return self.pilot_pattern

    def show(self):
        """Print all properties of the PUSCHConfig and children"""
        self.pusch_configs[0].carrier.show()
        Config.show(self.pusch_configs[0])
        for idx, p in enumerate(self.pusch_configs):
            print(f"---- UE {idx} ----")
            p.dmrs.show()
            p.tb.show()

    def forward(self, inputs):
        if self.return_bits:
            # inputs 定义 batch_size
            batch_size = inputs
            b = self.binary_source((batch_size, self._num_tx, self._tb_size))
        else:
            b = inputs

        # 编码传输块
        c = self.tb_encoder(b)

        # 映射到星座点
        x_map = self.mapper(c)

        # 映射到层
        x_layer = self.layer_mapper(x_map)

        # 资源网格映射
        x_grid = self.resource_grid_mapper(x_layer)

        # (可选) PUSCH 预编码
        if self._precoding == "codebook":
            x_pre = self.precoder(x_grid)
        else:
            x_pre = x_grid

        # (可选) OFDM 调制
        if self.output_domain == "time":
            x = self.ofdm_modulator(x_pre)
        else:
            x = x_pre

        if self.return_bits:
            return x, b
        else:
            return x
