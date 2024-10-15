import torch
import warnings
import numpy as np
from comcloak.utils import expand_to_rank
from comcloak.supplememt import get_real_dtype, gather_pytorch
from comcloak import PI
# def gather_pytorch(input_data, indices=None, batch_dims=0, axis=0):
#     input_data = torch.tensor(input_data)
#     indices = torch.tensor(indices)
#     if batch_dims == 0:
#         if axis < 0:
#             axis = len(input_data.shape) + axis
#         data = torch.index_select(input_data, axis, indices.flatten())
#         shape_input = list(input_data.shape)
#         # shape_ = delete(shape_input, axis)
#         # 连接列表
#         shape_output = shape_input[:axis] + \
#             list(indices.shape) + shape_input[axis + 1:]
#         data_output = data.reshape(shape_output)
#         return data_output
#     else:
#         data_output = []
#         for data,ind in zip(input_data, indices):
#             r = gather_pytorch(data, ind, batch_dims=batch_dims-1)
#             data_output.append(r)
#         return torch.stack(data_output)

def subcarrier_frequencies(num_subcarriers, subcarrier_spacing, dtype=torch.complex64):
    """
    计算“num_subcarriers”个子载波的基带频率,间隔为“subcarrier_spacing”

    参数
    ------
    num_subcarriers : int
        子载波数量

    subcarrier_spacing : float
        子载波间隔 [Hz]

    dtype : torch.dtype
        用于内部处理和输出的数据类型。
        如果提供的是复数数据类型，则使用相应的实数部分精度。
        默认为`torch.complex64` (`torch.float32`)。

    返回
    ------
    frequencies : [``num_subcarrier``], torch.float
        子载波的基带频率
    """
    if dtype in (torch.complex32,torch.complex64,torch.complex128):
        if dtype == torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("not find dtype type")
    elif dtype in (torch.float16,torch.float32,torch.float64):
        real_dtype = dtype
    else:
        raise AssertionError("dtype must be a complex or floating datatype")
    
    num_subcarriers_tensor = torch.tensor(num_subcarriers)  
    if torch.equal(torch.fmod(num_subcarriers_tensor,torch.tensor(2)),torch.tensor(0)):
        start = -num_subcarriers/2
        end = num_subcarriers/2
    else:
        start = -(num_subcarriers-1)/2
        end = (num_subcarriers-1)/2+1
    
    frequencies = torch.arange( start=start,
                                end=end,
                                dtype=real_dtype)
    frequencies = frequencies*subcarrier_spacing
    return frequencies
    # 测试函数
    """ num_subcarriers = 5
    subcarrier_spacing = 15e3  # 15 kHz
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing, dtype=torch.float32)
    print("Generated subcarrier frequencies:")
    print(frequencies)
    """

def time_frequency_vector(num_samples, sample_duration, dtype=torch.float32):
    # pylint: disable=line-too-long
    r"""
    Compute the time and frequency vector for a given number of samples
    and duration per sample in normalized time unit.

    >>> t = torch.cast(torch.linspace(-n_min, n_max, num_samples), dtype) * sample_duration
    >>> f = torch.cast(torch.linspace(-n_min, n_max, num_samples), dtype) * 1/(sample_duration*num_samples)

    Input
    ------
        num_samples : int
            Number of samples

        sample_duration : float
            Sample duration in normalized time

        dtype : torch.DType
            Datatype to use for internal processing and output.
            Defaults to `torch.float32`.

    Output
    ------
        t : [``num_samples``], ``dtype``
            Time vector

        f : [``num_samples``], ``dtype``
            Frequency vector
    """

    num_samples = int(num_samples)

    if num_samples % 2 == 0:  # 如果样本数为偶数
        n_min = torch.tensor(-(num_samples) / 2, dtype=torch.int32)
        n_max = torch.tensor((num_samples) / 2 - 1, dtype=torch.int32)
    else:  # 如果样本数为奇数
        n_min = torch.tensor(-(num_samples-1) / 2, dtype=torch.int32)
        n_max = torch.tensor((num_samples+1) / 2 - 1, dtype=torch.int32)

    # 时间向量
    t = torch.linspace(n_min, n_max, num_samples, dtype=dtype) \
        * torch.tensor(sample_duration, dtype=dtype)

    # 频率向量
    df = torch.tensor(1.0/sample_duration, dtype=dtype)/torch.tensor(num_samples, dtype=dtype)
    f = torch.linspace(n_min, n_max, num_samples, dtype=dtype) \
        * df

    return t, f

def time_lag_discrete_time_channel(bandwidth, maximum_delay_spread=3e-6):
    """
    Compute the smallest and largest time-lag for the discrete complex baseband channel,
    i.e., L_min and L_max.

    The smallest time-lag (L_min) returned is always -6, as this value
    was found small enough for all models included in Sionna.

    The largest time-lag (L_max) is computed from the `bandwidth`
    and `maximum_delay_spread` as follows:

    L_max = ceil(W * tau_max) + 6

    where L_max is the largest time-lag, W the `bandwidth`,
    and tau_max the `maximum_delay_spread`.

    The default value for the `maximum_delay_spread` is 3us, which was found
    to be large enough to include most significant paths with all channel models
    included in Sionna assuming a nominal delay spread of 100ns.

    Note:
    The values of L_min and L_max computed by this function are only recommended values.
    L_min and L_max should be set according to the considered channel model. For OFDM systems,
    one also needs to be careful that the effective length of the complex baseband channel
    is not larger than the cyclic prefix length.

    Args:
        bandwidth (float): Bandwidth (W) [Hz]
        maximum_delay_spread (float): Maximum delay spread [s]. Defaults to 3us.

    Returns:
        tuple: (l_min, l_max)
            l_min (int): Smallest time-lag (L_min) for the discrete complex baseband channel. Set to -6.
            l_max (int): Largest time-lag (L_max) for the discrete complex baseband channel.
    """
    l_min = torch.tensor(-6, dtype=torch.int32)
    l_max = torch.ceil(torch.tensor(maximum_delay_spread * bandwidth, dtype=torch.float32)) + 6
    l_max = torch.tensor(l_max, dtype=torch.int32)
    return l_min, l_max

    """ # 测试函数
    bandwidth = 1e6  # 1 MHz
    max_delay_spread = 3e-6  # 3 microseconds
    l_min, l_max = time_lag_discrete_time_channel(bandwidth, max_delay_spread)
    print(f"l_min: {l_min}, l_max: {l_max}") """

def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):
    """
    计算给定频率下的信道频率响应。

    参数
    ------
    frequencies : [fft_size], torch.float
        计算信道响应的频率

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], torch.complex
        路径系数

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], torch.float
        路径延迟

    normalize : bool
        如果设置为`True`,则信道响应将在资源网格上归一化以确保每个资源元素的平均能量为1。默认为`False`。

    返回
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], torch.complex
        在``frequencies``下的信道频率响应
    """
    tau = torch.tensor(tau)
    real_dtype = tau.dtype

    if len(tau.shape) ==4:
        # 扩展维度以与 h 广播。添加以下维度：
        # - 接收天线数量 (2)
        # - 发射天线数量 (4)
        tau = torch.unsqueeze(torch.unsqueeze(tau, dim=2),dim=4)
        # 手动进行部分广播
        tau = tau.repeat(1,1,1,1,a.shape[4],1)
    
    # 添加时间样本维度以进行广播
    tau = torch.unsqueeze(tau, dim=6)

    # 将所有张量调整为可广播的形状
    tau = torch.unsqueeze(tau, dim=-1)
    h = torch.unsqueeze(a, dim=-1)
    frequencies = expand_to_rank(frequencies, tau.dim(),axis=0)

    # 计算所有集群抽头的傅里叶变换
    # 指数组件
    e = torch.exp(torch.complex(torch.zeros((), dtype=real_dtype),
                                -2 * PI * frequencies * tau))
    h_f = h * e
    # 对所有集群求和以获得信道频率响应
    h_f = torch.sum(h_f, dim=-3)

    if normalize:   
        # 归一化以确保每个批次示例和链接的资源网格能量为 1
        # 平均发射天线、接收天线、OFDM 符号和子载波
        c = torch.mean(torch.square(torch.abs(h_f)), 
                       dim=(2,4,5,6),
                       keepdim=True)
        c = torch.complex(torch.sqrt(c),
                          torch.zeros((),dtype=real_dtype) )
        # h_f = torch.divide(h_f, c+ 1e-10)
        h_f = torch.nan_to_num(h_f / c, nan=0.0, posinf=0.0, neginf=0.0)

    return h_f
    # 测试函数
    """ from my_code.mysionna.utils import GLOBAL_SEED_NUMBER
    import tensorflow as torch

    fft_size = 64
    batch_size = 2
    num_rx = 2
    num_rx_ant = 4
    num_tx = 2
    num_tx_ant = 4
    num_paths = 3
    num_time_steps = 1

    torch.random.set_seed(GLOBAL_SEED_NUMBER)

    # 生成 TensorFlow 数据
    frequencies_torch = torch.linspace(-fft_size // 2, fft_size // 2 - 1, fft_size)
    frequencies_torch = torch.cast(frequencies_torch, torch.float64)
    a_real_torch = torch.random.uniform((batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps), dtype=torch.float64)
    a_imag_torch = torch.random.uniform((batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps), dtype=torch.float64)
    a_torch = torch.complex(a_real_torch, a_imag_torch)
    tau_torch = torch.random.uniform((batch_size, num_rx, num_tx, num_paths), dtype=torch.float64)

    # 将 TensorFlow 数据转换为 NumPy 数组
    frequencies_np = frequencies_torch.numpy()
    a_np = a_torch.numpy()
    tau_np = tau_torch.numpy()


    # 将 NumPy 数组转换为 PyTorch 张量
    frequencies = torch.from_numpy(frequencies_np)
    a = torch.from_numpy(a_np)
    tau = torch.from_numpy(tau_np)

    h_f = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    print("Generated OFDM channel frequency responses:")
    print(h_f)
    """

def cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=False):

    # tau = torch.tensor(tau)
    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = torch.unsqueeze(torch.unsqueeze(tau,dim=2),dim=4)
        # We therefore do part of it manually
        tau = tau.repeat(1,1,1,1,a.shape[4],1)
    
    # Add a time samples dimension for broadcasting
    tau = torch.unsqueeze(tau, dim=6)

    # Time lags for which to compute the channel taps
    l = torch.arange(l_min, l_max+1, dtype=real_dtype)

    # Bring tau and l to broadcastable shapes
    tau = torch.unsqueeze(tau, dim=-1)
    l = expand_to_rank(l, tau.dim(),axis=0)

    # sinc pulse shaping
    g = torch.sinc(l-tau*bandwidth)
    g = torch.complex(g, torch.tensor(0.,dtype=real_dtype))
    a = torch.unsqueeze(a, dim=-1)

    # For every tap, sum the sinc-weighted coefficients
    hm = torch.sum(a*g, dim=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = torch.mean(torch.sum(torch.square(torch.abs(hm)),
                                dim=6, 
                                keepdim=True),
                            dim=(2,4,5),
                            keepdim=True
                       )
        c = torch.complex(torch.sqrt(c), torch.tensor(0.,dtype=real_dtype))
        hm = torch.divide(hm, c+1e-10)
    
    return hm
    #测试函数
    """ import numpy as np
    def test_cir_to_time_channel():
        #设置种子
        torch.random.set_seed(GLOBAL_SEED_NUMBER)

        # 示例参数
        bandwidth = 20e6  # 20 MHz

        # 生成模拟数据
        batch_size = 1
        num_rx = 1
        num_rx_ant = 1
        num_tx = 1
        num_tx_ant = 1
        num_paths = 5
        num_time_steps = 10

        # 随机生成路径系数（复数）
        a_torch = torch.complex(torch.random.normal([batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]),
                    torch.random.normal([batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]))

        # 随机生成路径延迟（实数）
        tau_torch = torch.random.uniform([batch_size, num_rx, num_tx, num_paths])

        #将 TensorFlow 数据转换为 NumPy 数组
        a_np = a_torch.numpy()
        tau_np = tau_torch.numpy()
        # 将 NumPy 数组转换为 PyTorch 张量
        a = torch.from_numpy(a_np)
        tau = torch.from_numpy(tau_np)
        # 时间抽头范围
        l_min = -5
        l_max = 5

        # 调用函数
        hm = cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=True)

        print(hm)
        print(hm.shape)
        print(hm.dtype)
    test_cir_to_time_channel()
    """

def time_to_ofdm_channel(h_t, rg, l_min):
    r"""
    Compute the channel frequency response from the discrete complex-baseband
    channel impulse response.

    Given a discrete complex-baseband channel impulse response
    :math:`\bar{h}_{b,\ell}`, for :math:`\ell` ranging from :math:`L_\text{min}\le 0`
    to :math:`L_\text{max}`, the discrete channel frequency response is computed as

    .. math::

        \hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1

    where :math:`N` is the FFT size and :math:`b` is the time step.

    This function only produces one channel frequency response per OFDM symbol, i.e.,
    only values of :math:`b` corresponding to the start of an OFDM symbol (after
    cyclic prefix removal) are considered.

    Input
    ------
    h_t : [..., num_time_steps, l_max-l_min+1], torch.complex
        Tensor of discrete complex-baseband channel impulse responses

    resource_grid : ResourceGrid
        Resource grid

    l_min : int
        Smallest time-lag for the discrete complex baseband
        channel impulse response (:math:`L_{\text{min}}`)

    Output
    ------
    h_f : [..., num_ofdm_symbols, fft_size], torch.complex
        Tensor of discrete complex-baseband channel frequency responses
    """

    # Total length of an OFDM symbol including cyclic prefix
    ofdm_length = rg.fft_size + rg.cyclic_prefix_length

    # Downsample the impulse response to one sample per OFDM symbol
    h_t = h_t[..., rg.cyclic_prefix_length:rg.num_time_samples:ofdm_length, :]

    # Pad channel impulse response with zeros to the FFT size
    pad_dims = rg.fft_size - h_t.shape[-1]
    pad_shape = list(h_t.shape[:-1]) + [pad_dims]
    h_t = torch.cat([h_t, torch.zeros(pad_shape, dtype=h_t.dtype, device=h_t.device)], dim=-1)

    # Circular shift of negative time lags so that the channel impulse response
    # starts with h_{b,0}
    h_t = torch.roll(h_t, shifts=l_min, dims=-1)

    # Compute FFT
    h_f = torch.fft.fft(h_t, dim=-1)

    # Move the zero subcarrier to the center of the spectrum
    h_f = torch.fft.fftshift(h_f, dim=-1)

    return h_f

def deg_2_rad(x):
    r"""
    Convert degree to radian

    Input
    ------
        x : Tensor
            Angles in degree

    Output
    -------
        y : Tensor
            Angles ``x`` converted to radian
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    y=x*torch.tensor(PI/180.0)
    return y.to(dtype= x.dtype)

def rad_2_deg(x):
    r"""
    Convert radian to degree

    Input
    ------
        x : Tensor
            Angles in radian

    Output
    -------
        y : Tensor
            Angles ``x`` converted to degree
    """
    return x*torch.tensor(180.0/PI, dtype=x.dtype)

def wrap_angle_0_360(angle):
    r"""
    Wrap ``angle`` to (0,360)

    Input
    ------
        angle : Tensor
            Input to wrap

    Output
    -------
        y : Tensor
            ``angle`` wrapped to (0,360)
    """
    
    # return torch.fmod(angle, 360.)
    result = torch.fmod(angle, 360)
    result = torch.where(result < 0, result + 360, result)
    return result


def sample_bernoulli(shape, p, dtype = np.int32):
    r"""
    Sample a tensor with shape ``shape`` from a Bernoulli distribution with
    probability ``p``

    Input
    --------
    shape : Tensor shape
        Shape of the tensor to sample

    p : Broadcastable with ``shape``, tf.float
        Probability

    dtype : tf.DType
        Datatype to use for internal processing and output.

    Output
    --------
    : Tensor of shape ``shape``, bool
        Binary samples
    """
    z = np.random.uniform(size=shape, low=0.0, high=1.0)
    z = np.less(z, p)
    return z

def drop_uts_in_sector(batch_size, num_ut, min_bs_ut_dist, isd, dtype=torch.complex64):
    r"""
    Uniformly sample UT locations from a sector.

    The sector from which UTs are sampled is shown in the following figure.
    The BS is assumed to be located at the origin (0,0) of the coordinate
    system.

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    min_bs_ut_dist : float
        Minimum BS-UT distance [m]

    isd : float
        Inter-site distance, i.e., the distance between two adjacent BSs [m]

    dtype : torch.dtype
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `torch.complex64` (`torch.float32`).

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], torch.float
        UTs locations in the X-Y plane
    """

    # if torch.is_complex(dtype):
    #     real_dtype = dtype.to_real()
    # elif torch.is_floating_point(dtype):
    #     real_dtype = dtype
    # else:
    #     raise AssertionError("dtype must be a complex or floating datatype")
    
    # real_dtype = get_real_dtype(dtype)

    # r_min = torch.tensor(min_bs_ut_dist, dtype=real_dtype)

    # r = torch.tensor(isd * 0.5, dtype=real_dtype)

    # # Angles from (-pi/6, pi/6), covering half of the sector and denoted by
    # # alpha_half, are randomly sampled for all UTs.
    # # Then, the maximum distance UTs can be from the BS, denoted by r_max,
    # # is computed for each angle.
    # # Distance between UT - BS are then uniformly sampled from the range
    # # (r_min, r_max)
    # # Each UT is then randomly and uniformly pushed into a half of the sector
    # # by adding either PI/6 or PI/2 to the angle alpha_half

    # # Sample angles for half of the sector (which half will be decided randomly)
    # alpha_half = torch.rand([batch_size, num_ut], dtype=real_dtype) * (PI / 3) - (PI / 6)

    # # Maximum distance from BS at this angle to be in the sector
    # r_max = r / torch.cos(alpha_half)

    # # Randomly sample distance for the UTs
    # distance = torch.rand([batch_size, num_ut], dtype=real_dtype) * (r_max - r_min) + r_min

    # # Randomly assign the UTs to one of the two halves of the sector
    # side = sample_bernoulli([batch_size, num_ut],
    #                         torch.tensor(0.5, dtype=real_dtype),
    #                         real_dtype)
    # side = torch.tensor(side, dtype=real_dtype)
    # side = 2.*side+1.    
    # alpha = alpha_half + side * (PI / 6)

    # # Set UT location in X-Y coordinate system
    # ut_loc = torch.stack([distance * torch.cos(alpha),
    #                       distance * torch.sin(alpha)], dim=-1)
    real_dtype = np.float32
    np.random.seed(1)
    # Step 1: Initialize values
    r_min = np.float32(min_bs_ut_dist)
    r = np.float32(isd * 0.5)

    # Step 2: Sample angles for half of the sector
    alpha_half = np.random.uniform(low=-PI/6., high=PI/6., size=[batch_size, num_ut]).astype(real_dtype)
    # Step 3: Maximum distance from BS at this angle to be in the sector
    r_max = r / np.cos(alpha_half)

    # Step 4: Randomly sample distance for the UTs
    distance = np.random.uniform(low=r_min, high=r_max, size=[batch_size, num_ut]).astype(real_dtype)

    # Step 5: Randomly assign the UTs to one of the two halves of the sector
    side = np.random.binomial(1, 0.5, size=[batch_size, num_ut]).astype(real_dtype)
    side = 2. * side + 1.
    alpha = alpha_half + side * PI / 6.

    # Step 6: Set UT location in X-Y coordinate system
    ut_loc_x = distance * np.cos(alpha)
    ut_loc_y = distance * np.sin(alpha)
    ut_loc = torch.tensor(np.stack([ut_loc_x, ut_loc_y], axis=-1))
    return ut_loc

def set_3gpp_scenario_parameters(scenario,
                                 min_bs_ut_dist=None,
                                 isd=None,
                                 bs_height=None,
                                 min_ut_height=None,
                                 max_ut_height=None,
                                 indoor_probability=None,
                                 min_ut_velocity=None,
                                 max_ut_velocity=None,
                                 dtype=torch.complex64):
    r"""
    Set valid parameters for a specified 3GPP system level ``scenario``
    (RMa, UMi, or UMa).

    If a parameter is given, then it is returned. If it is set to `None`,
    then a parameter valid according to the chosen scenario is returned
    (see [TR38901]_).

    Input
    --------
    scenario : str
        System level model scenario. Must be one of "rma", "umi", or "uma".

    min_bs_ut_dist : None or torch.Tensor
        Minimum BS-UT distance [m]

    isd : None or torch.Tensor
        Inter-site distance [m]

    bs_height : None or torch.Tensor
        BS elevation [m]

    min_ut_height : None or torch.Tensor
        Minimum UT elevation [m]

    max_ut_height : None or torch.Tensor
        Maximum UT elevation [m]

    indoor_probability : None or torch.Tensor
        Probability of a UT to be indoor

    min_ut_velocity : None or torch.Tensor
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.Tensor
        Maximum UT velocity [m/s]

    dtype : torch.dtype
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `torch.complex64` (`torch.float32`).

    Output
    --------
    min_bs_ut_dist : torch.Tensor
        Minimum BS-UT distance [m]

    isd : torch.Tensor
        Inter-site distance [m]

    bs_height : torch.Tensor
        BS elevation [m]

    min_ut_height : torch.Tensor
        Minimum UT elevation [m]

    max_ut_height : torch.Tensor
        Maximum UT elevation [m]

    indoor_probability : torch.Tensor
        Probability of a UT to be indoor

    min_ut_velocity : torch.Tensor
        Minimum UT velocity [m/s]

    max_ut_velocity : torch.Tensor
        Maximum UT velocity [m/s]
    """

    assert scenario in ('umi', 'uma', 'rma'), \
        "`scenario` must be one of 'umi', 'uma', 'rma'"

    # if torch.is_complex(dtype):
    #     real_dtype = dtype.to_real() 
    # elif torch.is_floating_point(dtype):
    #     real_dtype = dtype
    # else:
    #     raise AssertionError("dtype must be a complex or floating datatype")
    real_dtype = get_real_dtype(dtype)

    # Default values for scenario parameters.
    # From TR38.901, sections 7.2 and 7.4.
    # All distances and heights are in meters
    # All velocities are in meters per second.
    default_scenario_par = {
        'umi': {
            'min_bs_ut_dist': torch.tensor(10.0, dtype=real_dtype),
            'isd': torch.tensor(200.0, dtype=real_dtype),
            'bs_height': torch.tensor(10.0, dtype=real_dtype),
            'min_ut_height': torch.tensor(1.5, dtype=real_dtype),
            'max_ut_height': torch.tensor(1.5, dtype=real_dtype),
            'indoor_probability': torch.tensor(0.8, dtype=real_dtype),
            'min_ut_velocity': torch.tensor(0.0, dtype=real_dtype),
            'max_ut_velocity': torch.tensor(0.0, dtype=real_dtype)
        },
        'uma': {
            'min_bs_ut_dist': torch.tensor(35.0, dtype=real_dtype),
            'isd': torch.tensor(500.0, dtype=real_dtype),
            'bs_height': torch.tensor(25.0, dtype=real_dtype),
            'min_ut_height': torch.tensor(1.5, dtype=real_dtype),
            'max_ut_height': torch.tensor(1.5, dtype=real_dtype),
            'indoor_probability': torch.tensor(0.8, dtype=real_dtype),
            'min_ut_velocity': torch.tensor(0.0, dtype=real_dtype),
            'max_ut_velocity': torch.tensor(0.0, dtype=real_dtype)
        },
        'rma': {
            'min_bs_ut_dist': torch.tensor(35.0, dtype=real_dtype),
            'isd': torch.tensor(5000.0, dtype=real_dtype),
            'bs_height': torch.tensor(35.0, dtype=real_dtype),
            'min_ut_height': torch.tensor(1.5, dtype=real_dtype),
            'max_ut_height': torch.tensor(1.5, dtype=real_dtype),
            'indoor_probability': torch.tensor(0.5, dtype=real_dtype),
            'min_ut_velocity': torch.tensor(0.0, dtype=real_dtype),
            'max_ut_velocity': torch.tensor(0.0, dtype=real_dtype)
        }
    }

    # Setting the scenario parameters
    if min_bs_ut_dist is None:
        min_bs_ut_dist = default_scenario_par[scenario]['min_bs_ut_dist']
    if isd is None:
        isd = default_scenario_par[scenario]['isd']
    if bs_height is None:
        bs_height = default_scenario_par[scenario]['bs_height']
    if min_ut_height is None:
        min_ut_height = default_scenario_par[scenario]['min_ut_height']
    if max_ut_height is None:
        max_ut_height = default_scenario_par[scenario]['max_ut_height']
    if indoor_probability is None:
        indoor_probability = default_scenario_par[scenario]['indoor_probability']
    if min_ut_velocity is None:
        min_ut_velocity = default_scenario_par[scenario]['min_ut_velocity']
    if max_ut_velocity is None:
        max_ut_velocity = default_scenario_par[scenario]['max_ut_velocity']

    return min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity

def relocate_uts(ut_loc, sector_id, cell_loc):
    r"""
    Relocate the UTs by rotating them into the sector with index ``sector_id``
    and transposing them to the cell centered on ``cell_loc``.

    ``sector_id`` gives the index of the sector to which the UTs are
    rotated to. The picture below shows how the three sectors of a cell are
    indexed.

    If ``sector_id`` is a scalar, then all UTs are relocated to the same
    sector indexed by ``sector_id``.
    If ``sector_id`` is a tensor, it should be broadcastable with
    [``batch_size``, ``num_ut``], and give the sector in which each UT or
    batch example is relocated to.

    When calling the function, ``ut_loc`` gives the locations of the UTs to
    relocate, which are all assumed to be in sector with index 0, and in the
    cell centered on the origin (0,0).

    Input
    --------
    ut_loc : [batch_size, num_ut, 2], torch.float
        UTs locations in the X-Y plane

    sector_id : Tensor broadcastable with [batch_size, num_ut], int
        Indexes of the sector to which to relocate the UTs

    cell_loc : Tensor broadcastable with [batch_size, num_ut], torch.float
        Center of the cell to which to transpose the UTs

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], torch.float
        Relocated UTs locations in the X-Y plane
    """

    # Ensure sector_id is of the same dtype as ut_loc
    sector_id = sector_id.to(ut_loc.dtype)
    while sector_id.dim() < 2:
        sector_id = sector_id.unsqueeze(0)


    # Ensure cell_loc is of the same dtype and rank as ut_loc
    cell_loc = cell_loc.to(ut_loc.dtype)
    while cell_loc.dim() < ut_loc.dim():
        cell_loc = cell_loc.unsqueeze(0)  # Expand rank to match ut_loc

    # Calculate the rotation angle
    rotation_angle = sector_id * 2. * PI / 3.0

    # Create the rotation matrix
    cos_angle = torch.cos(rotation_angle)
    sin_angle = torch.sin(rotation_angle)
    rotation_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
    rotation_matrix = torch.reshape(rotation_matrix,
                                 torch.Size(list(rotation_angle.shape)+[2,2]))
    rotation_matrix = torch.tensor(rotation_matrix, dtype=ut_loc.dtype)
    
    # Apply the rotation matrix
    ut_loc = ut_loc.unsqueeze(-1)  # Add a dimension for matrix multiplication
    ut_loc_rotated = torch.matmul(rotation_matrix, ut_loc).squeeze(-1)

    # Translate to the BS location
    ut_loc_rotated_translated = ut_loc_rotated + cell_loc

    return ut_loc_rotated_translated

def generate_uts_topology(batch_size,
                            num_ut,
                            drop_area,
                            cell_loc_xy,
                            min_bs_ut_dist,
                            isd,
                            min_ut_height,
                            max_ut_height,
                            indoor_probability,
                            min_ut_velocity,
                            max_ut_velocity,
                            dtype=torch.complex64):
    r"""
    Sample UTs location from a sector or a cell

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    drop_area : str
        'sector' or 'cell'. If set to 'sector', UTs are sampled from the
        sector with index 0 in the figure below

        .. figure:: ../figures/panel_array_sector_id.png
            :align: center
            :scale: 30%

    Indexing of sectors

    cell_loc_xy : Tensor broadcastable with[batch_size, num_ut, 3], torch.float
        Center of the cell(s)

    min_bs_ut_dist : None or torch.float
        Minimum BS-UT distance [m]

    isd : None or torch.float
        Inter-site distance [m]

    min_ut_height : None or torch.float
        Minimum UT elevation [m]

    max_ut_height : None or torch.float
        Maximum UT elevation [m]

    indoor_probability : None or torch.float
        Probability of a UT to be indoor

    min_ut_velocity : None or torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.float
        Maximum UT velocity [m/s]

    dtype : torch.DType
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `torch.complex64` (`torch.float32`).

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], torch.float
        UTs locations

    ut_orientations : [batch_size, num_ut, 3], torch.float
        UTs orientations [radian]

    ut_velocities : [batch_size, num_ut, 3], torch.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], torch.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """

    assert drop_area in ('sector', 'cell'),\
        "Drop area must be either 'sector' or 'cell'"

    # 确定实际使用的数据类型
    # if torch.is_complex(dtype):
    #     real_dtype = dtype.to_real() 
    # elif torch.is_floating_point(dtype):
    #     real_dtype = dtype
    # else:
    #     raise AssertionError("dtype must be a complex or floating datatype")
    real_dtype = get_real_dtype(dtype)
    # 随机生成UT的位置
    ut_loc_xy = drop_uts_in_sector(batch_size,
                                   num_ut,
                                   min_bs_ut_dist,
                                   isd,
                                   dtype)
    print('ut_loc_xy:---------------',ut_loc_xy)
    
    # if drop_area == 'sector':
    #     sectors = torch.zeros((batch_size, num_ut), dtype=torch.int32)
    # elif drop_area == 'cell':
    #     sectors = torch.randint(0, 3, (batch_size, num_ut), dtype=torch.int32)

    # ut_loc_xy = relocate_uts(ut_loc_xy, sectors, cell_loc_xy)

    # ut_loc_z = torch.rand((batch_size, num_ut, 1), dtype=real_dtype) * \
    #            (max_ut_height - min_ut_height) + min_ut_height
    # ut_loc = torch.cat([ut_loc_xy, ut_loc_z], dim=-1)

    # # 随机生成 UT 的室内/室外状态
    # in_state = sample_bernoulli(   [batch_size, num_ut], indoor_probability,
    #                                 real_dtype)
    # # 随机生成 UT 的速度
    # ut_vel_angle = torch.rand((batch_size, num_ut), dtype=real_dtype) * 2 * PI - PI
    # ut_vel_norm = torch.rand((batch_size, num_ut), dtype=real_dtype) * \
    #               (max_ut_velocity - min_ut_velocity) + min_ut_velocity
    # ut_velocities = torch.stack([ut_vel_norm * torch.cos(ut_vel_angle),
    #                              ut_vel_norm * torch.sin(ut_vel_angle),
    #                              torch.zeros((batch_size, num_ut), dtype=real_dtype)],
    #                              dim=-1)

    # # 随机生成 UT 的方向
    # ut_bearing = torch.rand((batch_size, num_ut), dtype=real_dtype) * PI - 0.5 * PI
    # ut_downtilt = torch.rand((batch_size, num_ut), dtype=real_dtype) * PI - 0.5 * PI
    # ut_slant = torch.rand((batch_size, num_ut), dtype=real_dtype) * PI - 0.5 * PI
    # ut_orientations = torch.stack([ut_bearing, ut_downtilt, ut_slant], dim=-1)
    #######################################################
    real_dtype = np.float32
    if drop_area == 'sector':
        sectors = torch.tensor(0, dtype=torch.int32)
    elif drop_area == 'cell':
        sectors = np.random.uniform(size=(batch_size, num_ut),
                                    low=0,
                                    high=3).astype(np.int32)
    ut_loc_xy = relocate_uts(ut_loc_xy,
                             sectors,
                             cell_loc_xy)
    # Randomly generate UT heights
    ut_loc_z = np.random.uniform(low=min_ut_height,
                                high=max_ut_height,
                                size=(batch_size, num_ut, 1)).astype(real_dtype)

    # Concatenate UT XY locations with Z
    ut_loc = torch.tensor(np.concatenate([ut_loc_xy, ut_loc_z], axis=-1))

    # Randomly generate the UT indoor/outdoor state using Bernoulli sampling
    in_state = sample_bernoulli([batch_size, num_ut], indoor_probability,
                                get_real_dtype(dtype))
    # Randomly generate the UT velocities
    ut_vel_angle = np.random.uniform(low=-np.pi,
                                    high=np.pi,
                                    size=(batch_size, num_ut)).astype(real_dtype)
    ut_vel_norm = np.random.uniform(low=min_ut_velocity,
                                    high=max_ut_velocity,
                                    size=(batch_size, num_ut)).astype(real_dtype)

    # Stack the UT velocities using cos and sin for the angle
    ut_velocities = torch.tensor(np.stack([ut_vel_norm * np.cos(ut_vel_angle),
                            ut_vel_norm * np.sin(ut_vel_angle),
                            np.zeros((batch_size, num_ut), dtype=real_dtype)],
                            axis=-1))

    # Randomly generate the UT orientations
    ut_bearing = np.random.uniform(low=-0.5 * np.pi,
                                high=0.5 * np.pi,
                                size=(batch_size, num_ut)).astype(real_dtype)
    ut_downtilt = np.random.uniform(low=-0.5 * np.pi,
                                    high=0.5 * np.pi,
                                    size=(batch_size, num_ut)).astype(real_dtype)
    ut_slant = np.random.uniform(low=-0.5 * np.pi,
                                high=0.5 * np.pi,
                                size=(batch_size, num_ut)).astype(real_dtype)

    # Stack the orientations along the last axis
    ut_orientations = torch.tensor(np.stack([ut_bearing, ut_downtilt, ut_slant], axis=-1))
    return ut_loc, ut_orientations, ut_velocities, in_state

def gen_single_sector_topology(batch_size,
                               num_ut,
                               scenario,
                               min_bs_ut_dist=None,
                               isd=None,
                               bs_height=None,
                               min_ut_height=None,
                               max_ut_height=None,
                               indoor_probability=None,
                               min_ut_velocity=None,
                               max_ut_velocity=None,
                               dtype=torch.complex64):
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin and ``num_ut`` UTs randomly and uniformly dropped in a cell sector.

    The following picture shows the sector from which UTs are sampled.

    .. figure:: ../figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    UTs orientations are randomly and uniformly set, whereas the BS orientation
    is set such that the it is oriented towards the center of the sector.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.channel.tr38901.UMi`, :class:`~sionna.channel.tr38901.UMa`,
    and :class:`~sionna.channel.tr38901.RMa`.

    Example
    --------
    >>> # Create antenna arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                      num_cols_per_panel = 4,
    ...                      polarization = 'dual',
    ...                      polarization_type = 'VH',
    ...                      antenna_pattern = '38.901',
    ...                      carrier_frequency = 3.5e9)
    >>>
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Create channel model
    >>> channel_model = UMi(carrier_frequency = 3.5e9,
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology(batch_size = 100,
    ...                                       num_ut = 4,
    ...                                       scenario = 'umi')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology.png

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    scenario : str
        System leven model scenario. Must be one of "rma", "umi", or "uma".

    min_bs_ut_dist : None or torch.float
        Minimum BS-UT distance [m]

    isd : None or torch.float
        Inter-site distance [m]

    bs_height : None or torch.float
        BS elevation [m]

    min_ut_height : None or torch.float
        Minimum UT elevation [m]

    max_ut_height : None or torch.float
        Maximum UT elevation [m]

    indoor_probability : None or torch.float
        Probability of a UT to be indoor

    min_ut_velocity : None or torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.float
        Maximim UT velocity [m/s]

    dtype : torch.DType
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `torch.complex64` (`torch.float32`).

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], torch.float
        UTs locations

    bs_loc : [batch_size, 1, 3], torch.float
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], torch.float
        UTs orientations [radian]

    bs_orientations : [batch_size, 1, 3], torch.float
        BS orientations [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], torch.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], torch.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """


    # 设定3GPP场景参数
    params = set_3gpp_scenario_parameters(scenario,
                                          min_bs_ut_dist,
                                          isd,
                                          bs_height,
                                          min_ut_height,
                                          max_ut_height,
                                          indoor_probability,
                                          min_ut_velocity,
                                          max_ut_velocity,
                                          dtype)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,\
    indoor_probability, min_ut_velocity, max_ut_velocity = params
    # print('params:----------------------------------------------------\n',params)
    # OK
    real_dtype = dtype.to_real()
    # 设置 BS 位置为 (0,0,bs_height)
    bs_loc = torch.stack([torch.zeros((batch_size, 1), dtype=real_dtype),
                          torch.zeros((batch_size, 1), dtype=real_dtype),
                          torch.full((batch_size, 1), bs_height, dtype=real_dtype)], dim=-1)

    # 设置 BS 的方向，使其向下倾斜朝向扇区中心
    sector_center = (min_bs_ut_dist + 0.5 * isd) * 0.5
    bs_downtilt = 0.5 * PI - torch.atan(sector_center / bs_height)
    bs_yaw = torch.tensor(PI / 3.0, dtype=real_dtype)
    bs_orientation = torch.stack([torch.full((batch_size, 1), bs_yaw, dtype=real_dtype),
                                  torch.full((batch_size, 1), bs_downtilt, dtype=real_dtype),
                                  torch.zeros((batch_size, 1), dtype=real_dtype)], dim=-1)

    # 生成 UT 拓扑
    ut_topology = generate_uts_topology(batch_size,
                                        num_ut,
                                        'sector',
                                        torch.zeros(2, dtype=real_dtype),
                                        min_bs_ut_dist,
                                        isd,
                                        min_ut_height,
                                        max_ut_height,
                                        indoor_probability,
                                        min_ut_velocity,
                                        max_ut_velocity,
                                        dtype)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology
    print('ut_topology---------------------\n\n',ut_topology)
    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state

def gen_single_sector_topology_interferers(batch_size,
                                            num_ut,
                                            num_interferer,
                                            scenario,
                                            min_bs_ut_dist=None,
                                            isd=None,
                                            bs_height=None,
                                            min_ut_height=None,
                                            max_ut_height=None,
                                            indoor_probability=None,
                                            min_ut_velocity=None,
                                            max_ut_velocity=None,
                                            dtype=torch.complex64):
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin, ``num_ut`` UTs randomly and uniformly dropped in a cell sector, and
    ``num_interferer`` interfering UTs randomly dropped in the adjacent cells.

    The following picture shows how UTs are sampled

    .. figure:: ../figures/drop_uts_in_sector_interferers.png
        :align: center
        :scale: 30%

    UTs orientations are randomly and uniformly set, whereas the BS orientation
    is set such that it is oriented towards the center of the sector it
    serves.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.channel.tr38901.UMi`, :class:`~sionna.channel.tr38901.UMa`,
    and :class:`~sionna.channel.tr38901.RMa`.

    In the returned ``ut_loc``, ``ut_orientations``, ``ut_velocities``, and
    ``in_state`` tensors, the first ``num_ut`` items along the axis with index
    1 correspond to the served UTs, whereas the remaining ``num_interferer``
    items correspond to the interfering UTs.

    Example
    --------
    >>> # Create antenna arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                      num_cols_per_panel = 4,
    ...                      polarization = 'dual',
    ...                      polarization_type = 'VH',
    ...                      antenna_pattern = '38.901',
    ...                      carrier_frequency = 3.5e9)
    >>>
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Create channel model
    >>> channel_model = UMi(carrier_frequency = 3.5e9,
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology_interferers(batch_size = 100,
    ...                                                   num_ut = 4,
    ...                                                   num_interferer = 4,
    ...                                                   scenario = 'umi')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology_inter.png

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    num_interferer : int
        Number of interfeering UTs per batch example

    scenario : str
        System leven model scenario. Must be one of "rma", "umi", or "uma".

    min_bs_ut_dist : None or torch.float
        Minimum BS-UT distance [m]

    isd : None or torch.float
        Inter-site distance [m]

    bs_height : None or torch.float
        BS elevation [m]

    min_ut_height : None or torch.float
        Minimum UT elevation [m]

    max_ut_height : None or torch.float
        Maximum UT elevation [m]

    indoor_probability : None or torch.float
        Probability of a UT to be indoor

    min_ut_velocity : None or torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.float
        Maximim UT velocity [m/s]

    dtype : torch.DType
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `torch.complex64` (`torch.float32`).

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], torch.float
        UTs locations. The first ``num_ut`` items along the axis with index
        1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.

    bs_loc : [batch_size, 1, 3], torch.float
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], torch.float
        UTs orientations [radian]. The first ``num_ut`` items along the
        axis with index 1 correspond to the served UTs, whereas the
        remaining ``num_interferer`` items correspond to the interfeering
        UTs.

    bs_orientations : [batch_size, 1, 3], torch.float
        BS orientation [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], torch.float
        UTs velocities [m/s]. The first ``num_ut`` items along the axis
        with index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.

    in_state : [batch_size, num_ut], torch.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. The first ``num_ut`` items along the axis with
        index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.
    """
    
    # 设定3GPP场景参数
    params = set_3gpp_scenario_parameters(scenario,
                                          min_bs_ut_dist,
                                          isd,
                                          bs_height,
                                          min_ut_height,
                                          max_ut_height,
                                          indoor_probability,
                                          min_ut_velocity,
                                          max_ut_velocity,
                                          dtype)

    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,\
    indoor_probability, min_ut_velocity, max_ut_velocity = params

    real_dtype = dtype.to_real()

    # 设置 BS 位置为 (0,0,bs_height)
    bs_loc = torch.stack([torch.zeros((batch_size, 1), dtype=real_dtype),
                          torch.zeros((batch_size, 1), dtype=real_dtype),
                          torch.full((batch_size, 1), bs_height, dtype=real_dtype)], dim=-1)

    # 设置 BS 的方向，使其向下倾斜朝向扇区中心
    sector_center = (min_bs_ut_dist + 0.5 * isd) * 0.5
    bs_downtilt = 0.5 * PI - torch.atan(sector_center / bs_height)
    bs_yaw = torch.tensor(PI / 3.0, dtype=real_dtype)
    bs_orientation = torch.stack([torch.full((batch_size, 1), bs_yaw, dtype=real_dtype),
                                  torch.full((batch_size, 1), bs_downtilt, dtype=real_dtype),
                                  torch.zeros((batch_size, 1), dtype=real_dtype)], dim=-1)

    # 生成服务 UT 的拓扑
    ut_topology = generate_uts_topology(batch_size,
                                        num_ut,
                                        'sector',
                                        torch.zeros(2, dtype=real_dtype),
                                        min_bs_ut_dist,
                                        isd,
                                        min_ut_height,
                                        max_ut_height,
                                        indoor_probability,
                                        min_ut_velocity,
                                        max_ut_velocity,
                                        dtype)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    # 生成邻近小区中的干扰 UT
    inter_cell_center = torch.tensor([[0.0, isd],
                                     [isd * torch.cos(PI / 6.0),
                                      isd * torch.sin(PI / 6.0)]], dtype=real_dtype)
    cell_index = torch.randint(0, 2, (batch_size, num_interferer), dtype=torch.int64)
    inter_cells = gather_pytorch(inter_cell_center,cell_index)

    inter_topology = generate_uts_topology(batch_size,
                                           num_interferer,
                                           'cell',
                                           inter_cells,
                                           min_bs_ut_dist,
                                           isd,
                                           min_ut_height,
                                           max_ut_height,
                                           indoor_probability,
                                           min_ut_velocity,
                                           max_ut_velocity,
                                           dtype)
    inter_loc, inter_orientations, inter_velocities, inter_in_state = inter_topology

    # 合并服务 UT 和干扰 UT 的数据
    ut_loc = torch.cat([ut_loc, inter_loc], dim=1)
    ut_orientations = torch.cat([ut_orientations, inter_orientations], dim=1)
    ut_velocities = torch.cat([ut_velocities, inter_velocities], dim=1)
    in_state = torch.cat([in_state, inter_in_state], dim=1)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state

def exp_corr_mat(a, n, dtype=torch.complex64):
    r"""Generate exponential correlation matrices.

    This function computes for every element :math:`a` of a complex-valued
    tensor :math:`\mathbf{a}` the corresponding :math:`n\times n` exponential
    correlation matrix :math:`\mathbf{R}(a,n)`, defined as (Eq. 1, [MAL2018]_):

    .. math::
        \mathbf{R}(a,n)_{i,j} = \begin{cases}
                    1 & \text{if } i=j\\
                    a^{i-j}  & \text{if } i>j\\
                    (a^\star)^{j-i}  & \text{if } j<i, j=1,\dots,n\\
                  \end{cases}

    where :math:`|a|<1` and :math:`\mathbf{R}\in\mathbb{C}^{n\times n}`.

    Input
    -----
    a : [n_0, ..., n_k], torch.complex
        A tensor of arbitrary rank whose elements
        have an absolute value smaller than one.

    n : int
        Number of dimensions of the output correlation matrices.

    dtype : torch.complex64, torch.complex128
        The dtype of the output.

    Output
    ------
    R : [n_0, ..., n_k, n, n], torch.complex
        A tensor of the same dtype as the input tensor :math:`\mathbf{a}`.
    """
    if dtype == torch.complex32:
        real_dtype = torch.float16
    elif dtype == torch.complex64:
        real_dtype = torch.float32
    elif dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        raise TypeError("Not found comfortable type")

    # Cast to desired output dtype and expand last dimension for broadcasting
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    a = a.to(dtype=dtype)
    a = a.unsqueeze(-1)

    # Check that a is valid
    if not torch.all(torch.abs(a) < 1):
        raise ValueError("The absolute value of the elements of `a` must be smaller than one")

    # Vector of exponents, adapt dtype and dimensions for broadcasting
    exp = torch.arange(0, n, dtype=real_dtype)
    exp = exp.to(dtype=dtype) 
    exp = expand_to_rank(exp, a.dim(),0)

    # First column of R
    col = torch.pow(a, exp)

    # For a=0, one needs to remove the resulting nans due to 0**0=nan
    col = torch.where(torch.isnan(col.real), torch.ones_like(col), col)

    # First row of R (equal to complex-conjugate of the first column)
    row = torch.conj(col)

    # Create Toeplitz matrix manually
    R = torch.zeros(*a.shape[:-1], n, n,dtype=dtype)
    for i in range(n):
        for j in range(n):
            if i == j:
                R[..., i, j] = 1
            elif i > j:
                R[..., i, j] = col[..., i-j]
            else:
                R[..., i, j] = row[..., j-i]

    return R
    """     # 测试例子
    a = torch.tensor([0.5 + 0.5j,0.1+0.6j], dtype=torch.complex128)
    n = 4
    result = exp_corr_mat(a, n)

    print("Result:")
    print(result)
    print(result.shape) """

def one_ring_corr_mat(phi_deg, num_ant, d_h=0.5, sigma_phi_deg=15, dtype=torch.complex64):
    r"""Generate covariance matrices from the one-ring model.

    This function generates approximate covariance matrices for the
    so-called `one-ring` model (Eq. 2.24) [BHS2017]_. A uniform
    linear array (ULA) with uniform antenna spacing is assumed. The elements
    of the covariance matrices are computed as:

    .. math::
        \mathbf{R}_{\ell,m} =
              \exp\left( j2\pi d_\text{H} (\ell -m)\sin(\varphi) \right)
              \exp\left( -\frac{\sigma_\varphi^2}{2}
              \left( 2\pi d_\text{H}(\ell -m)\cos(\varphi) \right)^2 \right)

    for :math:`\ell,m = 1,\dots, M`, where :math:`M` is the number of antennas,
    :math:`\varphi` is the angle of arrival, :math:`d_\text{H}` is the antenna
    spacing in multiples of the wavelength,
    and :math:`\sigma^2_\varphi` is the angular standard deviation.

    Input
    -----
    phi_deg : [n_0, ..., n_k], torch.float
        A tensor of arbitrary rank containing azimuth angles (deg) of arrival.

    num_ant : int
        Number of antennas

    d_h : float
        Antenna spacing in multiples of the wavelength. Defaults to 0.5.

    sigma_phi_deg : float
        Angular standard deviation (deg). Defaults to 15 (deg). Values greater
        than 15 should not be used as the approximation becomes invalid.

    dtype : torch.complex64, torch.complex128
        The dtype of the output.

    Output
    ------
    R : [n_0, ..., n_k, num_ant, num_ant], `dtype`
        Tensor containing the covariance matrices of the desired dtype.
    """

    if sigma_phi_deg > 15:
        warnings.warn("sigma_phi_deg should be smaller than 15.")
    # get real_dtype    
    if dtype == torch.complex32:
        real_dtype = torch.float16
    elif dtype == torch.complex64:
        real_dtype = torch.float32
    elif dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        raise TypeError("Not found comfortable type")
    
    if not torch.is_tensor(phi_deg):
        phi_deg = torch.tensor(phi_deg)
    # Convert all inputs to radians
    phi_deg = phi_deg.to(dtype=real_dtype)
    sigma_phi_deg = torch.tensor(sigma_phi_deg, dtype=real_dtype)
    phi = deg_2_rad(phi_deg)
    sigma_phi = deg_2_rad(sigma_phi_deg)

    # Add dimensions for broadcasting
    phi = phi.unsqueeze(-1)
    sigma_phi = sigma_phi.unsqueeze(-1)

    # Compute first column
    c = torch.tensor(2 * PI * d_h, dtype=real_dtype)
    d = c * torch.arange(0, num_ant, dtype=real_dtype)
    d = expand_to_rank(d, phi.dim(), 0)

    a = torch.complex(torch.zeros_like(d,dtype=real_dtype), d * torch.sin(phi))
    exp_a = torch.exp(a)  # First exponential term

    b = -0.5 * (sigma_phi * d * torch.cos(phi)) ** 2
    exp_b = torch.exp(b).to(dtype)  # Second exponential term

    col = exp_a * exp_b  # First column

    # First row is just the complex conjugate of first column
    row = torch.conj(col)

    # Create Toeplitz matrix using first column and first row
    toeplitz_matrix = torch.zeros((*col.shape[:-1], num_ant, num_ant), dtype=dtype)
    for i in range(num_ant):
        for j in range(num_ant):
            if i >= j:
                toeplitz_matrix[..., i, j] = col[..., i - j]
            else:
                toeplitz_matrix[..., i, j] = row[..., j - i]

    return toeplitz_matrix


