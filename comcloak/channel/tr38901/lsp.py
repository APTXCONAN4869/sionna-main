

"""
Class for sampling large scale parameters (LSPs) and pathloss following the
3GPP TR38.901 specifications and according to a channel simulation scenario.
"""



import torch


from comcloak.utils.tensors import matrix_sqrt

class LSP:
    r"""
    Class for conveniently storing LSPs

    Parameters
    -----------

    ds : [batch size, num tx, num rx], torch.float
        RMS delay spread [s]

    asd : [batch size, num tx, num rx], torch.float
        azimuth angle spread of departure [deg]

    asa : [batch size, num tx, num rx], torch.float
        azimuth angle spread of arrival [deg]

    sf : [batch size, num tx, num rx], torch.float
        shadow fading

    k_factor : [batch size, num tx, num rx], torch.float
        Rician K-factor. Only used for LoS.

    zsa : [batch size, num tx, num rx], torch.float
        Zenith angle spread of arrival [deg]

    zsd: [batch size, num tx, num rx], torch.float
        Zenith angle spread of departure [deg]
    """

    def __init__(self, ds, asd, asa, sf, k_factor, zsa, zsd):
        self.ds = ds
        self.asd = asd
        self.asa = asa
        self.sf = sf
        self.k_factor = k_factor
        self.zsa = zsa
        self.zsd = zsd

class LSPGenerator:
    """
    Sample large scale parameters (LSP) and pathloss given a channel scenario,
    e.g., UMa, UMi, RMa.

    This class implements steps 1 to 4 of the TR 38.901 specifications
    (section 7.5), as well as path-loss generation (Section 7.4.1) with O2I
    low- and high- loss models (Section 7.4.3).

    Note that a global scenario is set for the entire batches when instantiating
    this class (UMa, UMi, or RMa). However, each UT-BS link can have its
    specific state (LoS, NLoS, or indoor).

    The batch size is set by the ``scenario`` given as argument when
    constructing the class.

    Parameters
    ----------
    scenario : :class:`~sionna.channel.tr38901.SystemLevelScenario``
        Scenario used to generate LSPs

    Input
    -----
    None

    Output
    ------
    An `LSP` instance storing realization of LSPs.
    """

    def __init__(self, scenario):
        # if not isinstance(scenario,torch.Tensor):
        #     scenario = torch.tensor(scenario)
        self._scenario = scenario

        dtype = scenario.dtype
        # For AoA, AoD, ZoA, and ZoD, offset to add to cluster angles to get ray
        # angles. This is hardcoded from table 7.5-3 for 3GPP 38.901
        # specification.
        if dtype == torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("real_dtype must be in [torch.float16,torch.float32,torch.float64]")

        self._real_dtype = real_dtype

    def sample_pathloss(self):
        """
        Generate pathlosses [dB] for each BS-UT link.

        Input
        ------
        None

        Output
        -------
            A tensor with shape [batch size, number of BSs, number of UTs] of
                pathloss [dB] for each BS-UT link
        """

        # Pre-computed basic pathloss
        pl_b = self._scenario.basic_pathloss

        ## O2I penetration
        if self._scenario.o2i_model == 'low':
            pl_o2i = self._o2i_low_loss()
        elif self._scenario.o2i_model == 'high':
            pl_o2i = self._o2i_high_loss()

        ## Total path loss, including O2I penetration
        pl = pl_b + pl_o2i

        return pl

    def __call__(self):

        # LSPs are assumed to follow a log-normal distribution.
        # They are generated in the log-domain (where they follow a normal
        # distribution), where they are correlated as indicated in TR38901
        # specification (Section 7.5, step 4)

        s = torch.normal(mean=0.0, std=1.0,size=(self._scenario.batch_size,
            self._scenario.num_bs, self._scenario.num_ut, 7),
            dtype=self._real_dtype)

        ## Applyting cross-LSP correlation
        s = torch.unsqueeze(s, dim=4)
        s = self._cross_lsp_correlation_matrix_sqrt@s
        s = torch.squeeze(s, dim=4)

        ## Applying spatial correlation
        s = torch.unsqueeze(s.permute(0, 1, 3, 2), dim=3)
        # print(s.dim())
        # print(self._spatial_lsp_correlation_matrix_sqrt.T.shape)
        s = torch.matmul(s, self._spatial_lsp_correlation_matrix_sqrt.permute(0, 1, 2, 4, 3))
        s = s.squeeze(3).permute(0, 1, 3, 2) 

        ## Scaling and transposing LSPs to the right mean and variance
        lsp_log_mean = self._scenario.lsp_log_mean
        lsp_log_std = self._scenario.lsp_log_std
        lsp_log = lsp_log_std*s + lsp_log_mean

        ## Mapping to linear domain
        lsp = torch.pow(torch.tensor(10., dtype=self._real_dtype),
            lsp_log)

        # Limit the RMS azimuth arrival (ASA) and azimuth departure (ASD)
        # spread values to 104 degrees
        # Limit the RMS zenith arrival (ZSA) and zenith departure (ZSD)
        # spread values to 52 degrees
        lsp = LSP(
            ds=lsp[:, :, :, 0],
            asd=torch.minimum(lsp[:, :, :, 1], torch.tensor(104.0)),
            asa=torch.minimum(lsp[:, :, :, 2], torch.tensor(104.0)),
            sf=lsp[:, :, :, 3],
            k_factor=lsp[:, :, :, 4],
            zsa=torch.minimum(lsp[:, :, :, 5], torch.tensor(52.0)),
            zsd=torch.minimum(lsp[:, :, :, 6], torch.tensor(52.0))
        )

        return lsp


    def topology_updated_callback(self):
        """
        Updates internal quantities. Must be called at every update of the
        scenario that changes the state of UTs or their locations.

        Input
        ------
        None

        Output
        ------
        None
        """

        # Pre-computing these quantities avoid unnecessary calculations at every
        # generation of new LSPs

        # Compute cross-LSP correlation matrix
        self._compute_cross_lsp_correlation_matrix()

        # Compute LSP spatial correlation matrix
        self._compute_lsp_spatial_correlation_sqrt()

    ########################################
    # Internal utility methods
    ########################################

    def _compute_cross_lsp_correlation_matrix(self):
        """
        Compute and store as attribute the square-root of the  cross-LSPs
        correlation matrices for each BS-UT link, and then the corresponding
        matrix square root for filtering.

        The resulting tensor is of shape
        [batch size, number of BSs, number of UTs, 7, 7)
        7 being the number of LSPs to correlate.

        Input
        ------
        None

        Output
        -------
        None
        """

        # The following 7 LSPs are correlated:
        # DS, ASA, ASD, SF, K, ZSA, ZSD
        # We create the correlation matrix initialized to the identity matrix

        # 创建一个 7x7 的单位矩阵
        eye_matrix = torch.eye(7, dtype=self._real_dtype)
        # 扩展到批量形状
        batch_shape = [self._scenario.batch_size,self._scenario.num_bs, self._scenario.num_ut]
        cross_lsp_corr_mat = eye_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 添加三个维度
        cross_lsp_corr_mat = cross_lsp_corr_mat.expand(batch_shape + [7, 7])  # 扩展到批量形状


        # Tensors of bool indicating the state of UT-BS links
        # Indoor
        indoor_bool = torch.tile(torch.unsqueeze(self._scenario.indoor, dim=1),
            [1, self._scenario.num_bs, 1])
        # LoS
        los_bool = self._scenario.los
        # NLoS (outdoor)
        nlos_bool = torch.logical_and(torch.logical_not(self._scenario.los),
            torch.logical_not(indoor_bool))
        # Expand to allow broadcasting with the BS dimension
        indoor_bool = torch.unsqueeze(torch.unsqueeze(indoor_bool, dim=3),dim=4)
        los_bool = torch.unsqueeze(torch.unsqueeze(los_bool, dim=3),dim=4)
        nlos_bool = torch.unsqueeze(torch.unsqueeze(nlos_bool, dim=3),dim=4)

        # Internal function that adds to the correlation matrix ``mat``
        # ``cross_lsp_corr_mat`` the parameter ``parameter_name`` at location
        # (m,n)
        def _add_param(mat, parameter_name, m, n):

            # Mask to put the parameters in the right spot of the 7x7
            # correlation matrix

            # 定义目标形状
            shape = [7, 7]
            # 创建一个全零的张量
            mask = torch.zeros(shape, dtype=self._real_dtype)
            # 定义索引和对应值
            indices = torch.tensor([[m, n], [n, m]])
            # indices = [[m, n], [n, m]]
            values = torch.tensor([1.0, 1.0], dtype=self._real_dtype)
            # 使用 scatter_ 填充值
            # mask[indices[:, 0], indices[:, 1]] = values
            init_shape = torch.zeros([7,7])
            init_mask = init_shape.clone()
            init_mask[tuple(indices.t())] = values
            mask = torch.reshape(init_mask, [1,1,1,7,7])
            # Get the parameter value according to the link scenario
            update = self._scenario.get_param(parameter_name)
            update = torch.unsqueeze(torch.unsqueeze(update, dim=3), dim=4)
            # Add update
            mat = mat + update*mask
            return mat

        # Fill off-diagonal elements of the correlation matrices
        # ASD vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsDS', 0, 1)
        # ASA vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsDS', 0, 2)
        # ASA vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsSF', 3, 2)
        # ASD vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsSF', 3, 1)
        # DS vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrDSvsSF', 3, 0)
        # ASD vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsASA', 1,2)
        # ASD vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASDvsK', 1, 4)
        # ASA vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrASAvsK', 2, 4)
        # DS vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrDSvsK', 0, 4)
        # SF vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrSFvsK', 3, 4)
        # ZSD vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsSF', 3, 6)
        # ZSA vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsSF', 3, 5)
        # ZSD vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsK', 6, 4)
        # ZSA vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsK', 5, 4)
        # ZSD vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsDS', 6, 0)
        # ZSA vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsDS', 5, 0)
        # ZSD vs ASD
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsASD', 6,1)
        # ZSA vs ASD
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsASD', 5,1)
        # ZSD vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsASA', 6,2)
        # ZSA vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSAvsASA', 5,2)
        # ZSD vs ZSA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, 'corrZSDvsZSA', 5,6)

        # Compute and store the square root of the cross-LSP correlation
        # matrix
        self._cross_lsp_correlation_matrix_sqrt = matrix_sqrt(
                cross_lsp_corr_mat)
        
    def _compute_lsp_spatial_correlation_sqrt(self):
        """
        Compute the square root of the spatial correlation matrices of LSPs.

        The LSPs are correlated accross users according to the distance between
        the users. Each LSP is spatially correlated according to a different
        spatial correlation matrix.

        The links involving different BSs are not correlated.
        UTs in different state (LoS, NLoS, O2I) are not assumed to be
        correlated.

        The correlation of the LSPs X of two UTs in the same state related to
        the links of these UTs to a same BS is

        .. math::
            C(X_1,X_2) = exp(-d/D_X)

        where :math:`d` is the distance between the UTs in the X-Y plane (2D
        distance) and D_X the correlation distance of LSP X.

        The resulting tensor if of shape
        [batch size, number of BSs, 7, number of UTs, number of UTs)
        7 being the number of LSPs.

        Input
        ------
        None

        Output
        -------
        None
        """

        # Tensors of bool indicating which pair of UTs to correlate.
        # Pairs of UTs that are correlated are those that share the same state
        # (indoor, LoS, or NLoS).
        # Indoor
        indoor = torch.tile(torch.unsqueeze(self._scenario.indoor, dim=1),
                         [1, self._scenario.num_bs, 1])
        # LoS
        los_ut = self._scenario.los
        los_pair_bool = torch.logical_and(torch.unsqueeze(los_ut, dim=3),
                                       torch.unsqueeze(los_ut, dim=2))
        # NLoS
        nlos_ut = torch.logical_and(torch.logical_not(self._scenario.los),
                                 torch.logical_not(indoor))
        nlos_pair_bool = torch.logical_and(torch.unsqueeze(nlos_ut, dim=3),
                                        torch.unsqueeze(nlos_ut, dim=2))
        # O2I
        o2i_pair_bool = torch.logical_and(torch.unsqueeze(indoor, dim=3),
                                       torch.unsqueeze(indoor, dim=2))

        # Stacking the correlation matrix
        # One correlation matrix per LSP
        filtering_matrices = []
        distance_scaling_matrices = []
        for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
            'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
            # Matrix used for filtering and scaling the 2D distances
            # For each pair of UTs, the entry is set to 0 if the UTs are in
            # different states, -1/(correlation distance) otherwise.
            # The correlation distance is different for each LSP.

            eye_matrix = torch.eye(self._scenario.num_ut,self._scenario.num_ut, dtype=self._real_dtype)
            batch_shape=[self._scenario.batch_size,self._scenario.num_bs]
            filtering_matrix = eye_matrix.unsqueeze(0).unsqueeze(0)  # 添加两个维度
            filtering_matrix = filtering_matrix.expand(batch_shape +[self._scenario.num_ut,self._scenario.num_ut])  # 扩展到批量形状

            distance_scaling_matrix = self._scenario.get_param(parameter_name)
            distance_scaling_matrix = torch.tile(torch.unsqueeze(
                distance_scaling_matrix, dim=3),
                [1, 1, 1, self._scenario.num_ut])
            distance_scaling_matrix = -1./distance_scaling_matrix
            # LoS
            filtering_matrix = torch.where(los_pair_bool,
                torch.tensor(1.0, dtype=self._real_dtype),
                    filtering_matrix)
            # NLoS
            filtering_matrix = torch.where(nlos_pair_bool,
                torch.tensor(1.0, dtype=self._real_dtype),
                    filtering_matrix)
            # indoor
            filtering_matrix = torch.where(o2i_pair_bool,
                torch.tensor(1.0, dtype=self._real_dtype),
                    filtering_matrix)
            # Stacking
            filtering_matrices.append(filtering_matrix)
            distance_scaling_matrices.append(distance_scaling_matrix)
        filtering_matrices = torch.stack(filtering_matrices, dim=2)
        distance_scaling_matrices = torch.stack(distance_scaling_matrices, dim=2)

        ut_dist_2d = self._scenario.matrix_ut_distance_2d
        # Adding a dimension for broadcasting with BS
        ut_dist_2d = torch.unsqueeze(torch.unsqueeze(ut_dist_2d, dim=1), dim=2)

        # Correlation matrix
        spatial_lsp_correlation = (torch.exp(
            ut_dist_2d*distance_scaling_matrices)*filtering_matrices)

        # Compute and store the square root of the spatial correlation matrix
        self._spatial_lsp_correlation_matrix_sqrt = matrix_sqrt(
                spatial_lsp_correlation)


    def _o2i_low_loss(self):
        """
        Compute for each BS-UT link the pathloss due to the O2I penetration loss
        in dB with the low-loss model.
        See section 7.4.3.1 of 38.901 specification.

        UTs located outdoor (LoS and NLoS) get O2I pathloss of 0dB.

        Input
        -----
        None

        Output
        -------
            Tensor with shape
            [batch size, number of BSs, number of UTs]
            containing the O2I penetration low-loss in dB for each BS-UT link
        """

        fc = self._scenario.carrier_frequency/1e9 # Carrier frequency (GHz)
        batch_size = self._scenario.batch_size
        num_ut = self._scenario.num_ut
        num_bs = self._scenario.num_bs

        # Material penetration losses
        # fc must be in GHz
        l_glass = 2. + 0.2*fc
        l_concrete = 5. + 4.*fc

        # Path loss through external wall
        pl_tw = 5.0 - 10.*torch.log10(0.3*torch.math.pow(torch.tensor(10.,
            dtype=self._real_dtype), -l_glass/10.0) + 0.7*torch.math.pow(
                torch.tensor(10., dtype=self._real_dtype),
                    -l_concrete/10.0))

        # Filtering-out the O2I pathloss for UTs located outdoor
        indoor_mask = torch.where(self._scenario.indoor, torch.tensor(1.0,
            dtype=self._real_dtype), torch.zeros([batch_size, num_ut],
            dtype=self._real_dtype))
        indoor_mask = torch.unsqueeze(indoor_mask, dim=1)
        pl_tw = pl_tw*indoor_mask

        # Pathloss due to indoor propagation
        # The indoor 2D distance for outdoor UTs is 0
        pl_in = 0.5*self._scenario.distance_2d_in

        # Random path loss component
        # Gaussian distributed with standard deviation 4.4 in dB
        pl_rnd = torch.normal(mean=0.0 ,std=4.4, size=[batch_size, num_bs, num_ut], dtype= self._real_dtype)
        pl_rnd = pl_rnd*indoor_mask

        return pl_tw + pl_in + pl_rnd


    def _o2i_high_loss(self):
        """
        Compute for each BS-UT link the pathloss due to the O2I penetration loss
        in dB with the high-loss model.
        See section 7.4.3.1 of 38.901 specification.

        UTs located outdoor (LoS and NLoS) get O2I pathloss of 0dB.

        Input
        -----
        None

        Output
        -------
            Tensor with shape
            [batch size, number of BSs, number of UTs]
            containing the O2I penetration low-loss in dB for each BS-UT link
        """

        fc = self._scenario.carrier_frequency/1e9 # Carrier frequency (GHz)
        batch_size = self._scenario.batch_size
        num_ut = self._scenario.num_ut
        num_bs = self._scenario.num_bs

        # Material penetration losses
        # fc must be in GHz
        l_iirglass = 23. + 0.3*fc
        l_concrete = 5. + 4.*fc

        # Path loss through external wall
        pl_tw = 5.0 - 10.*torch.log10(0.7*torch.math.pow(torch.tensor(10.,
            dtype=self._real_dtype), -l_iirglass/10.0)
                + 0.3*torch.math.pow(torch.tensor(10.,
                dtype=self._real_dtype), -l_concrete/10.0))

        # Filtering-out the O2I pathloss for outdoor UTs
        indoor_mask = torch.where(self._scenario.indoor, 1.0,
            torch.zeros([batch_size, num_ut], dtype=self._real_dtype))
        indoor_mask = torch.unsqueeze(indoor_mask, dim=1)
        pl_tw = pl_tw*indoor_mask

        # Pathloss due to indoor propagation
        # The indoor 2D distance for outdoor UTs is 0
        pl_in = 0.5*self._scenario.distance_2d_in

        # Random path loss component
        # Gaussian distributed with standard deviation 6.5 in dB for the
        # high loss model
        pl_rnd = torch.normal(mean=0.0, stddev=6.5,size=[batch_size, num_bs, num_ut],
                                  dtype=self._real_dtype)
        pl_rnd = pl_rnd*indoor_mask

        return pl_tw + pl_in + pl_rnd