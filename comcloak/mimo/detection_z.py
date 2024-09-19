import warnings
import numpy as np
import torch
from comcloak.utils import expand_to_rank, matrix_sqrt_inv, flatten_last_dims, flatten_dims, split_dim, insert_dims, hard_decisions
from comcloak.mapping import Constellation, SymbolLogits2LLRs, LLRs2SymbolLogits, PAM2QAM, Demapper, SymbolDemapper, SymbolInds2Bits, DemapperWithPrior, SymbolLogits2Moments
from comcloak.mimo.utils import complex2real_channel, whiten_channel, List2LLR, List2LLRSimple, complex2real_matrix, complex2real_vector, real2complex_vector
from comcloak.mimo.equalization import lmmse_equalizer, zf_equalizer, mf_equalizer

import torch.nn as nn
import torch.nn.functional as F

class LinearDetector(nn.Module):
    def __init__(self,
                 equalizer,
                 output,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64,
                 **kwargs):
        super(LinearDetector, self).__init__()

        self._output = output
        self._hard_out = hard_out

        # Determine the equalizer to use
        if isinstance(equalizer, str):
            assert equalizer in ["lmmse", "zf", "mf"], "Unknown equalizer."
            if equalizer == "lmmse":
                self._equalizer = lmmse_equalizer
            elif equalizer == "zf":
                self._equalizer = zf_equalizer
            else:
                self._equalizer = mf_equalizer
        else:
            self._equalizer = equalizer

        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        # Determine the constellation to use
        self._constellation = Constellation.create_or_check_constellation(
            constellation_type, num_bits_per_symbol, constellation, dtype=dtype)

        # Determine the demapper to use
        if output == "bit":
            self._demapper = Demapper(demapping_method, 
                                      constellation=self._constellation,
                                      hard_out=hard_out,
                                      dtype=dtype)
        else:
            self._demapper = SymbolDemapper(constellation=self._constellation,
                                            hard_out=hard_out,
                                            dtype=dtype)

    def forward(self, inputs):
        x_hat, no_eff = self._equalizer(*inputs)
        z = self._demapper([x_hat, no_eff])

        # Reshape to the expected output shape
        num_streams = inputs[1].shape[-1]
        if self._output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            z = z.view(*z.shape[:-1], num_streams, num_bits_per_symbol)

        return z

class MaximumLikelihoodDetector(nn.Module):
    def __init__(self,
                 output,
                 demapping_method,
                 num_streams,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 with_prior=False,
                 dtype=torch.complex64):
        super(MaximumLikelihoodDetector, self).__init__()

        assert dtype in [torch.complex64, torch.complex128],\
            "dtype must be torch.complex64 or torch.complex128"

        assert output in ("bit", "symbol"), "Unknown output"

        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        self._output = output
        self._demapping_method = demapping_method
        self._hard_out = hard_out
        self._with_prior = with_prior

        # Determine the reduce function for LLR computation
        if self._demapping_method == "app":
            self._reduce = torch.logsumexp
        else:
            self._reduce = torch.max

        # Create constellation object
        self._constellation = self._create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)

        # Utility function to compute vecs and indices
        vecs, vecs_ind, c = self._build_vecs(num_streams)
        self._vecs = vecs.to(dtype)
        self._vecs_ind = vecs_ind.to(torch.int32)
        self._c = c.to(torch.int32)

        if output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            self._logits2llr = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=num_bits_per_symbol,
                hard_out=hard_out,
                dtype=dtype.real_dtype)
            self._llrs2logits = LLRs2SymbolLogits(
                num_bits_per_symbol=num_bits_per_symbol,
                hard_out=False,
                dtype=dtype.real_dtype)

    def _create_or_check_constellation(self, constellation_type, num_bits_per_symbol, constellation, dtype):
        # You need to define this function or replace it with a similar one for PyTorch
        pass

    def _build_vecs(self, num_streams):
        points = self._constellation.points
        num_points = points.shape[0]

        def _build_vecs_(n):
            if n == 1:
                vecs = points.unsqueeze(1)
                vecs_ind = torch.arange(num_points).unsqueeze(1)
            else:
                v, vi = _build_vecs_(n - 1)
                vecs = []
                vecs_ind = []
                for i, p in enumerate(points):
                    vecs.append(torch.cat([torch.full([v.shape[0], 1], p), v], dim=1))
                    vecs_ind.append(torch.cat([torch.full([v.shape[0], 1], i), vi], dim=1))
                vecs = torch.cat(vecs, dim=0)
                vecs_ind = torch.cat(vecs_ind, dim=0)
            return vecs, vecs_ind

        vecs, vecs_ind = _build_vecs_(num_streams)
        tx_ind = torch.arange(num_streams).unsqueeze(0).repeat(vecs_ind.shape[0], 1)
        vecs_ind = torch.stack([tx_ind, vecs_ind], dim=-1)

        c = []
        for p in points:
            c_ = []
            for j in range(num_streams):
                c_.append(torch.where(vecs[:, j] == p)[0])
            c_ = torch.stack(c_, dim=-1)
            c.append(c_)
        c = torch.stack(c, dim=-1)

        return vecs, vecs_ind, c

    def forward(self, *inputs):
        if self._with_prior:
            y, h, prior, s = inputs
            if self._output == 'bit':
                prior = self._llrs2logits(prior)
        else:
            y, h, s = inputs

        s_inv = torch.linalg.inv(torch.linalg.cholesky(s))

        y = y.unsqueeze(-1)
        y = (s_inv @ y).squeeze(-1)

        h = s_inv @ h
        h = h.unsqueeze(-3)
        y = y.unsqueeze(-2)

        vecs = self._vecs.unsqueeze(-1)
        vecs = vecs.expand_as(h)

        diff = y - (h @ vecs).squeeze(-1)
        exponents = -torch.sum(diff.abs() ** 2, dim=-1)

        if self._with_prior:
            prior = prior.expand_as(exponents)
            prior = prior.permute(-2, -1, *torch.arange(prior.dim() - 2).tolist())
            prior = torch.gather(prior, 1, self._vecs_ind)
            prior = prior.permute(*torch.arange(2, prior.dim()).tolist(), 0, 1)
            prior = torch.sum(prior, dim=-1)
            exponents = exponents + prior

        exp = torch.gather(exponents, 1, self._c)

        logits = self._reduce(exp, dim=-3)

        if self._output == 'bit':
            return self._logits2llr(logits)
        else:
            if self._hard_out:
                return torch.argmax(logits, dim=-1)
            else:
                return logits

class MaximumLikelihoodDetectorWithPrior(MaximumLikelihoodDetector):
    """
    MaximumLikelihoodDetectorWithPrior(output, demapping_method, num_streams, 
    constellation_type=None, num_bits_per_symbol=None, constellation=None, 
    hard_out=False, dtype=torch.complex64, **kwargs)

    MIMO maximum-likelihood (ML) detector, assuming prior
    knowledge on the bits or constellation points is available.

    This class is deprecated as the functionality has been integrated
    into MaximumLikelihoodDetector.

    This layer implements MIMO maximum-likelihood (ML) detection assuming the
    following channel model:

    y = Hx + n

    where y ∈ ℂ^M is the received signal vector,
    x ∈ ℂ^K is the vector of transmitted symbols which are uniformly 
    and independently drawn from the constellation ℂ,
    H ∈ ℂ^{M×K} is the known channel matrix,
    and n ∈ ℂ^M is a complex Gaussian noise vector.
    It is assumed that E[n] = 0 and
    E[nn^H] = S, where S has full rank.
    It is assumed that prior information of the transmitted signal x is available,
    provided either as LLRs on the bits modulated onto x or as logits on the individual
    constellation points forming x.

    Prior to demapping, the received signal is whitened:

    ẏ = S^(-1/2)y = S^(-1/2)Hx + S^(-1/2)n = Ĥx + ṅ

    The layer can compute ML detection of symbols or bits with either
    soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
    and not jointly for the entire vector x (or the underlying vector of bits).

    ML detection of bits:

    Soft-decisions on bits are called log-likelihood ratios (LLR).
    With the “app” demapping method, the LLR for the i-th bit
    of the k-th user is then computed according to

    LLR(k,i) = ln((Pr(b_{k,i}=1 | y,H))/(Pr(b_{k,i}=0 | y,H)))
             = ln(Σ[x ∈ ℂ_{k,i,1}] exp(-||ẏ - Ĥx||²)Pr(x) / Σ[x ∈ ℂ_{k,i,0}] exp(-||ẏ - Ĥx||²)Pr(x))

    where ℂ_{k,i,1} and ℂ_{k,i,0} are the
    sets of vectors of constellation points for which the i-th bit
    of the k-th user is equal to 1 and 0, respectively.
    Pr(x) is the prior distribution of the vector of
    constellation points x. Assuming that the constellation points and
    bit levels are independent, it is computed from the prior of the bits according to

    Pr(x) = Π_{k=1}^K Π_{i=1}^I σ(LLR_p(k,i))

    where LLR_p(k,i) is the prior knowledge of the i-th bit of the
    k-th user given as an LLR, and σ(·) is the sigmoid function.

    With the "maxlog" demapping method, the LLR for the i-th bit
    of the k-th user is approximated as:

    LLR(k,i) ≈ ln((max[x ∈ ℂ_{k,i,1}] exp(-||ẏ - Ĥx||²)Pr(x)) / (max[x ∈ ℂ_{k,i,0}] exp(-||ẏ - Ĥx||²)Pr(x)))
             ≈ min[x ∈ ℂ_{k,i,0}] (||ẏ - Ĥx||² - ln(Pr(x))) - min[x ∈ ℂ_{k,i,1}] (||ẏ - Ĥx||² - ln(Pr(x))).

    ML detection of symbols:

    Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

    With the “app” demapping method, the logit for the
    constellation point c ∈ ℂ of the k-th user  is computed according to

    logit(k,c) = ln(Σ[x : x_k = c] exp(-||ẏ - Ĥx||²)Pr(x)).

    With the "maxlog" demapping method, the logit for the constellation point c ∈ ℂ
    of the k-th user  is approximated as

    logit(k,c) ≈ max[x : x_k = c] (-||ẏ - Ĥx||² + ln(Pr(x))).

    When hard decisions are requested, this layer returns for the k-th stream

    ĥat{c}_k = argmax[c ∈ ℂ] (Σ[x : x_k = c] exp(-||ẏ - Ĥx||²)Pr(x))

    where ℂ is the set of constellation points.
    """

    def __init__(self, output, demapping_method, num_streams, 
                 constellation_type=None, num_bits_per_symbol=None, 
                 constellation=None, hard_out=False, dtype=torch.complex64, **kwargs):
        super().__init__(output=output, demapping_method=demapping_method, 
                         num_streams=num_streams, constellation_type=constellation_type, 
                         num_bits_per_symbol=num_bits_per_symbol, 
                         constellation=constellation, hard_out=hard_out, 
                         with_prior=True, dtype=dtype, **kwargs)

class KBestDetector(nn.Module):
    def __init__(self,
                 output,
                 num_streams,
                 k,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 use_real_rep=False,
                 list2llr="default",
                 dtype=torch.complex64,
                 **kwargs):
        super().__init__()
        assert dtype in [torch.complex64, torch.complex128],\
            "dtype must be torch.complex64 or torch.complex128."

        assert output in ("bit", "symbol"), "Unknown output"

        if constellation is None:
            assert constellation_type is not None and \
                   num_bits_per_symbol is not None, \
                   "You must provide either constellation or constellation_type and num_bits_per_symbol."
        else:
            assert constellation_type is None and \
                   num_bits_per_symbol is None, \
                   "You must provide either constellation or constellation_type and num_bits_per_symbol."

        if constellation is not None:
            assert constellation.dtype == dtype, \
                "Constellation has wrong dtype."

        self._output = output
        self._hard_out = hard_out
        self._use_real_rep = use_real_rep

        if self._use_real_rep:
            err_msg = "Only QAM can be used for the real-valued representation"
            if constellation_type is not None:
                assert constellation_type == "qam", err_msg
            else:
                assert constellation._constellation_type == "qam", err_msg

            self._num_streams = 2 * num_streams
            if num_bits_per_symbol is None:
                n = constellation.num_bits_per_symbol // 2
                self._num_bits_per_symbol = n
            else:
                self._num_bits_per_symbol = num_bits_per_symbol // 2

            c = Constellation("pam", self._num_bits_per_symbol, normalize=False, dtype=dtype)
            c._points /= torch.std(c.points) * torch.sqrt(torch.tensor(2.0, dtype=dtype))
            self._constellation = c.points

            self._pam2qam = PAM2QAM(2 * self._num_bits_per_symbol)

        else:
            self._num_streams = num_streams
            c = Constellation.create_or_check_constellation(
                constellation_type, num_bits_per_symbol, constellation, dtype=dtype)
            self._constellation = c.points
            self._num_bits_per_symbol = c.num_bits_per_symbol

        self._num_symbols = self._constellation.shape[0]
        self._k = min(k, self._num_symbols ** self._num_streams)
        if self._k < k:
            warnings.warn(f"KBestDetector: The provided value of k={k} is larger than the possible maximum number of paths. It has been set to k={self._k}.")

        num_paths = [1]
        for l in range(1, self._num_streams + 1):
            num_paths.append(min(self._k, self._num_symbols ** l))
        self._num_paths = torch.tensor(num_paths, dtype=torch.int32)

        indices = torch.zeros([self._num_streams, self._k * self._num_streams, 2], dtype=torch.int32)
        for l in range(self._num_streams):
            ind = torch.zeros([self._num_paths[l + 1], self._num_streams], dtype=torch.int32)
            ind[:, :l + 1] = 1
            ind = torch.stack(torch.where(ind), dim=-1)
            indices[l, :ind.shape[0], :ind.shape[1]] = ind
        self._indices = indices

        if self._output == "bit":
            if not self._hard_out:
                if list2llr == "default":
                    self.list2llr = List2LLRSimple(self._num_bits_per_symbol)
                else:
                    self.list2llr = list2llr
            else:
                n = 2 * self._num_bits_per_symbol if self._use_real_rep else self._num_bits_per_symbol
                self._symbolinds2bits = SymbolInds2Bits(n, dtype=dtype.real_dtype)
        else:
            assert self._hard_out, "Soft-symbols are not supported for this detector."

    @property
    def list2llr(self):
        return self._list2llr

    @list2llr.setter
    def list2llr(self, value):
        assert isinstance(value, List2LLR)
        self._list2llr = value

    def _preprocessing(self, inputs):
        y, h, s = inputs
        if self._use_real_rep:
            y, h, s = complex2real_channel(y, h, s)

        y, h = whiten_channel(y, h, s)

        h_norm = torch.sum(h.abs() ** 2, dim=1)
        column_order = torch.argsort(h_norm, dim=-1, descending=True)
        h = torch.gather(h, -1, column_order.unsqueeze(-2).expand(-1, -1, h.shape[-2]))

        q, r = torch.qr(h)

        y = torch.squeeze(torch.matmul(q.transpose(-2, -1).conj(), y.unsqueeze(-1)), -1)

        return y, r, column_order

    def _select_best_paths(self, dists, path_syms, path_inds):
        num_paths = path_syms.shape[1]
        k = min(num_paths, self._k)
        dists, ind = torch.topk(-dists, k=k, dim=1, sorted=True)
        dists = -dists

        path_syms = torch.gather(path_syms, 1, ind.unsqueeze(-1).expand(-1, -1, path_syms.shape[-1]))
        path_inds = torch.gather(path_inds, 1, ind.unsqueeze(-1).expand(-1, -1, path_inds.shape[-1]))

        return dists, path_syms, path_inds

    def _next_layer(self, y, r, dists, path_syms, path_inds, stream):
        batch_size = y.shape[0]
        stream_ind = self._num_streams - 1 - stream
        num_paths = self._num_paths[stream]

        dists_o = dists.clone()
        path_syms_o = path_syms.clone()
        path_inds_o = path_inds.clone()

        dists = dists[..., :num_paths]
        path_syms = path_syms[..., :num_paths, :stream]
        path_inds = path_inds[..., :num_paths, :stream]

        dists = dists.repeat(1, self._num_symbols)
        path_syms = path_syms.repeat(1, self._num_symbols, 1)
        path_inds = path_inds.repeat(1, self._num_symbols, 1)

        syms = self._constellation.view(1, -1)
        syms = syms.repeat(self._k, 1)
        syms = syms.view(1, -1, 1)
        syms = syms.repeat(batch_size, 1, 1)
        syms = syms[:, :num_paths * self._num_symbols]
        path_syms = torch.cat([path_syms, syms], dim=-1)

        inds = torch.arange(0, self._num_symbols).view(1, -1)
        inds = inds.repeat(self._k, 1)
        inds = inds.view(1, -1, 1)
        inds = inds.repeat(batch_size, 1, 1)
        inds = inds[:, :num_paths * self._num_symbols]
        path_inds = torch.cat([path_inds, inds], dim=-1)

        y = y[:, stream_ind].unsqueeze(-1)
        r = torch.flip(r[:, stream_ind, stream_ind:], [-1]).unsqueeze(1)
        delta = (y - torch.sum(r * path_syms, dim=-1)).abs().pow(2)

        dists += delta

        dists, path_syms, path_inds = self._select_best_paths(dists, path_syms, path_inds)

        dists = torch.transpose(dists_o, 1, 0)
        updates = torch.transpose(dists, 1, 0)
        indices = torch.arange(updates.shape[0], dtype=torch.int32).unsqueeze(-1)
        dists = dists_o.scatter(1, indices, updates)

        path_syms = torch.transpose(path_syms_o, 1, 2).contiguous()
        updates = torch.transpose(path_syms, 1, 2).contiguous().view(-1, batch_size)
        indices = self._indices[stream, :self._num_paths[stream + 1] * (stream + 1)]
        path_syms = path_syms_o.scatter(1, indices, updates)

        path_inds = torch.transpose(path_inds_o, 1, 2).contiguous()
        updates = torch.transpose(path_inds, 1, 2).contiguous().view(-1, batch_size)
        path_inds = path_inds_o.scatter(1, indices, updates)

        return dists, path_syms, path_inds

    def _unsort(self, column_order, tensor, transpose=True):
        unsort_inds = torch.argsort(column_order, dim=-1)
        if transpose:
            tensor = tensor.transpose(1, -1)
        tensor = torch.gather(tensor, -2, unsort_inds.unsqueeze(-2).expand(-1, -1, tensor.shape[-1]))
        if transpose:
            tensor = tensor.transpose(1, -1)
        return tensor

    def _logits2llrs(self, logits, path_inds):
        # Implementation details depend on List2LLR class
        llrs = self.list2llr(logits, path_inds)
        return llrs

    def forward(self, inputs):
        y, h, s = inputs
        y, r, column_order = self._preprocessing(inputs)

        dists = torch.zeros([y.shape[0], 1], dtype=y.dtype, device=y.device)
        path_syms = torch.zeros([y.shape[0], 1, 0], dtype=y.dtype, device=y.device)
        path_inds = torch.zeros([y.shape[0], 1, 0], dtype=torch.int32, device=y.device)

        for stream in range(self._num_streams):
            dists, path_syms, path_inds = self._next_layer(y, r, dists, path_syms, path_inds, stream)

        path_syms = torch.squeeze(path_syms[:, 0, :])
        path_inds = torch.squeeze(path_inds[:, 0, :])

        logits = path_syms.unsqueeze(-1) * self._constellation.view(1, 1, -1)
        logits = torch.sum(logits, dim=-2)

        if self._output == "symbol":
            return self._unsort(column_order, logits)

        if self._hard_out:
            return self._unsort(column_order, path_inds, transpose=False)
        else:
            llrs = self._logits2llrs(logits, path_inds)
            return llrs

class EPDetector(nn.Module):
    def __init__(self, output, num_bits_per_symbol, hard_out=False, l=10, beta=0.9, dtype=torch.complex64):
        super().__init__()
        assert dtype in [torch.complex64, torch.complex128], "Invalid dtype"
        self._cdtype = dtype
        self._rdtype = torch.float32 if dtype == torch.complex64 else torch.float64

        # Numerical stability threshold
        self._prec = 1e-6 if dtype == torch.complex64 else 1e-12

        assert output in ("bit", "symbol"), "Unknown output"
        self._output = output
        self._hard_out = hard_out

        if self._output == "symbol":
            self._pam2qam = PAM2QAM(num_bits_per_symbol, hard_out)
        else:
            self._symbollogits2llrs = SymbolLogits2LLRs("maxlog", num_bits_per_symbol // 2, hard_out=hard_out)
            self._demapper = Demapper("maxlog", "pam", num_bits_per_symbol // 2)

        assert l >= 1, "l must be a positive integer"
        self._l = l
        assert 0.0 <= beta <= 1.0, "beta must be in [0,1]"
        self._beta = beta

        # Create PAM constellations for real-valued detection
        self._num_bits_per_symbol = num_bits_per_symbol // 2
        points = Constellation("pam", self._num_bits_per_symbol).points

        # Scale constellation points to half the energy
        self._points = torch.tensor(points / np.sqrt(2.0), dtype=self._rdtype)

        # Average symbol energy
        self._es = torch.tensor(np.var(self._points), dtype=self._rdtype)

    def compute_sigma_mu(self, h_t_h, h_t_y, no, lam, gam):
        """Equations (28) and (29)"""
        lam = torch.diag_embed(lam)
        gam = gam.unsqueeze(-1)
        
        sigma = torch.linalg.inv(h_t_h + no * lam)
        mu = torch.squeeze(torch.matmul(sigma, h_t_y + no * gam), dim=-1)
        sigma *= no
        sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)
        
        return sigma, mu

    def compute_v_x_obs(self, sigma, mu, lam, gam):
        """Equations (31) and (32)"""
        v_obs = torch.clamp(1 / (1 / sigma - lam), min=self._prec)
        x_obs = v_obs * (mu / sigma - gam)
        return v_obs, x_obs

    def compute_v_x(self, v_obs, x_obs):
        """Equation (33)"""
        x_obs = x_obs.unsqueeze(-1)
        v_obs = v_obs.unsqueeze(-1)

        points = self._points.unsqueeze(0).expand_as(x_obs)
        logits = -torch.pow(x_obs - points, 2) / (2 * v_obs)
        pmf = F.softmax(logits, dim=-1)

        x = torch.sum(points * pmf, dim=-1, keepdim=True)
        v = torch.sum((points - x)**2 * pmf, dim=-1)
        v = torch.clamp(v, min=self._prec)
        x = torch.squeeze(x, dim=-1)

        return v, x, logits

    def update_lam_gam(self, v, v_obs, x, x_obs, lam, gam):
        """Equations (35), (36), (37), (38)"""
        lam_old = lam
        gam_old = gam

        lam = 1 / v - 1 / v_obs
        gam = x / v - x_obs / v_obs

        lam_new = torch.where(lam < 0, lam_old, lam)
        gam_new = torch.where(lam < 0, gam_old, gam)

        lam_damp = (1 - self._beta) * lam_new + self._beta * lam_old
        gam_damp = (1 - self._beta) * gam_new + self._beta * gam_old

        return lam_damp, gam_damp

    def forward(self, y, h, s):
        # Flatten the batch dimensions
        batch_shape = y.shape[:-1]
        num_batch_dims = len(batch_shape)
        if num_batch_dims > 1:
            y = y.view(-1, *y.shape[-2:])
            h = h.view(-1, *h.shape[-3:])
            s = s.view(-1, *s.shape[-3:])
        
        n_t = h.shape[-1]

        # Whiten channel
        y, h, s = whiten_channel(y, h, s)

        # Convert channel to real-valued representation
        y, h, s = complex2real_channel(y, h, s)

        # Convert all inputs to desired dtypes
        y = y.to(self._rdtype)
        h = h.to(self._rdtype)
        no = torch.tensor(0.5, dtype=self._rdtype)

        # Initialize gamma and lambda
        gam = torch.zeros(*y.shape[:-1], h.shape[-1], dtype=y.dtype)
        lam = torch.ones(*y.shape[:-1], h.shape[-1], dtype=y.dtype) / self._es

        # Precompute values
        h_t_h = torch.matmul(h, h.transpose(-2, -1))
        y = y.unsqueeze(-1)
        h_t_y = torch.matmul(h, y)
        no = no.expand_as(h_t_h)

        for _ in range(self._l):
            sigma, mu = self.compute_sigma_mu(h_t_h, h_t_y, no, lam, gam)
            v_obs, x_obs = self.compute_v_x_obs(sigma, mu, lam, gam)
            v, x, logits = self.compute_v_x(v_obs, x_obs)
            lam, gam = self.update_lam_gam(v, v_obs, x, x_obs, lam, gam)

        # Extract the logits for the 2 PAM constellations for each stream
        pam1_logits = logits[..., :n_t, :]
        pam2_logits = logits[..., n_t:, :]

        if self._output == "symbol" and self._hard_out:
            # Take hard decisions on PAM symbols
            pam1_ind = torch.argmax(pam1_logits, dim=-1)
            pam2_ind = torch.argmax(pam2_logits, dim=-1)

            # Transform to QAM indices
            qam_ind = self._pam2qam(pam1_ind, pam2_ind)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                qam_ind = qam_ind.view(*batch_shape, -1)

            return qam_ind

        elif self._output == "symbol" and not self._hard_out:
            qam_logits = self._pam2qam(pam1_logits, pam2_logits)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                qam_logits = qam_logits.view(*batch_shape, -1)

            return qam_logits

        elif self._output == "bit":
            # Compute LLRs for both PAM constellations
            llr1 = self._symbollogits2llrs(pam1_logits)
            llr2 = self._symbollogits2llrs(pam2_logits)

            # Put LLRs in the correct order and shape
            llr = torch.stack([llr1, llr2], dim=-1)
            llr = llr.view(*llr.shape[:-1], -1)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                llr = llr.view(*batch_shape, -1)

            return llr

class MMSEPICDetector(nn.Module):
    def __init__(self,
                 output,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=torch.complex64):
        super(MMSEPICDetector, self).__init__()

        assert isinstance(num_iter, int), "num_iter must be an integer"
        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        assert dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"

        self.num_iter = num_iter
        self.output = output
        self.epsilon = 1e-4
        self.realdtype = dtype.real
        self.demapping_method = demapping_method
        self.hard_out = hard_out

        # Create constellation object
        self.constellation = self.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype
        )

        # Soft symbol mapping
        self.llr_2_symbol_logits = LLRs2SymbolLogits(
            self.constellation.num_bits_per_symbol,
            dtype=self.realdtype
        )

        if self.output == "symbol":
            self.llr_2_symbol_logits_output = LLRs2SymbolLogits(
                self.constellation.num_bits_per_symbol,
                dtype=self.realdtype,
                hard_out=hard_out
            )
            self.symbol_logits_2_llrs = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=self.constellation.num_bits_per_symbol
            )
        self.symbol_logits_2_moments = SymbolLogits2Moments(
            constellation=self.constellation,
            dtype=self.realdtype
        )

        # soft output demapping
        self.bit_demapper = DemapperWithPrior(
            demapping_method=demapping_method,
            constellation=self.constellation,
            dtype=dtype
        )

    def create_or_check_constellation(self, constellation_type, num_bits_per_symbol, constellation, dtype):
        # Placeholder for constellation creation logic
        # Implement or import the actual constellation logic here
        pass

    def whiten_channel(self, y, h, s):
        # Placeholder for whitening channel logic
        # Implement or import the actual whitening logic here
        pass

    def call(self, inputs):
        y, h, prior, s = inputs
        
        # Preprocessing
        y, h = self.whiten_channel(y, h, s)

        # Matched filtering of y
        y_mf = torch.matmul(h, y.unsqueeze(-1)).squeeze(-1)

        # Step 1: compute Gramm matrix
        g = torch.matmul(h.transpose(-2, -1), h)

        # For XLA compatibility, this implementation performs the MIMO equalization in the real-valued domain
        hr = self.complex2real_matrix(h)
        gr = torch.matmul(hr.transpose(-2, -1), hr)

        # Compute a priori LLRs
        if self.output == "symbol":
            llr_a = self.symbol_logits_2_llrs(prior)
        else:
            llr_a = prior
        llr_shape = llr_a.shape

        def mmse_pic_self_iteration(llr_d, llr_a, it):
            # MMSE PIC takes in a priori LLRs
            llr_a = llr_d

            # Step 2: compute soft symbol estimates and variances
            x_logits = self.llr_2_symbol_logits(llr_a)
            x_hat, var_x = self.symbol_logits_2_moments(x_logits)

            # Step 3: perform parallel interference cancellation
            y_mf_pic = y_mf + g.unsqueeze(-1) * x_hat.unsqueeze(-2) - torch.matmul(g, x_hat.unsqueeze(-1)).squeeze(-1)

            # Step 4: compute A^-1 matrix
            var_x = torch.cat([var_x, var_x], dim=-1)
            var_x_row_vec = var_x.unsqueeze(-2)
            a = gr * var_x_row_vec

            a_inv = torch.linalg.inv(a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype))

            # Step 5: compute unbiased MMSE filter and outputs
            mu = torch.sum(a_inv * gr, dim=-1)

            y_mf_pic_trans = self.complex2real_vector(y_mf_pic.transpose(-2, -1))
            y_mf_pic_trans = torch.cat([y_mf_pic_trans, y_mf_pic_trans], dim=-2)

            x_hat = torch.sum(a_inv * y_mf_pic_trans, dim=-1) / mu.unsqueeze(-1)

            var_x = mu / torch.clamp(1 - var_x * mu, min=self.epsilon)
            var_x, _ = torch.split(var_x, 2, dim=-1)

            no_eff = 1. / var_x

            # Step 6: LLR demapping (extrinsic LLRs)
            llr_d = self.bit_demapper([x_hat, llr_a, no_eff]).reshape(llr_shape)

            return llr_d, llr_a, it

        def dec_stop(llr_d, llr_a, it):
            return it < self.num_iter

        it = torch.tensor(0)
        null_prior = torch.zeros_like(llr_a, dtype=self.realdtype)
        llr_d, llr_a, _ = self.iterative_loop(
            dec_stop,
            mmse_pic_self_iteration,
            (llr_a, null_prior, it)
        )
        llr_e = llr_d - llr_a
        if self.output == "symbol":
            out = self.llr_2_symbol_logits_output(llr_e)
        else:
            out = llr_e
            if self.hard_out:
                out = self.hard_decisions(out)

        return out

    def complex2real_matrix(self, x):
        return torch.cat([x.real, x.imag], dim=-1)

    def complex2real_vector(self, x):
        return torch.cat([x.real.unsqueeze(-1), x.imag.unsqueeze(-1)], dim=-1)

    def hard_decisions(self, x):
        return torch.argmax(x, dim=-1)

    def iterative_loop(self, stop_fn, body_fn, init_state):
        state = init_state
        while not stop_fn(*state):
            state = body_fn(*state)
        return state
