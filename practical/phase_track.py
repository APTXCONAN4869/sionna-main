import torch
import torch.nn.functional as F
import numpy as np

def phase_tracking(y, x_pilot, pilot_idx, pilot_sym_idx=[2, 11]):
    """
    OFDM 相位跟踪算法
    y:        (B, Rx, Ant, Nsym, Nfft)
    x_pilot:  (B, Rx, Ant, Nsym, Nfft)
    pilot_idx: 导频子载波索引 (list or tensor)
    pilot_sym_idx: 导频符号索引 (默认 [2, 11])
    """
    B, Rx, Ant, Nsym, Nfft = y.shape

    # Step 1: 导频符号相位估计
    phi_pilot = torch.zeros(B, Rx, Ant, len(pilot_sym_idx), device=y.device)
    for i, sym in enumerate(pilot_sym_idx):
        Yp = y[..., sym, pilot_idx]  # (B, Rx, Ant, Npilots)
        Xp = x_pilot[..., sym, pilot_idx]
        prod = Yp * torch.conj(Xp)
        phi_pilot[..., i] = torch.angle(prod.sum(dim=-1))  # (B, Rx, Ant)

    # Step 2: 相位展开 (unwrap)
    phi_pilot = torch.from_numpy(np.unwrap(phi_pilot.cpu().numpy(), axis=-1)).to(y.device)

    # Step 3: 线性插值到所有符号
    sym_idx = torch.arange(Nsym, device=y.device).float()
    phi_interp = torch.zeros(B, Rx, Ant, Nsym, device=y.device)
    n1, n2 = pilot_sym_idx

    phi1, phi2 = phi_pilot[..., 0], phi_pilot[..., 1]
    slope = (phi2 - phi1) / (n2 - n1)
    phi_interp = phi1.unsqueeze(-1) + slope.unsqueeze(-1) * (sym_idx - n1)

    # Step 4: 相位补偿
    phase_corr = torch.exp(-1j * phi_interp.unsqueeze(-1))  # (B, Rx, Ant, Nsym, 1)
    y_corr = y * phase_corr  # 补偿后的信号

    return y_corr, phi_interp
