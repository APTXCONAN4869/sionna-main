import torch
import math

def apply_residual_cfo_torch(y, fs, delta_f, phi0=0.0, pn_std_per_sample=0.0, device=None):
    """
    Apply residual CFO (+ optional phase noise) to time-domain signal y.
    y: complex tensor shape (B, R, A, T) (dtype=torch.cfloat)
    fs: sampling rate (Hz)
    delta_f: scalar or tensor broadcastable to (B, R, A) in Hz
    phi0: initial phase in radians, scalar or broadcastable to (B,R,A)
    pn_std_per_sample: std dev of Wiener increment per sample (radians). If 0 -> no PN.
    Returns: y_out same shape as y (complex)
    """
    if device is None:
        device = y.device
    B, R, A, T = y.shape
    # time index n (0..T-1)
    n = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
    # ensure delta_f and phi0 have shape (B,R,A,1) for broadcasting
    def _prep(x):
        tx = torch.as_tensor(x, device=device, dtype=torch.float32)
        while tx.dim() < 3:
            tx = tx.unsqueeze(0)
        # now tx shape <=3, expand to (B,R,A)
        tx = tx.expand(B, R, A)
        return tx.unsqueeze(-1)  # (B,R,A,1)

    delta_f_t = _prep(delta_f)    # (B,R,A,1)
    phi0_t = _prep(phi0)          # (B,R,A,1)

    # deterministic phase: 2*pi*delta_f * n / fs + phi0
    phase_det = 2.0 * math.pi * delta_f_t * (n / fs) + phi0_t  # (B,R,A,T)
    # optionally add Wiener phase noise (cumulative normal increments)
    if pn_std_per_sample > 0.0:
        # generate normal increments per sample and cumsum
        increments = torch.normal(0.0, pn_std_per_sample, size=(B, R, A, T), device=device)
        pn = torch.cumsum(increments, dim=-1)  # (B,R,A,T)
    else:
        pn = torch.zeros_like(phase_det)

    total_phase = phase_det + pn  # (B,R,A,T)
    phasor = torch.exp(1j * total_phase)  # complex phasor
    y_out = y * phasor
    return y_out
