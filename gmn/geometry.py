"""Differentiable kinematics and rotation utilities (PyTorch).

Everything here is parameter-free and operates on physical coordinates
(hand-span units, normalised gesture time), so the quantities it returns are
identifiable: e.g. a velocity estimate for "Move left" has a negative x
component in the sensor frame, whatever the network learns afterwards.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants import SL_DIR, SL_NORMAL, SL_PALM, SL_TIPS, IDX_FINGERS


# ─────────────────────────────────────────────────────────────────────────────
# Local polynomial (Savitzky–Golay) operators
# ─────────────────────────────────────────────────────────────────────────────
def local_poly_operators(K: int, dt: float, degree: int):
    """Design matrix A (K x d+1) with columns [1, tau, tau^2/2, ...] for the
    centred window tau = (-h..h)*dt, and its pseudo-inverse P (d+1 x K).
    ``P @ window`` returns least-squares estimates [p, v, a] at the window
    centre; ``(I - A P)`` projects a window onto the residual that no
    polynomial of degree <= d can explain."""
    h = K // 2
    tau = torch.arange(-h, h + 1, dtype=torch.float64) * dt
    A = torch.stack([tau ** d / math.factorial(d) for d in range(degree + 1)], 1)
    P = torch.linalg.pinv(A)
    R = torch.eye(K, dtype=torch.float64) - A @ P
    return A, P, R          # float64; cast to the input dtype at use time


def windowed_estimates(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """x: (B, C, T); P: (d+1, K). Returns (B, d+1, C, T) estimates at every
    frame, using replicate padding at the sequence ends."""
    B, C, T = x.shape
    K = P.shape[1]
    xp = F.pad(x, (K // 2, K // 2), mode="replicate")
    # depthwise conv with each row of P (flip because conv1d is correlation
    # and our windows are ordered from past to future -> no flip needed)
    w = P.to(x.dtype).unsqueeze(1)                       # (d+1, 1, K)
    out = F.conv1d(xp.reshape(B * C, 1, -1), w)          # (B*C, d+1, T)
    return out.reshape(B, C, -1, T).permute(0, 2, 1, 3)


def finite_differences(x: torch.Tensor, dt: float):
    """Central finite differences along time. x: (B, T, C)."""
    v = torch.gradient(x, spacing=dt, dim=1)[0]
    a = torch.gradient(v, spacing=dt, dim=1)[0]
    return v, a


# ─────────────────────────────────────────────────────────────────────────────
# Rotations
# ─────────────────────────────────────────────────────────────────────────────
def palm_frame(direction: torch.Tensor, normal: torch.Tensor, eps: float = 1e-6):
    """Orthonormal hand frame R = [e1 e2 e3] (columns) from the Leap hand
    direction (palm -> fingers) and palm normal, by Gram–Schmidt.
    direction, normal: (..., 3). Returns (..., 3, 3) with det = +1."""
    e1 = F.normalize(direction, dim=-1, eps=eps)
    n = normal - (normal * e1).sum(-1, keepdim=True) * e1
    e3 = F.normalize(n, dim=-1, eps=eps)
    e2 = torch.cross(e3, e1, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def rotmat_to_quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> unit quaternion (..., 4) as (w, x, y, z),
    numerically stable branch selection (Shepperd's method)."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = torch.stack([
        torch.stack([1 + t, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
        torch.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]], -1),
        torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1),
    ], -2)                                                      # (..., 4, 4)
    diag = torch.stack([1 + t, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                        1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                        1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    best = diag.argmax(-1)
    q = torch.gather(cands, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    return F.normalize(q, dim=-1, eps=eps)


def quat_hemisphere_continuity(q: torch.Tensor) -> torch.Tensor:
    """Remove the q / -q double cover along time (q: (B, T, 4)) so the
    sequence is continuous."""
    dots = (q[:, 1:] * q[:, :-1]).sum(-1)
    sign = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    cum = torch.cumprod(sign, dim=1)                     # flipping q_{t-1} flips the next dot
    return torch.cat([q[:, :1], q[:, 1:] * cum.unsqueeze(-1)], 1)


def body_angular_velocity(R: torch.Tensor, dt: float) -> torch.Tensor:
    """Angular velocity expressed in the hand frame, from consecutive frames:
    R_t^T R_{t+1} ~ I + [w]_x dt. Invariant to any constant global rotation
    of the sensor. R: (B, T, 3, 3) -> (B, T, 3)."""
    M = R[:, :-1].transpose(-1, -2) @ R[:, 1:]
    S = 0.5 * (M - M.transpose(-1, -2))
    w = torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], -1) / dt
    return torch.cat([w, w[:, -1:]], 1)


def random_rotation_about(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula; axis (3,), angle (B,) -> (B, 3, 3)."""
    k = F.normalize(axis, dim=0)
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
                     dtype=angle.dtype, device=angle.device)
    I = torch.eye(3, dtype=angle.dtype, device=angle.device)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return I + s * K + (1 - c) * (K @ K)


def rotate_sample(x: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
    """Apply a global rotation Rg (B,3,3) to every 3-vector of a 26-feature
    sequence x (B, T, 26). Positions are rotated about the sensor origin."""
    x = x.clone()
    groups = [SL_DIR, SL_PALM, SL_NORMAL] + [slice(11 + 3 * i, 14 + 3 * i) for i in range(5)]
    for sl in groups:
        x[:, :, sl] = torch.einsum("bij,btj->bti", Rg, x[:, :, sl])
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (parameter-free, differentiable)
# ─────────────────────────────────────────────────────────────────────────────
def split_features(x: torch.Tensor):
    """x: (B, T, 26) geometric-normalised. Returns dict of physical streams."""
    B, T, _ = x.shape
    palm = x[:, :, SL_PALM]
    tips = x[:, :, SL_TIPS].reshape(B, T, 5, 3)
    return dict(palm=palm, tips=tips, direction=x[:, :, SL_DIR],
                normal=x[:, :, SL_NORMAL], fingers=x[:, :, IDX_FINGERS:IDX_FINGERS + 1],
                hand_id=x[:, :, 0:1])


def world_stream(x: torch.Tensor):
    """Sensor-frame stream: palm + fingertip trajectories and the unit
    quaternion of the hand orientation (computed from the true rotation
    matrix, not a learned projection)."""
    f = split_features(x)
    pos = torch.cat([f["palm"].unsqueeze(2), f["tips"]], 2)       # (B,T,6,3)
    R = palm_frame(f["direction"], f["normal"])
    q = quat_hemisphere_continuity(rotmat_to_quat(R))
    extras = torch.cat([f["direction"], f["normal"], q, f["fingers"], f["hand_id"]], -1)
    return pos, f["normal"], extras                                # extras: 12


def palm_stream(x: torch.Tensor, dt: float):
    """Hand-frame stream, invariant to any constant global rotation (and
    translation) of the sensor: fingertips expressed in the palm frame,
    angular velocity and palm velocity expressed in the palm frame."""
    f = split_features(x)
    R = palm_frame(f["direction"], f["normal"])                    # (B,T,3,3)
    Rt = R.transpose(-1, -2)
    local_tips = torch.einsum("btij,btkj->btki", Rt, f["tips"] - f["palm"].unsqueeze(2))
    omega = body_angular_velocity(R, dt)
    v_palm = torch.gradient(f["palm"], spacing=dt, dim=1)[0]
    v_body = torch.einsum("btij,btj->bti", Rt, v_palm)
    axis = torch.zeros_like(f["normal"])
    axis[..., 2] = 1.0                                             # palm normal in its own frame
    extras = torch.cat([torch.asinh(omega), torch.asinh(v_body), f["fingers"], f["hand_id"]], -1)
    return local_tips, axis, extras                                # extras: 8
