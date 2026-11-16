"""Network layers.

GalileanFrameConv (replaces the v1 layer)
------------------------------------------------------
v1 computed  h_t = sum_k W_k (x_{t-k} + s_k k dt + 1/2 a_k (k dt)^2) + b,
where s_k, a_k are input-independent parameters. The added term is a
constant, so the layer was a convolution with a re-parameterised bias.

v2 makes the kinematics *input-dependent* and *physically identifiable*:
for every frame t and window of K frames of a joint trajectory x (in
hand-span units, on the physical coordinates rather than after a linear
embedding) we estimate by least squares

        x(t + tau) ~ p_t + v_t tau + 1/2 a_t tau^2,

and split the window into two streams:

  1. Co-moving-frame residual  r_t(tau) = x(t+tau) - p_t - v_t tau
     (frame_order=1, a Galilean boost into the local inertial frame) or
     additionally - 1/2 a_t tau^2 (frame_order=2, uniformly accelerating
     frame). A learned kernel W acts on r. Because r is exactly invariant to
     x -> x + c + u t (+ 1/2 g t^2 for order 2), this stream is invariant to
     the Galilean group (exactly on interior frames; tested in
     tests/test_invariance.py). Mathematically this is a convolution whose
     kernel is constrained to the orthogonal complement of low-degree
     polynomials (W_eff = W (I - A P)), i.e. a vanishing-moment constraint;
     we state this equivalence openly in the paper.
  2. Kinematic read-out of (v_t, a_t), their magnitudes and a chirality term
     chi_t = n_t . (v_t x a_t) / (|v_t| |a_t| + eps), the signed sense of
     rotation of the trajectory about the palm normal n_t (which separates
     clockwise from counter-clockwise circles; the v1 discussion identified
     its absence as the main CW/CCW failure).

The classifier as a whole is *not* Galilean-invariant (Move left vs Move
right differ only in velocity); invariance is a property of stream 1, and the
velocity information is passed explicitly and interpretably through stream 2.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import local_poly_operators, windowed_estimates


class GalileanFrameConv(nn.Module):
    def __init__(self, n_joints: int, out_ch: int, K: int, dt: float,
                 frame_order: int = 1, readout: bool = True, chirality: bool = True):
        super().__init__()
        assert K % 2 == 1 and K >= 3
        assert frame_order in (0, 1, 2)
        self.K, self.dt, self.J = K, dt, n_joints
        self.frame_order, self.readout, self.chirality = frame_order, readout, chirality
        _, P2, _ = local_poly_operators(K, dt, 2)
        self.register_buffer("P2", P2)                         # (3, K) -> p, v, a  (float64)
        if frame_order > 0 and K > frame_order + 1:
            _, _, Rf = local_poly_operators(K, dt, frame_order)
        else:
            Rf = torch.eye(K, dtype=torch.float64)
        self.register_buffer("Rf", Rf)                         # (K, K) residual projector
        C = n_joints * 3
        self.weight = nn.Parameter(torch.empty(out_ch, C, K))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        n_read = (8 + int(chirality)) * n_joints
        self.read_proj = nn.Linear(n_read, out_ch) if readout else None

    def effective_kernel(self) -> torch.Tensor:
        """The kernel actually applied to raw coordinates: W (I - A P)^T."""
        return torch.einsum("ock,kj->ocj", self.weight, self.Rf.to(self.weight.dtype))

    def kinematics(self, pos: torch.Tensor):
        """pos: (B, T, J, 3) -> v, a: (B, T, J, 3) least-squares estimates."""
        B, T, J, _ = pos.shape
        x = pos.reshape(B, T, J * 3).transpose(1, 2)            # (B, C, T)
        est = windowed_estimates(x, self.P2.to(x.dtype))                    # (B, 3, C, T)
        v = est[:, 1].transpose(1, 2).reshape(B, T, J, 3)
        a = est[:, 2].transpose(1, 2).reshape(B, T, J, 3)
        return v, a

    def forward(self, pos: torch.Tensor, axis: torch.Tensor):
        """pos: (B, T, J, 3) physical trajectories; axis: (B, T, 3) palm normal.
        Returns (B, T, out_ch) and a dict of interpretable kinematics."""
        B, T, J, _ = pos.shape
        x = pos.reshape(B, T, J * 3).transpose(1, 2)
        xp = F.pad(x, (self.K // 2, self.K // 2), mode="replicate")
        h = F.conv1d(xp, self.effective_kernel(), self.bias)   # invariant stream
        h = h.transpose(1, 2)                                   # (B, T, out)
        kin = {}
        if self.readout:
            v, a = self.kinematics(pos)
            sv, sa = v.norm(dim=-1, keepdim=True), a.norm(dim=-1, keepdim=True)
            feats = [torch.asinh(v), torch.asinh(a), torch.asinh(sv), torch.asinh(sa)]
            if self.chirality:
                cr = torch.cross(v, a, dim=-1)
                chi = (cr * axis.unsqueeze(2)).sum(-1, keepdim=True) / (sv * sa + 1e-6)
                feats.append(chi)
                kin["chirality"] = chi.squeeze(-1)
            r = torch.cat(feats, -1).reshape(B, T, -1)
            h = h + self.read_proj(r)
            kin.update(velocity=v, acceleration=a)
        return h, kin


class StdConvStem(nn.Module):
    """Ablation / control: an ordinary 1-D convolution over the same physical
    channels (optionally concatenated with finite-difference velocity and
    acceleration -- the "explicit feature engineering" control)."""

    def __init__(self, n_in: int, out_ch: int, K: int):
        super().__init__()
        self.conv = nn.Conv1d(n_in, out_ch, K, padding=K // 2, padding_mode="replicate")

    def forward(self, x: torch.Tensor):                         # x: (B, T, C)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that also returns attention maps
    (needed for attention rollout)."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n = self.n1(x)
        a, w = self.attn(n, n, n, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        return x + self.mlp(self.n2(x)), w


class ClassQueryPooling(nn.Module):
    """One learnable query per class attends over the frame sequence; each
    class score is read from its own pooled vector. Replaces the v1
    "cross-modal fusion" (which had a single modality, 10 prompts for 11
    classes and undocumented 256->768 projections). The attention weights
    give a per-class temporal evidence map: which frames support class c."""

    def __init__(self, d, n_classes, heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_classes, d))
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        # class-specific read-out (one weight vector per class): a shared
        # scorer would see near-identical pooled vectors for all classes at
        # initialisation and learn very slowly
        self.w = nn.Parameter(torch.randn(n_classes, d) * d ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, h):
        q = self.q.unsqueeze(0).expand(h.shape[0], -1, -1)
        z, w = self.attn(q, h, h, need_weights=True, average_attn_weights=True)
        logits = (self.norm(z) * self.w).sum(-1) + self.b
        return logits, w                                       # (B,C), (B,C,T)
