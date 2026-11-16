"""GMN v2, its ablations, re-implemented baselines and a faithful port of v1.

Every model takes the same input: (B, T, 26) geometric-normalised sequences
from ``data_leap.geometric_normalise``. Models that consume flat features
standardise them with robust statistics fitted on the TRAINING subjects of
the current fold only (``set_normaliser``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import INPUT_DIM, NUM_CLASSES, SEQ_LENGTH
from .geometry import palm_stream, world_stream, finite_differences
from .layers import (ClassQueryPooling, EncoderLayer, GalileanFrameConv,
                     SinusoidalPE, StdConvStem)


class FlatNormaliser(nn.Module):
    """Robust per-feature standardisation, statistics fitted on train only."""

    def __init__(self, dim=INPUT_DIM):
        super().__init__()
        self.register_buffer("med", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))

    def fit(self, X: torch.Tensor):
        flat = X.reshape(-1, X.shape[-1])
        q1, q2, q3 = torch.quantile(flat[:200_000].float(),
                                    torch.tensor([.25, .5, .75], device=X.device), dim=0)
        self.med.copy_(q2)
        self.scale.copy_(torch.where((q3 - q1) > 1e-6, q3 - q1, torch.ones_like(q1)))

    def forward(self, x):
        return (x - self.med) / self.scale


class BaseModel(nn.Module):
    uses_flat_normaliser = False

    def set_normaliser(self, X_train: torch.Tensor):
        if self.uses_flat_normaliser:
            self.norm_in.fit(X_train)


# ─────────────────────────────────────────────────────────────────────────────
# GMN v2
# ─────────────────────────────────────────────────────────────────────────────
class GMN(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES, seq_len=SEQ_LENGTH, d=128,
                 scales=(3, 5, 7, 9), frames=("world", "palm"), stem="galilean",
                 frame_order=1, readout=True, chirality=True, fd_inputs=False,
                 n_layers=4, heads=4, pooling="class_query", dropout=0.1):
        super().__init__()
        self.dt = 1.0 / (seq_len - 1)          # normalised gesture time in [0, 1]
        self.frames, self.stem_type, self.fd_inputs = tuple(frames), stem, fd_inputs
        assert d % len(scales) == 0
        dk = d // len(scales)
        n_joints = {"world": 6, "palm": 5}
        n_extra = {"world": 12, "palm": 8}
        self.stems = nn.ModuleDict()
        self.extra = nn.ModuleDict()
        for fr in self.frames:
            J = n_joints[fr]
            if stem == "galilean":
                blocks = [GalileanFrameConv(J, dk, K, self.dt, frame_order, readout, chirality)
                          for K in scales]
            else:
                n_in = J * 3 * (3 if fd_inputs else 1)
                blocks = [StdConvStem(n_in, dk, K) for K in scales]
            self.stems[fr] = nn.ModuleList(blocks)
            self.extra[fr] = nn.Linear(n_extra[fr], d)
        self.stem_norm = nn.LayerNorm(d)
        self.pe = SinusoidalPE(d)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.pooling = pooling
        if pooling == "class_query":
            self.pool = ClassQueryPooling(d, num_classes, heads, dropout)
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))
        self.drop = nn.Dropout(dropout)

    def _stream(self, x, fr):
        if fr == "world":
            pos, axis, extras = world_stream(x)
        else:
            pos, axis, extras = palm_stream(x, self.dt)
        outs, kin = [], {}
        for blk in self.stems[fr]:
            if self.stem_type == "galilean":
                h, k = blk(pos, axis)
                kin[blk.K] = k
            else:
                B, T, J, _ = pos.shape
                flat = pos.reshape(B, T, J * 3)
                if self.fd_inputs:
                    v, a = finite_differences(flat, self.dt)
                    flat = torch.cat([flat, torch.asinh(v), torch.asinh(a)], -1)
                h = blk(flat)
            outs.append(h)
        return torch.cat(outs, -1) + self.extra[fr](extras), kin

    def forward(self, x, return_aux=False):
        h, kin = 0, {}
        # kinematics involve 1/dt^2 ~ 1e4 factors: keep the physics stem in fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            for fr in self.frames:
                hf, k = self._stream(x.float(), fr)
                h = h + hf
                kin[fr] = k
        h = self.drop(F.gelu(self.stem_norm(h)))
        h = self.pe(h)
        attns = []
        for layer in self.layers:
            h, w = layer(h)
            attns.append(w)
        h = self.final_norm(h)
        if self.pooling == "class_query":
            logits, class_attn = self.pool(h)
        else:
            logits, class_attn = self.head(h.mean(1)), None
        if return_aux:
            return logits, dict(attentions=attns, class_attention=class_attn, kinematics=kin)
        return logits


class GMNJoints(BaseModel):
    """The same GMN design for datasets that provide only joint positions
    (LMDHG: 46 joints, two hands, no hand-orientation vectors). Identical
    width, scales, depth, heads, pooling and training recipe as ``GMN``; the
    only differences are the joint count of the Galilean stem and the absence
    of the palm-frame stream (no direction/normal vectors are provided).
    Input: (B, T, J*3 + 2) = flattened joints + per-hand visibility flags."""

    def __init__(self, num_classes=14, n_joints=46, seq_len=SEQ_LENGTH, d=128,
                 scales=(3, 5, 7, 9), stem="galilean", frame_order=1, readout=True,
                 fd_inputs=False, n_layers=4, heads=4, pooling="class_query", dropout=0.1):
        super().__init__()
        self.J, self.dt, self.stem_type, self.fd_inputs = n_joints, 1.0 / (seq_len - 1), stem, fd_inputs
        dk = d // len(scales)
        if stem == "galilean":
            # chirality needs a palm normal, which LMDHG does not provide
            self.stems = nn.ModuleList([GalileanFrameConv(n_joints, dk, K, self.dt, frame_order,
                                                          readout, chirality=False) for K in scales])
        else:
            n_in = n_joints * 3 * (3 if fd_inputs else 1)
            self.stems = nn.ModuleList([StdConvStem(n_in, dk, K) for K in scales])
        self.vis = nn.Linear(2, d)
        self.stem_norm = nn.LayerNorm(d)
        self.pe = SinusoidalPE(d)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.pooling = pooling
        if pooling == "class_query":
            self.pool = ClassQueryPooling(d, num_classes, heads, dropout)
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_aux=False):
        B, T, _ = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            pos = x[..., : self.J * 3].reshape(B, T, self.J, 3)
            axis = torch.zeros(B, T, 3, device=x.device)
            outs = []
            for blk in self.stems:
                if self.stem_type == "galilean":
                    outs.append(blk(pos, axis)[0])
                else:
                    flat = pos.reshape(B, T, -1)
                    if self.fd_inputs:
                        v, a = finite_differences(flat, self.dt)
                        flat = torch.cat([flat, torch.asinh(v), torch.asinh(a)], -1)
                    outs.append(blk(flat))
            h = torch.cat(outs, -1) + self.vis(x[..., self.J * 3:])
        h = self.pe(self.drop(F.gelu(self.stem_norm(h))))
        attns = []
        for layer in self.layers:
            h, w = layer(h)
            attns.append(w)
        h = self.final_norm(h)
        if self.pooling == "class_query":
            logits, ca = self.pool(h)
        else:
            logits, ca = self.head(h.mean(1)), None
        return (logits, dict(attentions=attns, class_attention=ca)) if return_aux else logits


# ─────────────────────────────────────────────────────────────────────────────
# Re-implemented baselines (same split, same preprocessing, same training)
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTM(BaseModel):
    uses_flat_normaliser = True

    def __init__(self, num_classes=NUM_CLASSES, hidden=128, layers=2, dropout=0.2,
                 input_dim=INPUT_DIM):
        super().__init__()
        self.norm_in = FlatNormaliser(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, num_classes))

    def forward(self, x, return_aux=False):
        h, _ = self.lstm(self.norm_in(x))
        logits = self.head(torch.cat([h.mean(1), h[:, -1]], -1))
        return (logits, {}) if return_aux else logits


class PlainTransformer(BaseModel):
    uses_flat_normaliser = True

    def __init__(self, num_classes=NUM_CLASSES, d=128, n_layers=4, heads=4, dropout=0.1,
                 input_dim=INPUT_DIM):
        super().__init__()
        self.norm_in = FlatNormaliser(input_dim)
        self.inp = nn.Linear(input_dim, d)
        self.pe = SinusoidalPE(d)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x, return_aux=False):
        h = self.pe(self.inp(self.norm_in(x)))
        attns = []
        for layer in self.layers:
            h, w = layer(h)
            attns.append(w)
        logits = self.head(self.norm(h).mean(1))
        return (logits, dict(attentions=attns)) if return_aux else logits


class STGCNLite(BaseModel):
    """Compact ST-GCN (Yan et al., 2018) on a 6-node hand graph: palm + five
    fingertips (the only joints LeapGestureDB provides). Edges: palm-tip and
    neighbouring tips; symmetric normalised adjacency with self loops."""

    def __init__(self, num_classes=NUM_CLASSES, channels=(64, 64, 128, 128), tk=9, dropout=0.2):
        super().__init__()
        A = torch.eye(6)
        for i in range(1, 6):
            A[0, i] = A[i, 0] = 1
        for i in range(1, 5):
            A[i, i + 1] = A[i + 1, i] = 1
        Dm = torch.diag(A.sum(1) ** -0.5)
        self.register_buffer("A", Dm @ A @ Dm)
        layers, c_in = [], 3
        for c in channels:
            layers.append(nn.ModuleDict(dict(
                gcn=nn.Conv2d(c_in, c, 1),
                tcn=nn.Sequential(nn.BatchNorm2d(c), nn.ReLU(),
                                  nn.Conv2d(c, c, (tk, 1), padding=(tk // 2, 0)),
                                  nn.BatchNorm2d(c), nn.Dropout(dropout)),
                res=nn.Conv2d(c_in, c, 1) if c_in != c else nn.Identity())))
            c_in = c
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x, return_aux=False):
        pos, _, _ = world_stream(x)                         # (B,T,6,3)
        h = pos.permute(0, 3, 1, 2)                         # (B,3,T,V)
        for l in self.layers:
            g = torch.einsum("bctv,vw->bctw", l["gcn"](h), self.A)
            h = F.relu(l["tcn"](g) + l["res"](h))
        logits = self.head(h.mean((2, 3)))
        return (logits, {}) if return_aux else logits


# ─────────────────────────────────────────────────────────────────────────────
# Faithful port of the v1 architecture (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
class _V1GalileanConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, pad=0, dt=0.01, reg=0.01):
        super().__init__()
        self.ks, self.pad, self.reg, self.in_ch = ks, pad, reg, in_ch
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, ks))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.vel = nn.Parameter(torch.randn(in_ch, ks) * 0.1)
        self.acc = nn.Parameter(torch.randn(in_ch, ks) * 0.01)
        self.register_buffer("t_off", torch.arange(ks).float() * dt)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode="replicate")
        xu = x.unfold(2, self.ks, 1)
        vel = self.vel.unsqueeze(0).unsqueeze(2) * self.t_off.view(1, 1, 1, -1)
        acc = .5 * self.acc.unsqueeze(0).unsqueeze(2) * (self.t_off ** 2).view(1, 1, 1, -1)
        xt = xu + vel + acc                    # input-independent offset
        B = xt.size(0)
        xt = xt.permute(0, 2, 1, 3).reshape(B, -1, self.in_ch * self.ks)
        out = torch.matmul(xt, self.weight.view(self.weight.size(0), -1).t())
        out = out.permute(0, 2, 1) + self.bias.view(1, -1, 1)
        return out, self.reg * (self.vel.pow(2).mean() + self.acc.pow(2).mean())


class _V1ViTBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        n = self.n1(x)
        a, w = self.attn(n, n, n)
        x = x + a
        return x + self.mlp(self.n2(x)), w


class GMNv1(BaseModel):
    """Exactly the architecture in the released v1 main.py (num_joints=9,
    10 prompt tokens, 768-d fusion, sequential Galilean blocks), so its
    parameter count and accuracy can be reported under the new protocol."""
    uses_flat_normaliser = True

    def __init__(self, num_classes=NUM_CLASSES, embed_dim=256, kernel_sizes=(3, 5, 7, 9),
                 num_vit=6, num_heads=8, lang_dim=768, dropout=0.2, num_joints=9,
                 physics_weight=1.0):
        super().__init__()
        self.norm_in = FlatNormaliser()
        self.physics_weight = physics_weight
        self.inp = nn.Linear(INPUT_DIM, embed_dim)
        self.gal = nn.ModuleList([nn.ModuleDict(dict(
            conv=_V1GalileanConv1D(embed_dim, embed_dim, ks, pad=ks // 2),
            norm=nn.LayerNorm(embed_dim), drop=nn.Dropout(dropout))) for ks in kernel_sizes])
        self.quat = nn.Linear(embed_dim, num_joints * 4)
        self.nj = num_joints
        self.fproj = nn.Linear(embed_dim + num_joints * 4, embed_dim)
        self.vit = nn.ModuleList([_V1ViTBlock(embed_dim, num_heads, dropout) for _ in range(num_vit)])
        self.prompt = nn.Parameter(torch.randn(1, 10, lang_dim))
        self.vp = nn.Sequential(nn.Linear(embed_dim, lang_dim), nn.LayerNorm(lang_dim), nn.GELU())
        self.ca = nn.MultiheadAttention(lang_dim, 8, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * lang_dim, lang_dim), nn.Sigmoid())
        self.fnorm = nn.LayerNorm(lang_dim)
        self.head = nn.Sequential(nn.Linear(lang_dim, 512), nn.ReLU(), nn.Dropout(.3),
                                  nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.3),
                                  nn.Linear(256, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self._phys = torch.tensor(0.0)

    def forward(self, x, return_aux=False):
        B = x.size(0)
        xe = self.inp(self.norm_in(x))
        xc, phys = xe.transpose(1, 2), 0.0
        for blk in self.gal:
            xc, p = blk["conv"](xc)
            phys = phys + p
            xc = blk["drop"](F.relu(blk["norm"](xc.transpose(1, 2))).transpose(1, 2))
        q = F.normalize(self.quat(xe).view(B, -1, self.nj, 4), dim=-1).view(B, -1, self.nj * 4)
        h = self.fproj(torch.cat([xc.transpose(1, 2), q], -1))
        attns = []
        for blk in self.vit:
            h, w = blk(h)
            attns.append(w)
        t = self.prompt.expand(B, -1, -1)
        vp = self.vp(h)
        f, _ = self.ca(t, vp, vp)
        g = self.gate(torch.cat([t, f], -1))
        out = self.fnorm(g * f + (1 - g) * t)
        logits = self.head(out.mean(1))
        self._phys = phys
        return (logits, dict(attentions=attns)) if return_aux else logits

    def extra_loss(self):
        return self.physics_weight * self._phys


# ─────────────────────────────────────────────────────────────────────────────
# Registry: every name used in the paper's tables maps to exactly one config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # proposed
    "gmn":               (GMN, dict()),
    # ablations (one change each)
    "gmn_stdconv":       (GMN, dict(stem="std")),                       # w/o Galilean frame
    "gmn_fd_control":    (GMN, dict(stem="std", fd_inputs=True)),       # explicit finite differences
    "gmn_no_readout":    (GMN, dict(readout=False)),                    # residual stream only
    "gmn_no_chirality":  (GMN, dict(chirality=False)),
    "gmn_frame_order0":  (GMN, dict(frame_order=0)),                    # no co-moving frame
    "gmn_frame_order2":  (GMN, dict(frame_order=2)),                    # accelerating frame
    "gmn_world_only":    (GMN, dict(frames=("world",))),                # w/o palm-frame stream
    "gmn_palm_only":     (GMN, dict(frames=("palm",))),                 # rotation-invariant only
    "gmn_meanpool":      (GMN, dict(pooling="mean")),                   # w/o class-query pooling
    "gmn_no_transformer": (GMN, dict(n_layers=0)),
    # baselines
    "bilstm":            (BiLSTM, dict()),
    "transformer":       (PlainTransformer, dict()),
    "stgcn":             (STGCNLite, dict()),
    "gmn_v1":            (GMNv1, dict()),
    # LMDHG (joint-only input, 46 joints + 2 visibility flags)
    "lmdhg_gmn":         (GMNJoints, dict()),
    "lmdhg_fd_control":  (GMNJoints, dict(stem="std", fd_inputs=True)),
    "lmdhg_stdconv":     (GMNJoints, dict(stem="std")),
    "lmdhg_transformer": (PlainTransformer, dict(num_classes=14, input_dim=46 * 3 + 2)),
    "lmdhg_bilstm":      (BiLSTM, dict(num_classes=14, input_dim=46 * 3 + 2)),
}


def build_model(name: str, **overrides) -> BaseModel:
    cls, kw = MODEL_REGISTRY[name]
    return cls(**{**kw, **overrides})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
