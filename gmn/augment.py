"""Training-time augmentation, applied identically to every model.

All operations act on geometric-normalised (B, T, 26) tensors on the GPU and
preserve 3-D geometry (unit vectors stay unit vectors).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import POSITION_IDX, SL_PALM, SL_TIPS
from .geometry import random_rotation_about, rotate_sample


@dataclass
class AugConfig:
    time_shift: int = 5            # +/- frames (temporal jitter)
    time_scale: tuple = (0.9, 1.1) # speed variation
    noise_std: float = 0.01        # hand-span units, positions only
    translate: float = 0.10        # hand-span units
    rotate_deg: float = 0.0        # random rotation about the sensor's vertical (y) axis
    p: float = 0.8                 # probability of augmenting a sample


def _time_warp(x, scale):
    B, T, C = x.shape
    base = torch.linspace(-1, 1, T, device=x.device)
    grid_t = (base[None, :] * scale[:, None]).clamp(-1, 1)        # (B,T)
    grid = torch.stack([grid_t, torch.zeros_like(grid_t)], -1).unsqueeze(1)  # (B,1,T,2)
    out = F.grid_sample(x.transpose(1, 2).unsqueeze(2), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.squeeze(2).transpose(1, 2)


def _time_shift(x, shift):
    B, T, C = x.shape
    idx = (torch.arange(T, device=x.device)[None, :] - shift[:, None]).clamp(0, T - 1)
    return torch.gather(x, 1, idx[..., None].expand(B, T, C))


def augment(x: torch.Tensor, cfg: AugConfig, gen: torch.Generator | None = None) -> torch.Tensor:
    B = x.shape[0]
    dev = x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    apply = (r(B) < cfg.p).float()[:, None, None]
    out = x
    if cfg.time_scale:
        lo, hi = cfg.time_scale
        out = _time_warp(out, lo + (hi - lo) * r(B))
    if cfg.time_shift:
        sh = torch.randint(-cfg.time_shift, cfg.time_shift + 1, (B,), device=dev, generator=gen)
        out = _time_shift(out, sh)
    if cfg.rotate_deg:
        ang = (2 * r(B) - 1) * torch.deg2rad(torch.tensor(cfg.rotate_deg, device=dev))
        out = rotate_sample(out, random_rotation_about(torch.tensor([0., 1., 0.], device=dev), ang))
    out = out.clone()
    if cfg.translate:
        t = (2 * r(B, 1, 3) - 1) * cfg.translate
        out[:, :, SL_PALM] += t
        out[:, :, SL_TIPS] += t.repeat(1, 1, 5)
    if cfg.noise_std:
        n = torch.randn(out[:, :, POSITION_IDX].shape, device=dev, generator=gen) * cfg.noise_std
        out[:, :, POSITION_IDX] += n
    return apply * out + (1 - apply) * x


def augment_positions_only(x: torch.Tensor, n_pos: int, cfg: AugConfig,
                           gen: torch.Generator | None = None) -> torch.Tensor:
    """Generic version for joint-only data (e.g. LMDHG): time warp, time
    shift, translation and noise on the first ``n_pos`` (= J*3) channels."""
    B = x.shape[0]
    dev = x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    apply = (r(B) < cfg.p).float()[:, None, None]
    out = _time_warp(x, cfg.time_scale[0] + (cfg.time_scale[1] - cfg.time_scale[0]) * r(B))
    out = _time_shift(out, torch.randint(-cfg.time_shift, cfg.time_shift + 1, (B,), device=dev, generator=gen))
    out = out.clone()
    J = n_pos // 3
    t = (2 * r(B, 1, 3) - 1) * cfg.translate
    out[..., :n_pos] += t.repeat(1, 1, J)
    out[..., :n_pos] += torch.randn(out[..., :n_pos].shape, device=dev, generator=gen) * cfg.noise_std
    return apply * out + (1 - apply) * x
