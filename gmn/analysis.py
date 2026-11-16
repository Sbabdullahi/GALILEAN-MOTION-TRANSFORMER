"""Post-hoc analyses on trained checkpoints, run over the WHOLE test set
(not hand-picked samples): rotation robustness, integrated gradients with
sanity checks, kinematic read-out summaries, attention rollout, profiling."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from .constants import FEATURE_GROUPS, NUM_CLASSES
from .geometry import random_rotation_about, rotate_sample


# ─────────────────────────────────────────────────────────────────────────────
# Rotation robustness (tests the invariance claim instead of asserting it)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rotation_robustness(model, X, y, angles_deg=(0, 15, 30, 45, 90, 180), axis=(0., 1., 0.)):
    """Accuracy when every test sequence is rotated about the sensor's
    vertical axis (rotation about the sensor origin, palm re-centred by the
    per-sample normalisation being translation-free in the palm stream)."""
    model.eval()
    ax = torch.tensor(axis, device=X.device)
    out = {}
    for a in angles_deg:
        R = random_rotation_about(ax, torch.full((len(X),), np.deg2rad(a), device=X.device))
        Xr = rotate_sample(X, R)
        pred = torch.cat([model(Xr[i:i + 256]).argmax(1) for i in range(0, len(X), 256)])
        out[int(a)] = float((pred.cpu().numpy() == y).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients (Sundararajan et al., 2017) + sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def integrated_gradients(model, x, target, baseline=None, steps=64):
    """x: (B,T,F). Baseline defaults to the 'resting hand': the first frame of
    each sample held constant (a zero vector is not a physically meaningful
    hand). Returns signed attributions (B,T,F) and the completeness gap
    |sum(attr) - (f(x) - f(baseline))| per sample."""
    model.eval()
    if baseline is None:
        baseline = x[:, :1].expand_as(x)
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)[1:]
    grads = torch.zeros_like(x)
    for a in alphas:
        xi = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        out = model(xi).gather(1, target[:, None]).sum()
        grads += torch.autograd.grad(out, xi)[0]
    attr = (x - baseline) * grads / steps
    with torch.no_grad():
        fx = model(x).gather(1, target[:, None]).squeeze(1)
        fb = model(baseline).gather(1, target[:, None]).squeeze(1)
    gap = (attr.sum((1, 2)) - (fx - fb)).abs()
    return attr.detach(), gap, (fx - fb).abs()


def group_attributions(attr: np.ndarray):
    """(N,T,F) signed -> per-group signed sum and absolute sum, (N, G)."""
    names = list(FEATURE_GROUPS)
    signed = np.stack([attr[:, :, FEATURE_GROUPS[g]].sum((1, 2)) for g in names], 1)
    absolute = np.stack([np.abs(attr[:, :, FEATURE_GROUPS[g]]).sum((1, 2)) for g in names], 1)
    return names, signed, absolute


def attribution_study(model, X, y, n_per_class=30, steps=64, seed=0):
    """Class-wise IG over (up to) n_per_class correctly-classified test
    samples per class, plus the model-parameter randomisation sanity check
    of Adebayo et al. (2018): attributions of the trained model should NOT
    be similar to those of a model with randomised weights."""
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 256]).argmax(1) for i in range(0, len(X), 256)]).cpu().numpy()
    sel = []
    for c in range(NUM_CLASSES):
        ok = np.where((y == c) & (pred == c))[0]
        sel += rng.choice(ok, min(n_per_class, len(ok)), replace=False).tolist()
    sel = np.array(sel)
    xs, ts = X[sel], torch.tensor(y[sel], device=X.device)
    attr, gap, delta = [], [], []
    for i in range(0, len(sel), 32):
        a, g, d = integrated_gradients(model, xs[i:i + 32], ts[i:i + 32], steps=steps)
        attr.append(a.cpu().numpy()); gap.append(g.cpu().numpy()); delta.append(d.cpu().numpy())
    attr, gap, delta = np.concatenate(attr), np.concatenate(gap), np.concatenate(delta)

    rand = copy.deepcopy(model)
    for m in rand.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for p in rand.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, std=0.02)
    attr_r = []
    for i in range(0, len(sel), 32):
        attr_r.append(integrated_gradients(rand, xs[i:i + 32], ts[i:i + 32], steps=steps)[0].cpu().numpy())
    attr_r = np.concatenate(attr_r)
    names, signed, absolute = group_attributions(attr)
    _, _, absolute_r = group_attributions(attr_r)
    rho = [spearmanr(absolute[i], absolute_r[i]).correlation for i in range(len(sel))]
    return dict(groups=names, classes=y[sel].tolist(), signed=signed, absolute=absolute,
                completeness_rel_gap=float(np.median(gap / np.maximum(delta, 1e-6))),
                randomisation_spearman_median=float(np.nanmedian(rho)),
                frame_profile=np.abs(attr).sum(2))           # (N,T): when evidence occurs


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic read-outs: are the estimated velocities physically sensible?
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kinematic_summary(model, X, y, scale=9):
    """Mean estimated palm velocity (sensor frame) and mean chirality of the
    index fingertip per class, from the GMN's own Galilean read-out."""
    model.eval()
    vs, chis = [], []
    for i in range(0, len(X), 256):
        _, aux = model(X[i:i + 256], return_aux=True)
        k = aux["kinematics"]["world"][scale]
        vs.append(k["velocity"][:, :, 0].mean(1).cpu().numpy())        # palm (B,3)
        chis.append(k["chirality"][:, :, 2].mean(1).cpu().numpy())      # index tip (B,)
    v, chi = np.concatenate(vs), np.concatenate(chis)
    return dict(palm_velocity_mean=[v[y == c].mean(0).tolist() for c in range(NUM_CLASSES)],
                palm_velocity_std=[v[y == c].std(0).tolist() for c in range(NUM_CLASSES)],
                index_chirality_mean=[float(chi[y == c].mean()) for c in range(NUM_CLASSES)],
                index_chirality_std=[float(chi[y == c].std()) for c in range(NUM_CLASSES)])


# ─────────────────────────────────────────────────────────────────────────────
# Attention rollout (Abnar & Zuidema, 2020)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def attention_rollout(model, x):
    _, aux = model(x, return_aux=True)
    A = None
    for w in aux.get("attentions", []):
        w = 0.5 * w + 0.5 * torch.eye(w.shape[-1], device=w.device)
        w = w / w.sum(-1, keepdim=True)
        A = w if A is None else w @ A
    return A, aux.get("class_attention")


# ─────────────────────────────────────────────────────────────────────────────
# Cost: parameters, FLOPs and latency with an explicit protocol
# ─────────────────────────────────────────────────────────────────────────────
def profile(model, seq_len=100, n_feat=26, device="cpu", batch=1, warmup=20, runs=200):
    from torch.utils.flop_counter import FlopCounterMode
    model = model.to(device).eval()
    x = torch.randn(batch, seq_len, n_feat, device=device)
    with FlopCounterMode(display=False) as fc:      # (no_grad breaks its module tracker)
        model(x)
    flops = fc.get_total_flops()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
    return dict(params=sum(p.numel() for p in model.parameters()), flops=int(flops),
                latency_ms_median=float(np.median(ts)), latency_ms_p95=float(np.percentile(ts, 95)),
                device=device if not device.startswith("cuda") else torch.cuda.get_device_name(0),
                batch=batch, warmup=warmup, runs=runs)
