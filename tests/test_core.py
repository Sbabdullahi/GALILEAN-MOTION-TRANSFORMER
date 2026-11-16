"""Unit tests for the properties. Run: pytest -q tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from gmn.data_leap import make_cv_splits, check_split, geometric_normalise
from gmn.geometry import (local_poly_operators, palm_frame, palm_stream, random_rotation_about,
                          rotate_sample, rotmat_to_quat)
from gmn.layers import GalileanFrameConv
from gmn.models import MODEL_REGISTRY, GMNv1, build_model, count_parameters
from gmn.metrics import choose_threshold, apply_deferral, fit_temperature

torch.manual_seed(0)
T, dt = 100, 1 / 99


def _random_sequence(B=4):
    x = torch.randn(B, T, 26) * 0.3
    x[:, :, 2:5] = torch.nn.functional.normalize(torch.randn(B, T, 3) + torch.tensor([0, 0, -3.]), dim=-1)
    x[:, :, 8:11] = torch.nn.functional.normalize(torch.randn(B, T, 3) + torch.tensor([0, -3., 0]), dim=-1)
    return x


# ── Galilean frame convolution ────────────────────────────────────────────
@pytest.mark.parametrize("K", [3, 5, 7, 9])
def test_residual_stream_galilean_invariant(K):
    """Residual stream is exactly invariant to x -> x + c + u t (interior)."""
    layer = GalileanFrameConv(6, 16, K, dt, frame_order=1, readout=False).double()
    pos = torch.randn(2, T, 6, 3, dtype=torch.float64)
    t = torch.arange(T, dtype=torch.float64)[None, :, None, None] * dt
    c, u = torch.randn(2, 1, 6, 3, dtype=torch.float64), torch.randn(2, 1, 6, 3, dtype=torch.float64)
    axis = torch.zeros(2, T, 3, dtype=torch.float64)
    h0, _ = layer(pos, axis)
    h1, _ = layer(pos + c + u * t, axis)
    h = K // 2
    assert torch.allclose(h0[:, h:-h], h1[:, h:-h], atol=1e-8)


def test_order2_invariant_to_uniform_acceleration():
    layer = GalileanFrameConv(6, 16, 7, dt, frame_order=2, readout=False).double()
    pos = torch.randn(1, T, 6, 3, dtype=torch.float64)
    t = torch.arange(T, dtype=torch.float64)[None, :, None, None] * dt
    g = torch.randn(1, 1, 6, 3, dtype=torch.float64)
    axis = torch.zeros(1, T, 3, dtype=torch.float64)
    assert torch.allclose(layer(pos, axis)[0][:, 3:-3], layer(pos + 0.5 * g * t ** 2, axis)[0][:, 3:-3], atol=1e-6)


def test_v1_layer_is_bias_only():
    """Documents that in v1 the 'kinematic' term is a
    constant, so changing the input never interacts with vel/acc."""
    from gmn.models import _V1GalileanConv1D
    l = _V1GalileanConv1D(4, 4, 5, pad=2)
    x1, x2 = torch.randn(1, 4, 20), torch.randn(1, 4, 20)
    with torch.no_grad():
        d_with = l(x1)[0] - l(x2)[0]
        l.vel.zero_(); l.acc.zero_()
        d_without = l(x1)[0] - l(x2)[0]
    assert torch.allclose(d_with, d_without, atol=1e-5)


def test_velocity_estimates_are_physical():
    """LS estimates recover the true velocity / acceleration of a parabola."""
    layer = GalileanFrameConv(1, 4, 9, dt).double()
    t = torch.arange(T, dtype=torch.float64) * dt
    v0, a0 = torch.tensor([1.5, -2.0, 0.3], dtype=torch.float64), torch.tensor([4., 0., -1.], dtype=torch.float64)
    pos = (v0 * t[:, None] + 0.5 * a0 * t[:, None] ** 2)[None, :, None, :]
    v, a = layer.kinematics(pos)
    mid = slice(10, -10)
    assert torch.allclose(a[0, mid, 0], a0.expand(80, 3), atol=1e-6)
    assert torch.allclose(v[0, mid, 0], (v0 + a0 * t[:, None])[mid], atol=1e-6)


def test_chirality_separates_cw_ccw():
    layer = GalileanFrameConv(1, 4, 9, dt).double()
    t = torch.arange(T, dtype=torch.float64) * dt * 2 * np.pi
    circ = lambda s: torch.stack([torch.cos(t), torch.zeros_like(t), s * torch.sin(t)], -1)[None, :, None]
    axis = torch.tensor([0, 1., 0], dtype=torch.float64).expand(1, T, 3)
    _, k_cw = layer(circ(1), axis)
    _, k_ccw = layer(circ(-1), axis)
    assert k_cw["chirality"][0, 10:-10].mean() * k_ccw["chirality"][0, 10:-10].mean() < 0


def test_poly_projector_annihilates_polynomials():
    A, P, R = local_poly_operators(7, dt, 1)
    assert torch.allclose(R @ A, torch.zeros_like(A), atol=1e-6)


# ── rotations ────────────────────────────────────────────────────────────
def test_palm_frame_is_rotation():
    R = palm_frame(torch.randn(50, 3), torch.randn(50, 3))
    I = torch.eye(3).expand(50, 3, 3)
    assert torch.allclose(R.transpose(-1, -2) @ R, I, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(50), atol=1e-5)


def test_quaternion_roundtrip():
    R = palm_frame(torch.randn(100, 3), torch.randn(100, 3))
    q = rotmat_to_quat(R)
    w, x, y, z = q.unbind(-1)
    R2 = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    assert torch.allclose(R, R2, atol=1e-5)


def test_palm_stream_rotation_invariant():
    x = _random_sequence()
    Rg = random_rotation_about(torch.randn(3), torch.rand(4) * 6.28)
    a = palm_stream(x, dt)
    b = palm_stream(rotate_sample(x, Rg), dt)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-4)


def test_palm_only_model_rotation_invariant():
    m = build_model("gmn_palm_only").eval()
    x = _random_sequence()
    Rg = random_rotation_about(torch.randn(3), torch.rand(4) * 6.28)
    with torch.no_grad():
        assert torch.allclose(m(x), m(rotate_sample(x, Rg)), atol=1e-3)


# ── models & protocol ────────────────────────────────────────────────────
@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_models_forward_backward(name):
    m = build_model(name)
    if name.startswith("lmdhg"):
        x, C = torch.randn(3, T, 46 * 3 + 2), 14
    else:
        x, C = _random_sequence(3), 11
    m.set_normaliser(x)
    out = m(x)
    assert out.shape == (3, C)
    out.sum().backward()


def test_v1_port_matches_released_code():
    """The port has exactly the parameters of the released v1 main.py.
    The released v1 code has 10,696,751 trainable parameters."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legacy_main", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy", "main.py"))
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    assert count_parameters(GMNv1()) == count_parameters(legacy.GalileanMotionTransformer()) == 10_696_751


def test_splits_are_subject_disjoint():
    subjects = np.repeat(np.arange(1, 121), 55)
    for s in make_cv_splits(subjects, 5):
        check_split(s, subjects)
        assert len(s["test"]) == 24


def test_geometric_normalise_no_cross_sample_dependency():
    X = np.random.randn(10, T, 26).astype(np.float32) * 50
    a = geometric_normalise(X)
    b = geometric_normalise(X[:3])
    assert np.allclose(a[:3], b)


def test_deferral_threshold_respects_target_on_val():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 11, 2000)
    conf = rng.rand(2000)
    pred = np.where(rng.rand(2000) < conf, y, (y + 1) % 11)
    thr = choose_threshold(y, pred, conf, 0.05)
    r = apply_deferral(y, pred, conf, thr)
    assert r["accepted_error"] <= 0.05 + 1e-12


def test_temperature_recovers_scale():
    rng = np.random.RandomState(0)
    logits = rng.randn(5000, 11) * 3
    y = np.array([rng.choice(11, p=np.exp(l) / np.exp(l).sum()) for l in logits])
    assert abs(fit_temperature(y, logits * 2.0) - 2.0) < 0.15
