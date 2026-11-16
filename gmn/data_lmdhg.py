"""LMDHG (Boulahia et al., 2017) under its standard protocol.

* Ground-truth segments only (no motion-energy re-segmentation, no KMeans
  pseudo-labels -- both used in v1 and both non-standard).
* Sequence split used in the literature: sequences 1-35 for training and
  36-50 for testing (Boulahia et al., 2017; Lupinetti et al., 2020). We hold
  out training sequences 31-35 for validation (early stopping, temperature,
  deferral threshold). Test sequences are never used for any choice.
* 13 gesture classes + idle (14 classes) by default; ``include_idle=False``
  gives the 13-class variant. Report which one you use.

VERIFY ON FIRST RUN: the .mat variable names below follow the public
release we know of ('skeleton', 'Anotations', 'labels'). ``inspect_mat``
prints what a file actually contains; adapt ``_read`` if they differ.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from .data_leap import resample

CLASS_NAMES = ["POINT_TO", "CATCH", "SHAKE_2H", "CATCH_2H", "SHAKE_DOWN", "SHAKE",
               "DRAW_C", "POINT_TO_2H", "ZOOM", "SCROLL", "DRAW_LINE", "SLICE", "ROTATE", "IDLE"]
N_JOINTS = 46


def inspect_mat(path):
    from scipy.io import loadmat
    m = loadmat(path)
    return {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in m.items() if not k.startswith("__")}


def _norm_label(s: str) -> str:
    s = re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")
    aliases = {"REPOS": "IDLE", "NOISE": "IDLE", "REST": "IDLE", "POINT_TO_WITH_TWO_HANDS": "POINT_TO_2H",
               "SHAKE_WITH_TWO_HANDS": "SHAKE_2H", "CATCH_WITH_TWO_HANDS": "CATCH_2H",
               "POINTTO": "POINT_TO", "SHAKEDOWN": "SHAKE_DOWN", "DRAWC": "DRAW_C",
               "DRAWLINE": "DRAW_LINE"}
    return aliases.get(s, s)


def _read_txt_skeleton(path):
    """Skeleton from the .txt export: one frame per line with 46*3 numbers;
    lines starting with '#' are markers/headers and are skipped."""
    frames = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) >= N_JOINTS * 3:
                try:
                    frames.append([float(v) for v in vals[:N_JOINTS * 3]])
                except ValueError:
                    continue
    return np.asarray(frames, np.float32) if frames else None


def _skeleton_to_array(skel):
    """Accept a numeric (frames x 138) matrix or a MATLAB cell array with one
    entry per frame (as in the public LMDHG .mat files). Empty / short cells
    (no hand tracked) become zeros and are forward-filled later by _clean."""
    skel = np.asarray(skel)
    if skel.dtype != object:
        return skel.astype(np.float32).reshape(len(skel), N_JOINTS, 3)
    frames = []
    for e in skel.ravel():
        v = np.asarray(e, dtype=np.float64).ravel() if np.size(e) else np.zeros(0)
        if v.size < N_JOINTS * 3:
            v = np.concatenate([v, np.zeros(N_JOINTS * 3 - v.size)])
        frames.append(v[:N_JOINTS * 3])
    return np.asarray(frames, np.float32).reshape(len(frames), N_JOINTS, 3)


def _first(m, names):
    for n in names:
        for k in m:
            if k.lower() == n.lower():
                return m[k]
    return None


def _read(path):
    from scipy.io import loadmat
    m = loadmat(path)
    skel = _first(m, ["skeleton", "skel", "data"])
    ann = _first(m, ["Anotations", "Annotations", "annotation", "Anotation"])
    lab = _first(m, ["labels", "label", "gestures"])
    if skel is None:                                   # fall back to the .txt twin
        txt = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt):
            skel = _read_txt_skeleton(txt)
    if skel is None or ann is None or lab is None:
        raise KeyError(f"{path}: unexpected variables {inspect_mat(path)} -- send this message")
    skel = _skeleton_to_array(skel)
    ann = np.asarray(ann).reshape(-1, 2).astype(int)                    # 1-based [start, end]
    labels = [_norm_label(np.asarray(l).squeeze()) for l in np.asarray(lab).ravel()]
    if len(labels) != len(ann):
        raise ValueError(f"{path}: {len(ann)} annotations but {len(labels)} labels")
    return skel, ann, labels


def _clean(skel):
    """Forward-fill hands that are not tracked (all-zero joints) and return a
    per-frame visibility flag per hand, so that tracking drop-outs do not
    appear to the kinematic layer as huge velocities."""
    skel = skel.copy()
    vis = np.zeros((len(skel), 2), np.float32)
    for h in range(2):
        js = slice(23 * h, 23 * (h + 1))
        ok = np.abs(skel[:, js]).sum((1, 2)) > 0
        vis[:, h] = ok
        if ok.any():
            idx = np.where(ok, np.arange(len(ok)), -1)
            np.maximum.accumulate(idx, out=idx)
            first = np.argmax(ok)
            idx[idx < 0] = first
            skel[:, js] = skel[idx, js]
    return skel, vis


def _normalise(seg):
    """Per-sample: translate by the first-frame centroid, isotropic scale by
    the mean joint-to-centroid distance (no cross-sample statistics)."""
    c = seg[0].mean(0, keepdims=True)
    s = np.linalg.norm(seg - seg.mean(1, keepdims=True), axis=-1).mean()
    return (seg - c) / (s if s > 1e-6 else 1.0)


def _files(root):
    files = sorted(glob.glob(os.path.join(root, "**", "DataFile*.mat"), recursive=True),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if not files:
        raise FileNotFoundError(f"No DataFile*.mat under {root}")
    return files


def discover_labels(root):
    """The label vocabulary actually used in the files (codes differ between
    releases, e.g. French names or single letters such as 'C'). Gesture
    classes are sorted; the idle class (REPOS/NOISE/IDLE) is last."""
    from collections import Counter
    cnt = Counter()
    for f in _files(root):
        _, _, labels = _read(f)
        cnt.update(labels)
    gestures = sorted(l for l in cnt if l != "IDLE")
    return gestures + (["IDLE"] if "IDLE" in cnt else []), dict(cnt)


def load_lmdhg_segments(root, T=100, include_idle=True, classes=None):
    if classes is None:
        classes, _ = discover_labels(root)
    names = [c for c in classes if include_idle or c != "IDLE"]
    X, V, y, seq = [], [], [], []
    for f in _files(root):
        sid = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
        skel, ann, labels = _read(f)
        skel, vis = _clean(skel)
        for (s, e), lab in zip(ann, labels):
            if lab not in names:
                continue                                  # only IDLE when include_idle=False
            seg = skel[max(s - 1, 0):e]
            if len(seg) < 3:
                continue
            X.append(resample(_normalise(seg).reshape(len(seg), -1), T))
            V.append(resample(vis[max(s - 1, 0):e], T))
            y.append(names.index(lab))
            seq.append(sid)
    return dict(X=np.stack(X), vis=np.stack(V), y=np.array(y), seq=np.array(seq), classes=names)


def lmdhg_split(seq):
    return dict(fold="lmdhg", train=[s for s in range(1, 31)], val=[s for s in range(31, 36)],
                test=[s for s in range(36, 51)])


def load_lmdhg_streams(root, classes):
    """Full sequences with per-frame ground truth for the unsegmented
    protocol (frames outside any annotation are labelled IDLE)."""
    out = {}
    idle = classes.index("IDLE") if "IDLE" in classes else None
    for f in _files(root):
        sid = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
        skel, ann, labels = _read(f)
        skel, vis = _clean(skel)
        gt = np.full(len(skel), -1 if idle is None else idle)
        for (s, e), lab in zip(ann, labels):
            if lab in classes:
                gt[max(s - 1, 0):e] = classes.index(lab)
        out[sid] = dict(skel=skel, vis=vis, gt=gt)
    return out
