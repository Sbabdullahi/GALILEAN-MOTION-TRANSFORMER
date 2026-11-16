"""LeapGestureDB loading, caching, subject-wise splitting and preprocessing.

Design rules (each one prevents a common evaluation error):
  * Parsing is cached once to ``.npz`` so every model sees byte-identical data.
  * All splits are made over SUBJECTS and written to JSON split files that are
    released with the paper; the same files are used by every model.
  * Preprocessing is per-sample and geometric (translation to the first palm
    position, isotropic scaling by hand span), so it needs no statistics from
    other samples and cannot leak across splits. Any statistic that *is*
    fitted (e.g. robust standardisation used by the flat-input baselines) is
    fitted on the training subjects of the fold only, inside the model.
  * Per-axis scaling (RobustScaler) and per-axis IQR clipping used in v1 are
    removed: they distort 3-D geometry (different scale per axis), which
    breaks any kinematic or rotational interpretation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from .constants import (GESTURE_MAPPING, INPUT_DIM, SEQ_LENGTH, SL_PALM,
                        SL_TIPS, IDX_HAND_ID)

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_TRIPLE = r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)"


# ─────────────────────────────────────────────────────────────────────────────
# Parsing (format of the LeapGestureDB .txt export; logic kept from v1)
# ─────────────────────────────────────────────────────────────────────────────
def _xyz(line: str, keyword: str):
    m = re.search(re.escape(keyword) + r"\s*" + _TRIPLE, line, re.IGNORECASE)
    return [float(m.group(i)) for i in (1, 2, 3)] if m else None


def _first_triple(line: str):
    m = re.search(_TRIPLE, line)
    return [float(m.group(i)) for i in (1, 2, 3)] if m else None


def parse_leap_file(filepath: str):
    """Return (frames x 26) float32 array of the *raw* recording (no resampling),
    plus the number of frames in which a field was missing."""
    frames, frame, in_frame, missing = [], {}, False, 0

    def _vec(d):
        nonlocal missing
        out = [d.get("hand_id", 0.0), d.get("finger_count", 0.0)]
        for k in ("hand_dir", "palm_pos", "palm_norm", "thumb", "index",
                  "middle", "ring", "pinky"):
            v = d.get(k)
            if v is None:
                missing += 1
                v = [np.nan, np.nan, np.nan]
            out += v
        return out

    with open(filepath, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Frame.id:"):
                if in_frame and frame:
                    frames.append(_vec(frame))
                frame, in_frame = {}, False
                hm = re.search(r"Hand_number:\s*(\d+)", line)
                in_frame = bool(hm and int(hm.group(1)) > 0)
                continue
            if not in_frame:
                continue
            if "hand_Id_type" in line or "hand direction" in line.lower():
                im = re.search(r"hand_Id_type:\s*(\d+)", line, re.I)
                fm = re.search(r"finger'?s?_number:?\s*(\d+)", line, re.I)
                frame["hand_id"] = float(im.group(1)) if im else 0.0
                frame["finger_count"] = float(fm.group(1)) if fm else 0.0
                frame["hand_dir"] = _xyz(line, "hand direction:")
                frame["palm_pos"] = _xyz(line, "Palm position:")
                frame["palm_norm"] = _xyz(line, "Palm normal:")
            elif "TYPE_THUMB" in line:
                frame["thumb"] = _first_triple(line)
            elif "TYPE_INDEX" in line:
                frame["index"] = _first_triple(line)
            elif "TYPE_MIDDLE" in line:
                frame["middle"] = _first_triple(line)
            elif "TYPE_RING" in line:
                frame["ring"] = _first_triple(line)
            elif "TYPE_PINKY" in line:
                frame["pinky"] = _first_triple(line)
    if in_frame and frame:
        frames.append(_vec(frame))
    if not frames:
        return None, missing
    return np.asarray(frames, dtype=np.float32), missing


def _fill_nan_forward_backward(seq: np.ndarray) -> np.ndarray:
    """Fill missing values per feature by carrying the last valid value
    forward (then backward for leading gaps). Counted and reported."""
    seq = seq.copy()
    for j in range(seq.shape[1]):
        col = seq[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            idx = np.where(~bad, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[:] = col[idx]
            first = np.argmax(~bad)
            col[:first] = col[first]
    return seq


def resample(seq: np.ndarray, target: int = SEQ_LENGTH) -> np.ndarray:
    """Linear-interpolation resampling to a fixed number of frames.
    (v1 used index sub-sampling for long and last-frame padding for short
    sequences, which changes the apparent speed differently per sample.)"""
    n = len(seq)
    if n == target:
        return seq.astype(np.float32)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target)
    return np.stack([np.interp(dst, src, seq[:, j]) for j in range(seq.shape[1])],
                    axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset cache
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LeapData:
    X: np.ndarray          # (N, T, 26) resampled, *unnormalised* (mm / unit vectors)
    y: np.ndarray          # (N,)
    subject: np.ndarray    # (N,) subject id
    orig_len: np.ndarray   # (N,) frames in the raw recording
    files: np.ndarray      # (N,) relative path
    report: dict           # parsing report (errors, missing values, ...)


def load_leapgesturedb(data_path: str, cache: str | None = None,
                       seq_length: int = SEQ_LENGTH, expect_n: int | None = None) -> LeapData:
    cache = cache or os.path.join(data_path, f"_gmn_cache_T{seq_length}.npz")
    if os.path.exists(cache) and expect_n is not None:
        z = np.load(cache, allow_pickle=True)
        if len(z["y"]) == expect_n:                  # validated cache: no need to touch raw files
            return LeapData(z["X"], z["y"], z["subject"], z["orig_len"], z["files"],
                            json.loads(str(z["report"])))
    n_txt = sum(f.lower().endswith(".txt") for _, _, fs in os.walk(data_path) for f in fs)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if len(z["y"]) == n_txt:
            return LeapData(z["X"], z["y"], z["subject"], z["orig_len"], z["files"],
                            json.loads(str(z["report"])))
        print(f"WARNING: cache has {len(z['y'])} samples but {n_txt} .txt files are visible "
              f"now -> re-parsing. (On Google Drive this usually means the folder listing was "
              f"incomplete; copy the dataset to local disk first.)")
    if expect_n is not None and n_txt != expect_n:
        raise RuntimeError(f"{n_txt} .txt files visible under {data_path}, expected {expect_n}. "
                           "The listing is incomplete (Drive not fully synced?) -- copy the data "
                           "to local disk and check the file count before running.")

    subjects = sorted(d for d in os.listdir(data_path)
                      if os.path.isdir(os.path.join(data_path, d)))
    X, y, subj, lens, files = [], [], [], [], []
    report = dict(n_subject_dirs=len(subjects), parse_failures=[],
                  unmatched_files=[], frames_with_missing_fields=0)
    # Subject id = position of the subject folder in sorted order (folder
    # names such as "1intissar", "2kamel" are recorded in the report). Files
    # are found recursively, and the gesture label is the last "G<k>" token
    # in the file's relative path (file name or an enclosing folder).
    report["subject_folders"] = subjects
    gtok = re.compile(r"(?<![A-Za-z])G(\d{1,2})(?!\d)", re.IGNORECASE)
    for si, sdir in enumerate(subjects):
        sid = si
        sroot = os.path.join(data_path, sdir)
        txts = sorted(os.path.relpath(os.path.join(dp, f), data_path)
                      for dp, _, fs in os.walk(sroot) for f in fs if f.lower().endswith(".txt"))
        for rel in txts:
            parts = rel.split(os.sep)[1:]                     # skip the subject folder itself
            key = None
            for part in reversed(parts):
                gm = gtok.findall(os.path.splitext(part)[0])
                if gm:
                    key = f"G{int(gm[-1])}"
                    break
            if key not in GESTURE_MAPPING:
                report["unmatched_files"].append(rel)
                continue
            seq, missing = parse_leap_file(os.path.join(data_path, rel))
            if seq is None:
                report["parse_failures"].append(rel)
                continue
            report["frames_with_missing_fields"] += int(missing)
            seq = _fill_nan_forward_backward(seq)
            X.append(resample(seq, seq_length))
            y.append(GESTURE_MAPPING[key])
            subj.append(sid)
            lens.append(len(seq))
            files.append(rel)
    if not X:
        raise ValueError(f"No samples parsed under {data_path}")
    data = LeapData(np.stack(X), np.asarray(y, np.int64), np.asarray(subj),
                    np.asarray(lens), np.asarray(files), report)
    report.update(n_samples=len(y), n_subjects=int(len(np.unique(subj))),
                  class_counts=np.bincount(data.y, minlength=11).tolist(),
                  sha1_X=hashlib.sha1(data.X.tobytes()).hexdigest())
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    np.savez_compressed(cache, X=data.X, y=data.y, subject=data.subject,
                        orig_len=data.orig_len, files=data.files,
                        report=json.dumps(report))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample geometric normalisation (no fitted statistics -> no leakage)
# ─────────────────────────────────────────────────────────────────────────────
def geometric_normalise(X: np.ndarray, drop_hand_id: bool = True) -> np.ndarray:
    """Translate all positions so the palm starts at the origin and scale them
    isotropically by the sample's median palm-to-fingertip span. Directions
    and normals (unit vectors) are left untouched, and 3-D geometry is
    preserved (same scale on all axes).

    The hand-id field is zeroed by default: in Leap SDK exports the hand id
    can be a tracking counter that correlates with recording order, i.e. a
    potential shortcut. Re-enable with ``drop_hand_id=False`` and report it
    as an ablation if you can confirm it only encodes left/right."""
    X = X.astype(np.float32).copy()
    palm0 = X[:, :1, SL_PALM]                                   # (N,1,3)
    tips = X[:, :, SL_TIPS].reshape(len(X), X.shape[1], 5, 3)
    span = np.linalg.norm(tips - X[:, :, None, SL_PALM], axis=-1).mean(-1)
    span = np.median(span, axis=1)                               # (N,)
    span = np.where(span > 1e-3, span, 1.0)[:, None, None]
    X[:, :, SL_PALM] = (X[:, :, SL_PALM] - palm0) / span
    tips = (tips - palm0[:, :, None, :]) / span[..., None]
    X[:, :, SL_TIPS] = tips.reshape(len(X), X.shape[1], 15)
    X[:, :, 1] = X[:, :, 1] / 5.0                                # finger count in [0,1]
    if drop_hand_id:
        X[:, :, IDX_HAND_ID] = 0.0
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Subject-wise splits (written to disk and released)
# ─────────────────────────────────────────────────────────────────────────────
def make_cv_splits(subjects: np.ndarray, n_folds: int = 5, val_frac: float = 0.1,
                   seed: int = 2024) -> list[dict]:
    """Subject-wise K-fold CV. In every fold the test subjects are disjoint
    from train/val subjects; validation subjects are drawn from the
    non-test subjects and are used ONLY for early stopping, temperature
    fitting and choosing the deferral threshold."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(np.unique(subjects)))
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, n_folds)
    splits = []
    for k in range(n_folds):
        test = np.sort(folds[k])
        rest = np.array([s for s in perm if s not in set(test)])
        rest = rng.permutation(rest)
        n_val = max(1, int(round(len(rest) * val_frac)))
        splits.append(dict(fold=k, test=test.tolist(),
                           val=np.sort(rest[:n_val]).tolist(),
                           train=np.sort(rest[n_val:]).tolist()))
    return splits


def make_legacy_split(subjects: np.ndarray) -> dict:
    """The single split used in v1 (87/9/24 subjects), kept so
    the v1 numbers can be compared. Not used for the main results."""
    from sklearn.model_selection import train_test_split
    uniq = np.unique(subjects)
    n = len(uniq)
    tv, te = train_test_split(uniq, test_size=max(1, int(n * .2)), random_state=42)
    tr, va = train_test_split(tv, test_size=max(1, int(len(tv) * .1)), random_state=42)
    return dict(fold="legacy", train=sorted(map(int, tr)), val=sorted(map(int, va)),
                test=sorted(map(int, te)))


def check_split(split: dict, subjects: np.ndarray) -> None:
    tr, va, te = map(set, (split["train"], split["val"], split["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te), "subject leakage"
    missing = set(np.unique(subjects).tolist()) - (tr | va | te)
    assert not missing, f"subjects not assigned to any split: {sorted(missing)[:10]}"


def indices(split: dict, subjects: np.ndarray):
    return {part: np.where(np.isin(subjects, split[part]))[0]
            for part in ("train", "val", "test")}


def deduplicate(data: LeapData):
    """Remove byte-identical recordings and merge subject folders that share
    them. LeapGestureDB contains identical recordings stored under different
    subject folders; with a subject-wise split such copies can land in both
    training and test (leakage). We (i) keep one copy of every identical
    group (all copies are dropped if they carry different labels) and
    (ii) merge subject folders connected by a shared recording into one
    split group, so a person (or a copied folder) never appears on both
    sides of a split. Returns the filtered data, the split-group id of every
    kept sample, and a report."""
    flat = data.X.reshape(len(data.X), -1)
    _, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    parent = {int(s): int(s) for s in np.unique(data.subject)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    keep = np.ones(len(inv), bool)
    n_conflict = 0
    for g in np.unique(inv):
        m = np.where(inv == g)[0]
        if len(m) < 2:
            continue
        subs = sorted(set(int(x) for x in data.subject[m]))
        for s2 in subs[1:]:
            parent[find(s2)] = find(subs[0])
        if len(set(data.y[m].tolist())) > 1:
            keep[m] = False
            n_conflict += 1
        else:
            keep[m[1:]] = False
    group = np.array([find(int(s)) for s in data.subject])
    merged = {}
    for s in np.unique(data.subject):
        merged.setdefault(find(int(s)), []).append(int(s))
    merged = {k: v for k, v in merged.items() if len(v) > 1}
    folders = data.report.get("subject_folders")
    rep = dict(removed_recordings=int((~keep).sum()), conflicting_label_groups=n_conflict,
               merged_subject_groups=[[folders[i] if folders else i for i in v] for v in merged.values()],
               n_split_groups=int(len(np.unique(group))), n_kept=int(keep.sum()))
    kept = LeapData(data.X[keep], data.y[keep], data.subject[keep], data.orig_len[keep],
                    data.files[keep], {**data.report, "dedup": rep})
    return kept, group[keep], rep
