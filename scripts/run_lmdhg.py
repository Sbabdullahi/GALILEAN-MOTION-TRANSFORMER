#!/usr/bin/env python
"""LMDHG under the standard protocol (train sequences 1-35, test 36-50),
using the SAME GMN design and training recipe as LeapGestureDB (no focal
loss, no label smoothing, no prototype loss, no SWA, no per-dataset tuning).

Segmented: classify ground-truth segments. Unsegmented: slide a window over
every test sequence and score per-frame predictions against the ground-truth
frame labels (no KMeans pseudo-labels). The unsegmented metrics are named
explicitly; they are NOT the HIF3D F-score unless that metric is
re-implemented.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import f1_score

from gmn.augment import augment_positions_only
from gmn.data_lmdhg import (N_JOINTS, _normalise, discover_labels, lmdhg_split,
                            load_lmdhg_segments, load_lmdhg_streams)
from gmn.data_leap import resample
from gmn.models import build_model
from gmn.train import TrainConfig, run_one

MODELS = ["lmdhg_gmn", "lmdhg_fd_control", "lmdhg_stdconv", "lmdhg_transformer", "lmdhg_bilstm"]


@torch.no_grad()
def unsegmented_eval(model, streams, test_ids, n_classes, win=60, stride=5, T=100, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    for sid in test_ids:
        s = streams[sid]
        n = len(s["skel"])
        votes = np.zeros((n, n_classes))
        batch, spans = [], []
        for start in range(0, max(1, n - win + 1), stride):
            seg = s["skel"][start:start + win]
            x = np.concatenate([_normalise(seg).reshape(len(seg), -1), s["vis"][start:start + win]], 1)
            batch.append(resample(x, T)); spans.append((start, start + len(seg)))
        X = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
        P = torch.cat([torch.softmax(model(X[i:i + 256]), -1) for i in range(0, len(X), 256)]).cpu().numpy()
        for (a, b), p in zip(spans, P):
            votes[a:b] += p
        y_true.append(s["gt"]); y_pred.append(votes.argmax(1))
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    return dict(frame_accuracy=float((yt == yp).mean()),
                frame_macro_f1=float(f1_score(yt, yp, average="macro")),
                majority_class_baseline=float(np.bincount(yt).max() / len(yt)),
                window=win, stride=stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--results", default="results_lmdhg")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)          # use the LeapGestureDB recipe
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()
    classes, counts = discover_labels(a.data_path)
    print("LMDHG labels found (annotation counts):", counts)
    n_gest = len([c for c in classes if c != "IDLE"])
    if n_gest != 13:
        print(f"WARNING: {n_gest} gesture labels found, the dataset paper describes 13 -- check the mapping")
    D = load_lmdhg_segments(a.data_path, classes=classes)
    X = np.concatenate([D["X"], D["vis"]], -1).astype(np.float32)
    split = lmdhg_split(D["seq"])
    os.makedirs(a.results, exist_ok=True)
    json.dump(dict(split=split, n_segments=int(len(D["y"])),
                   class_counts=np.bincount(D["y"], minlength=len(D["classes"])).tolist(),
                   classes=D["classes"], annotation_counts=counts), open(os.path.join(a.results, "data_report.json"), "w"), indent=1)
    streams = load_lmdhg_streams(a.data_path, D["classes"])
    ov = dict(num_classes=len(D["classes"]))
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay, batch_size=a.batch_size)
    aug = lambda xb, g: augment_positions_only(xb, N_JOINTS * 3, cfg.aug, g)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for m in a.models:
        for seed in a.seeds:
            out = os.path.join(a.results, "runs", m, f"foldlmdhg_seed{seed}")
            r = run_one(m, X, D["y"], D["seq"], split, seed, out, cfg, save_checkpoint=True,
                        n_classes=len(D["classes"]), augment_fn=aug, verbose=False, model_overrides=ov)
            uf = os.path.join(out, "unsegmented.json")
            if not os.path.exists(uf):
                model = build_model(m, **ov).to(dev)
                model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=dev))
                json.dump(unsegmented_eval(model, streams, split["test"], len(D["classes"]), device=dev), open(uf, "w"), indent=1)
            print(m, seed, "segmented test acc", round(r["test"]["accuracy"], 4),
                  "unsegmented", json.load(open(uf)))


if __name__ == "__main__":
    main()
