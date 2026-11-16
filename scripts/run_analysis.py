#!/usr/bin/env python
"""Post-hoc analyses on saved checkpoints (whole test sets, all folds):
rotation robustness, integrated gradients + sanity checks, kinematic
read-outs, and cost profiling. Writes <results>/analysis/*.json|npz."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gmn.analysis import (attribution_study, kinematic_summary, profile,
                          rotation_robustness)
from gmn.data_leap import deduplicate, geometric_normalise, load_leapgesturedb
from gmn.models import build_model

ROT_MODELS = ["gmn", "gmn_world_only", "gmn_palm_only", "gmn_fd_control", "transformer",
              "stgcn", "bilstm", "gmn_v1"]


def load(model_name, run_dir, X, device):
    m = build_model(model_name).to(device)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ig_per_class", type=int, default=30)
    ap.add_argument("--expect_n", type=int, default=None)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--cache", default=None, help="validated .npz cache (e.g. on Drive); raw files not needed if it exists")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_leapgesturedb(a.data_path, cache=a.cache, expect_n=a.expect_n)
    if a.dedup:                      # must match how run_cv.py was called
        data, groups, _ = deduplicate(data)
        data.subject = groups
    X = geometric_normalise(data.X)
    splits = json.load(open(os.path.join(a.results, "splits.json")))["cv"]
    out_dir = os.path.join(a.results, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    rot, kin, ig = {}, [], []
    for split in splits:
        te = np.where(np.isin(data.subject, split["test"]))[0]
        Xte = torch.tensor(X[te], device=dev)
        yte = data.y[te]
        for m in ROT_MODELS:
            rd = os.path.join(a.results, "runs", m, f"fold{split['fold']}_seed{a.seed}")
            if not os.path.exists(os.path.join(rd, "model.pt")):
                continue
            model = load(m, rd, X, dev)
            rot.setdefault(m, []).append(rotation_robustness(model, Xte, yte))
            if m == "gmn":
                kin.append(kinematic_summary(model, Xte, yte))
                s = attribution_study(model, Xte, yte, n_per_class=a.ig_per_class)
                ig.append(s)
        print("fold", split["fold"], "done")

    json.dump(rot, open(os.path.join(out_dir, "rotation.json"), "w"), indent=1)
    json.dump(kin, open(os.path.join(out_dir, "kinematics.json"), "w"), indent=1)
    if ig:
        np.savez_compressed(os.path.join(out_dir, "ig.npz"),
                            groups=np.array(ig[0]["groups"]),
                            classes=np.concatenate([s["classes"] for s in ig]),
                            signed=np.concatenate([s["signed"] for s in ig]),
                            absolute=np.concatenate([s["absolute"] for s in ig]),
                            frame_profile=np.concatenate([s["frame_profile"] for s in ig]))
        json.dump(dict(completeness_rel_gap=[s["completeness_rel_gap"] for s in ig],
                       randomisation_spearman_median=[s["randomisation_spearman_median"] for s in ig]),
                  open(os.path.join(out_dir, "ig_checks.json"), "w"), indent=1)

    prof = {}
    for m in ["gmn", "gmn_fd_control", "gmn_stdconv", "transformer", "bilstm", "stgcn", "gmn_v1"]:
        prof[m] = dict(cpu=profile(build_model(m), device="cpu"))
        if dev == "cuda":
            prof[m]["gpu"] = profile(build_model(m), device="cuda")
    json.dump(prof, open(os.path.join(out_dir, "profile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
