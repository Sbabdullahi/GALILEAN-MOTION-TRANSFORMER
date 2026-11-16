#!/usr/bin/env python
"""Repeated subject-wise cross-validation for every model in the paper.

Example (Colab, T4):
    python scripts/run_cv.py --data_path /content/LeapGestureDB \
        --results /content/drive/MyDrive/gmn_results --folds 5 --seeds 0 1 2

Runs are resumable: a run whose metrics.json exists is skipped, so the job
can be split across Colab sessions. Split files are written once to
<results>/splits.json and re-used by every model and every later session.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gmn.data_leap import (check_split, deduplicate, geometric_normalise, load_leapgesturedb,
                           make_cv_splits, make_legacy_split)
from gmn.train import TrainConfig, run_one

MAIN = ["gmn", "gmn_fd_control", "gmn_stdconv", "transformer", "bilstm", "stgcn", "gmn_v1"]
ABLATIONS = ["gmn_no_readout", "gmn_no_chirality", "gmn_frame_order0", "gmn_frame_order2",
             "gmn_world_only", "gmn_palm_only", "gmn_meanpool", "gmn_no_transformer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", default=MAIN + ABLATIONS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--legacy_split", action="store_true",
                    help="also run the single v1 split (87/9/24 subjects) for auditing")
    ap.add_argument("--epochs", type=int, default=60)
    # one recipe for ALL models, taken from the fold-0 validation sweep (sweep.csv)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--expect_n", type=int, default=None,
                    help="abort unless exactly this many .txt files are visible (6600 for LeapGestureDB)")
    ap.add_argument("--dedup", action="store_true",
                    help="drop identical recordings and merge subject folders that share them "
                         "(recommended: removes cross-subject leakage)")
    ap.add_argument("--exclude", default=None,
                    help="JSON list of relative file paths to drop (sensitivity analysis only)")
    ap.add_argument("--mc_passes", type=int, default=25)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--cache", default=None, help="validated .npz cache (e.g. on Drive); raw files not needed if it exists")
    a = ap.parse_args()

    data = load_leapgesturedb(a.data_path, cache=a.cache, expect_n=a.expect_n)
    print(json.dumps({k: v for k, v in data.report.items()
                      if k not in ("parse_failures", "unmatched_files")}, indent=1))
    if a.exclude:
        drop = set(json.load(open(a.exclude)))
        keep = np.array([f not in drop for f in data.files])
        print(f"excluding {int((~keep).sum())} recordings listed in {a.exclude}")
        data.X, data.y, data.subject, data.files = (data.X[keep], data.y[keep],
                                                     data.subject[keep], data.files[keep])
    os.makedirs(a.results, exist_ok=True)
    if a.dedup:
        data, groups, drep = deduplicate(data)
        print("dedup:", json.dumps(drep, indent=1))
        json.dump(drep, open(os.path.join(a.results, "dedup_report.json"), "w"), indent=1)
        data.subject = groups            # split unit = merged subject group
    X = geometric_normalise(data.X)
    json.dump(data.report, open(os.path.join(a.results, "data_report.json"), "w"), indent=1)

    split_file = os.path.join(a.results, "splits.json")
    if os.path.exists(split_file):
        splits = json.load(open(split_file))
    else:
        splits = dict(cv=make_cv_splits(data.subject, a.folds),
                      legacy=make_legacy_split(data.subject))
        json.dump(splits, open(split_file, "w"), indent=1)
    for s in splits["cv"] + [splits["legacy"]]:
        check_split(s, data.subject)

    todo = [s for s in splits["cv"] if a.only_folds is None or s["fold"] in a.only_folds]
    if a.legacy_split:
        todo.append(splits["legacy"])
    cfg = TrainConfig(epochs=a.epochs, mc_passes=a.mc_passes, lr=a.lr,
                      weight_decay=a.weight_decay, batch_size=a.batch_size)
    for model in a.models:
        for split in todo:
            for seed in a.seeds:
                out = os.path.join(a.results, "runs", model, f"fold{split['fold']}_seed{seed}")
                r = run_one(model, X, data.y, data.subject, split, seed, out, cfg,
                            save_checkpoint=not a.no_ckpt, verbose=not a.quiet)
                c = r["config"]
                if (c["lr"], c["weight_decay"], c["batch_size"], c["epochs"]) != \
                        (cfg.lr, cfg.weight_decay, cfg.batch_size, cfg.epochs):
                    raise SystemExit(f"{out} was trained with a different recipe "
                                     f"({c['lr']}, {c['weight_decay']}, {c['batch_size']}, {c['epochs']}); "
                                     "use a new --results folder instead of mixing recipes")
                print(f"{model:20s} fold {split['fold']} seed {seed}: "
                      f"test acc {r['test']['accuracy']:.4f}  macro-F1 {r['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
