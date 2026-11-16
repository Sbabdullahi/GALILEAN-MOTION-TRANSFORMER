#!/usr/bin/env python
"""Hyper-parameter sweep on VALIDATION subjects of fold 0 only (test
subjects are never loaded into the selection loop).

Also sweeps the v1 physics-regularisation weight lambda_phys.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from gmn.data_leap import geometric_normalise, load_leapgesturedb, make_cv_splits
from gmn.train import TrainConfig, run_one

GRIDS = {
    "gmn": dict(lr=[1e-3, 5e-4, 1e-4], weight_decay=[0.0, 1e-4, 1e-2], batch_size=[32, 64]),
    "gmn_v1": dict(physics_weight=[0.0, 0.01, 0.1, 1.0, 10.0]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", default=list(GRIDS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--expect_n", type=int, default=None)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--cache", default=None, help="validated .npz cache (e.g. on Drive); raw files not needed if it exists")
    a = ap.parse_args()
    data = load_leapgesturedb(a.data_path, cache=a.cache, expect_n=a.expect_n)
    X = geometric_normalise(data.X)
    split_file = os.path.join(a.results, "splits.json")
    splits = json.load(open(split_file)) if os.path.exists(split_file) else \
        dict(cv=make_cv_splits(data.subject))
    s0 = splits["cv"][0]
    # inner split: the fold-0 validation subjects play the role of "test" here
    inner = dict(fold="sweep", train=s0["train"], val=s0["val"], test=s0["val"])
    rows = []
    for model in a.models:
        grid = GRIDS[model]
        keys = list(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            hp = dict(zip(keys, vals))
            cfg_kw = {k: v for k, v in hp.items() if k in ("lr", "weight_decay", "batch_size")}
            ov = {k: v for k, v in hp.items() if k not in cfg_kw}
            tag = "_".join(f"{k}{v}" for k, v in hp.items())
            out = os.path.join(a.results, "sweep", model, tag)
            r = run_one(model, X, data.y, data.subject, inner, 0, out,
                        TrainConfig(epochs=a.epochs, mc_passes=0, **cfg_kw), model_overrides=ov,
                        verbose=False)
            rows.append(dict(model=model, **hp, val_acc=r["val"]["accuracy"],
                             val_f1=r["val"]["macro_f1"], best_epoch=r["best_epoch"]))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "sweep.csv"), index=False)


if __name__ == "__main__":
    main()
