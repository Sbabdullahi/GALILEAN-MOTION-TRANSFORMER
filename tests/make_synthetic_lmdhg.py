#!/usr/bin/env python
"""Synthetic LMDHG-format .mat files for smoke-testing the LMDHG pipeline only."""
import argparse
import os

import numpy as np
from scipy.io import savemat

LABELS = ["POINT_TO", "CATCH", "SHAKE_2H", "CATCH_2H", "SHAKE_DOWN", "SHAKE", "DRAW_C",
          "POINT_TO_2H", "ZOOM", "SCROLL", "DRAW_LINE", "SLICE", "ROTATE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(0)
    for sid in range(1, 51):
        frames, ann, labs, t = [], [], [], 0
        for c in rng.permutation(len(LABELS)):
            idle = rng.randint(20, 40)
            frames.append(rng.normal(0, 1, (idle, 46, 3)) * 2 + 100)
            ann.append([t + 1, t + idle]); labs.append("REPOS"); t += idle
            n = rng.randint(40, 80)
            tt = np.linspace(0, 1, n)[:, None, None]
            g = 100 + 30 * np.sin(2 * np.pi * tt * (1 + c / 5)) * np.cos(np.arange(46)[None, :, None] + c)
            g = g + rng.normal(0, 1, (n, 46, 3))
            if c % 3 == 0:
                g[:, 23:] = 0                       # second hand not tracked
            frames.append(g)
            ann.append([t + 1, t + n]); labs.append(LABELS[c]); t += n
        skel = np.concatenate(frames).reshape(t, 46 * 3)
        savemat(os.path.join(a.out, f"DataFile{sid}.mat"),
                dict(skeleton=skel, Anotations=np.array(ann, float),
                     labels=np.array(labs, dtype=object).reshape(-1, 1)))


if __name__ == "__main__":
    main()
