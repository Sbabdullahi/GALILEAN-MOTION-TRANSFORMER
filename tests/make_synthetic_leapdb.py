#!/usr/bin/env python
"""Write a small SYNTHETIC dataset in the LeapGestureDB .txt format, used
only to smoke-test the full pipeline (parser -> CV -> report). The numbers it
produces are meaningless."""
import argparse
import os

import numpy as np


def hand_frame(palm, d, n, spread, rng):
    e1, e3 = d / np.linalg.norm(d), n / np.linalg.norm(n)
    e2 = np.cross(e3, e1)
    tips = []
    for k, ang in enumerate(np.linspace(-0.6, 0.6, 5)):
        length = 70 + 10 * (k in (1, 2, 3)) + spread
        tips.append(palm + length * (np.cos(ang) * e1 + np.sin(ang) * e2) + rng.normal(0, 1.5, 3))
    return tips


def gesture(c, T, rng):
    t = np.linspace(0, 1, T)
    palm = np.zeros((T, 3)) + np.array([0, 200, 0])
    d = np.tile([0, 0, -1.0], (T, 1))
    n = np.tile([0, -1.0, 0], (T, 1))
    spread = np.zeros(T)
    A = rng.uniform(60, 90)
    if c in (1, 2):                                       # CW / CCW circle
        s = 1 if c == 1 else -1
        palm[:, 0] += A * np.cos(2 * np.pi * t); palm[:, 2] += s * A * np.sin(2 * np.pi * t)
    elif c in (7, 8):                                     # move left / right
        palm[:, 0] += (-1 if c == 7 else 1) * A * t
    elif c in (5, 6):                                     # zoom in / out
        spread = (1 if c == 5 else -1) * 25 * t
    elif c in (3, 4):                                     # contrast up / down
        palm[:, 1] += (1 if c == 3 else -1) * A * t
    elif c in (9, 10):                                    # previous / next: wrist rotation
        ang = (1 if c == 9 else -1) * 0.8 * np.sin(np.pi * t)
        d = np.stack([np.sin(ang), np.zeros(T), -np.cos(ang)], 1)
    else:                                                 # click
        palm[:, 1] -= 30 * np.sin(np.pi * t)
    palm += rng.normal(0, 2, palm.shape)
    return palm, d, n, spread


def fmt(v):
    return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    for s in range(1, a.subjects + 1):
        sd = os.path.join(a.out, f"Subject_{s}")
        os.makedirs(sd, exist_ok=True)
        yaw = rng.uniform(-0.3, 0.3)                     # subject-specific orientation
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        for c in range(11):
            for r in range(a.reps):
                T = rng.randint(70, 140)
                palm, d, n, spread = gesture(c, T, rng)
                with open(os.path.join(sd, f"G{c+1}_{r+1:03d}.txt"), "w") as f:
                    for i in range(T):
                        p, di, ni = Ry @ palm[i], Ry @ d[i], Ry @ n[i]
                        f.write(f"Frame.id: {i} timestamp: {i*10000} Hand_number: 1\n")
                        f.write(f"hand_Id_type: 1 finger's_number: 5 hand direction: {fmt(di)} "
                                f"Palm position: {fmt(p)} Palm normal: {fmt(ni)}\n")
                        for k, tip in enumerate(hand_frame(p, di, ni, spread[i], rng)):
                            f.write(f"Finger id: {k} TYPE_{names[k]} tip position: {fmt(tip)}\n")


if __name__ == "__main__":
    main()
