#!/usr/bin/env python
"""Tables for LMDHG (segmented test accuracy over seeds and the unsegmented
frame-level metrics), generated from saved runs only."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gmn.stats import load_runs

DISPLAY = {"lmdhg_gmn": "GMN v2 (joint-only stem)", "lmdhg_fd_control": "Same network, finite-difference $v,a$ input",
           "lmdhg_stdconv": "Same network, standard convolution", "lmdhg_transformer": "Transformer encoder",
           "lmdhg_bilstm": "BiLSTM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_lmdhg")
    ap.add_argument("--out", required=True, help="report dir of the main results")
    a = ap.parse_args()
    df = load_runs(a.results)
    rows, macros = [], []
    for m in DISPLAY:
        g = df[df.model == m]
        if not len(g):
            continue
        u = [json.load(open(os.path.join(p, "unsegmented.json"))) for p in g.path
             if os.path.exists(os.path.join(p, "unsegmented.json"))]
        fa = np.array([x["frame_accuracy"] for x in u]); ff = np.array([x["frame_macro_f1"] for x in u])
        maj = u[0]["majority_class_baseline"] if u else float("nan")
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        rows.append(f"{DISPLAY[m]} & {100*g.acc.mean():.2f} $\\pm$ {100*sd(g.acc.values):.2f} & "
                    f"{g.f1.mean():.3f} & {100*fa.mean():.2f} $\\pm$ {100*sd(fa):.2f} & {ff.mean():.3f} \\\\")
        key = "".join(w.capitalize() for w in m.split("_"))
        macros += [f"\\newcommand{{\\res{key}SegAcc}}{{{100*g.acc.mean():.2f}}}",
                   f"\\newcommand{{\\res{key}UnsegAcc}}{{{100*fa.mean():.2f}}}"]
    macros.append(f"\\newcommand{{\\resLmdhgMajority}}{{{100*maj:.2f}}}")
    rep = json.load(open(os.path.join(a.results, "data_report.json")))
    macros.append(f"\\newcommand{{\\resLmdhgSegments}}{{{rep['n_segments']}}}")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "tab_lmdhg.tex"), "w").write(
        "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Segmented (test seq. 36--50)} & "
        "\\multicolumn{2}{c}{Unsegmented (frame level)} \\\\\nModel & Accuracy (\\%) & Macro-F1 & "
        "Frame acc. (\\%) & Frame macro-F1 \\\\\n\\midrule\n" + "\n".join(rows) +
        f"\n\\midrule\nMajority class (frame level) & & & {100*maj:.2f} & \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    open(os.path.join(a.out, "macros_lmdhg.tex"), "w").write("\n".join(macros) + "\n")


if __name__ == "__main__":
    main()
