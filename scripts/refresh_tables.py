#!/usr/bin/env python
"""Re-derive the comparison tables and summary macros from an existing
report folder (all_runs.csv + generated tables), without the run folders.

Used to (i) apply the corrected-variance confidence intervals to reports
produced before that fix, and (ii) add summary macros (as-released results,
legacy-split audit, deferral error capture, rotation robustness, LMDHG class
inventory). Every value is computed here from generated files; nothing is
typed by hand. Writes report/macros_extra.tex and rewrites tab_main.tex and
tab_ablation.tex in place.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from gmn.stats import paired_comparisons, summarise
from scripts.make_report import ABL, DISPLAY, MAIN, macro_name

REF = "gmn"


def fmt_pm(m, s, scale=100, d=2):
    return f"{m*scale:.{d}f} $\\pm$ {s*scale:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--asis_report", default=None)
    ap.add_argument("--pipeline_log", default=None)
    ap.add_argument("--lmdhg_data_report", default=None)
    a = ap.parse_args()
    M = {}
    m = lambda k, v, f="{:.2f}": M.__setitem__(macro_name(k), f.format(v))

    df = pd.read_csv(os.path.join(a.report, "all_runs.csv"))
    df = df[df.fold.astype(str) != "legacy"].copy()
    S = summarise(df).set_index("model")

    # ── main table with corrected CIs ────────────────────────────────────────
    pc = paired_comparisons(df, REF, "acc", [x for x in MAIN if x != REF and x in S.index]).set_index("model")
    rows = []
    for k in [x for x in MAIN if x in S.index]:
        s = S.loc[k]
        if k == REF:
            d, pstr = "--", "--"
        else:
            p = pc.loc[k]
            d = f"{100*p.diff_mean:+.2f} [{100*(p.diff_mean-p.diff_ci95):+.2f}, {100*(p.diff_mean+p.diff_ci95):+.2f}]"
            pstr = f"{p.p_holm:.2f}"
            m(f"{k}_delta", 100 * p.diff_mean); m(f"{k}_delta_ci", 100 * p.diff_ci95)
            m(f"{k}_pholm", p.p_holm, "{:.2f}"); m(f"{k}_puncorr", p.p_corrected_t, "{:.3f}")
        rows.append(f"{DISPLAY[k]} & {s.n_params/1e6:.2f} & {fmt_pm(s.acc_mean, s.acc_sd)} & "
                    f"{fmt_pm(s.f1_mean, s.f1_sd, 1, 3)} & {fmt_pm(s.auprc_mean, s.auprc_sd, 1, 3)} & {d} & {pstr} \\\\")
    open(os.path.join(a.report, "tab_main.tex"), "w").write("\n".join([
        "\\begin{tabular}{lrcccrr}", "\\toprule",
        "Model & Params (M) & Accuracy (\\%) & Macro-F1 & Macro-AUPRC & $\\Delta$ acc. vs GMN [95\\% CI] & $p_{\\text{Holm}}$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

    # ── ablation table with corrected CIs ────────────────────────────────────
    pa = paired_comparisons(df, REF, "acc", [x for x in ABL if x != REF and x in S.index]).set_index("model")
    rows = []
    for k in [x for x in ABL if x in S.index]:
        s = S.loc[k]
        if k == REF:
            rows.append(f"{DISPLAY[k]} & {s.n_params/1e6:.3f} & {fmt_pm(s.acc_mean, s.acc_sd)} & -- & -- \\\\")
            continue
        p = pa.loc[k]
        m(f"abl_{k}_delta", 100 * p.diff_mean); m(f"abl_{k}_delta_ci", 100 * p.diff_ci95)
        rows.append(f"{DISPLAY[k]} & {s.n_params/1e6:.3f} & {fmt_pm(s.acc_mean, s.acc_sd)} & "
                    f"{100*p.diff_mean:+.2f} $\\pm$ {100*p.diff_ci95:.2f} & {p.p_holm:.2f} \\\\")
    open(os.path.join(a.report, "tab_ablation.tex"), "w").write("\n".join([
        "\\begin{tabular}{lrcrr}", "\\toprule",
        "Variant (one change each) & Params (M) & Accuracy (\\%) & $\\Delta$ vs GMN (95\\% CI) & $p_{\\text{Holm}}$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

    # ── human-in-the-loop: share of all errors intercepted by deferral ──────
    g = df[df.model == REF]
    err = 1 - g.acc
    caught = 1 - g.cov_msp * g.risk_msp / err
    m("hitl_caught", 100 * caught.mean(), "{:.0f}")
    m("hitl_overall_err", 100 * err.mean())
    m("hitl_cov_min", 100 * g.cov_msp.min(), "{:.1f}"); m("hitl_cov_max", 100 * g.cov_msp.max(), "{:.1f}")
    m("hitl_risk_mc", 100 * g.risk_mc.mean()); m("hitl_cov_mc", 100 * g.cov_mc.mean())
    m("train_time_min", g.train_time.mean() / 60, "{:.1f}")
    m("gpu_hours_total", df.train_time.sum() / 3600, "{:.1f}")
    for k in ("ece_before", "ece_after"):
        m(f"run_{k}", 100 * g[k].mean())

    # ── rotation robustness (from the generated table) ──────────────────────
    rot = os.path.join(a.report, "tab_rotation.tex")
    if os.path.exists(rot):
        lines = open(rot).read().splitlines()
        angles = re.findall(r"(\d+)\$\^\\circ\$", next(l for l in lines if l.startswith("Model")))
        inv = {v: k for k, v in DISPLAY.items()}
        for l in lines:
            parts = [x.strip().rstrip("\\").strip() for x in l.split("&")]
            if parts[0] in inv and len(parts) == len(angles) + 1:
                for ang, v in zip(angles, parts[1:]):
                    m(f"rot_{inv[parts[0]]}_{ang}", float(v), "{:.1f}")

    # ── as-released comparison and legacy split audit ───────────────────────
    if a.asis_report:
        da = pd.read_csv(os.path.join(a.asis_report, "all_runs.csv"))
        da = da[da.fold.astype(str) != "legacy"]
        for k, gg in da.groupby("model"):
            m(f"asis_{k}_acc", 100 * gg.acc.mean())
        d0 = df[df.seed == 0].groupby("model").acc.mean()
        for k in da.model.unique():
            if k in d0.index:
                m(f"asis_{k}_gain", 100 * (da[da.model == k].acc.mean() - d0[k]))
    if a.asis_report:
        rows = []
        for k in [x for x in MAIN if x in set(da.model)]:
            x0 = df[(df.model == k) & (df.seed == 0)].acc
            x1 = da[da.model == k].acc
            rows.append(f"{DISPLAY[k]} & {100*x0.mean():.2f} & {100*x1.mean():.2f} & {100*(x1.mean()-x0.mean()):+.2f} \\\\")
        open(os.path.join(a.report, "tab_asreleased.tex"), "w").write("\n".join([
            "\\begin{tabular}{lrrr}", "\\toprule",
            "Model & De-duplicated (\\%) & As released (\\%) & Difference \\\\", "\\midrule", *rows,
            "\\bottomrule", "\\end{tabular}"]))
    if a.pipeline_log and os.path.exists(a.pipeline_log):
        for k, acc in re.findall(r"^(\w+)\s+fold legacy seed 0: test acc ([\d.]+)", open(a.pipeline_log).read(), re.M):
            m(f"legacy_{k}_acc", 100 * float(acc))

    # ── LMDHG label inventory ───────────────────────────────────────────────
    if a.lmdhg_data_report and os.path.exists(a.lmdhg_data_report):
        r = json.load(open(a.lmdhg_data_report))
        cls = r["classes"]
        m("lmdhg_n_gestures", len([c for c in cls if c != "IDLE"]), "{:d}")
        m("lmdhg_n_classes", len(cls), "{:d}")
        small = [c for c, n in zip(cls, r["class_counts"]) if n < 30]
        m("lmdhg_n_small", len(small), "{:d}")

    # ── data audit ──────────────────────────────────────────────────────────
    au = os.path.join(a.report, "data_audit.json")
    if os.path.exists(au):
        d = json.load(open(au))
        for k in ("n_samples", "n_subjects", "exact_duplicate_recordings", "duplicate_groups_across_subjects",
                  "duplicate_groups_with_conflicting_labels", "recordings_shorter_than_20_frames",
                  "recordings_longer_than_2000_frames", "frames_with_missing_fields"):
            m(f"audit_{k}", int(d[k]), "{:d}")
        m("audit_min_len", int(d["raw_length_frames"]["min"]), "{:d}")
        m("audit_max_len", int(d["raw_length_frames"]["max"]), "{:d}")

    # ── cost table: mark LSTM FLOPs as not counted ──────────────────────────
    cost = os.path.join(a.report, "tab_cost.tex")
    if os.path.exists(cost):
        t = open(cost).read()
        t = re.sub(r"(BiLSTM & [\d.]+ & )0\.0( &)", r"\1n/c\2", t)
        t = t.replace("FLOPs (M) &", "FLOPs (M)$^a$ &", 1) if "$^a$" not in t else t
        open(cost, "w").write(t)

    open(os.path.join(a.report, "macros_extra.tex"), "w").write(
        "\n".join(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
    print(f"{len(M)} macros written; tables refreshed")


if __name__ == "__main__":
    main()
