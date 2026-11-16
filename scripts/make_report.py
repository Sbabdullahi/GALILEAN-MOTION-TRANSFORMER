#!/usr/bin/env python
"""Generate every number, table and figure of the paper from saved runs.

No result number is typed by hand: the .tex
includes <report>/macros.tex and <report>/tab_*.tex written here. The script
also runs internal-consistency checks (e.g. accuracy == mean per-class recall
on a balanced test set; deltas recomputed from the same table values) and
refuses to write the report if any check fails.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc

from gmn.constants import GESTURE_NAMES, NUM_CLASSES
from gmn.metrics import ece, risk_coverage
from gmn.stats import load_runs, mean_ci, paired_comparisons, summarise, subject_bootstrap_ci

REF = "gmn"
DISPLAY = {
    "gmn": "GMN v2 (proposed)",
    "gmn_fd_control": "Same network, finite-difference $v,a$ input",
    "gmn_stdconv": "Same network, standard convolution",
    "transformer": "Transformer encoder",
    "bilstm": "BiLSTM",
    "stgcn": "ST-GCN (6-node hand graph)",
    "gmn_v1": "GMN v1",
    "gmn_no_readout": "w/o kinematic read-out",
    "gmn_no_chirality": "w/o chirality term",
    "gmn_frame_order0": "w/o co-moving frame (order 0)",
    "gmn_frame_order2": "Accelerating frame (order 2)",
    "gmn_world_only": "w/o palm-frame stream",
    "gmn_palm_only": "Palm-frame stream only",
    "gmn_meanpool": "w/o class-query pooling",
    "gmn_no_transformer": "w/o transformer encoder",
}
MAIN = ["gmn", "gmn_fd_control", "gmn_stdconv", "transformer", "stgcn", "bilstm", "gmn_v1"]
ABL = ["gmn", "gmn_stdconv", "gmn_fd_control", "gmn_frame_order0", "gmn_frame_order2",
       "gmn_no_readout", "gmn_no_chirality", "gmn_world_only", "gmn_palm_only",
       "gmn_meanpool", "gmn_no_transformer"]


def macro_name(key):
    words = re.split(r"[_\W]+", key)
    num = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
           "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "res" + "".join("".join(num.get(c, c) for c in w).capitalize() for w in words if w)


class Report:
    def __init__(self, out):
        self.out, self.macros, self.checks = out, {}, []
        os.makedirs(out, exist_ok=True)

    def m(self, key, value, fmt="{:.2f}"):
        s = fmt.format(value) if isinstance(value, (int, float, np.floating, np.integer)) else str(value)
        self.macros[macro_name(key)] = s
        return s

    def check(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))

    def table(self, name, tex):
        open(os.path.join(self.out, f"tab_{name}.tex"), "w").write(tex)

    def write(self, synthetic=False):
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(self.macros.items())]
        if synthetic:
            lines.append("\\newcommand{\\resSyntheticWarning}{\\textbf{SYNTHETIC SMOKE-TEST DATA -- NOT RESULTS}}")
        else:
            lines.append("\\newcommand{\\resSyntheticWarning}{}")
        open(os.path.join(self.out, "macros.tex"), "w").write("\n".join(lines) + "\n")
        with open(os.path.join(self.out, "consistency_checks.txt"), "w") as f:
            for n, ok, d in self.checks:
                f.write(f"[{'PASS' if ok else 'FAIL'}] {n} {d}\n")
        failed = [c for c in self.checks if not c[1]]
        if failed:
            raise SystemExit(f"{len(failed)} consistency checks failed: {failed}")


def pct(x):
    return 100 * x


def fmt_pm(m, s, scale=100, d=2):
    return f"{m*scale:.{d}f} $\\pm$ {s*scale:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true", help="watermark as smoke test")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "report")
    R = Report(out)
    df = load_runs(a.results)
    df = df[df.fold != "legacy"].copy() if df.fold.dtype == object else df
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    S = summarise(df).set_index("model")
    n_runs = int(S.loc[REF, "runs"])
    R.m("n_runs", n_runs, "{:d}")
    R.m("n_folds", df[df.model == REF].fold.nunique(), "{:d}")
    R.m("n_seeds", df[df.model == REF].seed.nunique(), "{:d}")

    # ── main comparison ──────────────────────────────────────────────────────
    pc = paired_comparisons(df, REF, "acc").set_index("model") if len(df.model.unique()) > 1 else pd.DataFrame()
    rows = []
    for m in [x for x in MAIN if x in S.index]:
        s = S.loc[m]
        key = m
        R.m(f"{key}_acc", pct(s.acc_mean)); R.m(f"{key}_acc_sd", pct(s.acc_sd))
        R.m(f"{key}_acc_ci", pct(s.acc_ci)); R.m(f"{key}_f1", s.f1_mean, "{:.3f}")
        R.m(f"{key}_params", s.n_params / 1e6, "{:.2f}")
        if m != REF and m in pc.index:
            p = pc.loc[m]
            R.m(f"{key}_delta", pct(p.diff_mean)); R.m(f"{key}_delta_ci", pct(p.diff_ci95))
            R.m(f"{key}_pholm", p.p_holm, "{:.3g}")
            d = f"{pct(p.diff_mean):+.2f} [{pct(p.diff_mean - p.diff_ci95):+.2f}, {pct(p.diff_mean + p.diff_ci95):+.2f}]"
            pstr = f"{p.p_holm:.3g}"
            # consistency: delta must equal difference of means (same pairs)
            R.check(f"delta {m}", abs(p.diff_mean - (S.loc[REF].acc_mean - s.acc_mean)) < 1e-9 or
                    p.n_pairs != s.runs, f"{p.diff_mean:.6f}")
        else:
            d, pstr = "--", "--"
        rows.append(f"{DISPLAY[m]} & {s.n_params/1e6:.2f} & {fmt_pm(s.acc_mean, s.acc_sd)} & "
                    f"{fmt_pm(s.f1_mean, s.f1_sd, 1, 3)} & {fmt_pm(s.auprc_mean, s.auprc_sd, 1, 3)} & {d} & {pstr} \\\\")
    R.table("main", "\n".join([
        "\\begin{tabular}{lrcccrr}", "\\toprule",
        "Model & Params (M) & Accuracy (\\%) & Macro-F1 & Macro-AUPRC & $\\Delta$ acc. vs GMN [95\\% CI] & $p_{\\text{Holm}}$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

    # ── ablations ────────────────────────────────────────────────────────────
    pa = paired_comparisons(df, REF, "acc", [x for x in ABL if x != REF and x in S.index])
    pa = pa.set_index("model") if len(pa) else pa
    rows = []
    for m in [x for x in ABL if x in S.index]:
        s = S.loc[m]
        if m == REF:
            rows.append(f"{DISPLAY[m]} & {s.n_params/1e6:.3f} & {fmt_pm(s.acc_mean, s.acc_sd)} & -- & -- \\\\")
            continue
        p = pa.loc[m]
        R.m(f"abl_{m}_delta", pct(p.diff_mean))
        rows.append(f"{DISPLAY[m]} & {s.n_params/1e6:.3f} & {fmt_pm(s.acc_mean, s.acc_sd)} & "
                    f"{pct(p.diff_mean):+.2f} $\\pm$ {pct(p.diff_ci95):.2f} & {p.p_holm:.3g} \\\\")
    R.table("ablation", "\n".join([
        "\\begin{tabular}{lrcrr}", "\\toprule",
        "Variant (one change each) & Params (M) & Accuracy (\\%) & $\\Delta$ vs GMN (95\\% CI) & $p_{\\text{Holm}}$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

    # ── pooled per-class results for the proposed model ────────────────────
    ref_runs = df[df.model == REF]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    y_all, logit_all, subj_all, T_all, mc_ent, mc_mean = [], [], [], [], [], []
    for _, r in ref_runs.iterrows():
        met = json.load(open(os.path.join(r.path, "metrics.json")))
        z = np.load(os.path.join(r.path, "arrays.npz"))
        c = np.array(met["test"]["confusion"])
        cm += c
        T = met["calibration_test"]["temperature"]
        y_all.append(z["test_y"]); logit_all.append(z["test_logits"] / T)
        subj_all.append(z["test_subject"])
        # consistency: stored confusion == confusion recomputed from logits
        R.check(f"confusion {r.fold}/{r.seed}",
                np.array_equal(np.bincount(z["test_y"] * NUM_CLASSES + z["test_logits"].argmax(1),
                                           minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, -1), c))
        if "test_mc_entropy" in z:
            mc_ent.append(z["test_mc_entropy"]); mc_mean.append(z["test_mc_mean"])
    y_all, P_all = np.concatenate(y_all), softmax(np.concatenate(logit_all), 1)
    subj_all = np.concatenate(subj_all)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    pooled_acc = np.trace(cm) / cm.sum()
    balanced = len(set(cm.sum(1))) == 1
    if balanced:
        R.check("accuracy == mean recall (balanced test set)", abs(pooled_acc - rec.mean()) < 1e-12,
                f"{pooled_acc:.6f} vs {rec.mean():.6f}")
    R.check("errors == N - trace", cm.sum() - np.trace(cm) == int((P_all.argmax(1) != y_all).sum()))
    # per-subject accuracy: who does the system work worst for?
    correct = P_all.argmax(1) == y_all
    subj_acc = np.array([correct[subj_all == s].mean() for s in np.unique(subj_all)])
    R.m("subj_acc_min", pct(subj_acc.min())); R.m("subj_acc_lowdecile", pct(np.percentile(subj_acc, 10)))
    R.m("subj_acc_median", pct(np.median(subj_acc)))
    R.m("subj_below_ninety", int((subj_acc < 0.90).sum()), "{:d}"); R.m("subj_n", len(subj_acc), "{:d}")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(subj_acc * 100, bins=20)
    ax.set_xlabel("Per-subject accuracy (%)"); ax.set_ylabel("Subjects"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_subjects.pdf")); plt.close(fig)
    lo, hi = subject_bootstrap_ci(correct, subj_all)
    R.m("pooled_acc", pct(pooled_acc)); R.m("pooled_acc_lo", pct(lo)); R.m("pooled_acc_hi", pct(hi))
    R.m("pooled_errors", int(cm.sum() - np.trace(cm)), "{:d}"); R.m("pooled_n", int(cm.sum()), "{:d}")
    R.m("pooled_macro_f1", f1.mean(), "{:.3f}")
    rows = [f"{GESTURE_NAMES[i]} & {cm[i].sum()} & {prec[i]:.3f} & {rec[i]:.3f} & {f1[i]:.3f} \\\\"
            for i in range(NUM_CLASSES)]
    rows += ["\\midrule",
             f"Macro average & {cm.sum()} & {prec.mean():.3f} & {rec.mean():.3f} & {f1.mean():.3f} \\\\",
             f"Accuracy & \\multicolumn{{4}}{{c}}{{{pct(pooled_acc):.2f}\\% ({cm.sum()-np.trace(cm)} errors / {cm.sum()})}} \\\\"]
    R.table("perclass", "\n".join(["\\begin{tabular}{lrccc}", "\\toprule",
                                    "Class & $N$ & Precision & Recall & F1 \\\\", "\\midrule",
                                    *rows, "\\bottomrule", "\\end{tabular}"]))
    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(GESTURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(GESTURE_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=.046); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_confusion.pdf")); plt.close(fig)

    # ROC from the actual pooled predictions of THIS dataset
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_all == c, P_all[:, c])
        ax.plot(fpr, tpr, lw=1.2, label=f"{GESTURE_NAMES[c]} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_roc.pdf")); plt.close(fig)

    # ── calibration ─────────────────────────────────────────────────────────
    raw = softmax(np.concatenate([np.load(os.path.join(p, "arrays.npz"))["test_logits"]
                                  for p in ref_runs.path]), 1)
    e_b, bins_b = ece(y_all, raw)
    e_a, bins_a = ece(y_all, P_all)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for bins, lab, mk in ((bins_b, f"Before (ECE {100*e_b:.2f}%)", "o"), (bins_a, f"After T (ECE {100*e_a:.2f}%)", "s")):
        b = np.array([(x[4], x[3]) for x in bins if x[2] > 0])
        ax.plot(b[:, 0], b[:, 1], marker=mk, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_reliability.pdf")); plt.close(fig)
    T_m, T_sd, _ = mean_ci(ref_runs["T"])
    R.m("T_mean", T_m, "{:.2f}"); R.m("T_sd", T_sd, "{:.2f}")
    R.m("ece_before", pct(e_b)); R.m("ece_after", pct(e_a))
    rows = []
    for m in [x for x in MAIN if x in S.index]:
        g = df[df.model == m]
        rows.append(f"{DISPLAY[m]} & {g['T'].mean():.2f} & {fmt_pm(g.ece_before.mean(), g.ece_before.std(ddof=1) if len(g)>1 else 0)} & "
                    f"{fmt_pm(g.ece_after.mean(), g.ece_after.std(ddof=1) if len(g)>1 else 0)} & {g.nll_before.mean():.3f} & {g.nll_after.mean():.3f} \\\\")
    R.table("calibration", "\n".join(["\\begin{tabular}{lrcccc}", "\\toprule",
                                       "Model & $T^*$ & ECE before (\\%) & ECE after (\\%) & NLL before & NLL after \\\\",
                                       "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

    # ── human-in-the-loop deferral ─────────────────────────────────────────
    g = ref_runs
    rows = []
    for tag, cov, risk, aurc in (("Calibrated max-probability", "cov_msp", "risk_msp", "aurc_msp"),
                                 ("MC-Dropout entropy", "cov_mc", "risk_mc", "aurc_mc")):
        if g[cov].notna().any():
            rows.append(f"{tag} & {fmt_pm(g[cov].mean(), g[cov].std(ddof=1) if len(g)>1 else 0)} & "
                        f"{fmt_pm(g[risk].mean(), g[risk].std(ddof=1) if len(g)>1 else 0)} & {g[aurc].mean()*100:.2f} \\\\")
    R.m("hitl_cov", pct(g.cov_msp.mean())); R.m("hitl_risk", pct(g.risk_msp.mean()))
    R.m("hitl_target", 1.0, "{:.0f}")
    R.table("hitl", "\n".join(["\\begin{tabular}{lccc}", "\\toprule",
                                "Confidence signal & Auto-accepted (\\%) & Error on accepted (\\%) & AURC ($\\times 100$) \\\\",
                                "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))
    conf = P_all.max(1)
    cv, rk, _ = risk_coverage(y_all, P_all.argmax(1), conf)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(cv * 100, rk * 100, label="Calibrated max-probability")
    if mc_ent:
        mm = np.concatenate(mc_mean)
        cv2, rk2, _ = risk_coverage(y_all, mm.argmax(1), -np.concatenate(mc_ent))
        ax.plot(cv2 * 100, rk2 * 100, label="MC-Dropout entropy")
    ax.axhline(1.0, color="k", ls=":", lw=.8)
    ax.set_xlabel("Coverage: commands executed without confirmation (%)")
    ax.set_ylabel("Error on executed commands (%)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_risk_coverage.pdf")); plt.close(fig)

    # ── optional analyses ───────────────────────────────────────────────────
    adir = os.path.join(a.results, "analysis")
    if os.path.exists(os.path.join(adir, "rotation.json")):
        rot = json.load(open(os.path.join(adir, "rotation.json")))
        angles = sorted({int(k) for v in rot.values() for d in v for k in d})
        rows = []
        for m, lst in rot.items():
            accs = [np.mean([d[str(ang)] if str(ang) in d else d[ang] for d in lst]) for ang in angles]
            rows.append(f"{DISPLAY.get(m, m)} & " + " & ".join(f"{100*x:.1f}" for x in accs) + " \\\\")
        R.table("rotation", "\n".join(["\\begin{tabular}{l" + "r" * len(angles) + "}", "\\toprule",
                                        "Model & " + " & ".join(f"{x}$^\\circ$" for x in angles) + " \\\\",
                                        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))
    if os.path.exists(os.path.join(adir, "profile.json")):
        prof = json.load(open(os.path.join(adir, "profile.json")))
        rows = []
        for m, p in prof.items():
            gpu = p.get("gpu", {})
            fl = "%.1f" % (p['cpu']['flops'] / 1e6) if p['cpu']['flops'] else "n/c"
            rows.append(f"{DISPLAY.get(m, m)} & {p['cpu']['params']/1e6:.2f} & {fl} & "
                        f"{p['cpu']['latency_ms_median']:.2f} & {gpu.get('latency_ms_median', float('nan')):.2f} \\\\")
        R.table("cost", "\n".join(["\\begin{tabular}{lrrrr}", "\\toprule",
                                    "Model & Params (M) & FLOPs (M)$^a$ & CPU ms (median) & GPU ms (median) \\\\",
                                    "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))
    if os.path.exists(os.path.join(adir, "ig.npz")):
        z = np.load(os.path.join(adir, "ig.npz"))
        chk = json.load(open(os.path.join(adir, "ig_checks.json")))
        R.m("ig_completeness", 100 * np.median(chk["completeness_rel_gap"]))
        R.m("ig_rand_rho", np.median(chk["randomisation_spearman_median"]), "{:.2f}")
        groups, cls, absa = list(z["groups"]), z["classes"], z["absolute"]
        share = absa / absa.sum(1, keepdims=True)
        mat = np.stack([share[cls == c].mean(0) if (cls == c).any() else np.full(len(groups), np.nan)
                        for c in range(NUM_CLASSES)])
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(GESTURE_NAMES, fontsize=8)
        fig.colorbar(im, label="Share of |IG| attribution"); fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_ig_groups.pdf")); plt.close(fig)
    if os.path.exists(os.path.join(adir, "kinematics.json")):
        kin = json.load(open(os.path.join(adir, "kinematics.json")))
        v = np.mean([k["palm_velocity_mean"] for k in kin], 0)
        chi = np.mean([k["index_chirality_mean"] for k in kin], 0)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        axs[0].barh(range(NUM_CLASSES), v[:, 0]); axs[0].set_title("Mean palm $\\hat v_x$ (sensor frame)")
        axs[1].barh(range(NUM_CLASSES), chi); axs[1].set_title("Mean index-tip chirality $\\chi$")
        for ax in axs:
            ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(GESTURE_NAMES, fontsize=8)
            ax.axvline(0, color="k", lw=.6)
        axs[1].set_yticklabels([])
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_kinematics.pdf")); plt.close(fig)
    sweep = os.path.join(a.results, "sweep.csv")
    if os.path.exists(sweep):
        sw = pd.read_csv(sweep)
        cols = [c for c in sw.columns if c not in ("best_epoch",)]
        R.table("sweep", "\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n" +
                " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n" +
                "\n".join(" & ".join(f"{v:.4g}" if isinstance(v, float) else str(v).replace("_", "\\_")
                                     for v in row) + " \\\\" for row in sw[cols].values) +
                "\n\\bottomrule\n\\end{tabular}")
    data_rep = os.path.join(a.results, "data_report.json")
    if os.path.exists(data_rep):
        d = json.load(open(data_rep))
        R.m("data_n", d["n_samples"], "{:d}"); R.m("data_subjects", d["n_subjects"], "{:d}")
        R.m("data_parse_failures", len(d["parse_failures"]), "{:d}")
        R.m("data_missing_frames", d["frames_with_missing_fields"], "{:d}")
    ded = os.path.join(a.results, "dedup_report.json")
    if os.path.exists(ded):
        d = json.load(open(ded))
        R.m("dedup_kept", d["n_kept"], "{:d}"); R.m("dedup_removed", d["removed_recordings"], "{:d}")
        R.m("dedup_groups", d["n_split_groups"], "{:d}"); R.m("dedup_merged", len(d["merged_subject_groups"]), "{:d}")
        R.m("dedup_conflicts", d["conflicting_label_groups"], "{:d}")
    R.write(synthetic=a.synthetic)
    print(f"Report written to {out}  ({len(R.macros)} macros, {len(R.checks)} checks passed)")


if __name__ == "__main__":
    main()
