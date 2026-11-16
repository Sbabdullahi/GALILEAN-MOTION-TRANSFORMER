"""Statistics over repeated subject-wise cross-validation.

Paired comparisons use the (fold, seed) runs as the pairing unit. Because CV
training sets overlap, the naive paired t-test is anti-conservative; the
primary test is therefore the corrected resampled t-test of Nadeau & Bengio
(2003), with the Wilcoxon signed-rank test as a non-parametric check and
Holm (1979) correction across all comparisons against the proposed model.
"""
from __future__ import annotations

import json
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_runs(results_dir: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(results_dir, "runs", "*", "*", "metrics.json")):
        r = json.load(open(f))
        rows.append(dict(
            model=r["model"], fold=r["fold"], seed=r["seed"], n_params=r["n_params"],
            acc=r["test"]["accuracy"], bal_acc=r["test"]["balanced_accuracy"],
            f1=r["test"]["macro_f1"], auprc=r["test"]["macro_auprc"], auroc=r["test"]["macro_auroc"],
            n_test=r["split_sizes"]["test"], n_train=r["split_sizes"]["train"],
            ece_before=r["calibration_test"]["before"]["ece"],
            ece_after=r["calibration_test"]["after"]["ece"],
            nll_before=r["calibration_test"]["before"]["nll"],
            nll_after=r["calibration_test"]["after"]["nll"],
            T=r["calibration_test"]["temperature"],
            cov_msp=r["deferral_msp"]["coverage"], risk_msp=r["deferral_msp"]["accepted_error"],
            aurc_msp=r["aurc_msp"],
            cov_mc=r.get("deferral_mc_entropy", {}).get("coverage", np.nan),
            risk_mc=r.get("deferral_mc_entropy", {}).get("accepted_error", np.nan),
            aurc_mc=r.get("aurc_mc_entropy", np.nan),
            epochs=r["epochs_run"], train_time=r["train_time_s"], path=os.path.dirname(f)))
    return pd.DataFrame(rows)


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.5 + conf / 2, len(x) - 1)
    return float(m), float(x.std(ddof=1)), float(h)


def corrected_resampled_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired
    differences d from J = k*r resampling runs."""
    d = np.asarray(d, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(p)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_comparisons(df: pd.DataFrame, reference: str, metric: str = "acc", others=None):
    ref = df[df.model == reference].set_index(["fold", "seed"])
    others = others or [m for m in df.model.unique() if m != reference]
    rows = []
    for m in others:
        o = df[df.model == m].set_index(["fold", "seed"])
        common = ref.index.intersection(o.index)
        if len(common) < 2:
            continue
        d = ref.loc[common, metric].values - o.loc[common, metric].values
        t, p = corrected_resampled_ttest(d, ref.loc[common, "n_train"].mean(),
                                         ref.loc[common, "n_test"].mean())
        try:
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            pw = np.nan
        mean, sd, _ = mean_ci(d)
        # CI with the SAME Nadeau-Bengio corrected variance as the test, so the
        # interval and the p-value can never disagree
        J = len(d)
        ratio = ref.loc[common, "n_test"].mean() / ref.loc[common, "n_train"].mean()
        h = float(stats.t.ppf(0.975, J - 1) * np.sqrt((1.0 / J + ratio) * np.var(d, ddof=1))) if J > 1 else np.nan
        rows.append(dict(model=m, n_pairs=J, diff_mean=mean, diff_sd=sd,
                         diff_ci95=h, t=t, p_corrected_t=p, p_wilcoxon=float(pw)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_corrected_t.values)
    return out


def summarise(df: pd.DataFrame, metrics=("acc", "f1", "auprc", "auroc")) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("model"):
        row = dict(model=m, runs=len(g), n_params=int(g.n_params.iloc[0]))
        for k in metrics:
            row[k + "_mean"], row[k + "_sd"], row[k + "_ci"] = mean_ci(g[k])
        rows.append(row)
    return pd.DataFrame(rows)


def subject_bootstrap_ci(correct: np.ndarray, subject: np.ndarray, B=2000, seed=0):
    """Cluster bootstrap over test subjects for pooled accuracy."""
    rng = np.random.RandomState(seed)
    subs = np.unique(subject)
    per = {s: correct[subject == s] for s in subs}
    boots = []
    for _ in range(B):
        draw = rng.choice(subs, len(subs), replace=True)
        boots.append(np.concatenate([per[s] for s in draw]).mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
