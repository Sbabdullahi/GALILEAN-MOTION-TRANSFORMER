"""Metrics, calibration, uncertainty and selective prediction (HITL).

All functions take numpy arrays so they can be re-run from the saved
per-run predictions without the model (the paper's tables are generated
from these saved files only).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

_trapz = getattr(np, "trapezoid", None) or np.trapz


def classification_metrics(y, logits, n_classes):
    p = softmax(logits, 1)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    yb = label_binarize(y, classes=list(range(n_classes)))
    return dict(
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(recall.mean()),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        # AUPRC / AUROC: one-vs-rest, macro-averaged over classes, on [0, 1]
        macro_auprc=float(average_precision_score(yb, p, average="macro")),
        macro_auroc=float(roc_auc_score(yb, p, average="macro", multi_class="ovr")),
        n_errors=int((pred != y).sum()), n=int(len(y)),
        confusion=cm.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
def nll(y, logits, T=1.0):
    return float(-log_softmax(logits / T, 1)[np.arange(len(y)), y].mean())


def fit_temperature(y_val, logits_val):
    """Single global temperature (Guo et al., 2017) minimising validation NLL.
    Bounded scalar search -> deterministic, no optimiser state."""
    res = minimize_scalar(lambda t: nll(y_val, logits_val, t), bounds=(0.05, 20.0),
                          method="bounded", options=dict(xatol=1e-4))
    return float(res.x)


def ece(y, probs, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
            bins.append((float(lo), float(hi), int(m.sum()), float(acc[m].mean()), float(conf[m].mean())))
        else:
            bins.append((float(lo), float(hi), 0, float("nan"), float("nan")))
    return float(out), bins


def brier(y, probs):
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def calibration_report(y, logits, T):
    out = {}
    for tag, t in (("before", 1.0), ("after", T)):
        p = softmax(logits / t, 1)
        e, bins = ece(y, p)
        out[tag] = dict(ece=e, nll=nll(y, logits, t), brier=brier(y, p), bins=bins)
    out["temperature"] = T
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty
# ─────────────────────────────────────────────────────────────────────────────
def predictive_entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def mc_dropout_summary(mc_probs):
    """mc_probs: (S, N, C) softmax outputs of S stochastic passes.
    Returns mean probs, total entropy, expected entropy, mutual information
    (epistemic part, Gal & Ghahramani 2016; Kendall & Gal 2017)."""
    mean = mc_probs.mean(0)
    total = predictive_entropy(mean)
    expected = predictive_entropy(mc_probs).mean(0)
    return dict(mean=mean, entropy=total, mutual_info=total - expected,
                std=mc_probs.std(0))


# ─────────────────────────────────────────────────────────────────────────────
# Selective prediction / human-in-the-loop deferral
# ─────────────────────────────────────────────────────────────────────────────
def risk_coverage(y, pred, confidence):
    """Risk-coverage curve (Geifman & El-Yaniv, 2017): sort by confidence,
    accept the most confident fraction, report error on the accepted set."""
    order = np.argsort(-confidence, kind="stable")
    err = (pred[order] != y[order]).astype(float)
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(err) / k
    coverage = k / len(y)
    aurc = float(_trapz(risk, coverage)) if len(y) > 1 else float(risk[0])
    return coverage, risk, aurc


def choose_threshold(y_val, pred_val, conf_val, target_risk):
    """Lowest confidence threshold whose accepted-set error on VALIDATION is
    <= target_risk (maximises coverage subject to the risk constraint)."""
    order = np.argsort(-conf_val, kind="stable")
    err = (pred_val[order] != y_val[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    ok = np.where(risk <= target_risk)[0]
    if len(ok) == 0:
        return float("inf")                     # defer everything
    return float(conf_val[order][ok.max()])


def apply_deferral(y, pred, conf, threshold, always_confirm=()):
    """Accept prediction iff conf >= threshold and the predicted class is not
    on the clinician-defined always-confirm list. Everything else is routed
    to a human confirmation step."""
    accept = (conf >= threshold) & ~np.isin(pred, list(always_confirm))
    n_acc = int(accept.sum())
    return dict(
        threshold=float(threshold),
        coverage=float(accept.mean()),
        accepted_error=float((pred[accept] != y[accept]).mean()) if n_acc else float("nan"),
        deferred_error=float((pred[~accept] != y[~accept]).mean()) if (~accept).any() else float("nan"),
        n_accepted=n_acc, n_deferred=int((~accept).sum()),
        errors_caught=int(((pred != y) & ~accept).sum()), errors_total=int((pred != y).sum()),
    )
