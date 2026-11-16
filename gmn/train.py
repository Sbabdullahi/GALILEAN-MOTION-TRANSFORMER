"""One training run = (model, split/fold, seed). Saves everything needed to
recompute every number in the research without re-training."""
from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import AugConfig, augment
from .metrics import (calibration_report, choose_threshold, apply_deferral,
                      classification_metrics, fit_temperature, mc_dropout_summary,
                      risk_coverage)
from .models import build_model, count_parameters


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    patience: int = 12
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    select_on: str = "val_macro_f1"
    amp: bool = True
    mc_passes: int = 25
    target_risk: float = 0.01          # HITL: max error tolerated on auto-accepted commands
    checkpoint_every: int = 5          # epochs between resumable checkpoints (partial.pt)
    aug: AugConfig = field(default_factory=AugConfig)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_info():
    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, gpu=(torch.cuda.get_device_name(0)
                                              if torch.cuda.is_available() else None),
                platform=platform.platform())


@torch.no_grad()
def predict_logits(model, X, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs]).float() for i in range(0, len(X), bs)]).cpu().numpy()


@torch.no_grad()
def mc_dropout(model, X, passes, bs=256):
    """Dropout layers in train mode, everything else (BatchNorm) in eval."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    out = []
    for _ in range(passes):
        out.append(torch.cat([F.softmax(model(X[i:i + bs]).float(), -1)
                              for i in range(0, len(X), bs)]).cpu().numpy())
    model.eval()
    return np.stack(out)


def run_one(model_name: str, X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
            split: dict, seed: int, out_dir: str, cfg: TrainConfig = TrainConfig(),
            model_overrides: dict | None = None, save_checkpoint: bool = False,
            device: str | None = None, verbose: bool = True, n_classes: int = 11,
            augment_fn=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "metrics.json")
    if os.path.exists(done):                       # resumable
        return json.load(open(done))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    idx = {k: np.where(np.isin(subjects, split[k]))[0] for k in ("train", "val", "test")}
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    Xtr, ytr = T(X[idx["train"]]), T(y[idx["train"]], torch.long)
    Xv, yv = T(X[idx["val"]]), y[idx["val"]]
    Xte, yte = T(X[idx["test"]]), y[idx["test"]]

    model = build_model(model_name, **(model_overrides or {})).to(device)
    model.set_normaliser(Xtr)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(Xtr) / cfg.batch_size))
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=device).manual_seed(seed)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    augment_fn = augment_fn or (lambda xb, g: augment(xb, cfg.aug, g))

    best, best_state, bad, hist = -1.0, None, 0, []
    t0, ep0, elapsed0 = time.time(), 0, 0.0
    # ── resume an interrupted run (Colab disconnects) ─────────────────────
    partial = os.path.join(out_dir, "partial.pt")
    if os.path.exists(partial):
        try:
            ck = torch.load(partial, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
            best, best_state, bad, hist = ck["best"], ck["best_state"], ck["bad"], ck["hist"]
            ep0, elapsed0 = ck["epoch"], ck.get("elapsed", 0.0)
            if verbose:
                print(f"resuming {out_dir} from epoch {ep0}")
        except Exception as e:                       # corrupt file from a kill mid-write
            print(f"could not resume {partial} ({e}); restarting this run")
    stopped = bad >= cfg.patience
    for ep in range(ep0, cfg.epochs):
        if stopped:
            break
        model.train()
        perm = torch.randperm(len(Xtr), device=device, generator=gen)
        tl, n = 0.0, 0
        for i in range(0, len(Xtr), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            xb = augment_fn(Xtr[b], gen)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = crit(model(xb), ytr[b])
                if hasattr(model, "extra_loss"):
                    loss = loss + model.extra_loss()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(b)
            n += len(b)
        vm = classification_metrics(yv, predict_logits(model, Xv), n_classes)
        score = vm["macro_f1"] if cfg.select_on == "val_macro_f1" else vm["accuracy"]
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_acc=vm["accuracy"],
                         val_f1=vm["macro_f1"]))
        if verbose:
            print(f"[{model_name} f{split['fold']} s{seed}] ep {ep+1:3d} loss {tl/n:.4f} "
                  f"val acc {vm['accuracy']:.4f} f1 {vm['macro_f1']:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        stopped = bad >= cfg.patience
        if (ep + 1) % cfg.checkpoint_every == 0 and not stopped and ep + 1 < cfg.epochs:
            tmp = partial + ".tmp"
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), gen=gen.get_state(), best=best,
                            best_state=best_state, bad=bad, hist=hist, epoch=ep + 1,
                            elapsed=elapsed0 + time.time() - t0), tmp)
            os.replace(tmp, partial)                  # atomic: never leaves a half-written file
    train_time = elapsed0 + time.time() - t0
    model.load_state_dict(best_state)

    # ── evaluation (test touched exactly once, after model selection) ──────
    lv, lt = predict_logits(model, Xv), predict_logits(model, Xte)
    Tstar = fit_temperature(yv, lv)
    res = dict(model=model_name, fold=split["fold"], seed=seed, n_params=n_params,
               epochs_run=len(hist), best_epoch=int(np.argmax([h["val_f1"] for h in hist]) + 1),
               train_time_s=train_time, config=asdict(cfg), overrides=model_overrides or {},
               env=environment_info(), split_sizes={k: int(len(v)) for k, v in idx.items()},
               val=classification_metrics(yv, lv, n_classes),
               test=classification_metrics(yte, lt, n_classes),
               calibration_test=calibration_report(yte, lt, Tstar))

    # HITL deferral with calibrated max-probability; threshold chosen on val
    from scipy.special import softmax
    pv, pt = softmax(lv / Tstar, 1), softmax(lt / Tstar, 1)
    thr = choose_threshold(yv, pv.argmax(1), pv.max(1), cfg.target_risk)
    res["deferral_msp"] = apply_deferral(yte, pt.argmax(1), pt.max(1), thr)
    _, _, res["aurc_msp"] = risk_coverage(yte, pt.argmax(1), pt.max(1))
    arrays = dict(val_logits=lv, test_logits=lt, val_y=yv, test_y=yte,
                  test_subject=subjects[idx["test"]], test_index=idx["test"])

    if cfg.mc_passes:
        mcv = mc_dropout_summary(mc_dropout(model, Xv, cfg.mc_passes))
        mct = mc_dropout_summary(mc_dropout(model, Xte, cfg.mc_passes))
        conf_v, conf_t = -mcv["entropy"], -mct["entropy"]
        pred_v, pred_t = mcv["mean"].argmax(1), mct["mean"].argmax(1)
        thr_e = choose_threshold(yv, pred_v, conf_v, cfg.target_risk)
        res["deferral_mc_entropy"] = apply_deferral(yte, pred_t, conf_t, thr_e)
        _, _, res["aurc_mc_entropy"] = risk_coverage(yte, pred_t, conf_t)
        res["mc_test"] = dict(mean_entropy=float(mct["entropy"].mean()),
                              mean_mutual_info=float(mct["mutual_info"].mean()),
                              entropy_correct=float(mct["entropy"][pred_t == yte].mean()),
                              entropy_wrong=float(mct["entropy"][pred_t != yte].mean())
                              if (pred_t != yte).any() else float("nan"))
        arrays.update(test_mc_mean=mct["mean"], test_mc_entropy=mct["entropy"],
                      test_mc_mi=mct["mutual_info"])

    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), **arrays)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    if save_checkpoint:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    tmp = done + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, done)                              # metrics.json appears only when complete
    if os.path.exists(partial):
        os.remove(partial)
    return res
