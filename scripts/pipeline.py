#!/usr/bin/env python
"""One resumable driver for the whole study (designed for free Colab GPUs).

Re-running the same command after a disconnect continues where it stopped:
  * the parsed dataset is cached once on Drive (validated: exactly 6,600
    recordings) -> later sessions never touch the raw files again;
  * every training run is skipped if finished, and resumed from its last
    epoch checkpoint (partial.pt, every 5 epochs) if it was interrupted;
  * one-off stages (audit, recipe, analysis) leave a marker file;
  * everything is logged to <res>/pipeline.log.

    python scripts/pipeline.py --status          # what is done / what is left
    python scripts/pipeline.py                   # run (or continue) everything
    python scripts/pipeline.py --stages main     # only one stage
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAIN = ["gmn", "gmn_fd_control", "gmn_stdconv", "transformer", "stgcn", "bilstm", "gmn_v1"]
ABL = ["gmn_no_readout", "gmn_no_chirality", "gmn_frame_order0", "gmn_frame_order2",
       "gmn_world_only", "gmn_palm_only", "gmn_meanpool", "gmn_no_transformer"]
LMDHG_MODELS = ["lmdhg_gmn", "lmdhg_fd_control", "lmdhg_stdconv", "lmdhg_transformer", "lmdhg_bilstm"]
STAGES = ["data", "audit", "recipe", "main", "ablations", "analysis", "asreleased", "lmdhg", "paper"]
D = "/content/drive/MyDrive"


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leap_drive", default=f"{D}/GMN_Datasets/LeapGestureDB")
    ap.add_argument("--lmdhg_zip", default=f"{D}/GMN_Datasets/LMDHG.zip")
    ap.add_argument("--sweep_csv", default=f"{D}/gmn_results/sweep.csv")
    ap.add_argument("--res", default=f"{D}/gmn_results_v2_dedup")
    ap.add_argument("--res_asis", default=f"{D}/gmn_results_v2_asreleased")
    ap.add_argument("--local", default="/content", help="fast local disk for raw data copies")
    ap.add_argument("--expect_n", type=int, default=6600)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ablation_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--main_models", nargs="+", default=MAIN)
    ap.add_argument("--ablation_models", nargs="+", default=ABL)
    ap.add_argument("--lmdhg_models", nargs="+", default=LMDHG_MODELS)
    ap.add_argument("--status", action="store_true")
    return ap.parse_args()


class P:
    def __init__(self, a):
        self.a = a
        self.cache = os.path.join(a.res, "_cache", f"leapgesturedb_T100_n{a.expect_n}.npz")
        self.marks = os.path.join(a.res, "_stages")
        self.leap_local = os.path.join(a.local, "LeapGestureDB")
        os.makedirs(self.marks, exist_ok=True)
        self.log = open(os.path.join(a.res, "pipeline.log"), "a")

    def say(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n"); self.log.flush()

    def run(self, *cmd):
        cmd = [sys.executable, *cmd] if cmd[0].endswith(".py") else list(cmd)
        self.say("$ " + " ".join(cmd))
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if "Warning" in line or "warnings.warn" in line or "sched.step()" in line:
                continue
            print(line, end="", flush=True)
            self.log.write(line)
        self.log.flush()
        if p.wait() != 0:
            raise SystemExit(f"step failed (exit {p.returncode}); see {self.a.res}/pipeline.log. "
                             "Fix the cause and re-run the same command: finished work is kept.")

    def done(self, name):
        return os.path.exists(os.path.join(self.marks, name + ".done"))

    def mark(self, name, info=""):
        open(os.path.join(self.marks, name + ".done"), "w").write(info or time.ctime())

    def data_args(self):
        return ["--data_path", self.leap_local if os.path.isdir(self.leap_local) else self.a.leap_drive,
                "--cache", self.cache, "--expect_n", str(self.a.expect_n)]

    def recipe(self):
        r = json.load(open(os.path.join(self.a.res, "recipe.json")))
        return ["--lr", str(r["lr"]), "--weight_decay", str(r["weight_decay"]),
                "--batch_size", str(r["batch_size"]), "--epochs", str(self.a.epochs)]

    # ── stages ───────────────────────────────────────────────────────────────
    def data(self):
        from gmn.data_leap import load_leapgesturedb
        if os.path.exists(self.cache):
            import numpy as np
            if len(np.load(self.cache, allow_pickle=True)["y"]) == self.a.expect_n:
                self.say(f"data: validated cache found ({self.cache}) -- raw files not needed")
                return
        self.say("data: copying raw files from Drive to local disk (once per dataset)")
        os.makedirs(self.leap_local, exist_ok=True)
        shutil.copytree(self.a.leap_drive, self.leap_local, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("_gmn_cache*"))
        n = sum(f.lower().endswith(".txt") for _, _, fs in os.walk(self.leap_local) for f in fs)
        if n != self.a.expect_n:
            raise SystemExit(f"only {n}/{self.a.expect_n} files copied (Drive listing incomplete). "
                             "Wait a minute and re-run; nothing has been cached.")
        d = load_leapgesturedb(self.leap_local, cache=self.cache, expect_n=self.a.expect_n)
        self.say(f"data: parsed and cached {len(d.y)} recordings, {d.report['n_subjects']} subjects")

    def audit(self):
        if self.done("audit"):
            return self.say("audit: done earlier")
        self.run("scripts/data_audit.py", *self.data_args(), "--results", self.a.res)
        self.mark("audit")

    def recipe_stage(self):
        f = os.path.join(self.a.res, "recipe.json")
        if os.path.exists(f):
            return self.say(f"recipe: {open(f).read().strip()}")
        import pandas as pd
        sw = pd.read_csv(self.a.sweep_csv)
        g = sw[sw.model == "gmn"].sort_values("val_f1", ascending=False, kind="stable")
        b = g.iloc[0]
        r = dict(lr=float(b.lr), weight_decay=float(b.weight_decay), batch_size=int(b.batch_size),
                 rule="highest fold-0 validation macro-F1 of gmn; ties -> first row", val_f1=float(b.val_f1))
        shutil.copy(self.a.sweep_csv, os.path.join(self.a.res, "sweep.csv"))
        json.dump(r, open(f, "w"), indent=1)
        self.say(f"recipe: {r}")

    def cv(self, res, models, seeds, extra=()):
        self.run("scripts/run_cv.py", *self.data_args(), "--results", res, *self.recipe(),
                 "--folds", str(self.a.folds), "--seeds", *map(str, seeds), "--quiet",
                 "--models", *models, *extra)

    def analysis(self):
        need = len(self.a.main_models) * self.a.folds * len(self.a.seeds)
        have = self.count(self.a.res, self.a.main_models, self.a.seeds)
        if have < need:
            return self.say(f"analysis: waiting for main runs ({have}/{need})")
        tag = f"analysis_{have}_{self.count(self.a.res, self.a.ablation_models, self.a.ablation_seeds)}"
        if not self.done(tag):
            self.run("scripts/run_analysis.py", *self.data_args(), "--results", self.a.res, "--dedup")
            self.mark(tag)
        self.run("scripts/make_report.py", "--results", self.a.res)
        self.refresh()

    def refresh(self):
        rep = os.path.join(self.a.res, "report")
        args = ["scripts/refresh_tables.py", "--report", rep,
                "--pipeline_log", os.path.join(self.a.res, "pipeline.log")]
        asis = os.path.join(self.a.res, "report_asreleased")
        if os.path.exists(os.path.join(asis, "all_runs.csv")):
            args += ["--asis_report", asis]
        lm = os.path.join(self.a.res + "_lmdhg", "data_report.json")
        if os.path.exists(lm):
            args += ["--lmdhg_data_report", lm]
        self.run(*args)

    def asreleased(self):
        self.cv(self.a.res_asis, self.a.main_models, [0], ["--mc_passes", "0"])
        self.cv(self.a.res_asis, ["gmn", "gmn_v1"], [0], ["--mc_passes", "0", "--legacy_split", "--only_folds"])
        self.run("scripts/make_report.py", "--results", self.a.res_asis,
                 "--out", os.path.join(self.a.res, "report_asreleased"))

    def lmdhg(self):
        if not os.path.exists(self.a.lmdhg_zip):
            return self.say("lmdhg: zip not found, skipped")
        root = os.path.join(self.a.local, "LMDHG")
        if not glob.glob(root + "/**/DataFile*.mat", recursive=True):
            os.makedirs(root, exist_ok=True)
            self.run("unzip", "-q", "-o", self.a.lmdhg_zip, "-d", root)
        r = json.load(open(os.path.join(self.a.res, "recipe.json")))
        res = self.a.res + "_lmdhg"
        self.run("scripts/run_lmdhg.py", "--data_path", root, "--results", res,
                 "--lr", str(r["lr"]), "--weight_decay", str(r["weight_decay"]),
                 "--batch_size", str(r["batch_size"]), "--epochs", str(self.a.epochs),
                 "--seeds", *map(str, self.a.seeds), "--models", *self.a.lmdhg_models)
        self.run("scripts/make_lmdhg_report.py", "--results", res, "--out", os.path.join(self.a.res, "report"))
        self.refresh()

    def paper(self):
        rep = os.path.join(self.a.res, "report")
        if not os.path.exists(os.path.join(rep, "macros.tex")):
            return self.say("paper: report not generated yet")
        if not os.path.isfile(os.path.join(ROOT, "manuscript", "gmn_v2.tex")):
            return self.say("paper: no LaTeX sources in this copy -- the report folder is complete")
        link = os.path.join(ROOT, "manuscript", "report")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(rep, link)
        if shutil.which("latexmk") is None:
            subprocess.run("apt-get -qq update && apt-get -qq install -y latexmk texlive-latex-extra "
                           "texlive-fonts-recommended texlive-bibtex-extra", shell=True)
        if shutil.which("latexmk") is None:
            return self.say("paper: LaTeX not available here -- the report folder is complete; "
                            "build manuscript/gmn_v2.tex locally or on Overleaf")
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "gmn_v2.tex"],
                       cwd=os.path.join(ROOT, "manuscript"), stdout=subprocess.DEVNULL)
        pdf = os.path.join(ROOT, "manuscript", "gmn_v2.pdf")
        if os.path.exists(pdf):
            shutil.copy(pdf, os.path.join(self.a.res, "gmn_v2.pdf"))
            self.say(f"paper: {self.a.res}/gmn_v2.pdf")

    # ── status ──────────────────────────────────────────────────────────────
    def count(self, res, models, seeds, folds=None):
        folds = range(self.a.folds) if folds is None else folds
        return sum(os.path.exists(os.path.join(res, "runs", m, f"fold{f}_seed{s}", "metrics.json"))
                   for m in models for f in folds for s in seeds)

    def status(self):
        a = self.a
        print(f"code version {__import__('gmn').__version__}")
        print(f"dataset cache : {'OK' if os.path.exists(self.cache) else 'not yet'}  ({self.cache})")
        print(f"recipe        : {'OK' if os.path.exists(os.path.join(a.res, 'recipe.json')) else 'not yet'}")
        for title, res, models, seeds in (("main (dedup)", a.res, a.main_models, a.seeds),
                                          ("ablations", a.res, a.ablation_models, a.ablation_seeds),
                                          ("as released", a.res_asis, a.main_models, [0])):
            print(f"\n{title}:")
            for m in models:
                n = self.count(res, [m], seeds)
                part = len(glob.glob(os.path.join(res, "runs", m, "*", "partial.pt")))
                print(f"  {m:20s} {n:3d}/{a.folds * len(seeds)}" + (f"  (+{part} interrupted, will resume)" if part else ""))
        lm = a.res + "_lmdhg"
        print("\nLMDHG:")
        for m in a.lmdhg_models:
            n = sum(os.path.exists(os.path.join(lm, "runs", m, f"foldlmdhg_seed{s}", "metrics.json")) for s in a.seeds)
            print(f"  {m:20s} {n}/{len(a.seeds)}")
        print("\nreport:", "OK" if os.path.exists(os.path.join(a.res, "report", "macros.tex")) else "not yet")


def main():
    a = args()
    os.makedirs(a.res, exist_ok=True)
    p = P(a)
    if a.status:
        return p.status()
    p.say(f"=== pipeline start, stages: {a.stages}")
    for st in STAGES:
        if st not in a.stages:
            continue
        p.say(f"--- stage {st}")
        if st == "data":
            p.data()
        elif st == "audit":
            p.audit()
        elif st == "recipe":
            p.recipe_stage()
        elif st == "main":
            p.cv(a.res, a.main_models, a.seeds, ["--dedup"])
        elif st == "ablations":
            p.cv(a.res, a.ablation_models, a.ablation_seeds, ["--dedup"])
        elif st == "analysis":
            p.analysis()
        elif st == "asreleased":
            p.asreleased()
        elif st == "lmdhg":
            p.lmdhg()
        elif st == "paper":
            p.paper()
    p.say("=== pipeline finished")
    p.status()


if __name__ == "__main__":
    main()
