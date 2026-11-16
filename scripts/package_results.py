#!/usr/bin/env python
"""Bundle the results folders for archiving (e.g. Zenodo) WITHOUT any data
derived from the datasets themselves.

Included: split files, recipe, sweep, de-duplication report (folder names
only), per-run metrics/predictions/histories/weights, analysis outputs, the
generated report and the pipeline log.
Excluded: the parsed-dataset cache (_cache/), unfinished-run checkpoints
(partial.pt) and stage markers -- the cache is a copy of the recordings and
may only be redistributed under the dataset providers' terms.

    python scripts/package_results.py --res /content/drive/MyDrive/gmn_results_v2_dedup \
        --extra /content/drive/MyDrive/gmn_results_v2_asreleased \
                /content/drive/MyDrive/gmn_results_v2_dedup_lmdhg \
        --out /content/drive/MyDrive/GMN_v2_results.zip
"""
import argparse
import hashlib
import json
import os
import zipfile

EXCLUDE_DIRS = {"_cache", "_stages", "__pycache__"}
EXCLUDE_FILES = {"partial.pt", "partial.pt.tmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_weights", action="store_true", help="omit model.pt files (much smaller)")
    ap.add_argument("--keep_names", action="store_true",
                    help="keep subject folder names (first names); default replaces them by S001, S002, ...")
    a = ap.parse_args()
    names = []
    for root in [a.res, *a.extra]:
        f = os.path.join(root, "data_report.json")
        if os.path.exists(f):
            names = json.load(open(f)).get("subject_folders", []) or names
    mapping = {} if a.keep_names else {n: f"S{i+1:03d}" for i, n in enumerate(names)}
    # longest names first so that e.g. "118nairouz" is replaced before "18nairouz"
    order = sorted(mapping, key=len, reverse=True)

    def scrub(data: bytes) -> bytes:
        if not order:
            return data
        t = data.decode("utf-8", errors="surrogateescape")
        for n in order:
            t = t.replace(n, mapping[n])
        return t.encode("utf-8", errors="surrogateescape")
    manifest, total = [], 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root in [a.res, *a.extra]:
            base = os.path.basename(root.rstrip("/"))
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
                for f in fns:
                    if f in EXCLUDE_FILES or f.endswith(".tmp") or (a.no_weights and f == "model.pt"):
                        continue
                    if f.startswith("_gmn_cache") or f.endswith(".npz") and "_cache" in dp:
                        continue
                    src = os.path.join(dp, f)
                    arc = os.path.join(base, os.path.relpath(src, root))
                    data = open(src, "rb").read()
                    if f.endswith((".json", ".log", ".csv", ".txt", ".tex")):
                        data = scrub(data)
                    z.writestr(arc, data)
                    manifest.append(dict(path=arc, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                    total += len(data)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
        z.writestr("README.txt", (
            "Results of GMN v2 (see the code repository README for how they were produced).\n"
            "runs/<model>/fold<k>_seed<s>/: metrics.json, arrays.npz (logits, labels, test indices),\n"
            "history.json, model.pt. report/: every table, macro and figure used in the paper.\n"
            "No recordings from LeapGestureDB or LMDHG are included; obtain them from their providers.\n"
            "MANIFEST.json lists every file with its SHA-256 checksum.\n"
            + ("Subject folder names were replaced by S001..S120 in sorted-folder order.\n" if mapping else "")))
    print(f"{len(manifest)} files, {total/1e6:.1f} MB uncompressed -> {a.out}")


if __name__ == "__main__":
    main()
