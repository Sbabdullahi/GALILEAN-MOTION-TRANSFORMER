#!/usr/bin/env python
"""Dataset audit that is actually connected to the pipeline that is run:
samples per subject and class, raw recording lengths, missing fields,
tracking dropouts and duplicate recordings. Replaces the v1 Shapiro-Wilk /
IQR / collinearity analysis (which did not inform the model and whose
conclusions were not supported by tests). Writes
<results>/report/tab_data.tex and data_audit.json."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gmn.constants import GESTURE_NAMES
from gmn.data_leap import load_leapgesturedb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--expect_n", type=int, default=None)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--cache", default=None, help="validated .npz cache (e.g. on Drive); raw files not needed if it exists")
    a = ap.parse_args()
    d = load_leapgesturedb(a.data_path, cache=a.cache, expect_n=a.expect_n)
    out = os.path.join(a.results, "report")
    os.makedirs(out, exist_ok=True)
    subj, y = d.subject, d.y
    per_subj = np.array([np.sum(subj == s) for s in np.unique(subj)])
    per_cell = np.array([[np.sum((subj == s) & (y == c)) for c in range(11)] for s in np.unique(subj)])
    flat = d.X.reshape(len(d.X), -1)
    _, first, inverse, counts = np.unique(flat, axis=0, return_index=True, return_inverse=True,
                                          return_counts=True)
    inverse = np.asarray(inverse).ravel()
    dup_groups = []
    for g in np.where(counts > 1)[0]:
        members = np.where(inverse == g)[0]
        dup_groups.append(dict(files=d.files[members].tolist(),
                               subjects=sorted(set(int(x) for x in subj[members])),
                               labels=sorted(set(int(x) for x in y[members])),
                               raw_lengths=d.orig_len[members].tolist()))
    cross_subject = [g for g in dup_groups if len(g["subjects"]) > 1]
    cross_label = [g for g in dup_groups if len(g["labels"]) > 1]
    json.dump(dict(groups=dup_groups), open(os.path.join(out, "duplicates.json"), "w"), indent=1)
    short = np.where(d.orig_len < 20)[0]
    long_ = np.where(d.orig_len > 2000)[0]
    json.dump(dict(short=[(str(d.files[i]), int(d.orig_len[i])) for i in short],
                   long=[(str(d.files[i]), int(d.orig_len[i])) for i in long_]),
              open(os.path.join(out, "length_outliers.json"), "w"), indent=1)
    static = np.array([np.allclose(x[0], x[-1]) and np.ptp(x[:, 5:8], 0).max() < 1e-3 for x in d.X])
    rep = dict(
        n_samples=int(len(y)), n_subjects=int(len(np.unique(subj))),
        class_counts=dict(zip(GESTURE_NAMES, np.bincount(y, minlength=11).tolist())),
        samples_per_subject=dict(min=int(per_subj.min()), max=int(per_subj.max()),
                                 median=float(np.median(per_subj))),
        samples_per_subject_class=dict(min=int(per_cell.min()), max=int(per_cell.max())),
        raw_length_frames=dict(min=int(d.orig_len.min()), median=float(np.median(d.orig_len)),
                               max=int(d.orig_len.max())),
        exact_duplicate_recordings=int((counts > 1).sum()),
        duplicate_files_total=int(sum(len(g["files"]) for g in dup_groups)),
        duplicate_groups_across_subjects=len(cross_subject),
        duplicate_groups_with_conflicting_labels=len(cross_label),
        recordings_shorter_than_20_frames=int(len(short)),
        recordings_longer_than_2000_frames=int(len(long_)),
        static_recordings=int(static.sum()),
        parse_failures=len(d.report["parse_failures"]),
        unmatched_files=len(d.report["unmatched_files"]),
        frames_with_missing_fields=d.report["frames_with_missing_fields"],
    )
    json.dump(rep, open(os.path.join(out, "data_audit.json"), "w"), indent=1)
    rows = [
        ("Recordings parsed", rep["n_samples"]), ("Subjects", rep["n_subjects"]),
        ("Recordings per subject (min / median / max)",
         f"{per_subj.min()} / {np.median(per_subj):.0f} / {per_subj.max()}"),
        ("Recordings per subject and class (min / max)", f"{per_cell.min()} / {per_cell.max()}"),
        ("Raw length in frames (min / median / max)",
         f"{d.orig_len.min()} / {np.median(d.orig_len):.0f} / {d.orig_len.max()}"),
        ("Files that failed to parse", rep["parse_failures"]),
        ("Frames with a missing field (forward-filled)", rep["frames_with_missing_fields"]),
        ("Groups of identical recordings (files involved)",
         f"{rep['exact_duplicate_recordings']} ({rep['duplicate_files_total']})"),
        ("  of which span different subjects", rep["duplicate_groups_across_subjects"]),
        ("  of which carry different labels", rep["duplicate_groups_with_conflicting_labels"]),
        ("Recordings $<$20 / $>$2000 raw frames",
         f"{rep['recordings_shorter_than_20_frames']} / {rep['recordings_longer_than_2000_frames']}"),
        ("Static recordings (no palm motion)", rep["static_recordings"]),
    ]
    open(os.path.join(out, "tab_data.tex"), "w").write(
        "\\begin{tabular}{lr}\n\\toprule\nCheck & Value \\\\\n\\midrule\n" +
        "\n".join(f"{k} & {v} \\\\" for k, v in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
