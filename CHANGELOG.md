# Changelog

## v2.0.0
- Galilean-frame convolution with input-dependent least-squares kinematics, exact invariance and unit tests.
- Palm-frame rotation-invariant stream from measured orientation; true quaternions.
- Class-query attention pooling.
- Leakage-free protocol: de-duplication of identical recordings and merging of subject folders that share them; released split files.
- Re-implemented baselines (BiLSTM, Transformer, ST-GCN, v1) and controls trained by one script; 5-fold subject-wise CV × 3 seeds; corrected resampled tests and confidence intervals with Holm correction.
- Calibration, MC-Dropout, validated human-in-the-loop deferral, rotation robustness, attribution sanity checks, cost profiling.
- LMDHG under the standard sequence split with ground-truth segments and the label inventory found in the files.
- Every number, table and figure generated from saved per-run predictions, with consistency checks.
- Resumable pipeline (`scripts/pipeline.py`) for interrupted GPU sessions.

## v1.0 — first public release
Preserved unchanged in `legacy/` and under the git tag `v1.0-published`.
