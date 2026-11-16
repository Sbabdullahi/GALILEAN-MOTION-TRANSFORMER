#!/usr/bin/env bash
# Full reproduction of every number in the paper.
# Usage: bash run_all.sh /path/to/LeapGestureDB /path/to/LMDHG /path/to/results
# On a Colab T4 the LeapGestureDB part takes several sessions; every step is
# resumable (finished runs are skipped), so just re-run the same command.
set -euo pipefail
LEAP=${1:?LeapGestureDB path}; LMDHG=${2:-}; RES=${3:-results}

pytest -q tests                                                     # properties the paper claims
python scripts/data_audit.py   --data_path "$LEAP" --results "$RES"
python scripts/sweep.py        --data_path "$LEAP" --results "$RES" # fold-0 validation only
python scripts/run_cv.py       --data_path "$LEAP" --results "$RES" --folds 5 --seeds 0 1 2
python scripts/run_cv.py       --data_path "$LEAP" --results "$RES" --legacy_split --only_folds \
                               --seeds 0 --models gmn gmn_v1       # audit of the v1 split
python scripts/run_analysis.py --data_path "$LEAP" --results "$RES"
python scripts/make_report.py  --results "$RES"
if [ -n "$LMDHG" ]; then
  python scripts/run_lmdhg.py  --data_path "$LMDHG" --results "${RES}_lmdhg"
  python scripts/make_lmdhg_report.py --results "${RES}_lmdhg" --out "$RES/report"
fi
