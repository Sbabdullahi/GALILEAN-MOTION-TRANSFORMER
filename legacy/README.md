# legacy/ — v1 code, unchanged

The files of the first public release, kept for reference.
Note: `evaluate.py` and `xai_dashboard.py` import `GESTURE_ICONS`, which `main.py` does not define, so they fail at import time.
Do not use this code for new results; `gmn.models.GMNv1` is a faithful, tested port (identical parameter count).
