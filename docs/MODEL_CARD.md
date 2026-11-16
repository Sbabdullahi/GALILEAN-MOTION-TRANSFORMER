# Model card — GMN v2 (Galilean-frame gesture classifier)

Format after Mitchell et al. (2019).

## Model details
- Developed by the S.B. Abdullahi. Version 2.0. Licence: MIT (code). Weights: released with the research.
- Type: a skeleton-sequence classifier. It has a Galilean-frame convolution stem (input-dependent least-squares kinematics) and a palm-frame rotation-invariant stream, followed by a 4-layer transformer encoder and class-query pooling. Temperature scaling and a deferral threshold are fitted on validation subjects.

## Intended use
- **Research only.** It is a benchmark model for touchless gesture recognition from Leap Motion skeletal data, plus a reference implementation of selective prediction with a human in the loop.
- **Out of scope:** any clinical use, control of devices in a patient-care setting, or use as a medical device. The model has never been tested with surgeons, in an operating room, in a closed real-time loop, or on gestures outside its 11-command vocabulary.

## Human-in-the-loop design
- The model **proposes** command scheme; it does not execute them. A command is auto-accepted only when its calibrated confidence is above a threshold, chosen on validation subjects for a target error that people set in advance (default 1 %). All other commands go to a confirmation step.
- An `always_confirm` list (`gmn.metrics.apply_deferral`) lets clinicians require confirmation for specific commands (e.g. contrast changes) whatever the model's confidence. It is empty by default because choosing it is a clinical decision.
- The cost of deferral is extra interaction time. Report coverage together with error. Never report error alone.

## Training data
- LeapGestureDB: 120 subjects, 11 commands, one Leap Motion controller, laboratory setting. See `results/report/data_audit.json`.
- Known gaps: no surgical gloves, no OR lighting or occlusion, no surgeons, limited demographic reporting in the source dataset.

## Evaluation
- Subject-wise 5-fold CV × 3 seeds. Metrics: accuracy, macro-F1, macro AUPRC/AUROC, ECE/NLL/Brier before and after calibration, coverage and accepted-set error of the deferral policy, AURC, per-subject accuracy distribution, and accuracy under sensor rotation.
- Disaggregation: by class (confusion matrix) and by subject (`fig_subjects.pdf`). Demographic disaggregation is not possible with the available metadata.

## Ethical considerations
- Hand-motion trajectories can identify people (biometric). Treat recordings as personal data, and obtain consent before collecting data from clinical staff.
- Over-trust: calibrated confidence can still be wrong on new users or sensors. The deferral rate and errors should be monitored after any change of setting.
- The hand-id field is zeroed by default, because it may encode recording order (a shortcut).

## Caveats
- Velocities are in hand-spans per normalised gesture duration. Absolute speed is lost when recordings are resampled to 100 frames.
- Attributions (Integrated Gradients) describe the model. They do not prove it reasons correctly.
