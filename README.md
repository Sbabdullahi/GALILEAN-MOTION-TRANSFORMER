# GALILEAN-MOTION-TRANSFORMER
A physics-informed, interpretable skeleton-based gesture recognition system
# Galilean Motion Network (GMN)
### Physics-Informed Gesture Recognition for Surgical Interfaces

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-LeapGestureDB-lightgrey)](https://example.com)

---

## Overview

GMN is an end-to-end deep learning architecture that recognises contactless hand gestures from Leap Motion skeletal data. It embeds discrete-time **velocity and acceleration** as learnable convolutional parameters (Galilean motion encoding), couples these with **quaternion rotational invariance**, **Vision Transformer** temporal attention, and **cross-modal task-prompt fusion** into a single trainable pipeline.

The integrated XAI suite (Integrated Gradients, Attention Rollout, MC-Dropout, Temperature Calibration) provides per-sample transparency without post-hoc surrogate models.

---

## Repository Structure

```
gmn/
├── main.py            # Data loading · Architecture · Training · Test evaluation
├── evaluate.py        # Confusion matrix · ROC · Per-class F1 · HTML report
├── xai_dashboard.py   # Integrated Gradients · MC-Dropout · Streamlit dashboard
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Dataset

**LeapGestureDB** — 6,600 recordings from 120 subjects, 11 surgical command classes (600 per class), 100 frames per sequence, 26 semantic Leap Motion features per frame.

```
LeapGestureDB/
├── Subject_1/
│   ├── G1_001.txt      # Click
│   ├── G2_001.txt      # Left rotation (CW)
│   └── ...
├── Subject_2/
└── ...
```

Each `.txt` file encodes one gesture recording in the Leap Motion SDK output format. The parser in `main.py` extracts 26 features per frame:

| Index | Feature | Index | Feature |
|---|---|---|---|
| 0 | HandID | 1 | FingerCount |
| 2–4 | HandDirection (X,Y,Z) | 5–7 | PalmPosition (X,Y,Z) |
| 8–10 | PalmNormal (X,Y,Z) | 11–13 | ThumbTip (X,Y,Z) |
| 14–16 | IndexTip (X,Y,Z) | 17–19 | MiddleTip (X,Y,Z) |
| 20–22 | RingTip (X,Y,Z) | 23–25 | PinkyTip (X,Y,Z) |

---

## Installation

```bash
git clone https://github.com/Sbabdullahi/gmn.git
cd gmn
pip install -r requirements.txt
```

> **GPU:** PyTorch will use CUDA automatically if available.  
> **Google Colab:** All three scripts run without modification; see Colab notes below.

---

## Usage

### 1 — Train and evaluate

```bash
python main.py --data_path /path/to/LeapGestureDB
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `/content/LeapGestureDB` | Path to dataset root |
| `--epochs` | `60` | Maximum training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--patience` | `12` | Early stopping patience |
| `--pw` | `1.0` | Physics loss weight |
| `--ckpt` | `best_gmn.pt` | Checkpoint save path |

**Outputs:** `best_gmn.pt` · `training_curves.png` · `training_history.json`

---

### 2 — Evaluate (plots + HTML report)

```bash
# Interactive plots only
python evaluate.py --data_path /path/to/LeapGestureDB

# + save self-contained HTML report
python evaluate.py --data_path /path/to/LeapGestureDB --save_html
```

**Outputs:** confusion matrix · ROC curves · per-class F1 · cross-class confidence · `GMN_Evaluation_Report.html`

---

### 3a — Offline XAI (single sample)

```bash
python xai_dashboard.py --data_path /path/to/LeapGestureDB --sample_idx 0
```

**Outputs:** IG heatmap · feature importance ranking · attention rollout · MC-Dropout uncertainty gauge · 3-D trajectory · `xai_report.html`

---

### 3b — Real-time Streamlit dashboard

```bash
streamlit run xai_dashboard.py
```

Then open `http://localhost:8501` in your browser.

**Dashboard features:**
- Simulate any of 11 gesture classes or upload a real `.npy` sequence (shape `100 × 26`)
- Calibrated confidence bar chart with CW/CCW highlighted
- MC-Dropout uncertainty with entropy gauge
- Gradient × input feature attribution (fast approximation)
- 3-D hand trajectory viewer

---

## Google Colab

**Train:**
```python
!python main.py --data_path /content/LeapGestureDB --epochs 60
```

**Dashboard via ngrok (new cell):**
```python
!pip install pyngrok -q
from pyngrok import ngrok, conf
conf.get_default().auth_token = "YOUR_NGROK_TOKEN"   # free at ngrok.com
import subprocess, time
subprocess.Popen(["streamlit","run","xai_dashboard.py",
                  "--server.port","8501","--server.headless","true"])
time.sleep(5)
print(ngrok.connect(8501,"http"))   # click the printed URL
```

---

## Architecture

```
Input (B, 100, 26)
      │
      ▼
Linear projection → embed_dim=256
      │
      ▼
4× Galilean Conv1D  (k=3,5,7,9)    ← learnable velocity + acceleration
      │
      ▼
QuaternionLayer (9 joints × 4)      ← rotational invariance
      │
      ▼
Fusion projection  (256+36 → 256)
      │
      ▼
6× Vision Transformer Block         ← temporal self-attention
      │
      ▼
CrossModal Fusion  (task prompt)    ← lang_dim=768
      │
      ▼
MLP classifier  (768→512→256→11)
      │
      ▼
Logits  →  Temperature Scaling  →  Calibrated probabilities
                                   MC-Dropout uncertainty
```

**Total parameters:** ~10.5M · **Model size:** 41.0 MB · **Training:** 26 epochs, ~45 min on Tesla T4

---


--

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{gmn2025,
  title   = {Galilean Motion Network: Physics-Informed Gesture Recognition
             for Contactless Surgical Interfaces},
  author  = {S.B. Abdullahi, et al.},
  journal = {YOUR JOURNAL},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
