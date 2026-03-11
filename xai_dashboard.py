"""
xai_dashboard.py — GMN Interpretability & Real-Time Dashboard
==============================================================
Covers: Integrated Gradients · Attention Rollout · MC-Dropout
        Temperature Calibration · Streamlit live dashboard

Usage (offline XAI on a single sample):
    python xai_dashboard.py --data_path /path/to/LeapGestureDB --sample_idx 0

Usage (launch Streamlit dashboard):
    streamlit run xai_dashboard.py

Usage (Google Colab — new cell):
    !pip install pyngrok -q
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = "YOUR_TOKEN"
    import subprocess, time
    subprocess.Popen(["streamlit","run","xai_dashboard.py",
                      "--server.port","8501","--server.headless","true"])
    time.sleep(5)
    print(ngrok.connect(8501,"http"))

Outputs (offline mode):
    xai_report.html   — self-contained interactive XAI report
"""

import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import shared components
from main import (GalileanMotionTransformer, load_leapgesturedb,
                  GESTURE_NAMES, GESTURE_ICONS, FEATURE_NAMES,
                  NUM_CLASSES, INPUT_DIM, SEQ_LENGTH)

# ─────────────────────────────────────────────────────────────────────────────
#  Dark theme
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(plot_bgcolor='#0a0f1e', paper_bgcolor='#0a0f1e',
            font=dict(color='#8fb8d8', family='Courier New, monospace'))

# ─────────────────────────────────────────────────────────────────────────────
#  XAI utilities
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.to(logits.device)

    def fit(self, model, val_loader, device, lr=0.01, max_iter=50):
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits_list.append(model(x)['logits'].cpu())
                labels_list.append(y.cpu())
        la  = torch.cat(logits_list)
        ll  = torch.cat(labels_list)
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        crit = nn.CrossEntropyLoss()
        def step():
            opt.zero_grad()
            loss = crit(la / self.temperature, ll)
            loss.backward(); return loss
        opt.step(step)
        print(f"  [Calibration] Optimal temperature: {self.temperature.item():.4f}")
        return self


def integrated_gradients(model, x, target_cls, steps=40):
    """
    Compute Integrated Gradients attribution.
    x : (1, T, 26) tensor on device
    Returns ndarray (T, 26) — one attribution per frame per feature.
    """
    baseline = torch.zeros_like(x)
    grads    = []
    for alpha in torch.linspace(0, 1, steps):
        inp   = (baseline + alpha * (x - baseline)).clone().detach().requires_grad_(True)
        score = model(inp)['logits'][0, target_cls]
        model.zero_grad(); score.backward()
        grads.append(inp.grad.cpu().numpy()[0])
    avg_g = np.mean(grads, axis=0)
    return (x.cpu().numpy()[0] - baseline.cpu().numpy()[0]) * avg_g  # (T, 26)


def mc_dropout_predict(model, x, n_samples=25):
    """
    MC-Dropout uncertainty estimation.
    Returns mean_probs (B,C), std_probs (B,C), entropy (B,).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(F.softmax(model(x)['logits'], dim=-1).cpu().numpy())
    model.eval()
    s = np.stack(samples)
    mean_p  = s.mean(0); std_p = s.std(0)
    entropy = -np.sum(mean_p * np.log(mean_p + 1e-9), axis=-1)
    return mean_p, std_p, entropy


def get_attention_rollout(model, x):
    """Capture ViT attention weights via hooks and compute rollout."""
    attns = []; hooks = []
    for blk in model.vit:
        def _hook(mod, inp, out):
            if isinstance(out, tuple) and out[1] is not None:
                attns.append(out[1].detach().cpu())
        hooks.append(blk.attn.register_forward_hook(_hook))
    model.eval()
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    if not attns: return None
    result = attns[0]
    for a in attns[1:]: result = torch.bmm(result, a)
    result = result / (result.sum(-1, keepdim=True) + 1e-9)
    return result.numpy()  # (B, T, T)


# ─────────────────────────────────────────────────────────────────────────────
#  XAI plot builders
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_bar(cal_probs, pred_cls):
    si     = np.argsort(cal_probs)[::-1]
    colors = ['#00d4ff' if k == 0 else '#1e3a5f' for k in range(NUM_CLASSES)]
    fig    = go.Figure(go.Bar(
        x=cal_probs[si],
        y=[f"{GESTURE_ICONS[i]}  {GESTURE_NAMES[i]}" for i in si],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v*100:.1f}%' for v in cal_probs[si]],
        textposition='outside'))
    fig.update_layout(**DARK,
        title=dict(text=f'<b>Calibrated Confidence</b>  — '
                        f'{GESTURE_ICONS[pred_cls]} {GESTURE_NAMES[pred_cls]} '
                        f'({cal_probs.max()*100:.1f}%)',
                   font=dict(color='#00d4ff', size=14)),
        xaxis=dict(range=[0,1.18], tickformat='.0%',
                   gridcolor='#122030', color='#5a7fa8'),
        yaxis=dict(categoryorder='array',
                   categoryarray=[f"{GESTURE_ICONS[i]}  {GESTURE_NAMES[i]}"
                                   for i in si][::-1], color='#8fb8d8'),
        height=430, margin=dict(l=220, r=70, t=70, b=40))
    return fig


def plot_ig_heatmap(attrs):
    """attrs: (T, 26) numpy array."""
    T = attrs.shape[0]
    fig = go.Figure(go.Heatmap(
        z=attrs.T,  # (26, T)
        x=[f't{i}' for i in range(T)],
        y=FEATURE_NAMES,
        colorscale='RdBu_r', zmid=0,
        colorbar=dict(title='Attribution',
                      tickfont=dict(color='#8fb8d8', size=9))))
    fig.update_layout(**DARK,
        title=dict(text='<b>Integrated Gradients</b>  — Feature × Frame',
                   font=dict(color='#00d4ff', size=14)),
        xaxis=dict(title='Frame', color='#5a7fa8', nticks=10, gridcolor='#122030'),
        yaxis=dict(title='Feature', color='#8fb8d8', autorange='reversed'),
        height=640, margin=dict(l=170, r=60, t=70, b=60))
    return fig


def plot_feature_importance(attrs):
    """attrs: (T, 26) — mean |IG| per feature."""
    fi     = np.abs(attrs).mean(0)   # (26,)
    sort_f = np.argsort(fi)[::-1]
    fig    = go.Figure(go.Bar(
        x=[FEATURE_NAMES[f] for f in sort_f],
        y=fi[sort_f],
        marker=dict(color=fi[sort_f], colorscale='Plasma', line=dict(width=0)),
        text=[f'{v:.4f}' for v in fi[sort_f]], textposition='outside'))
    fig.update_layout(**DARK,
        title=dict(text='<b>Feature Importance</b>  (mean |IG| across frames)',
                   font=dict(color='#00d4ff', size=14)),
        xaxis=dict(tickangle=-55, color='#5a7fa8'),
        yaxis=dict(title='|Attribution|', color='#8fb8d8', gridcolor='#122030'),
        height=480, margin=dict(l=60, r=40, t=70, b=160))
    return fig


def plot_attention_rollout(rollout):
    fig = go.Figure(go.Heatmap(
        z=rollout[0], colorscale='Viridis',
        colorbar=dict(title='Attn Weight',
                      tickfont=dict(color='#8fb8d8', size=9))))
    fig.update_layout(**DARK,
        title=dict(text='<b>ViT Attention Rollout</b>  — Frame-to-Frame Affinity',
                   font=dict(color='#00d4ff', size=14)),
        xaxis=dict(title='Key Frame', color='#5a7fa8', gridcolor='#122030'),
        yaxis=dict(title='Query Frame', color='#8fb8d8',
                   autorange='reversed', gridcolor='#122030'),
        height=500, margin=dict(l=80, r=60, t=70, b=60))
    return fig


def plot_mc_dropout(mean_p, std_p, ent_val):
    max_ent = np.log(NUM_CLASSES)
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Mean Probability ± Epistemic Uncertainty',
                        'Prediction Entropy'],
        column_widths=[0.68, 0.32],
        specs=[[{'type':'xy'}, {'type':'indicator'}]])
    fig.add_trace(go.Bar(
        x=[f"{GESTURE_ICONS[i]} {GESTURE_NAMES[i][:12]}" for i in range(NUM_CLASSES)],
        y=mean_p,
        error_y=dict(type='data', array=std_p, color='#ff6b6b', thickness=2, width=8),
        marker=dict(color=mean_p, colorscale='Blues', cmin=0, cmax=1),
        name='Mean Prob'), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=ent_val / max_ent * 100,
        title=dict(text='Uncertainty %', font=dict(color='#8fb8d8')),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor='#5a7fa8'),
            bar=dict(color='#00d4ff'),
            steps=[dict(range=[0,33],  color='#0a3d2e'),
                   dict(range=[33,66], color='#3d3a0a'),
                   dict(range=[66,100],color='#3d0a0a')],
            threshold=dict(line=dict(color='#ff6b6b',width=4), value=66)),
        number=dict(suffix='%', font=dict(color='#e0e0e0'))), row=1, col=2)
    fig.update_layout(**DARK,
        title=dict(text='<b>MC-Dropout Uncertainty</b>  (25 passes)',
                   font=dict(color='#00d4ff', size=14)),
        height=440, showlegend=False,
        margin=dict(l=60, r=40, t=70, b=100))
    fig.update_xaxes(tickangle=-35, color='#5a7fa8', row=1, col=1)
    fig.update_yaxes(range=[0,1], color='#8fb8d8', gridcolor='#122030', row=1, col=1)
    return fig


def plot_3d_trajectory(xi_np):
    KEY = {'HandDirection':2, 'PalmPosition':5,
           'ThumbTip':11, 'IndexTip':14, 'PinkyTip':23}
    COL = {'HandDirection':'#00cfff','PalmPosition':'#ff6060',
           'ThumbTip':'#4dffb4','IndexTip':'#ff9f40','PinkyTip':'#a259ff'}
    fig = go.Figure()
    for name, start in KEY.items():
        fig.add_trace(go.Scatter3d(
            x=xi_np[:,start], y=xi_np[:,start+2], z=xi_np[:,start+1],
            mode='lines+markers', name=name,
            line=dict(color=COL[name],width=3), marker=dict(size=2)))
    fig.update_layout(**DARK,
        title=dict(text='<b>3-D Feature Trajectory</b>  — Key Hand Features',
                   font=dict(color='#00d4ff', size=14)),
        scene=dict(
            bgcolor='#060b14',
            xaxis=dict(backgroundcolor='#060b14',gridcolor='#122030',
                       color='#5a7fa8',title='X'),
            yaxis=dict(backgroundcolor='#060b14',gridcolor='#122030',
                       color='#5a7fa8',title='Z'),
            zaxis=dict(backgroundcolor='#060b14',gridcolor='#122030',
                       color='#5a7fa8',title='Y')),
        height=500, margin=dict(l=0,r=0,t=60,b=0),
        legend=dict(bgcolor='#0d1a30'))
    return fig


def save_xai_html(figs_dict, sample_info):
    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>GMN XAI Report</title>
<style>
  body{{background:#060b14;color:#8fb8d8;
       font-family:'Courier New',monospace;margin:0;padding:0;}}
  h1{{text-align:center;padding:2.5rem 1rem .5rem;font-size:2.2rem;
      background:linear-gradient(100deg,#00d4ff,#5b8cff,#a259ff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
  .sub{{text-align:center;color:#2d5a7a;font-size:.72rem;
        letter-spacing:.22em;text-transform:uppercase;margin-bottom:2rem;}}
  h2{{padding:.4rem 2rem;color:#00d4ff;font-size:1rem;
      border-left:3px solid #00d4ff;margin:2rem 2rem .3rem;}}
  .fig{{margin:0 2rem 2rem;border:1px solid #1a3060;
        border-radius:8px;overflow:hidden;}}
  .meta{{display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;
         padding:.5rem 2rem 1.5rem;font-size:.75rem;}}
  .meta span{{background:#0d1a30;border:1px solid #1a3060;
              border-radius:6px;padding:.3rem .8rem;}}
</style></head><body>
<h1>GMN · XAI Report</h1>
<div class="sub">Integrated Gradients · Attention Rollout · MC-Dropout · Calibration</div>
<div class="meta">
  <span>True: <b style="color:#4dffb4">{sample_info['true_name']}</b></span>
  <span>Predicted: <b style="color:#00d4ff">{sample_info['pred_name']}</b></span>
  <span>Confidence: <b style="color:#ff9f40">{sample_info['confidence']*100:.1f}%</b></span>
  <span>Entropy: <b style="color:#a259ff">{sample_info['entropy']:.3f} nats</b></span>
  <span>T*: <b style="color:#ff6060">{sample_info['T_opt']:.4f}</b></span>
</div>"""]
    for name, fig in figs_dict.items():
        div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        html.append(f'<h2>{name}</h2><div class="fig">{div}</div>')
    html.append("</body></html>")
    with open('xai_report.html', 'w') as f: f.write('\n'.join(html))
    print("✓ XAI report saved → xai_report.html")


# ─────────────────────────────────────────────────────────────────────────────
#  Offline XAI runner
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_xai(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_leapgesturedb(args.data_path)
    Xv   = torch.tensor(data['X_val'],  dtype=torch.float32).to(device)
    yv   = torch.tensor(data['y_val'],  dtype=torch.int64).to(device)
    Xte  = torch.tensor(data['X_test'], dtype=torch.float32).to(device)
    yte  = torch.tensor(data['y_test'], dtype=torch.int64).to(device)

    model = GalileanMotionTransformer().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Calibration
    v_loader = DataLoader(TensorDataset(Xv, yv), batch_size=32)
    calib    = TemperatureScaler()
    calib.fit(model, v_loader, device)
    T_OPT = calib.temperature.item()

    # Select sample
    xi       = Xte[args.sample_idx:args.sample_idx+1]
    xi_np    = xi.cpu().numpy()[0]
    true_cls = int(yte[args.sample_idx].item())

    with torch.no_grad():
        raw     = model(xi)['logits']
        cal_p   = F.softmax(calib(raw), dim=-1).cpu().numpy()[0]

    pred_cls   = int(cal_p.argmax())
    confidence = float(cal_p.max())
    correct    = pred_cls == true_cls

    print(f"\n  Sample {args.sample_idx}:")
    print(f"    True      : {GESTURE_ICONS[true_cls]}  {GESTURE_NAMES[true_cls]}")
    print(f"    Predicted : {GESTURE_ICONS[pred_cls]}  {GESTURE_NAMES[pred_cls]}")
    print(f"    Confidence: {confidence*100:.1f}%  "
          f"({'✓ correct' if correct else '✗ wrong'})")

    # IG
    print("  Computing Integrated Gradients…")
    attrs   = integrated_gradients(model, xi, pred_cls, steps=40)

    # Attention rollout
    print("  Computing attention rollout…")
    rollout = get_attention_rollout(model, xi)

    # MC-Dropout
    print("  Running MC-Dropout (25 passes)…")
    mean_p, std_p, ent = mc_dropout_predict(model, xi, n_samples=25)
    mean_p = mean_p[0]; std_p = std_p[0]; ent_val = float(ent[0])
    max_ent = np.log(NUM_CLASSES)
    print(f"  Entropy: {ent_val:.3f} / {max_ent:.3f} nats")

    # Figures
    figs = {}
    figs['Calibrated Confidence']  = plot_confidence_bar(cal_p, pred_cls)
    figs['Integrated Gradients']   = plot_ig_heatmap(attrs)
    figs['Feature Importance']     = plot_feature_importance(attrs)
    if rollout is not None:
        figs['Attention Rollout']  = plot_attention_rollout(rollout)
    figs['MC-Dropout Uncertainty'] = plot_mc_dropout(mean_p, std_p, ent_val)
    figs['3-D Trajectory']         = plot_3d_trajectory(xi_np)

    for fig in figs.values(): fig.show()

    sample_info = dict(true_name=GESTURE_NAMES[true_cls],
                       pred_name=GESTURE_NAMES[pred_cls],
                       confidence=confidence, entropy=ent_val, T_opt=T_OPT)
    save_xai_html(figs, sample_info)


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit dashboard  (runs when: streamlit run xai_dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_streamlit_dashboard():
    import streamlit as st

    st.set_page_config(page_title="GMN Dashboard", page_icon="🤲", layout="wide")
    st.markdown("""
    <style>
    body,[class*="css"]{background:#060b14;color:#8fb8d8;
                        font-family:'Courier New',monospace;}
    [data-testid="stSidebar"]{background:#0a0f1e;
                               border-right:1px solid #1a3060;}
    .stButton>button{background:#0d2040;color:#00d4ff;
                     border:1px solid #1a3060;border-radius:6px;width:100%;}
    .cw{background:#1a3060;color:#00d4ff;border-radius:4px;
        padding:.2rem .6rem;font-size:.75rem;display:inline-block;margin:.2rem 0;}
    </style>""", unsafe_allow_html=True)

    CW_CLASS  = 1   # Left rotation
    CCW_CLASS = 2   # Right rotation

    def simulate(g, T=SEQ_LENGTH, noise=0.06):
        t   = np.linspace(0, 2*np.pi, T)
        f   = 0.8 + g * 0.4
        seq = np.zeros((T, INPUT_DIM), dtype=np.float32)
        seq[:,2] = np.sin(f*t + g*0.5)
        seq[:,3] = np.cos(f*t + g*0.5)
        seq[:,4] = 0.4*np.sin(2*f*t)
        seq[:,5] = 50*np.sin(f*t)
        seq[:,6] = 200 + 30*np.cos(f*t)
        seq[:,7] = 300 + 20*np.sin(2*f*t)
        for fi, start in enumerate([11,14,17,20,23]):
            seq[:,start]   = seq[:,5] + fi*10
            seq[:,start+1] = seq[:,6] + fi*5
            seq[:,start+2] = seq[:,7] - fi*8
        return (seq + np.random.randn(*seq.shape).astype(np.float32)*noise)

    @st.cache_resource
    def load_model():
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m   = GalileanMotionTransformer().to(dev)
        if os.path.exists("best_gmn.pt"):
            m.load_state_dict(torch.load("best_gmn.pt", map_location=dev))
        m.eval()
        return m, dev

    model, dev = load_model()

    def predict(x_np):
        x = torch.tensor(x_np).unsqueeze(0).to(dev)
        with torch.no_grad():
            return F.softmax(model(x)['logits'],-1).cpu().numpy()[0]

    def mc_unc(x_np, n=20):
        x = torch.tensor(x_np).unsqueeze(0).to(dev)
        model.train()
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()
        s = []
        with torch.no_grad():
            for _ in range(n):
                s.append(F.softmax(model(x)['logits'],-1).cpu().numpy()[0])
        model.eval()
        s = np.array(s); return s.mean(0), s.std(0)

    def approx_ig(x_np, pc):
        x = torch.tensor(x_np).unsqueeze(0).to(dev).requires_grad_(True)
        model(x)['logits'][0,pc].backward()
        return (x.grad.cpu().numpy()[0] * x_np).mean(0)  # (26,)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤲 GMN · Gesture Intelligence")
        st.markdown("---")
        st.markdown("""**Rotation Classes**
<div class="cw">↺ CW  — Left rotation  (class 1)</div><br>
<div class="cw">↻ CCW — Right rotation (class 2)</div>""",
                    unsafe_allow_html=True)
        st.markdown("---")
        mode = st.radio("Input", ["Simulate gesture", "Upload .npy (100×26)"])
        if mode == "Simulate gesture":
            g      = st.selectbox("Gesture", range(NUM_CLASSES),
                       format_func=lambda i: f"{GESTURE_ICONS[i]} {GESTURE_NAMES[i]}")
            noise  = st.slider("Noise", 0.0, 0.3, 0.06, 0.01)
            x_np   = simulate(g, noise=noise); true_g = g
        else:
            up = st.file_uploader("Upload .npy  (100 × 26)", type="npy")
            if up:
                x_np = np.load(up).astype(np.float32)
                if x_np.shape != (SEQ_LENGTH, INPUT_DIM):
                    st.error(f"Expected ({SEQ_LENGTH},{INPUT_DIM}), got {x_np.shape}")
                    st.stop()
                true_g = None
            else:
                st.info("Upload a .npy file or switch to simulate mode.")
                st.stop()
        n_mc = st.slider("MC-Dropout samples", 10, 50, 20)
        run  = st.button("▶  RUN ANALYSIS")

    # ── Main ─────────────────────────────────────────────────────────────────
    st.markdown("<h1 style='color:#00d4ff;font-family:serif;'>"
                "GMN · Real-Time Gesture Dashboard</h1>", unsafe_allow_html=True)
    st.caption("LeapGestureDB · 11 classes · 26 features/frame · "
               "Temperature-calibrated confidence · MC-Dropout uncertainty")

    if not run:
        st.info("Configure options in the sidebar and click ▶ RUN ANALYSIS")
        st.stop()

    probs = predict(x_np)
    pc    = int(probs.argmax())

    c1,c2,c3,c4 = st.columns([2,1,1,1])
    c1.markdown(f"### {GESTURE_ICONS[pc]}  {GESTURE_NAMES[pc]}")
    c2.metric("Confidence", f"{probs.max()*100:.1f}%")
    if true_g is not None:
        c3.metric("Result", "✅ Correct" if pc==true_g else "❌ Wrong")
    c4.metric("↺ CW / ↻ CCW",
              f"{probs[CW_CLASS]*100:.1f}% / {probs[CCW_CLASS]*100:.1f}%")
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs(
        ["📊 Confidence","🎲 Uncertainty","🔬 Feature IG","🦴 3-D Trajectory"])

    with tab1:
        si   = np.argsort(probs)[::-1]
        cols = ["#00d4ff" if i==CW_CLASS else
                "#a259ff" if i==CCW_CLASS else "#1e3a5f" for i in si]
        fig  = go.Figure(go.Bar(
            x=probs[si],
            y=[f"{GESTURE_ICONS[i]} {GESTURE_NAMES[i]}" for i in si],
            orientation='h', marker=dict(color=cols),
            text=[f"{v*100:.1f}%" for v in probs[si]], textposition='outside'))
        fig.update_layout(**DARK, height=420,
            xaxis=dict(range=[0,1.15], tickformat='.0%', gridcolor='#122030'),
            yaxis=dict(categoryorder='array',
                       categoryarray=[f"{GESTURE_ICONS[i]} {GESTURE_NAMES[i]}"
                                       for i in si][::-1]))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("↺ CW (blue) · ↻ CCW (purple) highlighted. "
                   "These classes share identical trajectory topology "
                   "and differ only in rotation sign.")

    with tab2:
        mp,sp = mc_unc(x_np, n=n_mc)
        fig   = go.Figure(go.Bar(
            x=[f"{GESTURE_ICONS[i]}" for i in range(NUM_CLASSES)],
            y=mp,
            error_y=dict(type='data',array=sp,color='#ff6b6b',thickness=2,width=8),
            marker=dict(color=mp, colorscale='Blues')))
        fig.update_layout(**DARK, height=400,
            xaxis=dict(color='#5a7fa8',title='Gesture'),
            yaxis=dict(range=[0,1], gridcolor='#122030', title='Mean Prob'))
        st.plotly_chart(fig, use_container_width=True)
        ent = -np.sum(mp*np.log(mp+1e-9))
        st.caption(f"Predictive entropy = {ent:.3f} nats  "
                   f"(max = {np.log(NUM_CLASSES):.3f} nats)  |  "
                   f"{'Zero uncertainty — safe to execute immediately.' if ent < 0.01 else 'Non-zero uncertainty — consider secondary confirmation.'}")

    with tab3:
        ig   = approx_ig(x_np, pc)
        sf   = np.argsort(np.abs(ig))[::-1]
        fig  = go.Figure(go.Bar(
            x=[FEATURE_NAMES[f] for f in sf],
            y=ig[sf],
            marker=dict(color=ig[sf], colorscale='RdBu', cmid=0,
                        line=dict(width=0))))
        fig.update_layout(**DARK, height=460,
            xaxis=dict(tickangle=-55, color='#5a7fa8'),
            yaxis=dict(title='Attribution', gridcolor='#122030'),
            margin=dict(l=60,r=40,t=40,b=180))
        st.plotly_chart(fig, use_container_width=True)
        top3 = ", ".join([FEATURE_NAMES[sf[i]] for i in range(3)])
        st.caption(f"Top-3 discriminative features: {top3}. "
                   "Positive = supports prediction · Negative = suppresses it.")

    with tab4:
        fig = plot_3d_trajectory(x_np)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("3-D trajectories of key hand features over 100 frames.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',  type=str, default='/content/LeapGestureDB')
    p.add_argument('--ckpt',       type=str, default='best_gmn.pt')
    p.add_argument('--sample_idx', type=int, default=0)
    return p.parse_args()


# Detect whether running via Streamlit or directly
if __name__ == '__main__':
    # Check if launched by streamlit
    try:
        import streamlit as st
        _is_streamlit = st.runtime.exists()
    except Exception:
        _is_streamlit = False

    if _is_streamlit or 'streamlit' in sys.modules:
        run_streamlit_dashboard()
    else:
        run_offline_xai(parse_args())
else:
    # Always run dashboard when imported by streamlit runner
    run_streamlit_dashboard()
