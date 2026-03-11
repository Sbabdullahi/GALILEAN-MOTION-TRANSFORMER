"""
evaluate.py — GMN Evaluation Suite
====================================
Produces all quantitative evaluation plots from a trained checkpoint.

Requires:
    best_gmn.pt             — trained model checkpoint (from main.py)
    training_history.json   — training history (from main.py)
    LeapGestureDB/          — raw dataset folder

Usage:
    python evaluate.py --data_path /path/to/LeapGestureDB
    python evaluate.py --data_path /path/to/LeapGestureDB --save_html

Outputs:
    GMN_Evaluation_Report.html   — self-contained interactive report
    confusion_matrix.png
    roc_curves.png
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, accuracy_score, f1_score)
from sklearn.preprocessing import label_binarize

# Import shared components from main
from main import (GalileanMotionTransformer, load_leapgesturedb,
                  evaluate, GESTURE_NAMES, GESTURE_ICONS,
                  NUM_CLASSES, INPUT_DIM, SEQ_LENGTH)

# ─────────────────────────────────────────────────────────────────────────────
#  Dark Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(
    plot_bgcolor='#0a0f1e', paper_bgcolor='#0a0f1e',
    font=dict(color='#8fb8d8', family='Courier New, monospace'))

def _dark(fig, title='', height=460):
    fig.update_layout(**DARK,
        title=dict(text=title, font=dict(color='#00d4ff', size=14)),
        margin=dict(l=60,r=40,t=60,b=60), height=height)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  Temperature calibration
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*1.5)
    def forward(self, logits):
        return logits / self.temperature.to(logits.device)
    def fit(self, model, val_loader, device, lr=0.01, max_iter=50):
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x,y in val_loader:
                logits_list.append(model(x)['logits'].cpu())
                labels_list.append(y.cpu())
        la,ll = torch.cat(logits_list), torch.cat(labels_list)
        opt  = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        crit = nn.CrossEntropyLoss()
        def step():
            opt.zero_grad()
            loss = crit(la/self.temperature, ll)
            loss.backward(); return loss
        opt.step(step)
        print(f"  Optimal temperature: {self.temperature.item():.4f}")
        return self

# ─────────────────────────────────────────────────────────────────────────────
#  Plot builders
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history):
    ep  = list(range(1, len(history['train_loss'])+1))
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Loss','Accuracy (%)','Val Macro F1'])
    for col,(tk,vk) in enumerate([('train_loss','val_loss'),
                                   ('train_acc','val_acc'),(None,'val_f1')],1):
        if tk:
            fig.add_trace(go.Scatter(x=ep,y=history[tk],name='Train',
                line=dict(color='#00d4ff',width=2)),row=1,col=col)
        fig.add_trace(go.Scatter(x=ep,y=history[vk],
            name='Val' if tk else 'F1',
            line=dict(color='#ff6b6b' if tk else '#4dffb4',width=2)),row=1,col=col)
    fig.update_layout(**DARK,
        title=dict(text='<b>GMN Training History</b>',
                   font=dict(color='#00d4ff',size=15)),
        height=400, showlegend=True,
        legend=dict(bgcolor='#0d1a30',bordercolor='#1a3a60'))
    fig.update_xaxes(gridcolor='#122030',color='#5a7fa8')
    fig.update_yaxes(gridcolor='#122030',color='#8fb8d8')
    return fig


def plot_confusion_matrix(labels, preds, acc):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float)/(cm.sum(1,keepdims=True)+1e-9)
    ann     = [[f'{cm[i,j]}<br>({cm_norm[i,j]*100:.0f}%)'
                for j in range(NUM_CLASSES)] for i in range(NUM_CLASSES)]
    sn      = [n[:14] for n in GESTURE_NAMES]
    fig     = ff.create_annotated_heatmap(
        z=cm_norm, x=sn, y=sn, annotation_text=ann,
        colorscale='Blues', showscale=True)
    fig.update_layout(**DARK,
        title=dict(text=f'<b>Confusion Matrix</b>  — Test Acc: {acc:.1f}%',
                   font=dict(color='#00d4ff',size=14)),
        height=560, margin=dict(l=150,r=40,t=70,b=130))
    fig.update_xaxes(tickangle=-35,color='#5a7fa8',title='Predicted')
    fig.update_yaxes(color='#8fb8d8',autorange='reversed',title='True')
    return fig


def plot_roc_curves(labels, probs):
    y_bin   = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    fig     = go.Figure()
    for i,name in enumerate(GESTURE_NAMES):
        fpr,tpr,_ = roc_curve(y_bin[:,i], probs[:,i])
        au        = auc(fpr,tpr)
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',
            name=f'{GESTURE_ICONS[i]} {name[:14]} ({au:.2f})',
            line=dict(color=palette[i%len(palette)],width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',
        line=dict(color='#444',width=1,dash='dash')))
    fig.update_layout(**DARK,
        title=dict(text='<b>Per-Class ROC Curves  (One-vs-Rest)</b>',
                   font=dict(color='#00d4ff',size=14)),
        xaxis=dict(title='FPR',gridcolor='#122030',color='#5a7fa8'),
        yaxis=dict(title='TPR',gridcolor='#122030',color='#8fb8d8'),
        legend=dict(bgcolor='#0d1a30',bordercolor='#1a3a60',font=dict(size=9)),
        height=520)
    return fig


def plot_per_class_f1(labels, preds):
    report = classification_report(labels,preds,
                                    target_names=GESTURE_NAMES,
                                    output_dict=True, zero_division=0)
    f1s    = [report[c]['f1-score'] for c in GESTURE_NAMES]
    clrs   = ['#ff6b6b' if v < 0.90 else '#00d4ff' for v in f1s]
    fig    = go.Figure(go.Bar(
        x=GESTURE_NAMES, y=f1s,
        marker=dict(color=clrs, line=dict(width=0)),
        text=[f'{v:.3f}' for v in f1s], textposition='outside'))
    fig.update_layout(**DARK,
        title=dict(text='<b>Per-Class F1-Score</b>',
                   font=dict(color='#00d4ff',size=14)),
        xaxis=dict(tickangle=-35,color='#5a7fa8'),
        yaxis=dict(title='F1',range=[0,1.1],gridcolor='#122030',color='#8fb8d8'),
        height=430, margin=dict(l=60,r=40,t=70,b=120))
    return fig


def plot_cross_class_confidence(model, X_test, y_test, calib_scaler, device):
    rows = []
    for g in range(NUM_CLASSES):
        idxs = np.where(y_test.cpu().numpy()==g)[0][:10]
        if len(idxs)==0: rows.append(np.zeros(NUM_CLASSES)); continue
        xg = X_test[idxs]
        with torch.no_grad():
            lg  = calib_scaler(model(xg)['logits'])
            row = F.softmax(lg,-1).cpu().numpy().mean(0)
        rows.append(row)
    mat = np.array(rows)
    fig = go.Figure(go.Heatmap(
        z=mat,
        x=[f"{GESTURE_ICONS[i]}" for i in range(NUM_CLASSES)],
        y=[f"{GESTURE_ICONS[i]}  {GESTURE_NAMES[i][:16]}" for i in range(NUM_CLASSES)],
        colorscale='Blues', showscale=True,
        text=[[f'{v:.2f}' for v in row] for row in mat],
        texttemplate='%{text}',
        colorbar=dict(title='Avg P',tickfont=dict(color='#8fb8d8'))))
    fig.update_layout(**DARK,
        title=dict(text='<b>Cross-Class Confidence Matrix</b>',
                   font=dict(color='#00d4ff',size=14)),
        xaxis=dict(title='Predicted',color='#5a7fa8'),
        yaxis=dict(title='True',color='#8fb8d8',autorange='reversed'),
        height=540, margin=dict(l=220,r=60,t=70,b=60))
    return fig


def save_html_report(figs_dict, te, T_OPT, total_params):
    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>GMN Evaluation Report</title>
<style>
  body{{background:#060b14;color:#8fb8d8;font-family:'Courier New',monospace;margin:0;}}
  h1{{text-align:center;padding:2rem 1rem .5rem;font-size:2rem;
      background:linear-gradient(100deg,#00d4ff,#5b8cff,#a259ff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
  .sub{{text-align:center;color:#2d5a7a;font-size:.72rem;letter-spacing:.2em;
        text-transform:uppercase;margin-bottom:1.5rem;}}
  h2{{padding:.4rem 2rem;color:#00d4ff;font-size:1rem;
      border-left:3px solid #00d4ff;margin:2rem 2rem .3rem;}}
  .fig{{margin:0 2rem 2rem;border:1px solid #1a3060;border-radius:8px;overflow:hidden;}}
  .meta{{display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;
         padding:.5rem 2rem 1.5rem;font-size:.75rem;}}
  .meta span{{background:#0d1a30;border:1px solid #1a3060;
              border-radius:6px;padding:.3rem .8rem;}}
</style></head><body>
<h1>GMN · Evaluation Report</h1>
<div class="sub">LeapGestureDB · 11 classes · 26 features/frame</div>
<div class="meta">
  <span>Test Acc: <b style="color:#00d4ff">{te['acc']:.2f}%</b></span>
  <span>Macro F1: <b style="color:#4dffb4">{te['f1']:.4f}</b></span>
  <span>Calibration T: <b style="color:#a259ff">{T_OPT:.4f}</b></span>
  <span>Params: <b style="color:#ff6060">{total_params:,}</b></span>
</div>"""]
    for name, fig in figs_dict.items():
        div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        html.append(f'<h2>{name}</h2><div class="fig">{div}</div>')
    html.append("</body></html>")
    path = 'GMN_Evaluation_Report.html'
    with open(path,'w') as f: f.write('\n'.join(html))
    print(f"✓ Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GMN Evaluation")
    p.add_argument('--data_path', type=str, default='/content/LeapGestureDB')
    p.add_argument('--ckpt',      type=str, default='best_gmn.pt')
    p.add_argument('--history',   type=str, default='training_history.json')
    p.add_argument('--save_html', action='store_true')
    p.add_argument('--device',    type=str, default='auto')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
                          if args.device=='auto' else args.device)

    # Load data & model
    data   = load_leapgesturedb(args.data_path)
    Xv  = torch.tensor(data['X_val'],   dtype=torch.float32).to(device)
    yv  = torch.tensor(data['y_val'],   dtype=torch.int64).to(device)
    Xte = torch.tensor(data['X_test'],  dtype=torch.float32).to(device)
    yte = torch.tensor(data['y_test'],  dtype=torch.int64).to(device)

    v_loader  = DataLoader(TensorDataset(Xv,yv),   batch_size=32)
    te_loader = DataLoader(TensorDataset(Xte,yte),  batch_size=32)

    model = GalileanMotionTransformer().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    total_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    te = evaluate(model, te_loader, criterion)
    print(f"\nTest accuracy : {te['acc']:.2f}%")
    print(f"Macro F1      : {te['f1']:.4f}")
    print(classification_report(te['labels'],te['preds'],
                                  target_names=GESTURE_NAMES,digits=4,zero_division=0))

    # Calibration
    calib = TemperatureScaler()
    calib.fit(model, v_loader, device)
    T_OPT = calib.temperature.item()

    # Load training history
    history = {}
    if os.path.exists(args.history):
        with open(args.history) as f: history = json.load(f)

    # Build figures
    figs = {}
    if history: figs['Training History'] = plot_training_history(history)
    figs['Confusion Matrix']          = plot_confusion_matrix(te['labels'],te['preds'],te['acc'])
    figs['ROC Curves']                = plot_roc_curves(te['labels'],te['probs'])
    figs['Per-Class F1']              = plot_per_class_f1(te['labels'],te['preds'])
    figs['Cross-Class Confidence']    = plot_cross_class_confidence(
                                            model,Xte,yte,calib,device)

    for name,fig in figs.items():
        fig.show()

    if args.save_html:
        save_html_report(figs, te, T_OPT, total_params)


if __name__ == '__main__':
    main()
