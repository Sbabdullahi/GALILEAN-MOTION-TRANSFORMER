"""
main.py — Galilean Motion Network (GMN)
=======================================
Handles: data loading · model definition · training · test evaluation

Usage (local):
    python main.py --data_path /path/to/LeapGestureDB

Usage (Google Colab):
    !python main.py --data_path /content/LeapGestureDB --epochs 60

Outputs:
    best_gmn.pt          — best model checkpoint (by val accuracy)
    training_curves.png  — loss / accuracy / F1 training history
"""

import os, re, sys, json, time, zipfile, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
GESTURE_MAPPING = {
    'G1': 0, 'G2': 1, 'G3': 2,  'G4': 3,  'G5': 4,
    'G6': 5, 'G7': 6, 'G8': 7,  'G9': 8,  'G10': 9, 'G11': 10
}
GESTURE_NAMES = [
    "Click",            "Left rotation",     "Right rotation",
    "Increase contrast","Decrease contrast", "Zoom in",
    "Zoom out",         "Move left",         "Move right",
    "Previous",         "Next",
]
FEATURE_NAMES = [
    "HandID",       "FingerCount",
    "HandDirection_X","HandDirection_Y","HandDirection_Z",
    "PalmPosition_X", "PalmPosition_Y", "PalmPosition_Z",
    "PalmNormal_X",   "PalmNormal_Y",   "PalmNormal_Z",
    "ThumbTip_X",     "ThumbTip_Y",     "ThumbTip_Z",
    "IndexTip_X",     "IndexTip_Y",     "IndexTip_Z",
    "MiddleTip_X",    "MiddleTip_Y",    "MiddleTip_Z",
    "RingTip_X",      "RingTip_Y",      "RingTip_Z",
    "PinkyTip_X",     "PinkyTip_Y",     "PinkyTip_Z",
]

NUM_CLASSES = 11
INPUT_DIM   = 26   # 26 semantic features per frame (see FEATURE_NAMES)
SEQ_LENGTH  = 100  # frames per sequence

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def parse_leap_file(filepath, target_frames=SEQ_LENGTH):
    """
    Parse one LeapGestureDB .txt file into a (target_frames, 26) float32 array.

    Feature layout:
      [0]    HandID          [1]    FingerCount
      [2-4]  HandDirection   [5-7]  PalmPosition
      [8-10] PalmNormal      [11-13] ThumbTip
      [14-16] IndexTip       [17-19] MiddleTip
      [20-22] RingTip        [23-25] PinkyTip
    """
    def _xyz(line, keyword):
        m = re.search(
            re.escape(keyword) +
            r'\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)',
            line, re.IGNORECASE)
        return [float(m.group(i)) for i in (1,2,3)] if m else [0.,0.,0.]

    def _finger(line):
        m = re.search(
            r'\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*'
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)', line)
        return [float(m.group(i)) for i in (1,2,3)] if m else [0.,0.,0.]

    def _vec(d):
        return ([d.get('hand_id',0.), d.get('finger_count',0.)]
              + d.get('hand_dir',[0.,0.,0.]) + d.get('palm_pos',[0.,0.,0.])
              + d.get('palm_norm',[0.,0.,0.]) + d.get('thumb',[0.,0.,0.])
              + d.get('index',[0.,0.,0.])  + d.get('middle',[0.,0.,0.])
              + d.get('ring',[0.,0.,0.])   + d.get('pinky',[0.,0.,0.]))

    try:
        frames, in_frame, frame = [], False, {}
        with open(filepath, 'r', errors='replace') as fh:
            for raw in fh:
                line = raw.strip()
                if not line: continue
                if line.startswith('Frame.id:'):
                    if in_frame and frame: frames.append(_vec(frame))
                    frame, in_frame = {}, False
                    hm = re.search(r'Hand_number:\s*(\d+)', line)
                    if hm and int(hm.group(1)) > 0: in_frame = True
                    continue
                if not in_frame: continue
                if 'hand_Id_type' in line or 'hand direction' in line.lower():
                    im = re.search(r'hand_Id_type:\s*(\d+)', line, re.I)
                    fm = re.search(r"finger'?s?_number:?\s*(\d+)", line, re.I)
                    frame['hand_id']      = float(im.group(1)) if im else 0.
                    frame['finger_count'] = float(fm.group(1)) if fm else 0.
                    frame['hand_dir']  = _xyz(line, 'hand direction:')
                    frame['palm_pos']  = _xyz(line, 'Palm position:')
                    frame['palm_norm'] = _xyz(line, 'Palm normal:')
                elif 'TYPE_THUMB'  in line: frame['thumb']  = _finger(line)
                elif 'TYPE_INDEX'  in line: frame['index']  = _finger(line)
                elif 'TYPE_MIDDLE' in line: frame['middle'] = _finger(line)
                elif 'TYPE_RING'   in line: frame['ring']   = _finger(line)
                elif 'TYPE_PINKY'  in line: frame['pinky']  = _finger(line)
        if in_frame and frame: frames.append(_vec(frame))
        if not frames: return None
        seq = np.array(frames, dtype=np.float32)
        if len(seq) < target_frames:
            seq = np.vstack([seq, np.repeat(seq[-1:], target_frames-len(seq), axis=0)])
        elif len(seq) > target_frames:
            seq = seq[np.linspace(0, len(seq)-1, target_frames, dtype=int)]
        return seq
    except Exception:
        return None


def load_leapgesturedb(data_path, seq_length=SEQ_LENGTH):
    """
    Load full dataset with subject-aware split and robust preprocessing.

    Preprocessing (DQA-motivated):
      - IQR clipping (k=1.5): handles outlier rates up to 12.1% in proximal features
      - RobustScaler: preferred over z-score given universal non-normality (SW p<0.001)

    Returns dict: X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays)
    """
    subjects = sorted([
        os.path.join(data_path, d) for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])
    print(f"Found {len(subjects)} subjects in {data_path}")

    X_list, y_list, subj_ids = [], [], []
    errors = 0
    for si, sfolder in enumerate(tqdm(subjects, desc="Loading subjects")):
        sid_m = re.search(r'(\d+)', os.path.basename(sfolder))
        sid   = int(sid_m.group(1)) if sid_m else si
        for fname in os.listdir(sfolder):
            if not fname.endswith('.txt'): continue
            gm = re.search(r'G(\d+)', fname)
            if not gm: errors += 1; continue
            gkey = f'G{gm.group(1)}'
            if gkey not in GESTURE_MAPPING: errors += 1; continue
            seq = parse_leap_file(os.path.join(sfolder, fname), seq_length)
            if seq is None: errors += 1; continue
            X_list.append(seq); y_list.append(GESTURE_MAPPING[gkey]); subj_ids.append(sid)

    print(f"Loaded {len(X_list)} samples  |  Errors: {errors}")
    if not X_list: raise ValueError("No samples loaded — check data_path.")

    X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    y = np.array(y_list, dtype=np.int64)
    subj_ids = np.array(subj_ids)

    # Subject-aware split
    unique_subj = np.unique(subj_ids)
    n = len(unique_subj)
    if n < 10:
        Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        Xtr, Xv, ytr, yv   = train_test_split(Xtv, ytv, test_size=.1, random_state=42, stratify=ytv)
    else:
        tv, te_s = train_test_split(unique_subj, test_size=max(1,int(n*.2)), random_state=42)
        tr_s, v_s = train_test_split(tv, test_size=max(1,int(len(tv)*.1)), random_state=42)
        Xtr,ytr = X[np.isin(subj_ids,tr_s)], y[np.isin(subj_ids,tr_s)]
        Xv, yv  = X[np.isin(subj_ids,v_s)],  y[np.isin(subj_ids,v_s)]
        Xte,yte = X[np.isin(subj_ids,te_s)], y[np.isin(subj_ids,te_s)]

    # IQR clip + RobustScale (fit on train only)
    F = INPUT_DIM
    flat_tr = Xtr.reshape(-1, F)
    Q1,Q3   = np.percentile(flat_tr,25,axis=0), np.percentile(flat_tr,75,axis=0)
    IQR     = Q3-Q1; lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
    clip    = lambda a: np.clip(a.reshape(-1,F), lo, hi)
    scaler  = RobustScaler()
    Xtr = scaler.fit_transform(clip(Xtr)).reshape(len(Xtr), seq_length, F)
    Xv  = scaler.transform(clip(Xv)).reshape(len(Xv),  seq_length, F)
    Xte = scaler.transform(clip(Xte)).reshape(len(Xte), seq_length, F)

    print(f"Train:{len(Xtr)}  Val:{len(Xv)}  Test:{len(Xte)}")
    return dict(X_train=Xtr,y_train=ytr, X_val=Xv,y_val=yv, X_test=Xte,y_test=yte)

# ─────────────────────────────────────────────────────────────────────────────
#  Architecture
# ─────────────────────────────────────────────────────────────────────────────

class GalileanConv1D(nn.Module):
    """Physics-informed 1-D conv with learnable velocity and acceleration."""
    def __init__(self, in_ch, out_ch, ks, stride=1, pad=0, dt=0.01, reg=0.01):
        super().__init__()
        self.ks,self.stride,self.pad,self.reg = ks,stride,pad,reg
        self.in_ch = in_ch
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, ks))
        self.bias   = nn.Parameter(torch.zeros(out_ch))
        self.vel    = nn.Parameter(torch.randn(in_ch, ks)*0.1)
        self.acc    = nn.Parameter(torch.randn(in_ch, ks)*0.01)
        self.register_buffer('t_off', torch.arange(ks).float()*dt)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if self.pad > 0: x = F.pad(x,(self.pad,self.pad),mode='replicate')
        xu  = x.unfold(2, self.ks, self.stride)
        vel = self.vel.unsqueeze(0).unsqueeze(2)*self.t_off.view(1,1,1,-1)
        acc = .5*self.acc.unsqueeze(0).unsqueeze(2)*(self.t_off**2).view(1,1,1,-1)
        xt  = xu+vel+acc
        B   = xt.size(0)
        xt  = xt.permute(0,2,1,3).reshape(B,-1,self.in_ch*self.ks)
        out = torch.matmul(xt, self.weight.view(self.weight.size(0),-1).t())
        out = out.permute(0,2,1)+self.bias.view(1,-1,1)
        phys = self.reg*(self.vel.pow(2).mean()+self.acc.pow(2).mean())
        return out, phys


class QuaternionLayer(nn.Module):
    def __init__(self, dim, n_joints=9):
        super().__init__()
        self.proj = nn.Linear(dim, n_joints*4)
        self.nj   = n_joints
    def forward(self, x):
        B,T,_ = x.shape
        q = self.proj(x).view(B,T,self.nj,4)
        return F.normalize(q,p=2,dim=-1).view(B,T,-1)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.n1   = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2   = nn.LayerNorm(dim)
        md = int(dim*mlp_ratio)
        self.mlp  = nn.Sequential(nn.Linear(dim,md),nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(md,dim),nn.Dropout(dropout))
    def forward(self, x):
        n=self.n1(x); a,w=self.attn(n,n,n); x=x+a; x=x+self.mlp(self.n2(x))
        return x, w


class CrossModalFusion(nn.Module):
    def __init__(self, vdim=256, ldim=768, heads=8):
        super().__init__()
        self.vp   = nn.Sequential(nn.Linear(vdim,ldim),nn.LayerNorm(ldim),nn.GELU())
        self.ca   = nn.MultiheadAttention(ldim,heads,batch_first=True)
        self.gate = nn.Sequential(nn.Linear(ldim*2,ldim),nn.Sigmoid())
        self.norm = nn.LayerNorm(ldim)
    def forward(self, v, t):
        vp=self.vp(v); f,_=self.ca(t,vp,vp); g=self.gate(torch.cat([t,f],-1))
        return self.norm(g*f+(1-g)*t)


class GalileanMotionTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, seq_length=SEQ_LENGTH,
                 embed_dim=256, num_gal_blocks=4, kernel_sizes=None,
                 num_vit=6, num_heads=8, lang_dim=768,
                 num_classes=NUM_CLASSES, dropout=0.2, num_joints=9):
        super().__init__()
        if kernel_sizes is None: kernel_sizes=[3,5,7,9]
        self.inp = nn.Linear(input_dim, embed_dim)
        self.gal = nn.ModuleList([nn.ModuleDict({
            'conv': GalileanConv1D(embed_dim,embed_dim,ks,pad=ks//2),
            'norm': nn.LayerNorm(embed_dim),
            'drop': nn.Dropout(dropout),
            'act':  nn.ReLU()}) for ks in kernel_sizes])
        self.quat   = QuaternionLayer(embed_dim, num_joints)
        self.fproj  = nn.Linear(embed_dim+num_joints*4, embed_dim)
        self.vit    = nn.ModuleList([ViTBlock(embed_dim,num_heads,4.,dropout) for _ in range(num_vit)])
        self.prompt = nn.Parameter(torch.randn(1,10,lang_dim))
        self.fusion = CrossModalFusion(embed_dim, lang_dim, 8)
        self.head   = nn.Sequential(
            nn.Linear(lang_dim,512),nn.ReLU(),nn.Dropout(.3),
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(.3),
            nn.Linear(256,num_classes))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self, x, return_attentions=False):
        B     = x.size(0)
        xe    = self.inp(x)
        xc    = xe.transpose(1,2)
        pl    = []
        for blk in self.gal:
            xc,p = blk['conv'](xc); pl.append(p)
            xc   = blk['act'](blk['norm'](xc.transpose(1,2))).transpose(1,2)
            xc   = blk['drop'](xc)
        fused = self.fproj(torch.cat([xc.transpose(1,2), self.quat(xe)],-1))
        aws   = []
        for blk in self.vit:
            fused,aw = blk(fused); aws.append(aw)
        out   = self.fusion(fused, self.prompt.expand(B,-1,-1))
        logits= self.head(out.mean(1))
        res   = {'logits':logits, 'physics_loss':sum(pl)}
        if return_attentions: res['attentions']=aws
        return res


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, pw=1.0):
    model.train()
    tot=ce_t=ph_t=correct=total=0
    for X,y in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        out  = model(X)
        ce   = criterion(out['logits'], y)
        loss = ce + pw*out['physics_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot+=loss.item(); ce_t+=ce.item(); ph_t+=out['physics_loss'].item()
        _,pred=out['logits'].max(1); total+=y.size(0); correct+=pred.eq(y).sum().item()
    n=len(loader)
    return dict(loss=tot/n, ce=ce_t/n, phys=ph_t/n, acc=100.*correct/total)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tl=0; preds,labels,probs=[],[],[]
    for X,y in loader:
        out   = model(X)
        tl   += criterion(out['logits'],y).item()
        p     = F.softmax(out['logits'],-1)
        preds.extend(p.argmax(1).cpu().numpy())
        probs.extend(p.cpu().numpy())
        labels.extend(y.cpu().numpy())
    return dict(loss=tl/len(loader),
                acc=accuracy_score(labels,preds)*100,
                f1=f1_score(labels,preds,average='macro',zero_division=0),
                preds=preds, labels=labels, probs=np.array(probs))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GMN Training — LeapGestureDB")
    p.add_argument('--data_path',  type=str, default='/content/LeapGestureDB')
    p.add_argument('--epochs',     type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--patience',   type=int, default=12)
    p.add_argument('--pw',         type=float, default=1.0,  help='Physics loss weight')
    p.add_argument('--ckpt',       type=str, default='best_gmn.pt')
    p.add_argument('--device',     type=str, default='auto')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
        if args.device == 'auto' else args.device)
    print(f"Device: {device}")

    # Data
    data = load_leapgesturedb(args.data_path)
    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype).to(device)
    Xtr,ytr = to_tensor(data['X_train']), to_tensor(data['y_train'],torch.int64)
    Xv, yv  = to_tensor(data['X_val']),   to_tensor(data['y_val'],  torch.int64)
    Xte,yte = to_tensor(data['X_test']),  to_tensor(data['y_test'], torch.int64)

    tr_loader = DataLoader(TensorDataset(Xtr,ytr), batch_size=args.batch_size, shuffle=True)
    v_loader  = DataLoader(TensorDataset(Xv, yv),  batch_size=args.batch_size)
    te_loader = DataLoader(TensorDataset(Xte,yte), batch_size=args.batch_size)

    # Model
    model = GalileanMotionTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  ({total_params*4/1024**2:.1f} MB)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = dict(train_loss=[],train_acc=[],val_loss=[],val_acc=[],val_f1=[])
    best_val, patience_ctr = 0., 0
    print(f"\n{'='*65}\nTRAINING\n{'='*65}")
    for epoch in range(args.epochs):
        tr = train_epoch(model, tr_loader, optimizer, criterion, device, args.pw)
        va = evaluate(model, v_loader, criterion)
        scheduler.step()
        for k,v in [('train_loss',tr['loss']),('train_acc',tr['acc']),
                    ('val_loss',va['loss']),('val_acc',va['acc']),('val_f1',va['f1'])]:
            history[k].append(v)
        marker=''
        if va['acc']>best_val:
            best_val=va['acc']; patience_ctr=0
            torch.save(model.state_dict(), args.ckpt); marker=' ← best'
        else: patience_ctr+=1
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={tr['loss']:.4f}  train={tr['acc']:.1f}%  "
              f"val={va['acc']:.1f}%  F1={va['f1']:.3f}{marker}")
        if patience_ctr>=args.patience:
            print(f"Early stop at epoch {epoch+1}"); break

    # Test evaluation
    print(f"\n{'='*65}\nTEST EVALUATION\n{'='*65}")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    te = evaluate(model, te_loader, criterion)
    print(f"Test accuracy : {te['acc']:.2f}%")
    print(f"Macro F1      : {te['f1']:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(te['labels'],te['preds'],
                                 target_names=GESTURE_NAMES, digits=4, zero_division=0))

    # Save history for evaluation script
    with open('training_history.json','w') as f:
        json.dump({**history,
                   'test_acc':te['acc'], 'test_f1':te['f1'],
                   'test_labels':te['labels'], 'test_preds':te['preds']}, f)

    # Training curves
    ep = range(1, len(history['train_loss'])+1)
    fig, axes = plt.subplots(1,3,figsize=(16,4))
    fig.suptitle("GMN Training History", fontsize=14, fontweight='bold')
    axes[0].plot(ep,history['train_loss'],label='Train')
    axes[0].plot(ep,history['val_loss'],  label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(ep,history['train_acc'], label='Train')
    axes[1].plot(ep,history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(ep,history['val_f1'],color='green')
    axes[2].set_title('Val Macro F1'); axes[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: best_gmn.pt  |  training_curves.png  |  training_history.json")


if __name__ == '__main__':
    main()
