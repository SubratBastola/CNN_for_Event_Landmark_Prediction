#  Actual Deep learning happens here

import os, glob, random, math, time, collections
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

# =============== Config ===============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 64
BASE_LR = 2e-3
WARMUP_EPOCHS = 5
VAL_SPLIT = 0.2
SEED = 1337
PATIENCE = 15
MIN_EPOCHS = 20

SMOOTH_FS = 500.0
N_SAMPLES = 256               # fixed input length within strict window
MA_OPT_FAST = 20.0            # ms
MA_OPT_SLOW = 80.0            # ms
MA_ELEC     = 8.0             # ms (for electrical post-logic)
STRICT_PAD_MS = 50.0
ORDER_W = 0.3

# >>> NEW (refinement windows) <<<
REFINE_OPT_PEAK_MS  = 10.0    # snap optical 'step' to raw max within ±10 ms
REFINE_ELEC_PEAK_MS = 10.0    # snap electrical entry/exit peaks to raw extrema within ±10 ms

OPT_KEYS  = ["opt_base","opt_plateau","opt_end"]
OPT_IDX   = {k:i for i,k in enumerate(OPT_KEYS)}
ELEC_KEYS = ["elec_entry_base","elec_entry_peak","elec_exit_peak","elec_exit_base"]

# =============== Utils ===============
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def moving_average_same_len(x, win):
    win = int(max(1, round(win)))
    if win % 2 == 0: win -= 1
    pad = win // 2
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), k, mode="valid").astype(np.float32, copy=False)

def stride_decimate(x, fs_in, fs_target):
    dec = max(1, int(round(float(fs_in) / float(fs_target))))
    return x[::dec].astype(np.float32, copy=False), float(fs_in / dec)

def _bytes_to_str_list(a):
    out=[]
    for v in a:
        if isinstance(v,(bytes,bytearray)):
            try: out.append(v.decode())
            except: out.append(str(v))
        else: out.append(str(v))
    return out

def load_npz_all(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    with z:
        x_raw=z["x_raw"]; fs=float(z.get("fs",5000.0)); t0=float(z.get("t0",0.0))
        L=int(z.get("length", x_raw.shape[-1])); L=min(L, x_raw.shape[-1])
        lm_names=_bytes_to_str_list(z["landmark_names"]) if "landmark_names" in z.files else []
        lm_idx  = z["landmarks_idx"].astype(int) if "landmarks_idx" in z.files else np.array([], int)
        lm_time = z["landmarks_time_s"].astype(float) if "landmarks_time_s" in z.files else np.array([], float)
        t_start=float(z.get("t_start", np.nan))
        t_end  =float(z.get("t_end",   np.nan))
        event_step_time=float(z.get("event_step_time_s", np.nan))
    t   = t0 + np.arange(L, dtype=np.float32) / float(fs)
    elec = x_raw[0,:L].astype(np.float32)
    opt  = x_raw[1,:L].astype(np.float32)
    lm = {k: None for k in (OPT_KEYS + ELEC_KEYS)}
    name_to_i = {str(n): i for i, n in enumerate(lm_names)}
    for k in name_to_i:
        i = name_to_i[k]
        if k in lm and i < len(lm_time) and lm_time[i] >= 0: lm[k] = float(lm_time[i])
        elif k in lm and i < len(lm_idx)  and lm_idx[i]  >= 0: lm[k] = t0 + float(lm_idx[i]) / fs
    if lm["opt_plateau"] is None and not np.isnan(event_step_time): lm["opt_plateau"] = float(event_step_time)
    if lm["opt_base"]    is None and not np.isnan(t_start):        lm["opt_base"]    = float(t_start)
    if lm["opt_end"]     is None and not np.isnan(t_end):          lm["opt_end"]     = float(t_end)
    return t, elec, opt, fs, lm

def to_fixed_length(series_t, series_x, t_lo, t_hi, n_out):
    if t_hi <= t_lo: t_hi = t_lo + 1e-6
    grid = np.linspace(t_lo, t_hi, n_out, dtype=np.float32)
    x = np.interp(grid, series_t, series_x).astype(np.float32)
    return grid, x

def robust_norm(x, axis=1):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1e-9 + mad)

# >>> NEW: snap a predicted time to nearest raw local extremum within ±window_ms
def refine_raw_extremum(t, y, t_est, window_ms, mode="max"):
    w = window_ms/1000.0
    m = (t >= t_est - w) & (t <= t_est + w)
    if not np.any(m): return float(t_est)
    seg = y[m]; i0 = np.nonzero(m)[0][0]
    if seg.size < 3:
        idx = int(np.argmax(seg) if mode=="max" else np.argmin(seg))
        return float(t[i0 + idx])
    if mode == "max":
        p,_ = find_peaks(seg)
        if p.size:
            best = p[np.argmax(seg[p])]
            return float(t[i0 + best])
        return float(t[i0 + int(np.argmax(seg))])
    else:  # "min" → downside extremum
        segi = -seg
        p,_ = find_peaks(segi)
        if p.size:
            best = p[np.argmax(segi[p])]
            return float(t[i0 + best])
        return float(t[i0 + int(np.argmin(seg))])

# =============== Inputs (optical only) ===============
def build_opt_inputs(t, opt, fs, t_lo, t_hi):
    # 500 Hz decimation for stability
    opt_d, fs_d = stride_decimate(opt, fs, SMOOTH_FS)
    td = t[0] + np.arange(opt_d.size)/fs_d
    # smoothers
    w20 = max(1, int(round((MA_OPT_FAST/1000.0)*fs_d)))
    w80 = max(1, int(round((MA_OPT_SLOW/1000.0)*fs_d)))
    opt_ma20 = moving_average_same_len(opt_d, w20)
    opt_ma80 = moving_average_same_len(opt_d, w80)
    # resample to strict window, fixed length
    grid,  opt_raw = to_fixed_length(td, opt_d,    t_lo, t_hi, N_SAMPLES)
    _,     opt_f   = to_fixed_length(td, opt_ma20, t_lo, t_hi, N_SAMPLES)
    _,     opt_s   = to_fixed_length(td, opt_ma80, t_lo, t_hi, N_SAMPLES)
    # position channel helps localization
    pos = np.linspace(0,1,N_SAMPLES, dtype=np.float32)
    x = np.stack([opt_raw, opt_f, opt_s, pos], axis=0)  # (4,N)
    x = robust_norm(x, axis=1)
    return grid, x

# =============== Electrical landmarks (post, with snap-to-raw) ===============
def compute_electrical_from_opt(t, elec, fs, t_lo, t_hi, opt_pred):
    m = (t>=t_lo)&(t<=t_hi)
    if not np.any(m): 
        return {k: float(t_lo) for k in ELEC_KEYS}
    ed, fs_ed = stride_decimate(elec[m], fs, SMOOTH_FS)
    elec_td = t_lo + np.arange(ed.size)/fs_ed
    w = max(1, int(round((MA_ELEC/1000.0)*fs_ed)))
    eS = moving_average_same_len(ed, w)
    b = max(6, int(0.20*len(eS))); base = float(np.median(eS[:b])); sig = float(max(1e-9, 1.4826*np.median(np.abs(eS[:b]-np.median(eS[:b])))))
    run = max(6, int(round((8e-3)*fs_ed)))
    # entry_base
    mask_left = elec_td <= opt_pred["opt_base"]
    if np.any(mask_left):
        idx=None; c=0; left_idx=np.nonzero(mask_left)[0]
        for i in left_idx[::-1]:
            if abs(eS[i]-base) < 2.5*sig: c+=1
            else: c=0
            if c>=run: idx=i; break
        if idx is None: idx = left_idx[-1]
        entry_base = elec_td[idx]
    else:
        entry_base = float(t_lo)
    # entry_peak: max in [step-20ms, step]
    step_t = opt_pred["opt_plateau"]; radius=0.020
    seg = (elec_td>=step_t-radius)&(elec_td<=step_t)
    if not np.any(seg): entry_peak = step_t
    else:
        s=eS[seg]; i0=np.nonzero(seg)[0][0]
        p,_=find_peaks(s); entry_peak = elec_td[i0 + (int(p[np.argmax(s[p])]) if p.size else int(np.argmax(s)))]
    # exit_peak: min in [end-20ms, end]
    end_t = opt_pred["opt_end"]
    seg2=(elec_td>=end_t-0.020)&(elec_td<=end_t)
    if not np.any(seg2): exit_peak = end_t
    else:
        s=-eS[seg2]; i0=np.nonzero(seg2)[0][0]
        p,_=find_peaks(s); exit_peak = elec_td[i0 + (int(p[np.argmax(s[p])]) if p.size else int(np.argmax(s)))]
    # exit_base: first sustained baseline after exit_peak
    start_idx = int(np.searchsorted(elec_td, exit_peak, side="left"))
    near = np.abs(eS - base) < 2.5*sig
    eb_idx=None; c=0
    for i in range(start_idx, len(near)):
        if near[i]: c+=1
        else: c=0
        if c>=run: eb_idx=i; break
    exit_base = elec_td[eb_idx] if eb_idx is not None else t_hi

    # >>> SNAP TO RAW for peaks (±10 ms), then clamp & order
    entry_peak_raw = refine_raw_extremum(t, elec, entry_peak, REFINE_ELEC_PEAK_MS, mode="max")
    exit_peak_raw  = refine_raw_extremum(t, elec, exit_peak,  REFINE_ELEC_PEAK_MS, mode="min")
    def clamp(v): return float(min(max(v, np.nextafter(t_lo,t_hi)), np.nextafter(t_hi,t_lo)))
    entry_peak = min(clamp(entry_peak_raw), clamp(step_t))
    exit_peak  = min(clamp(exit_peak_raw),  clamp(end_t))
    entry_base = clamp(entry_base)
    exit_base  = max(clamp(exit_base), exit_peak)

    return {"elec_entry_base":entry_base,"elec_entry_peak":entry_peak,
            "elec_exit_peak":exit_peak,"elec_exit_base":exit_base}

# =============== Dataset (strict ±50 ms) ===============
class StrictOptDataset(Dataset):
    def __init__(self, paths):
        self.items=[]
        for p in paths:
            t, elec, opt, fs, lm = load_npz_all(p)
            if lm.get("opt_base") is None or lm.get("opt_end") is None: continue
            if not all(lm.get(k) is not None for k in OPT_KEYS): continue
            t_lo = lm["opt_base"] - STRICT_PAD_MS/1000.0
            t_hi = lm["opt_end"]  + STRICT_PAD_MS/1000.0
            self.items.append((p,t,elec,opt,fs,lm,t_lo,t_hi))
        print(f"Prepared {len(self.items)} strict windows.")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p,t,elec,opt,fs,lm,t_lo,t_hi = self.items[idx]
        _, x = build_opt_inputs(t,opt,fs,t_lo,t_hi)
        dur = max(t_hi - t_lo, 1e-6)
        y = np.array([(lm[k]-t_lo)/dur for k in OPT_KEYS], dtype=np.float32)
        y = np.clip(y, 0.0, 1.0)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor([t_lo,t_hi],dtype=torch.float32), idx

# =============== Model (bigger + dropout) ===============
class StrongOpt1D(nn.Module):
    def __init__(self, in_ch=4, hidden=96, n_out=3, p_drop=0.1):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 9, padding=4), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, 128), nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(128, n_out),
            nn.Sigmoid()
        )
    def forward(self,x): return self.head(self.fe(x))

# =============== Loss / Sched / Train utils ===============
def masked_smooth_l1(pred, target):
    return torch.nn.functional.smooth_l1_loss(pred, target, reduction="mean")

def order_penalty_opt(y):
    p1 = torch.relu(y[:,OPT_IDX["opt_base"]] - y[:,OPT_IDX["opt_plateau"]]).mean()
    p2 = torch.relu(y[:,OPT_IDX["opt_plateau"]] - y[:,OPT_IDX["opt_end"]]).mean()
    return p1+p2

class CosineWithWarmup:
    def __init__(self, optimizer, base_lr, epochs, warmup_epochs):
        self.opt = optimizer; self.base_lr=base_lr
        self.epochs=epochs; self.warm=warmup_epochs; self.step_idx=0
    def step(self, epoch):
        if epoch < self.warm:
            lr = self.base_lr * (epoch+1)/max(1,self.warm)
        else:
            frac = (epoch - self.warm) / max(1,(self.epochs - self.warm))
            lr = self.base_lr*(0.1 + 0.9*0.5*(1+math.cos(math.pi*(1-frac))))
        for g in self.opt.param_groups: g['lr'] = lr
        return lr

def moving_avg(val, buf, k=5):
    buf.append(val)
    if len(buf)>k: buf.pop(0)
    return sum(buf)/len(buf)

# =============== Metrics (per-label MAE in ms) ===============
@torch.no_grad()
def compute_mae_ms(model, ds, idxs):
    model.eval()
    opt_err = np.zeros(3); opt_cnt = np.zeros(3)
    elec_err= np.zeros(4); elec_cnt= np.zeros(4)
    per_file = []
    for i in idxs:
        p,t,elec,opt,fs,lm,t_lo,t_hi = ds.items[i]
        # predict optical (fractions)
        _, x = build_opt_inputs(t,opt,fs,t_lo,t_hi)
        x = torch.from_numpy(x[None]).to(DEVICE, dtype=torch.float32)
        frac = model(x).cpu().numpy()[0]
        dur = (t_hi - t_lo)
        pred_opt = {
            "opt_base":     t_lo + float(frac[OPT_IDX["opt_base"]])*dur,
            "opt_plateau":  t_lo + float(frac[OPT_IDX["opt_plateau"]])*dur,
            "opt_end":      t_lo + float(frac[OPT_IDX["opt_end"]])*dur,
        }
        # order/clamp
        def clamp(v): return float(min(max(v, np.nextafter(t_lo,t_hi)), np.nextafter(t_hi,t_lo)))
        ob=clamp(pred_opt["opt_base"]); os=clamp(pred_opt["opt_plateau"]); oe=clamp(pred_opt["opt_end"])
        os=max(ob,os); oe=max(os,oe)
        pred_opt={"opt_base":ob,"opt_plateau":os,"opt_end":oe}

        # >>> SNAP optical STEP to RAW opt max within ±REFINE_OPT_PEAK_MS
        snapped_step = refine_raw_extremum(t, opt, pred_opt["opt_plateau"], REFINE_OPT_PEAK_MS, mode="max")
        pred_opt["opt_plateau"] = clamp(min(max(snapped_step, pred_opt["opt_base"]), pred_opt["opt_end"]))

        # optical MAE
        for j,k in enumerate(OPT_KEYS):
            if lm.get(k) is None: continue
            if not (t_lo <= lm[k] <= t_hi): continue
            opt_err[j] += abs(pred_opt[k] - lm[k]) * 1e3; opt_cnt[j] += 1

        # electrical MAE if GT exists (compute w/ snap-to-raw for peaks)
        pred_elec = compute_electrical_from_opt(t, elec, fs, t_lo, t_hi, pred_opt)
        for j,k in enumerate(ELEC_KEYS):
            if lm.get(k) is None: continue
            if not (t_lo <= lm[k] <= t_hi): continue
            elec_err[j] += abs(pred_elec[k] - lm[k]) * 1e3; elec_cnt[j] += 1

        per_file.append((p, pred_opt, pred_elec))

    opt_mae = {k: float(opt_err[OPT_IDX[k]]/max(opt_cnt[OPT_IDX[k]],1)) for k in OPT_KEYS}
    elec_mae= {k: float(elec_err[j]/max(elec_cnt[j],1)) for j,k in enumerate(ELEC_KEYS)}
    opt_avg = float(np.sum(opt_err)/max(np.sum(opt_cnt),1))
    elec_avg= float(np.sum(elec_err)/max(np.sum(elec_cnt),1))
    return opt_mae, elec_mae, opt_avg, elec_avg, per_file

# =============== Training loop ===============
def train_model(model, train_ds, val_ds, save_dir):
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    optim = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    sched = CosineWithWarmup(optim, BASE_LR, EPOCHS, WARMUP_EPOCHS)

    best_score = float("inf"); best_path = os.path.join(save_dir, "best_opt_strict.pt")
    loss_buf = []
    for epoch in range(1, EPOCHS+1):
        lr = sched.step(epoch-1)
        # --- train ---
        model.train(); epoch_loss = 0.0; nobs = 0
        for xb,yb,win,_ in train_ld:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            pred = model(xb)
            loss = masked_smooth_l1(pred, yb) + ORDER_W*order_penalty_opt(pred)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
            epoch_loss += loss.item()*xb.size(0); nobs += xb.size(0)
        tr_loss = epoch_loss / max(1,nobs)
        tr_loss_ma = moving_avg(tr_loss, loss_buf, k=5)

        # --- metrics (MAE in ms) ---
        tr_opt, tr_elec, tr_opt_avg, tr_elec_avg, _ = compute_mae_ms(model, train_ds, list(range(len(train_ds)))) if len(train_ds) else ({},{},float('nan'),float('nan'),[])
        va_opt, va_elec, va_opt_avg, va_elec_avg, _ = compute_mae_ms(model, val_ds,   list(range(len(val_ds))))   if len(val_ds)   else ({},{},float('nan'),float('nan'),[])

        print(f"ep {epoch:03d} | lr {lr:.4g} | loss {tr_loss:.4f} (ma {tr_loss_ma:.4f}) | "
              f"TR opt {tr_opt_avg:6.2f}ms / elec {tr_elec_avg:6.2f}ms | "
              f"VA opt {va_opt_avg:6.2f}ms / elec {va_elec_avg:6.2f}ms")

        # --- early stopping keyed on val optical+electrical average ---
        if len(val_ds):
            score = (va_opt_avg if np.isfinite(va_opt_avg) else 0.0) + (va_elec_avg if np.isfinite(va_elec_avg) else 0.0)
            if score < best_score:
                best_score = score
                torch.save({"model":model.state_dict(),"opt_keys":OPT_KEYS}, best_path)
                best_epoch = epoch
            if epoch - best_epoch > PATIENCE and epoch >= MIN_EPOCHS:
                print(f"Early stopping at epoch {epoch} (best @ {best_epoch}, score={best_score:.2f}ms).")
                break
        else:
            torch.save({"model":model.state_dict(),"opt_keys":OPT_KEYS}, best_path); best_epoch = epoch

    print("Saved best to:", best_path)
    return best_path

# =============== GUI folder picker ===============
def pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        try: root.call('wm','attributes','.','-topmost',True)
        except: pass
        d = filedialog.askdirectory(title="Select folder with .npz files")
        root.destroy()
        return d
    except Exception as e:
        print("Tkinter failed:", e); return None

# =============== Main ===============
set_seed()
folder = pick_folder() or input("Folder with .npz: ").strip()
paths = sorted(glob.glob(os.path.join(folder, "*.npz"))) if folder else []
if not paths: raise FileNotFoundError("No .npz files found.")

# split
random.shuffle(paths)
n_total = len(paths)
n_val = max(1, int(round(n_total*VAL_SPLIT))) if n_total > 4 else 1 if n_total>=2 else 0
val_paths = paths[:n_val] if n_val>0 else []
train_paths = paths[n_val:] if n_val>0 else paths

train_ds = StrictOptDataset(train_paths)
val_ds   = StrictOptDataset(val_paths) if n_val>0 else StrictOptDataset([])

print(f"Train files: {len(train_ds)} | Val files: {len(val_ds)} | Device: {DEVICE}")
model = StrongOpt1D().to(DEVICE)
best_ckpt = train_model(model, train_ds, val_ds, folder)

# Final evaluation with best weights
ck = torch.load(best_ckpt, map_location="cpu"); model.load_state_dict(ck["model"]); model.to(DEVICE).eval()
tr_opt, tr_elec, tr_opt_avg, tr_elec_avg, _ = compute_mae_ms(model, train_ds, list(range(len(train_ds)))) if len(train_ds) else ({},{},float('nan'),float('nan'),[])
va_opt, va_elec, va_opt_avg, va_elec_avg, _ = compute_mae_ms(model, val_ds,   list(range(len(val_ds))))   if len(val_ds)   else ({},{},float('nan'),float('nan'),[])
print("\n=== FINAL (best ckpt) ===")
print("Train Optical MAE (ms):", tr_opt, " | avg:", f"{tr_opt_avg:.2f}")
print("Train Electr. MAE (ms):", tr_elec, " | avg:", f"{tr_elec_avg:.2f}")
print("Val   Optical MAE (ms):", va_opt, " | avg:", f"{va_opt_avg:.2f}")
print("Val   Electr. MAE (ms):", va_elec, " | avg:", f"{va_elec_avg:.2f}")
