# FAST 2-PHASE PIPELINE
# Phase A: Excel+ABF ->  (exact window + stride decimation, optional LPF)
# Phase B: DL inference (optical) + ±5 ms refinement + electrical + ±5 ms refinement -> one Excel per source Excel
#
# Needs: pandas, numpy, scipy, torch, pyabf, openpyxl/xlrd, tkinter

import os, glob, math, random, shutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
import pyabf
from scipy import signal
from scipy.signal import find_peaks
import torch, torch.nn as nn

# ============================== CONFIG ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NPZ build (fast, like your script)
PAD_S          = 0.01  # 10 ms on each side of (window_start, window_end)
LPF_HZ         = 100.0 # set None to disable LPF
BESSEL_ORDER   = 6
TARGET_FS      = 5000.0  
MIN_SAMPLES    = 32
MAX_WINDOW_S   = 60.0

# 
USE_FIXED_CHANNELS = True
CH_ELEC, CH_OPT, CH_OPTREF = 0, 2, 3

# DL preprocess 
SMOOTH_FS    = 500.0
N_SAMPLES    = 256
MA_OPT_FAST  = 20.0  # ms
MA_OPT_SLOW  = 80.0  # ms
MA_ELEC      = 8.0   # ms

# Refinement windows (WIDENED: optical ±10 ms, electrical ±20 ms)
REFINE_MS_OPT  = 10.0
REFINE_MS_ELEC = 20.0

# Excel columns
REQUIRED_IN_COLS = ["event_id","file_name","sensor","analytes","solution","window_start","window_end"]
OUT_COLS = [
    "event_id","file_name","sensor","analytes","solution",
    "event_start (s)","event_end (s)","notes",
    "Base (V)","Step (V)","RefBase (V)","RefStep (V)",
    "entry Base (pA)","entry Peak (pA)","exit Peak (pA)","exit Base (pA)",
    "duration","OSC","RefOSC","entry spike","exit spike",
    "event_plateau","event_plateau_t","entry_base_t","entry_peak_t","exit_peak_t","exit_base_t"
]

# ============================== UTILS ==============================
def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def moving_average_same_len(x, win):
    win = int(max(1, round(win))); 
    if win % 2 == 0: win -= 1
    pad = win // 2
    k = np.ones(win, dtype=np.float32)/float(win)
    return np.convolve(np.pad(x, (pad,pad), mode="edge"), k, mode="valid").astype(np.float32, copy=False)

def stride_decimate(x, fs_in, fs_target):
    if fs_target is None: return x.astype(np.float32), float(fs_in)
    dec = max(1, int(round(float(fs_in) / float(fs_target))))
    return x[::dec].astype(np.float32), float(fs_in / dec)

def design_sos(order, cutoff_hz, fs):
    if cutoff_hz is None: return None
    try:
        return signal.butter(order, cutoff_hz, btype="low", output="sos", fs=fs)
    except Exception:
        wn = float(cutoff_hz) / (fs * 0.5)
        return signal.butter(order, wn, btype="low", output="sos")

def filt(x, sos):
    x = np.asarray(x, dtype=np.float32)
    if sos is None or len(x) < 8: return x
    try:
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        y = signal.sosfilt(sos, x)
        y = signal.sosfilt(sos, y[::-1])[::-1]
        return y.astype(np.float32)

def robust_norm(x, axis=1):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1e-9 + mad)

# ============================== DL MODEL + INPUTS ==============================
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
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    def forward(self,x): return self.head(self.fe(x))

OPT_KEYS  = ["opt_base","opt_plateau","opt_end"]
OPT_IDX   = {k:i for i,k in enumerate(OPT_KEYS)}
ELEC_KEYS = ["elec_entry_base","elec_entry_peak","elec_exit_peak","elec_exit_base"]

def to_fixed_length(series_t, series_x, t_lo, t_hi, n_out):
    if t_hi <= t_lo: t_hi = t_lo + 1e-6
    grid = np.linspace(t_lo, t_hi, n_out, dtype=np.float32)
    x = np.interp(grid, series_t, series_x).astype(np.float32)
    return grid, x

def build_opt_inputs(t, opt, fs, t_lo, t_hi):
    opt_d, fs_d = stride_decimate(opt, fs, SMOOTH_FS)
    td = t[0] + np.arange(opt_d.size)/fs_d
    w20 = max(1, int(round((MA_OPT_FAST/1000.0)*fs_d)))
    w80 = max(1, int(round((MA_OPT_SLOW/1000.0)*fs_d)))
    opt_ma20 = moving_average_same_len(opt_d, w20)
    opt_ma80 = moving_average_same_len(opt_d, w80)
    grid,  opt_raw = to_fixed_length(td, opt_d,    t_lo, t_hi, N_SAMPLES)
    _,     opt_f   = to_fixed_length(td, opt_ma20, t_lo, t_hi, N_SAMPLES)
    _,     opt_s   = to_fixed_length(td, opt_ma80, t_lo, t_hi, N_SAMPLES)
    pos = np.linspace(0,1,N_SAMPLES, dtype=np.float32)
    x = np.stack([opt_raw, opt_f, opt_s, pos], axis=0)
    x = robust_norm(x, axis=1)
    return grid, x

@torch.no_grad()
def infer_opt_only(model, t, opt, fs, t_lo, t_hi):
    _, x = build_opt_inputs(t,opt,fs,t_lo,t_hi)
    x = torch.from_numpy(x[None]).to(DEVICE, dtype=torch.float32)
    frac = model(x).cpu().numpy()[0]
    dur = (t_hi - t_lo)
    ob, os, oe = [t_lo + float(frac[OPT_IDX[k]])*dur for k in OPT_KEYS]
    def clamp(v): return float(min(max(v, np.nextafter(t_lo,t_hi)), np.nextafter(t_hi,t_lo)))
    ob=clamp(ob); os=clamp(os); oe=clamp(oe)
    os = max(ob, os); oe = max(os, oe)
    return {"opt_base":ob, "opt_plateau":os, "opt_end":oe}

# ============================== REFINEMENT (± window) ==============================
def refine_optical_on_raw(opt_t, opt_y, pred):
    def widx(c, ms):
        r = ms/1000.0; m = (opt_t>=c-r)&(opt_t<=c+r); return np.nonzero(m)[0]
    out = dict(pred)
    # base/end -> lowest-variance 5-pt window
    for k in ["opt_base","opt_end"]:
        idx = widx(pred[k], REFINE_MS_OPT)
        if idx.size >= 5:
            v=[]; win=5
            for i in range(idx[0], idx[-1]-win+2): v.append(np.var(opt_y[i:i+win]))
            if v: out[k] = float(opt_t[idx[0] + int(np.argmin(v))])
    # step -> tallest local peak within window
    idx = widx(pred["opt_plateau"], REFINE_MS_OPT)
    if idx.size >= 3:
        seg = opt_y[idx]; pk,_ = find_peaks(seg)
        j = int(pk[np.argmax(seg[pk])]) if pk.size else int(np.argmax(seg))
        out["opt_plateau"] = float(opt_t[idx[j]])
    return out

def compute_electrical_from_opt(t, elec, fs, t_lo, t_hi, opt_pred):
    m=(t>=t_lo)&(t<=t_hi)
    if not np.any(m): 
        return {k:float(t_lo) for k in ELEC_KEYS}
    ed, fs_ed = stride_decimate(elec[m], fs, SMOOTH_FS)
    elec_td = t_lo + np.arange(ed.size)/fs_ed
    w = max(1, int(round((MA_ELEC/1000.0)*fs_ed)))
    eS = moving_average_same_len(ed, w)
    b = max(6, int(0.20*len(eS)))
    base = float(np.median(eS[:b])); sig = float(max(1e-9, 1.4826*np.median(np.abs(eS[:b]-np.median(eS[:b])))))
    run = max(6, int(round((8e-3)*fs_ed)))
    # entry_base: last sustained baseline before start
    mask_left = elec_td <= opt_pred["opt_base"]
    if np.any(mask_left):
        idx=None; c=0; left=np.nonzero(mask_left)[0]
        for i in left[::-1]:
            if abs(eS[i]-base) < 2.5*sig: c+=1
            else: c=0
            if c>=run: idx=i; break
        if idx is None: idx = left[-1]
        entry_base = elec_td[idx]
    else:
        entry_base = float(t_lo)
    # entry_peak: max in [step-20ms, step]
    step_t = opt_pred["opt_plateau"]
    seg = (elec_td>=step_t-0.020)&(elec_td<=step_t)
    if not np.any(seg): entry_peak = step_t
    else:
        s=eS[seg]; i0=np.nonzero(seg)[0][0]; pk,_=find_peaks(s)
        entry_peak = elec_td[i0 + (int(pk[np.argmax(s[pk])]) if pk.size else int(np.argmax(s)))]
    # exit_peak: min in [end-20ms, end]
    end_t = opt_pred["opt_end"]
    seg2 = (elec_td>=end_t-0.020)&(elec_td<=end_t)
    if not np.any(seg2): exit_peak = end_t
    else:
        s=-eS[seg2]; i0=np.nonzero(seg2)[0][0]; pk,_=find_peaks(s)
        exit_peak = elec_td[i0 + (int(pk[np.argmax(s[pk])]) if pk.size else int(np.argmax(-s)))]
    # exit_base: first sustained baseline after exit_peak
    start = int(np.searchsorted(elec_td, exit_peak, side="left"))
    near = np.abs(eS - base) < 2.5*sig
    eb_idx=None; c=0
    for i in range(start, len(near)):
        if near[i]: c+=1
        else: c=0
        if c>=run: eb_idx=i; break
    exit_base = elec_td[eb_idx] if eb_idx is not None else t_hi
    # clamp/order
    def clamp(v): return float(min(max(v, np.nextafter(t_lo,t_hi)), np.nextafter(t_hi,t_lo)))
    entry_peak = min(clamp(entry_peak), clamp(step_t))
    exit_peak  = min(clamp(exit_peak),  clamp(end_t))
    entry_base = clamp(entry_base)
    exit_base  = max(clamp(exit_base), exit_peak)
    return {"elec_entry_base":entry_base,"elec_entry_peak":entry_peak,
            "elec_exit_peak":exit_peak,"elec_exit_base":exit_base}

def refine_electrical_on_raw(elec_t, elec_y, pred):
    def widx(c, ms):
        r=ms/1000.0; m=(elec_t>=c-r)&(elec_t<=c+r); return np.nonzero(m)[0]
    out=dict(pred)
    idx = widx(pred["elec_entry_peak"], REFINE_MS_ELEC)
    if idx.size>=3:
        seg=elec_y[idx]; pk,_=find_peaks(seg)
        j=int(pk[np.argmax(seg[pk])]) if pk.size else int(np.argmax(seg))
        out["elec_entry_peak"]=float(elec_t[idx[j]])
    idx = widx(pred["elec_exit_peak"], REFINE_MS_ELEC)
    if idx.size>=3:
        seg=-elec_y[idx]; pk,_=find_peaks(seg)
        j=int(pk[np.argmax(seg[pk])]) if pk.size else int(np.argmax(-seg))
        out["elec_exit_peak"]=float(elec_t[idx[j]])
    for k in ["elec_entry_base","elec_exit_base"]:
        idx=widx(pred[k], REFINE_MS_ELEC)
        if idx.size>=5:
            v=[]; win=5
            for i in range(idx[0], idx[-1]-win+2): v.append(np.var(elec_y[i:i+win]))
            if v: out[k]=float(elec_t[idx[0]+int(np.argmin(v))])
    eb,ep,xp,xb=out["elec_entry_base"],out["elec_entry_peak"],out["elec_exit_peak"],out["elec_exit_base"]
    if ep<eb: ep=eb
    if xp<ep: xp=ep
    if xb<xp: xb=xp
    out.update({"elec_entry_base":eb,"elec_entry_peak":ep,"elec_exit_peak":xp,"elec_exit_base":xb})
    return out

# ============================== ABF FAST CUT ==============================
def resolve_sweep_range(abf: pyabf.ABF):
    """Return list of (sweep_idx, t0, t1) in absolute seconds across whole file."""
    ranges=[]
    t = 0.0
    for s in range(abf.sweepCount):
        abf.setSweep(sweepNumber=s, channel=0)
        dur = float(abf.sweepLengthSec)
        ranges.append((s, t, t+dur))
        t += dur
    return ranges

def cut_exact_from_sweep(abf: pyabf.ABF, fs: float, sweep_idx: int, t0_abs: float, t1_abs: float):
    """Cut [t0_abs, t1_abs] but **within a single sweep** (fast)."""
    s, s0, s1 = sweep_idx
    abf.setSweep(sweepNumber=s, channel=0); dur = float(abf.sweepLengthSec)
    fs_ok = float(abf.dataRate) if hasattr(abf,"dataRate") else fs
    # local sweep times
    t0_loc = max(0.0, t0_abs - s0)
    t1_loc = min(dur, t1_abs - s0)
    i0 = int(round(t0_loc * fs_ok))
    i1 = int(round(t1_loc * fs_ok))
    if i1 <= i0: i1 = i0 + MIN_SAMPLES
    # fetch channels quickly
    ys=[]
    for ch in range(abf.channelCount):
        abf.setSweep(sweepNumber=s, channel=ch)
        y = abf.sweepY
        ys.append(np.asarray(y[i0:i1], dtype=np.float32))
    return ys, fs_ok, (s0 + t0_loc)

def pick_channels(names, units):
    if USE_FIXED_CHANNELS:
        return CH_ELEC, CH_OPT, CH_OPTREF
    # heuristic fallback if needed
    cand_e = [i for i,u in enumerate(units) if "pa" in str(u).lower() or str(u).lower().endswith("a")]
    elec = cand_e[0] if cand_e else 0
    volt = [i for i,u in enumerate(units) if str(u).lower()=="v"]
    if volt:
        ref=None
        for i in volt:
            if "ref" in str(names[i]).lower(): ref=i; break
        if ref is None and len(volt)>=2: ref=volt[1]
        if ref is None: ref=volt[0]
        opt = volt[0] if volt[0]!=ref else (volt[1] if len(volt)>=2 else volt[0])
    else:
        others=[i for i in range(len(names)) if i!=elec]
        opt = others[0] if others else elec
        ref = others[1] if len(others)>1 else opt
    return int(elec), int(opt), int(ref)

# ============================== SCAN + GUI PREVIEW ==============================
def _excel_paths(folder):
    pats=["*.xlsx","*.xls","*.xlsm","*.csv"]
    out=[]
    for pat in pats: out+=glob.glob(os.path.join(folder,pat))
    return sorted(set(out))

def scan_folder(folder):
    excels=_excel_paths(folder)
    abfs=sorted(glob.glob(os.path.join(folder,"**","*.abf"), recursive=True))
    abf_map={os.path.splitext(os.path.basename(p))[0].lower():p for p in abfs}
    matches=[]; unmatched=[]; mentioned=set()
    for xp in excels:
        try: df = pd.read_csv(xp) if xp.lower().endswith(".csv") else pd.read_excel(xp)
        except Exception: df=pd.DataFrame()
        bases=[]
        if not df.empty:
            col=None
            for c in df.columns:
                if str(c).strip().lower()=="file_name": col=c; break
            if col is not None:
                for v in df[col].dropna().astype(str).values:
                    b=os.path.splitext(os.path.basename(v.strip()))[0].lower()
                    if b: bases.append(b)
        bases=sorted(set(bases))
        this=[]
        for b in bases:
            if b in abf_map:
                this.append((b,abf_map[b])); mentioned.add(b)
            else:
                unmatched.append((xp,b))
        matches.append((xp,this))
    orphans=[p for b,p in abf_map.items() if b not in mentioned]
    print(f"[scan] Found {len(excels)} Excel(s), {len(abfs)} ABF(s), "
          f"{sum(len(m) for _,m in matches)} matched pair(s).")
    return excels, abfs, matches, unmatched, orphans

def preview_gui(folder, excels, abfs, matches, unmatched, orphans):
    root=tk.Tk(); root.title("Preview: Excel/ABF scan"); root.geometry("1100x600")
    try: root.call('wm','attributes','.','-topmost',True)
    except: pass
    frm=ttk.Frame(root,padding=10); frm.pack(fill="both",expand=True)
    info=f"Folder: {folder}\nExcels: {len(excels)}   ABFs: {len(abfs)}   Matched pairs: {sum(len(m) for _,m in matches)}   Unmatched refs: {len(unmatched)}   Orphans: {len(orphans)}"
    ttk.Label(frm,text=info).pack(anchor="w",pady=(0,8))
    paned=ttk.PanedWindow(frm,orient="horizontal"); paned.pack(fill="both",expand=True)
    left=ttk.Frame(paned,padding=6)
    lf1=ttk.Labelframe(left,text="Excels")
    lb1=tk.Listbox(lf1); sb1=ttk.Scrollbar(lf1,orient="vertical",command=lb1.yview); lb1.config(yscrollcommand=sb1.set)
    lb1.pack(side="left",fill="both",expand=True); sb1.pack(side="right",fill="y")
    for p in excels: lb1.insert("end", os.path.basename(p))
    lf1.pack(fill="both",expand=True,pady=(0,6))
    lf2=ttk.Labelframe(left,text="ABFs"); lb2=tk.Listbox(lf2); sb2=ttk.Scrollbar(lf2,orient="vertical",command=lb2.yview); lb2.config(yscrollcommand=sb2.set)
    lb2.pack(side="left",fill="both",expand=True); sb2.pack(side="right",fill="y")
    for p in abfs: lb2.insert("end", p)
    lf2.pack(fill="both",expand=True)
    paned.add(left,weight=1)
    mid=ttk.Frame(paned,padding=6); lf3=ttk.Labelframe(mid,text="Matches (excel → base → abf)")
    lb3=tk.Listbox(lf3); sb3=ttk.Scrollbar(lf3,orient="vertical",command=lb3.yview); lb3.config(yscrollcommand=sb3.set)
    lb3.pack(side="left",fill="both",expand=True); sb3.pack(side="right",fill="y")
    for xp,ml in matches:
        x=os.path.basename(xp)
        if not ml: lb3.insert("end", f"[{x}] — no matches")
        for b,ap in ml: lb3.insert("end", f"[{x}] {b} → {ap}")
    lf3.pack(fill="both",expand=True); paned.add(mid,weight=2)
    right=ttk.Frame(paned,padding=6)
    lf4=ttk.Labelframe(right,text="Unmatched Excel refs"); lb4=tk.Listbox(lf4); sb4=ttk.Scrollbar(lf4,orient="vertical",command=lb4.yview); lb4.config(yscrollcommand=sb4.set)
    lb4.pack(side="left",fill="both",expand=True); sb4.pack(side="right",fill="y")
    for xp,b in unmatched: lb4.insert("end", f"[{os.path.basename(xp)}]  {b}")
    lf4.pack(fill="both",expand=True,pady=(0,6))
    lf5=ttk.Labelframe(right,text="Orphan ABFs"); lb5=tk.Listbox(lf5); sb5=ttk.Scrollbar(lf5,orient="vertical",command=lb5.yview); lb5.config(yscrollcommand=sb5.set)
    lb5.pack(side="left",fill="both",expand=True); sb5.pack(side="right",fill="y")
    for ap in orphans: lb5.insert("end", ap)
    lf5.pack(fill="both",expand=True); paned.add(right,weight=1)
    proceed={"ok":False}
    b=ttk.Frame(frm); b.pack(fill="x",pady=8)
    ttk.Button(b,text="Cancel",command=lambda:(root.destroy())).pack(side="right",padx=(0,6))
    ttk.Button(b,text="Proceed",command=lambda:(proceed.update(ok=True),root.destroy())).pack(side="right")
    root.mainloop()
    return proceed["ok"]

def pick_folder_dialog():
    root=tk.Tk(); root.withdraw()
    try: root.call('wm','attributes','.','-topmost',True)
    except: pass
    d=filedialog.askdirectory(title="Select folder with Excel + ABF")
    root.destroy(); return d

def pick_ckpt_dialog():
    root=tk.Tk(); root.withdraw()
    try: root.call('wm','attributes','.','-topmost',True)
    except: pass
    p=filedialog.askopenfilename(title="Select trained checkpoint (.pt)", filetypes=[("PyTorch","*.pt"),("All","*.*")])
    root.destroy(); return p

# ============================== PHASE A: FAST NPZ CREATION ==============================
def save_fast_npz(abf_path, row, out_dir):
    base = os.path.splitext(os.path.basename(abf_path))[0]
    # open ABF & metadata (no concat)
    A = pyabf.ABF(abf_path)
    fs = float(A.dataRate) if hasattr(A,"dataRate") else (1.0/float(A.dataSecPerPoint))
    ranges = resolve_sweep_range(A)

    # window from Excel (+/- 10ms)
    ws = float(row["window_start"]); we = float(row["window_end"])
    t0 = max(0.0, ws - PAD_S); t1 = we + PAD_S
    if (t1 - t0) > MAX_WINDOW_S:
        raise ValueError("window too long")

    # find the sweep that fully contains this window
    sweep = None
    for s_idx, s_lo, s_hi in ranges:
        if t0 >= s_lo and t1 <= s_hi:
            sweep = (s_idx, s_lo, s_hi); break
    if sweep is None:
        # fallback: pick the sweep where most overlap occurs
        overlaps = [(s_idx, max(0.0, min(s_hi,t1)-max(s_lo,t0)), s_lo, s_hi) for s_idx,s_lo,s_hi in ranges]
        s_idx, ov, s_lo, s_hi = max(overlaps, key=lambda z:z[1])
        if ov <= 0:
            raise ValueError("window not in any sweep")
        sweep = (s_idx, s_lo, s_hi)
        # clamp the cut to the overlapping portion
        t0 = max(t0, s_lo); t1 = min(t1, s_hi)

    # cut only that sweep's arrays
    ys, fs_ok, seg_t0_abs = cut_exact_from_sweep(A, fs, sweep, t0, t1)
    names = [A.adcNames[i] if hasattr(A,"adcNames") else f"ch{i}" for i in range(A.channelCount)]
    units = [A.adcUnits[i] if hasattr(A,"adcUnits") else "" for i in range(A.channelCount)]
    elec_i, opt_i, ref_i = pick_channels(names, units)

    # select 3 channels
    elec = ys[elec_i]
    opt  = ys[opt_i]
    ref  = ys[ref_i]
    L = min(len(elec), len(opt), len(ref))
    elec, opt, ref = elec[:L], opt[:L], ref[:L]

    # LPF + stride decimation (no interpolation)
    sos = design_sos(BESSEL_ORDER, LPF_HZ, fs_ok)
    elec = filt(elec, sos); opt = filt(opt, sos); ref = filt(ref, sos)
    elec, fs_out = stride_decimate(elec, fs_ok, TARGET_FS)
    opt,  _      = stride_decimate(opt,  fs_ok, TARGET_FS)
    ref,  _      = stride_decimate(ref,  fs_ok, TARGET_FS)
    L = min(len(elec), len(opt), len(ref))
    elec, opt, ref = elec[:L], opt[:L], ref[:L]

    if L < MIN_SAMPLES:
        raise ValueError("too few samples after cut/decimate")

    x_raw = np.stack([elec, opt, ref], axis=0).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{base}__{int(round(1e6*seg_t0_abs))}_{int(round(1e6*(seg_t0_abs + L/fs_out)))}.npz")
    np.savez_compressed(
        npz_path,
        x_raw=x_raw,
        fs=float(fs_out),
        t0=float(seg_t0_abs),
        length=int(L),
        base_name=base
    )
    return npz_path

# ============================== PHASE B: INFERENCE ==============================
def run_inference_on_npz(model, npz_path):
    z = np.load(npz_path, allow_pickle=True)
    with z:
        x_raw=z["x_raw"].astype(np.float32); fs=float(z["fs"]); t0=float(z["t0"]); L=int(z["length"])
        base=str(z.get("base_name", os.path.basename(npz_path)))
    t = t0 + np.arange(L, dtype=np.float32)/fs
    elec, opt, optr = x_raw[0], x_raw[1], x_raw[2]
    t_lo, t_hi = float(t[0]), float(t[-1])

    # optical DL -> refine
    opt_pred  = infer_opt_only(model, t, opt, fs, t_lo, t_hi)
    opt_pred  = refine_optical_on_raw(t, opt, opt_pred)
    # electrical -> refine
    elec_pred = compute_electrical_from_opt(t, elec, fs, t_lo, t_hi, opt_pred)
    elec_pred = refine_electrical_on_raw(t, elec, elec_pred)

    def interp(sig, tt): return float(np.interp(tt, t, sig))
    v_base=interp(opt,  opt_pred["opt_base"])
    v_step=interp(opt,  opt_pred["opt_plateau"])
    vr_b =interp(optr, opt_pred["opt_base"])
    vr_s =interp(optr, opt_pred["opt_plateau"])
    eb_pa=interp(elec, elec_pred["elec_entry_base"])
    ep_pa=interp(elec, elec_pred["elec_entry_peak"])
    xp_pa=interp(elec, elec_pred["elec_exit_peak"])
    xb_pa=interp(elec, elec_pred["elec_exit_base"])

    duration = opt_pred["opt_end"] - opt_pred["opt_base"]
    osc      = v_step - v_base
    refosc   = vr_s - vr_b
    ent_spk  = ep_pa - eb_pa
    ex_spk   = xp_pa - xb_pa

    out = {
        "event_start (s)": opt_pred["opt_base"],
        "event_end (s)":   opt_pred["opt_end"],
        "notes": "",
        "Base (V)": v_base, "Step (V)": v_step,
        "RefBase (V)": vr_b, "RefStep (V)": vr_s,
        "entry Base (pA)": eb_pa, "entry Peak (pA)": ep_pa,
        "exit Peak (pA)":  xp_pa, "exit Base (pA)":  xb_pa,
        "duration": duration, "OSC": osc, "RefOSC": refosc,
        # CHANGED: event_plateau is now the TIME (seconds)
        "event_plateau": opt_pred["opt_plateau"],
        # Kept for compatibility: also output the plateau time here
        "event_plateau_t": opt_pred["opt_plateau"],
        "entry_base_t": elec_pred["elec_entry_base"],
        "entry_peak_t": elec_pred["elec_entry_peak"],
        "exit_peak_t":  elec_pred["elec_exit_peak"],
        "exit_base_t":  elec_pred["elec_exit_base"],
        "_base_name": base
    }
    return out

# ============================== MAIN ==============================
def main():
    set_seed()
    folder = pick_folder_dialog()
    ckpt   = pick_ckpt_dialog()
    if not folder or not os.path.isdir(folder): raise FileNotFoundError("Folder not found")
    if not ckpt   or not os.path.isfile(ckpt):  raise FileNotFoundError("Checkpoint not found")

    excels, abfs, matches, unmatched, orphans = scan_folder(folder)
    ok = preview_gui(folder, excels, abfs, matches, unmatched, orphans)
    if not ok:
        print("Cancelled."); return

    # Phase A: build ALL NPZs (fast)
    npz_dir = os.path.join(folder, "_tmp_npz")
    # ---- overwrite temp folder if it already exists ----
    try:
        if os.path.exists(npz_dir):
            if os.path.isdir(npz_dir):
                shutil.rmtree(npz_dir)
            else:
                os.remove(npz_dir)
        os.makedirs(npz_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare temp folder '{npz_dir}': {e}")
    # ----------------------------------------------------

    # job registry grouped by source Excel so we can write one output per Excel
    jobs_by_excel = {}  # xp -> list of dict(row=..., npz=..., note=...)
    for xp, pair_list in matches:
        if not pair_list: continue
        df = pd.read_csv(xp) if xp.lower().endswith(".csv") else pd.read_excel(xp)
        miss = [c for c in REQUIRED_IN_COLS if c not in df.columns]
        if miss:
            print(f"[{os.path.basename(xp)}] missing required cols {miss} -> skip file.")
            continue
        base_to_abf = {os.path.splitext(os.path.basename(ap))[0].lower(): ap for _,ap in pair_list}
        jobs=[]
        print(f"\n[Phase A] Building NPZs for {os.path.basename(xp)} ...")
        for _, row in df.iterrows():
            rec = {k: row.get(k, None) for k in REQUIRED_IN_COLS}
            base = os.path.splitext(os.path.basename(str(rec["file_name"]).strip()))[0].lower()
            if base not in base_to_abf:
                jobs.append({"row":rec, "npz":None, "note":"abf not found"}); continue
            try:
                npz_path = save_fast_npz(base_to_abf[base], rec, npz_dir)
                jobs.append({"row":rec, "npz":npz_path, "note":""})
            except Exception as e:
                jobs.append({"row":rec, "npz":None, "note":f"npz build failed: {e}"})
        jobs_by_excel[xp]=jobs
        print(f"  -> {sum(1 for j in jobs if j['npz'])} NPZ(s)")

    # Phase B: load model once, run inference+refine
    print("\n[Phase B] Inference with trained model ...")
    ck = torch.load(ckpt, map_location="cpu")
    model = StrongOpt1D().to(DEVICE).eval()
    model.load_state_dict(ck["model"])

    for xp, jobs in jobs_by_excel.items():
        out_rows=[]
        for j in jobs:
            row=j["row"]; note=j["note"]
            if not j["npz"]:
                out_rows.append({
                    "event_id": row["event_id"], "file_name": row["file_name"],
                    "sensor": row["sensor"], "analytes": row["analytes"], "solution": row["solution"],
                    "event_start (s)": None, "event_end (s)": None, "notes": note,
                    "Base (V)": None, "Step (V)": None, "RefBase (V)": None, "RefStep (V)": None,
                    "entry Base (pA)": None,"entry Peak (pA)": None,"exit Peak (pA)": None,"exit Base (pA)": None,
                    "duration": None,"OSC": None,"RefOSC": None,"entry spike": None,"exit spike": None,
                    "event_plateau": None,"event_plateau_t": None,"entry_base_t": None,"entry_peak_t": None,"exit_peak_t": None,"exit_base_t": None
                })
                continue
            try:
                pred = run_inference_on_npz(model, j["npz"])
                out_rows.append({
                    "event_id": row["event_id"],
                    "file_name": row["file_name"],
                    "sensor": row["sensor"],
                    "analytes": row["analytes"],
                    "solution": row["solution"],
                    **{k: pred[k] for k in OUT_COLS if k in pred}
                })
            except Exception as e:
                out_rows.append({
                    "event_id": row["event_id"], "file_name": row["file_name"],
                    "sensor": row["sensor"], "analytes": row["analytes"], "solution": row["solution"],
                    "event_start (s)": None, "event_end (s)": None, "notes": f"infer fail: {e}",
                    "Base (V)": None, "Step (V)": None, "RefBase (V)": None, "RefStep (V)": None,
                    "entry Base (pA)": None,"entry Peak (pA)": None,"exit Peak (pA)": None,"exit Base (pA)": None,
                    "duration": None,"OSC": None,"RefOSC": None,"entry spike": None,"exit spike": None,
                    "event_plateau": None,"event_plateau_t": None,"entry_base_t": None,"entry_peak_t": None,"exit_peak_t": None,"exit_base_t": None
                })

        df_out = pd.DataFrame(out_rows, columns=OUT_COLS)
        out_xlsx = os.path.splitext(xp)[0] + "__predicted.xlsx"
        df_out.to_excel(out_xlsx, index=False)
        print(f"[write] {os.path.basename(out_xlsx)}  ({len(df_out)} rows)")

    print("\nDone. Temp NPZs:", npz_dir)

if __name__ == "__main__":
    main()
