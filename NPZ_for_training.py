# Creating Training Dataset with NPZ for Model training
import os, glob, warnings
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")

# ----------------- CONFIG -----------------
DATASET_ROOT = r"E:\Neural Network Trained"
CACHE_DIR    = os.path.join(DATASET_ROOT, "_preprocessed_cache_exact")
PAD_S        = 0.5
LPF_HZ       = 100.0          # set None to disable LPF
TARGET_FS    = 5000.0         # set None to disable
BESSEL_ORDER = 6

CH_ELEC      = 0
CH_OPT       = 2
CH_OPTREF    = 3

MIN_SAMPLES  = 32
MAX_WINDOW_S = 60.0
OVERWRITE    = True           # False -> skip if NPZ exists
# ------------------------------------------

LANDMARK_NAMES = np.array([
    "opt_base", "opt_plateau", "opt_end",
    "elec_entry_base", "elec_entry_peak", "elec_exit_peak", "elec_exit_base"
], dtype=object)

REQ_COLS = [
    "event_start (s)", "event_end (s)", "Base (V)", "Step (V)",
    "RefBase (V)", "RefStep (V)", "entry Base (pA)", "entry Peak (pA)",
    "exit Peak (pA)", "exit Base (pA)"
]

def log(s): print(s, flush=True)

# ----------------- I/O helpers -----------------
def find_file_pairs(root):
    abfs = glob.glob(os.path.join(root, "*.abf"))
    tables = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        tables += glob.glob(os.path.join(root, ext))
    by_abf = {os.path.splitext(os.path.basename(a))[0]: a for a in abfs}
    by_tbl = {os.path.splitext(os.path.basename(t))[0]: t for t in tables}
    common = sorted(set(by_abf) & set(by_tbl))
    return [(by_abf[k], by_tbl[k]) for k in common]

def load_table(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing} in {os.path.basename(path)}")
    return df

# ----------------- signal utils -----------------
def design_sos(order, cutoff_hz, fs):
    if cutoff_hz is None:
        return None
    try:
        return signal.butter(order, cutoff_hz, btype="low", output="sos", fs=fs)
    except Exception:
        wn = float(cutoff_hz) / (fs * 0.5)
        return signal.butter(order, wn, btype="low", output="sos")

def filt(x, sos):
    x = np.asarray(x, dtype=np.float32)
    if sos is None or len(x) < 8:
        return x
    try:
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        y = signal.sosfilt(sos, x)
        y = signal.sosfilt(sos, y[::-1])[::-1]
        return y.astype(np.float32)

def decimate_stride(x, fs_in, target_fs):
    if target_fs is None:
        return x.astype(np.float32), float(fs_in)
    dec = max(1, int(round(fs_in / target_fs)))
    return x[::dec].astype(np.float32), float(fs_in / dec)

def exact_cut(abf, fs, t0, t1, ch):
    abf.setSweep(sweepNumber=0, channel=ch)
    y = abf.sweepY
    n = len(y)
    i0 = max(0, min(int(round(t0*fs)), n-1))
    i1 = max(i0+MIN_SAMPLES, min(int(round(t1*fs)), n))
    return np.asarray(y[i0:i1], dtype=np.float32)

def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    sd  = np.std(x) + 1e-6
    return (x - med) / sd

# ---- landmark helpers (no interpolation) ----
def time_to_idx(t_sec: float, t0: float, fs: float, length: int) -> int:
    idx = int(round((t_sec - t0) * fs))
    return max(0, min(idx, length - 1))

def idx_to_time(idx: int, t0: float, fs: float) -> float:
    return t0 + idx / max(1e-12, fs)

def find_nearest_value_index(sig: np.ndarray, fs: float, t0: float,
                             time_window: Tuple[float, float], target_value: float
                             ) -> Optional[int]:
    n = len(sig)
    i0 = int(round((time_window[0] - t0) * fs))
    i1 = int(round((time_window[1] - t0) * fs))
    i0 = max(0, min(i0, n - 1))
    i1 = max(i0 + 1, min(i1, n))
    if i1 <= i0:
        return None
    seg = sig[i0:i1]
    return i0 + int(np.argmin(np.abs(seg - target_value)))

# ----------------- one-event save -----------------
def save_npz_for_event(abf, fs, base, abf_path, row, out_dir, idx):
    # parse table values
    t_start = float(row["event_start (s)"])
    t_end   = float(row["event_end (s)"])
    base_V  = float(row["Base (V)"])
    step_V  = float(row["Step (V)"])
    refbase_V = float(row["RefBase (V)"])
    refstep_V = float(row["RefStep (V)"])
    e_base  = float(row["entry Base (pA)"])
    e_peak  = float(row["entry Peak (pA)"])
    x_peak  = float(row["exit Peak (pA)"])
    x_base  = float(row["exit Base (pA)"])

    # exact bounds (clamped by exact_cut)
    t0 = max(0.0, t_start - PAD_S)
    t1 = t_end + PAD_S
    if (t1 - t0) > MAX_WINDOW_S:
        return False, "window too long"

    # cut three channels
    e = exact_cut(abf, fs, t0, t1, CH_ELEC)
    o = exact_cut(abf, fs, t0, t1, CH_OPT)
    r = exact_cut(abf, fs, t0, t1, CH_OPTREF)
    Lmin = min(len(e), len(o), len(r))
    if Lmin < MIN_SAMPLES:
        return False, "too few samples"
    e, o, r = e[:Lmin], o[:Lmin], r[:Lmin]

    # filter + stride decimate
    sos = design_sos(BESSEL_ORDER, LPF_HZ, fs)
    e = filt(e, sos); o = filt(o, sos); r = filt(r, sos)
    e, fs_out = decimate_stride(e, fs, TARGET_FS)
    o, _      = decimate_stride(o, fs, TARGET_FS)
    r, _      = decimate_stride(r, fs, TARGET_FS)
    L = min(len(e), len(o), len(r))
    e, o, r = e[:L], o[:L], r[:L]

    x_raw  = np.stack([e, o, r], axis=0).astype(np.float32)
    x_norm = np.stack([zscore(e), zscore(o), zscore(r)], axis=0).astype(np.float32)

    # ---- compute landmarks on decimated timebase ----
    duration = max(0.0, t_end - t_start)
    t_mid    = t_start + 0.5 * duration

    # Optical:
    opt_base_idx = time_to_idx(t_start, t0, fs_out, L)
    opt_end_idx  = time_to_idx(t_end,   t0, fs_out, L)
    opt_plateau_idx = find_nearest_value_index(
        o, fs_out, t0, (t_start, t_end), step_V
    )

    # Electrical (nearest-to-target within sub-windows)
    elec_entry_base_idx = find_nearest_value_index(
        e, fs_out, t0, (t0, t_start + 0.3 * duration), e_base
    )
    elec_entry_peak_idx = find_nearest_value_index(
        e, fs_out, t0, (t_start, t_mid), e_peak
    )
    elec_exit_peak_idx = find_nearest_value_index(
        e, fs_out, t0, (t_mid, t_end), x_peak
    )
    elec_exit_base_idx = find_nearest_value_index(
        e, fs_out, t0, (t_start + 0.7 * duration, t_end + PAD_S), x_base
    )

    lm_idx = [
        opt_base_idx,
        (-1 if opt_plateau_idx is None else int(opt_plateau_idx)),
        opt_end_idx,
        (-1 if elec_entry_base_idx is None else int(elec_entry_base_idx)),
        (-1 if elec_entry_peak_idx is None else int(elec_entry_peak_idx)),
        (-1 if elec_exit_peak_idx is None else int(elec_exit_peak_idx)),
        (-1 if elec_exit_base_idx is None else int(elec_exit_base_idx)),
    ]
    lm_idx = np.array(lm_idx, dtype=np.int32)

    # times for each (None -> -1)
    lm_time = np.array([
        idx_to_time(i, t0, fs_out) if i >= 0 else -1.0 for i in lm_idx
    ], dtype=np.float64)

    # dedicated "event_step_time" (same as optical plateau time)
    event_step_time_s = (idx_to_time(lm_idx[1], t0, fs_out) if lm_idx[1] >= 0 else np.nan)

    # save
    out_path = os.path.join(out_dir, f"{base}_window_{idx:04d}.npz")
    if (not OVERWRITE) and os.path.exists(out_path):
        return False, "exists"

    np.savez_compressed(
        out_path,
        # signals
        x_norm=x_norm,
        x_raw=x_raw,
        fs=fs_out,
        length=L,

        # window/time
        t0=t0,
        t1=t1,
        t_start=t_start,
        t_end=t_end,

        # landmark info
        landmark_names=LANDMARK_NAMES,
        landmarks_idx=lm_idx,
        landmarks_time_s=lm_time,

        # table targets (saved flat)
        base_V=base_V,
        step_V=step_V,
        refbase_V=refbase_V,
        refstep_V=refstep_V,
        entry_base_pA=e_base,
        entry_peak_pA=e_peak,
        exit_peak_pA=x_peak,
        exit_base_pA=x_base,
        event_step_time_s=event_step_time_s,

        # provenance
        abf_file=getattr(abf, "_abfFilePath", getattr(abf, "abfFilePath", "")),
        base_name=base,
    )
    return True, out_path

# ----------------- main loop -----------------
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        import pyabf
    except ImportError:
        raise SystemExit("pyabf not installed. Run: pip install pyabf")

    pairs = find_file_pairs(DATASET_ROOT)
    if not pairs:
        log("No ABF/table pairs found.")
        return

    total_saved = total_skipped = total_events = 0
    log(f"Found {len(pairs)} ABF/table pairs")

    for abf_path, tbl_path in pairs:
        base = os.path.splitext(os.path.basename(abf_path))[0]
        log(f"\n[Pair] {base}")
        try:
            df = load_table(tbl_path)
        except Exception as e:
            log(f"  ! table load failed: {e}")
            continue

        # open ABF & fs
        try:
            A = pyabf.ABF(abf_path)
            if hasattr(A, "dataRate") and A.dataRate:
                fs = float(A.dataRate)
            elif hasattr(A, "dataSecPerPoint"):
                fs = 1.0 / float(A.dataSecPerPoint)
            else:
                fs = 50000.0
                log("  ! fs fallback to 50kHz")
        except Exception as e:
            log(f"  ! ABF open failed: {e}")
            continue

        saved_here = skipped_here = 0
        for i, row in df.iterrows():
            total_events += 1
            ok, msg = save_npz_for_event(A, fs, base, abf_path, row, CACHE_DIR, i)
            if ok:
                saved_here += 1; total_saved += 1
            else:
                skipped_here += 1; total_skipped += 1
                if msg not in ("exists",):
                    log(f"    - skip window {i:04d}: {msg}")

        try: del A
        except Exception: pass

        log(f"  Saved: {saved_here} | Skipped: {skipped_here}")

    log("\n========== SUMMARY ==========")
    log(f"Pairs:          {len(pairs)}")
    log(f"Events seen:    {total_events}")
    log(f"NPZ saved:      {total_saved}")
    log(f"Windows skipped:{total_skipped}")
    log(f"Cache dir:      {CACHE_DIR}")
    log("=============================")

if __name__ == "__main__":
    main()
