# Window Tagger GUI

import os
os.environ["DASH_NO_JUPYTER"] = "1"

import datetime as _dt
import socket
import tkinter as tk
from tkinter import filedialog, simpledialog

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pyabf
import webbrowser
from dash import Dash, dcc, html, Input, Output, State
from dash import dash_table
from scipy import signal

# =============================================================================
# Constants
# =============================================================================
FILTER_ORDER = 8
FILTER_CUTOFF_HZ = 100.0  # matches UI text
WIN_SEC = 30.0
MIN_WIN_SEC = 0.1
OVERVIEW_FS = 1000.0
OPTICAL_FS = 10000.0
ELECTRICAL_FS = 10000.0

# =============================================================================
# File selection & helpers
# =============================================================================
def get_abf_file() -> str:
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Select ABF File", filetypes=[("ABF Files", "*.abf")])
    if not path:
        raise ValueError("No ABF file selected.")
    return path

def get_event_file() -> str | None:
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Event File (optional, CSV/XLSX)",
        filetypes=[("Event Files", "*.xlsx;*.csv"), ("Excel", "*.xlsx"), ("CSV", "*.csv")]
    )
    return path or None

def find_free_port(start: int = 8050, end: int = 9000) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found.")

def ask_role(prompt: str, default: str, keys: list[str]) -> str:
    root = tk.Tk(); root.withdraw()
    val = simpledialog.askstring(
        "Assign Channel Role",
        f"{prompt}\nType the channel name (case-sensitive) or leave blank for default [{default}]:"
    )
    val = (val.strip() if val else default)
    return val if val in keys else default

# =============================================================================
# Legacy <-> Internal schema
# =============================================================================
# INTERNAL SCHEMA: used by all plotting & app logic
base_columns = [
    "event_start", "event_plateau", "event_end",
    "optical_base", "optical_rise", "optical_end",
    "opticalre_base", "opticalre_rise", "opticalre_end",
    "entry_base", "entry_peak", "exit_peak", "exit_base",
    "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
    "event_id", "file_name", "sensor", "analytes", "solution", "notes",
    "window_start", "window_end"
]

# LEGACY VIEW / SAVE ORDER (exactly as you requested)
LEGACY_SAVE_ORDER = [
    "event_id", "file_name", "sensor", "analytes", "solution",
    "event_start (s)", "event_end (s)", "notes",
    "Base (V)", "Step (V)", "RefBase (V)", "RefStep (V)",
    "entry Base (pA)", "entry Peak (pA)", "exit Peak (pA)", "exit Base (pA)",
    "duration", "OSC", "RefOSC", "entry spike", "exit spike",
    "event_plateau",
    "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
    "window_start", "window_end"
]

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    for ch in [" ", "\t", "(", ")", ",", "[", "]"]:
        s = s.replace(ch, "")
    return s

def load_legacy_or_internal_events(path: str) -> pd.DataFrame:
    """Load legacy CSV/XLSX or an internal-format file and return INTERNAL columns."""
    if path.lower().endswith(".csv"):
        df_raw = pd.read_csv(path)
    else:
        df_raw = pd.read_excel(path)

    file_cols = {_norm(c): c for c in df_raw.columns}

    def get_series(*candidates, as_float=False):
        for cand in candidates:
            key = _norm(cand)
            if key in file_cols:
                s = df_raw[file_cols[key]]
                return pd.to_numeric(s, errors="coerce") if as_float else s
        return pd.Series([np.nan] * len(df_raw))

    out = pd.DataFrame(index=range(len(df_raw)))

    # Meta
    out["event_id"]  = get_series("event_id")
    out["file_name"] = get_series("file_name", "file name")
    out["sensor"]    = get_series("sensor")
    out["analytes"]  = get_series("analytes")
    out["solution"]  = get_series("solution")
    out["notes"]     = get_series("notes")

    # Times -> INTERNAL names
    out["event_start"]   = get_series("event_start (s)", "event_start_s", "event_start", as_float=True)
    out["event_end"]     = get_series("event_end (s)", "event_end_s", "event_end", as_float=True)
    out["event_plateau"] = get_series("event_plateau", as_float=True)

    # Optical/Ref amplitudes
    out["optical_base"]   = get_series("Base (V)", "base (v)", "optical_base", as_float=True)
    out["optical_rise"]   = get_series("Step (V)", "step (v)", "optical_rise", as_float=True)
    out["optical_end"]    = pd.Series([np.nan] * len(df_raw))
    out["opticalre_base"] = get_series("RefBase (V,)", "refbase (v,)", "opticalre_base", as_float=True)
    out["opticalre_rise"] = get_series("RefStep (V)", "refstep (v)", "opticalre_rise", as_float=True)
    out["opticalre_end"]  = pd.Series([np.nan] * len(df_raw))

    # Electrical amplitudes
    out["entry_base"] = get_series("entry Base (pA)", "entry base (pa)", "entry_base", as_float=True)
    out["entry_peak"] = get_series("entry Peak (pA)", "entry peak (pa)", "entry_peak", as_float=True)
    out["exit_peak"]  = get_series("exit Peak (pA)",  "exit peak (pa)",  "exit_peak",  as_float=True)
    out["exit_base"]  = get_series("exit Base (pA)",  "exit base (pa)",  "exit_base",  as_float=True)

    # Electrical times (optional)
    out["entry_base_t"] = get_series("entry_base_t", as_float=True)
    out["entry_peak_t"] = get_series("entry_peak_t", as_float=True)
    out["exit_peak_t"]  = get_series("exit_peak_t",  as_float=True)
    out["exit_base_t"]  = get_series("exit_base_t",  as_float=True)

    # Window columns
    out["window_start"] = get_series("window_start", as_float=True)
    out["window_end"]   = get_series("window_end", as_float=True)

    for c in base_columns:
        if c not in out.columns:
            out[c] = np.nan

    return out

def internal_to_legacy_saveframe(df_internal: pd.DataFrame, abf_file: str) -> pd.DataFrame:
    """Map INTERNAL dataframe to the requested LEGACY columns/order."""
    df = df_internal.copy()

    abf_name = os.path.basename(abf_file)
    if "file_name" not in df or df["file_name"].isna().all():
        df["file_name"] = abf_name
    else:
        df["file_name"] = df["file_name"].fillna(abf_name)

    if "event_id" not in df or df["event_id"].isna().all():
        df["event_id"] = np.arange(1, len(df) + 1)
    else:
        missing_mask = df["event_id"].isna()
        df.loc[missing_mask, "event_id"] = np.arange(1, missing_mask.sum() + 1)

    for meta in ["sensor", "analytes", "solution", "notes"]:
        if meta not in df:
            df[meta] = ""
        else:
            df[meta] = df[meta].fillna("")

    base = pd.to_numeric(df.get("optical_base", np.nan), errors="coerce")
    step = pd.to_numeric(df.get("optical_rise", np.nan), errors="coerce")
    refbase = pd.to_numeric(df.get("opticalre_base", np.nan), errors="coerce")
    refstep = pd.to_numeric(df.get("opticalre_rise", np.nan), errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        osc = ((step - base) / ((step + base) / 2.0)) * 100.0
        refosc = ((refstep - refbase) / ((refstep + refbase) / 2.0)) * 100.0

    duration = pd.to_numeric(df.get("event_end", np.nan), errors="coerce") - \
               pd.to_numeric(df.get("event_start", np.nan), errors="coerce")

    entry_spike = pd.to_numeric(df.get("entry_peak", np.nan), errors="coerce") - \
                  pd.to_numeric(df.get("entry_base", np.nan), errors="coerce")
    exit_spike = pd.to_numeric(df.get("exit_peak", np.nan), errors="coerce") - \
                 pd.to_numeric(df.get("exit_base", np.nan), errors="coerce")

    legacy = pd.DataFrame({
        "event_id": df["event_id"],
        "file_name": df["file_name"],
        "sensor": df["sensor"],
        "analytes": df["analytes"],
        "solution": df["solution"],
        "event_start (s)": df.get("event_start"),
        "event_end (s)": df.get("event_end"),
        "notes": df["notes"],
        "Base (V)": df.get("optical_base"),
        "Step (V)": df.get("optical_rise"),
        "RefBase (V)": df.get("opticalre_base"),
        "RefStep (V)": df.get("opticalre_rise"),
        "entry Base (pA)": df.get("entry_base"),
        "entry Peak (pA)": df.get("entry_peak"),
        "exit Peak (pA)": df.get("exit_peak"),
        "exit Base (pA)": df.get("exit_base"),
        "duration": duration,
        "OSC": osc,
        "RefOSC": refosc,
        "entry spike": entry_spike,
        "exit spike": exit_spike,
        "event_plateau": df.get("event_plateau"),
        "entry_base_t": df.get("entry_base_t"),
        "entry_peak_t": df.get("entry_peak_t"),
        "exit_peak_t": df.get("exit_peak_t"),
        "exit_base_t": df.get("exit_base_t"),
        "window_start": df.get("window_start"),
        "window_end": df.get("window_end"),
    })

    legacy = legacy.reindex(columns=LEGACY_SAVE_ORDER)
    return legacy

# =============================================================================
# Load ABF & preprocess
# =============================================================================
abf_file = get_abf_file()
abf = pyabf.ABF(abf_file)

abf_stem = os.path.splitext(abf_file)[0]
output_file = f"{abf_stem}_event.csv"

print("Available channels:")
for i in range(abf.channelCount):
    print(f"{i}: {abf.adcNames[i]}")

root = tk.Tk(); root.withdraw()
ch_input = simpledialog.askstring("Channels", "Enter up to 3 channel indices (comma-separated):")
channels = [int(ch.strip()) for ch in (ch_input or "").split(",")[:3]] if ch_input else []
if not channels:
    raise ValueError("No channels selected.")

abf.setSweep(0)
original_time = abf.sweepX.astype(float)
original_fs = float(abf.dataRate)

# Filter signals (8-pole Bessel @ 100 Hz); fallback to raw if filter fails
raw_signals: dict[str, np.ndarray] = {}
for ch in channels:
    abf.setSweep(0, channel=ch)
    raw = abf.sweepY.astype(float)
    try:
        wn = FILTER_CUTOFF_HZ / (original_fs / 2.0)
        sos = signal.bessel(FILTER_ORDER, wn, output="sos")
        filtered = signal.sosfilt(sos, raw)
    except Exception:
        filtered = raw
    raw_signals[abf.adcNames[ch]] = filtered

# Ensure exactly 3 keys for layout
while len(raw_signals) < 3:
    raw_signals[f"Blank-{len(raw_signals)+1}"] = np.full_like(original_time, np.nan)

signal_keys = list(raw_signals.keys())

def infer_keys(keys: list[str]) -> tuple[str, str, str]:
    optical = opticalre = electrical = None

    for k in keys:
        lk = k.lower()
        if optical is None and ("optical" in lk and "re" not in lk):
            optical = k
        if opticalre is None and ("opticalre" in lk or ("optical" in lk and "re" in lk)):
            opticalre = k

    for k in keys:
        if k not in (optical, opticalre) and electrical is None:
            electrical = k

    if optical is None:
        optical = ask_role("Which channel is OPTICAL (e.g., 'Optical')?", keys[0], keys)
    if opticalre is None:
        default_optre = keys[1] if len(keys) > 1 else keys[0]
        opticalre = ask_role("Which channel is OPTICAL REF (e.g., 'OpticalRe')?", default_optre, keys)
    if electrical is None:
        default_el = keys[2] if len(keys) > 2 else keys[-1]
        electrical = ask_role("Which channel is ELECTRICAL (e.g., 'Ipatch')?", default_el, keys)

    if opticalre == optical:
        opticalre = next((k for k in keys if k != optical), keys[0])
    if electrical in (optical, opticalre):
        electrical = next((k for k in keys if k not in (optical, opticalre)), keys[-1])

    return optical, opticalre, electrical

optical_key, opticalre_key, electrical_key = infer_keys(signal_keys)
display_keys = [optical_key, opticalre_key, electrical_key]
index_of = {k: i for i, k in enumerate(display_keys)}

# =============================================================================
# Sampling policies
# =============================================================================
# never show negative time
data_start = max(0.0, float(original_time[0]))
data_end = float(original_time[-1]) if float(original_time[-1]) > data_start else data_start + 0.1

def clamp_window_pair(x0: float, x1: float) -> tuple[float, float]:
    # Always cap to WIN_SEC (30 s) and never go below 0
    span = max(x1 - x0, MIN_WIN_SEC)
    span = min(span, WIN_SEC)
    x0 = max(x0, data_start)
    x1 = x0 + span
    if x1 > data_end:
        x1 = data_end
        x0 = max(data_start, x1 - span)
    # ensure non-negative time labels
    return round(max(x0, 0.0), 6), round(max(x1, 0.0), 6)

overview_fs = min(OVERVIEW_FS, original_fs)
overview_time = np.arange(data_start, data_end, 1.0 / overview_fs)
if overview_time.size == 0 or overview_time[-1] < data_end:
    overview_time = np.append(overview_time, data_end)

overview_signals: dict[str, np.ndarray] = {}
for k in display_keys:
    y = raw_signals.get(k, None)
    if y is None or not np.isfinite(y).any():
        overview_signals[k] = np.full_like(overview_time, np.nan)
    else:
        overview_signals[k] = np.interp(overview_time, original_time, y).astype(float)

def get_windowed_signal(signal_key: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_signals.get(signal_key)
    if raw is None or not np.isfinite(raw).any():
        return np.array([t0, t1]), np.array([np.nan, np.nan])
    target_fs = min(ELECTRICAL_FS, original_fs) if signal_key == electrical_key else min(OPTICAL_FS, original_fs)
    if t1 <= t0:
        t1 = t0 + MIN_WIN_SEC
    # never sample < 0
    t0 = max(t0, 0.0)
    t = np.arange(t0, t1, 1.0 / target_fs)
    if t.size == 0 or t[-1] < t1:
        t = np.append(t, t1)
    y = np.interp(t, original_time, raw)
    return t, y

# =============================================================================
# Events IO (legacy + internal)
# =============================================================================
event_file = get_event_file()
if event_file:
    try:
        intervals_df = load_legacy_or_internal_events(event_file)
    except Exception:
        intervals_df = pd.DataFrame(columns=base_columns)
else:
    intervals_df = pd.DataFrame(columns=base_columns)

for c in base_columns:
    if c not in intervals_df.columns:
        intervals_df[c] = np.nan

# Store (internal) and Table (legacy-view) records
initial_intervals_records = intervals_df[base_columns].to_dict("records")
# Note: abf_file is already defined, safe to use for the mapper
# (abf_file name is used only for default file_name fill when empty)
initial_table_records = internal_to_legacy_saveframe(intervals_df, abf_file).to_dict("records")

# =============================================================================
# Dash UI
# =============================================================================
app = Dash(__name__)
app.title = "ABF Annotator — Legacy CSV-Compatible"

# default window always starts within 30 s
default_start, default_end = clamp_window_pair(data_start, data_start + WIN_SEC)

POINT_LABELS = {
    "O1": "Event start", "O2": "Event plateau", "O3": "Event end",
    "E1": "Entry base",  "E2": "Entry peak",   "E3": "Exit peak", "E4": "Exit base"
}
def label_texts(short_labels: list[str], long: bool = False) -> list[str]:
    if not long:
        return short_labels
    return [f"{s} ({POINT_LABELS.get(s, '')})" for s in short_labels]

# -- Left controls for per-signal vertical zoom + moved X-zoom
left_controls = html.Div(
    [
        html.Div("Y Zoom", style={"fontWeight": "700", "marginBottom": "4px"}),
        html.Div([
            html.Div("Optical", style={"fontSize": "12px", "marginTop": "6px"}),
            html.Div([
                html.Button("＋", id="y1-plus", n_clicks=0, title="Optical: zoom in"),
                html.Button("－", id="y1-minus", n_clicks=0, style={"marginLeft": "6px"}, title="Optical: zoom out"),
                html.Button("Auto", id="y1-auto", n_clicks=0, style={"marginLeft": "6px"}, title="Optical: auto range"),
            ]),
            html.Div("OpticalR", style={"fontSize": "12px", "marginTop": "10px"}),
            html.Div([
                html.Button("＋", id="y2-plus", n_clicks=0, title="OpticalR: zoom in"),
                html.Button("－", id="y2-minus", n_clicks=0, style={"marginLeft": "6px"}, title="OpticalR: zoom out"),
                html.Button("Auto", id="y2-auto", n_clicks=0, style={"marginLeft": "6px"}, title="OpticalR: auto range"),
            ]),
            html.Div("Electrical", style={"fontSize": "12px", "marginTop": "10px"}),
            html.Div([
                html.Button("＋", id="y3-plus", n_clicks=0, title="Electrical: zoom in"),
                html.Button("－", id="y3-minus", n_clicks=0, style={"marginLeft": "6px"}, title="Electrical: zoom out"),
                html.Button("Auto", id="y3-auto", n_clicks=0, style={"marginLeft": "6px"}, title="Electrical: auto range"),
            ]),
        ]),
        html.Hr(style={"margin": "12px 0"}),
        html.Div("X Zoom", style={"fontWeight": 700, "marginBottom": "4px"}),
        html.Div([
            html.Button("X＋", id="x-in", n_clicks=0, title="Horizontal zoom in (shrink window from left edge)"),
            html.Button("X－", id="x-out", n_clicks=0, style={"marginLeft": "6px"}, title="Horizontal zoom out (expand window from left edge)"),
            html.Button("X Auto", id="x-auto", n_clicks=0, style={"marginLeft": "6px"}, title="Reset horizontal window"),
        ]),
    ],
    style={"flex": "0 0 160px", "padding": "6px 8px", "borderRight": "1px solid #eee"}
)

app.layout = html.Div([
    html.Div([
        html.Div(
            f"File: {os.path.basename(abf_file)} | Optical: {optical_key} | OpticalRe: {opticalre_key} | Electrical: {electrical_key}",
            style={"fontWeight": "bold", "marginBottom": "6px"}
        ),
        html.Div(
            ("Overview: %.0f Hz | Window: %.1f–%.0fs | Optical in window: %.0f Hz | "
             "Electrical in window: %.0f kHz | Filter: %d-pole Bessel %.0f Hz | "
             "Save → %s (legacy CSV)")
            % (OVERVIEW_FS, MIN_WIN_SEC, WIN_SEC, OPTICAL_FS, ELECTRICAL_FS/1000, FILTER_ORDER, FILTER_CUTOFF_HZ, os.path.basename(output_file)),
            style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"}
        ),
    ], style={"margin": "10px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id="interaction-mode",
                options=[
                    {"label": "Zoom/Inspect", "value": "zoom"},
                    {"label": "Add Event (3 Optical → 4 Electrical)", "value": "add"},
                    {"label": "Add Window (drag rectangle)", "value": "window"},
                    {"label": "Delete Events (drag rectangle on Optical)", "value": "delete"},
                    {"label": "Edit Points (select row then drag guides)", "value": "edit"}
                ],
                value="zoom",
                style={"width": "460px"}
            ),
            html.Div(
                "Add: click 3 points on Optical/OpticalRe (start, plateau, end), then 4 points on Electrical (entry base, entry peak, exit peak, exit base). "
                "Window: drag a rectangle to save only window_start and window_end times. "
                "Delete: drag a rectangle on Optical to remove events in that x-range. "
                "Edit: select a row in the table to show vertical guides; drag to adjust times.",
                style={"fontSize": "12px", "color": "#444", "marginTop": "6px"}
            ),
            dcc.Checklist(
                id="long-labels",
                options=[{"label": " Show long labels (e.g., O1 → Event start)", "value": "on"}],
                value=[],
                style={"marginTop": "6px"}
            ),
            dcc.Checklist(
                id="fit-mode",
                options=[{"label": " Fit to screen (auto-Y on pan/zoom)", "value": "on"}],
                value=[],
                style={"marginTop": "6px"}
            ),
        ], style={"flex": "1 1 auto", "minWidth": "520px"}),

        html.Div([
            html.Button("Toggle Side Panel (T)", id="toggle-side", n_clicks=0),
        ], style={"flex": "0 0 auto", "paddingLeft": "16px"})
    ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start", "margin": "0 10px 10px 10px"}),

    html.Div([
        left_controls,
        html.Div([
            dcc.Graph(
                id="main-plot",
                style={"height": "85vh"},
                config={
                    "scrollZoom": True,
                    "doubleClick": "reset",
                    "displaylogo": False,
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    "edits": {"shapePosition": True},
                    "responsive": True
                }
            ),
            html.Div([
                html.Button("Save Reviewed Events", id="save-btn", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("Undo Last Point", id="undo-btn", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("Discard Pending Event", id="discard-btn", n_clicks=0),
            ], style={"margin": "10px 10px 0 10px"}),
            html.Div(id="msg", style={"margin": "8px 10px", "color": "green"}),
        ], style={"flex": "1 1 auto", "minWidth": "700px"}),

        html.Div([
            html.Div("Events (click a row to navigate; Delete to remove):", style={"fontWeight": "600", "margin": "6px 0"}),
            dash_table.DataTable(
                id="events-table",
                columns=[{"name": c, "id": c} for c in LEGACY_SAVE_ORDER],
                data=initial_table_records,
                editable=False,
                row_selectable="single",
                page_size=25,  # show > 10
                sort_action="native",
                filter_action="native",
                style_table={"height": "520px", "overflowY": "auto", "minWidth": "420px"},
                style_cell={"fontSize": 12, "padding": "6px", "textAlign": "left", "minWidth": "80px", "maxWidth": "220px",
                            "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
                style_header={"fontWeight": "600"}
            ),
            html.Div([
                html.Button("Delete Selected", id="delete-row", n_clicks=0, style={"marginRight": "10px"}),
                dcc.Input(id="export-name", placeholder="export file name (no extension)",
                          style={"width": "260px", "marginRight": "10px"}),
                html.Button("Export CSV", id="export-csv", n_clicks=0),
                dcc.Download(id="download-csv")
            ], style={"marginTop": "8px"}),
            html.Div(id="table-msg", style={"marginTop": "6px", "fontSize": "12px", "color": "#444"})
        ], id="side-panel", style={"flex": "0 0 520px", "paddingLeft": "14px", "borderLeft": "1px solid #eee"})
    ], style={"display": "flex", "gap": "12px", "alignItems": "stretch", "margin": "0 10px"}),

    # Stores
    dcc.Store(id="intervals-store", data=initial_intervals_records),  # INTERNAL rows
    dcc.Store(id="click-state", data={
        "stage": "idle",
        "optical_times": [],
        "optical_vals": [],
        "electrical_times": [],
        "electrical_vals": [],
        "pending_event_id": 0
    }),
    dcc.Store(id="current-mode", data="zoom"),
    dcc.Store(id="current-window", data=[default_start, default_end]),
    dcc.Store(id="stored-zoom", data={"y1": None, "y2": None, "y3": None}),
    dcc.Store(id="selected-row-idx", data=None),
    dcc.Store(id="side-visible", data=True),
])

# =============================================================================
# Figure builder
# =============================================================================
def _has_plateau(row: dict) -> bool:
    return pd.notna(row.get("event_plateau"))

def _has_all_electrical_times(row: dict) -> bool:
    return all(pd.notna(row.get(k)) for k in ["entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t"])

def _collect_opt_times_and_labels(row: dict) -> tuple[list[float], list[str]]:
    times, labels = [], []
    t1 = row.get("event_start")
    tp = row.get("event_plateau")
    t3 = row.get("event_end")
    if pd.notna(t1):
        times.append(t1); labels.append("O1")
    if pd.notna(tp):
        times.append(tp); labels.append("O2")
    if pd.notna(t3):
        times.append(t3); labels.append("O3")
    return times, labels

def _yrange_from_data(y: np.ndarray) -> list[float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size:
        q1, q99 = np.percentile(y, [1, 99])
        margin = (q99 - q1) * 0.2 if (q99 - q1) > 0 else 1.0
        return [q1 - margin, q99 + margin]
    return [-1, 1]

def _event_center(row: dict) -> float | None:
    t_candidates = [row.get("event_start"), row.get("event_plateau"), row.get("event_end"),
                    row.get("entry_base_t"), row.get("entry_peak_t"), row.get("exit_peak_t"), row.get("exit_base_t")]
    t_candidates = [t for t in t_candidates if pd.notna(t)]
    if not t_candidates:
        return None
    return float(np.mean([min(t_candidates), max(t_candidates)]))

def _guide_shapes_for_event(row: dict, y_ranges: list[list[float]]) -> list[dict]:
    shapes = []
    # Optical & OpticalR guides (dot)
    for key in ["event_start", "event_plateau", "event_end"]:
        t = row.get(key)
        if pd.notna(t):
            shapes.append(dict(type="line", x0=max(t, 0.0), x1=max(t, 0.0), y0=y_ranges[0][0], y1=y_ranges[0][1], xref="x", yref="y",
                               line=dict(width=1.5, dash="dot")))
            shapes.append(dict(type="line", x0=max(t, 0.0), x1=max(t, 0.0), y0=y_ranges[1][0], y1=y_ranges[1][1], xref="x", yref="y2",
                               line=dict(width=1.5, dash="dot")))
    # Electrical guides (dash)
    for key in ["entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t"]:
        t = row.get(key)
        if pd.notna(t):
            shapes.append(dict(type="line", x0=max(t, 0.0), x1=max(t, 0.0), y0=y_ranges[2][0], y1=y_ranges[2][1], xref="x", yref="y3",
                               line=dict(width=1.5, dash="dash")))
    return shapes

def _add_mode_text(click_state) -> str:
    click_state = click_state or {}
    stage = click_state.get("stage", "optical")
    n_opt = len(click_state.get("optical_times", []) or [])
    n_el  = len(click_state.get("electrical_times", []) or [])
    if stage == "optical":
        prompts = ["Add Optical Start (O1)", "Add Optical Plateau (O2)", "Add Optical End (O3)"]
        idx = min(n_opt, 2)
        return prompts[idx] + f" — {n_opt}/3"
    else:
        prompts = ["Add Entry Base (E1)", "Add Entry Peak (E2)", "Add Exit Peak (E3)", "Add Exit Base (E4)"]
        idx = min(n_el, 3)
        return prompts[idx] + f" — {n_el}/4"

def build_figure(intervals_df: pd.DataFrame, window: list[float], click_state=None,
                 zoom=None, mode="zoom", long_labels=False, selected_row=None) -> go.Figure:
    zoom = zoom or {"y1": None, "y2": None, "y3": None}
    w0, w1 = clamp_window_pair(window[0], window[1])

    fig = go.Figure()

    # Overview traces (downsampled)
    for i, k in enumerate(display_keys):
        fig.add_trace(go.Scatter(
            x=overview_time, y=overview_signals[k],
            name=f"{k} (overview)",
            yaxis=f"y{i+1}",
            mode="lines",
            line=dict(width=1),
            hoverinfo="skip",
            showlegend=False
        ))

    # High-res window traces
    detail_ys = []
    for i, k in enumerate(display_keys):
        tx, ty = get_windowed_signal(k, w0, w1)
        detail_ys.append(ty)
        fig.add_trace(go.Scatter(
            x=tx, y=ty,
            name=f"{k} (detail)",
            yaxis=f"y{i+1}",
            mode="lines",
            line=dict(width=1.2),
            hoverinfo="x+y+name",
            showlegend=False
        ))

    y_ranges = [
        zoom.get("y1") if zoom.get("y1") else _yrange_from_data(detail_ys[0]),
        zoom.get("y2") if zoom.get("y2") else _yrange_from_data(detail_ys[1]),
        zoom.get("y3") if zoom.get("y3") else _yrange_from_data(detail_ys[2]),
    ]

    # Pending-click markers (in-progress)
    if click_state:
        # Optical/OpticalRef (O1..O3)
        opt_ts = click_state.get("optical_times", []) or []
        if opt_ts:
            oi = index_of.get(optical_key, 0)
            ori = index_of.get(opticalre_key, 1)
            y_opt = [float(np.interp(t, original_time, raw_signals[optical_key])) for t in opt_ts]
            y_ref = [float(np.interp(t, original_time, raw_signals[opticalre_key])) for t in opt_ts]
            fig.add_trace(go.Scatter(
                x=[max(t, 0.0) for t in opt_ts], y=y_opt, yaxis=f"y{oi+1}",
                mode="markers+text",
                text=label_texts([f"O{i+1}" for i in range(len(opt_ts))], long_labels),
                textposition="top center",
                marker=dict(size=9, symbol="circle-open"),
                showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=[max(t, 0.0) for t in opt_ts], y=y_ref, yaxis=f"y{ori+1}",
                mode="markers+text",
                text=label_texts([f"R{i+1}" for i in range(len(opt_ts))], long_labels),
                textposition="top center",
                marker=dict(size=9, symbol="diamond-open"),
                showlegend=False, hoverinfo="skip"
            ))

        # Electrical (E1..E4)
        elec_ts = click_state.get("electrical_times", []) or []
        if elec_ts:
            ei = index_of.get(electrical_key, 2)
            y_elec = [float(np.interp(t, original_time, raw_signals[electrical_key])) for t in elec_ts]
            fig.add_trace(go.Scatter(
                x=[max(t, 0.0) for t in elec_ts], y=y_elec, yaxis=f"y{ei+1}",
                mode="markers+text",
                text=label_texts([f"E{i+1}" for i in range(len(elec_ts))], long_labels),
                textposition="bottom center",
                marker=dict(size=9, symbol="x-open"),
                showlegend=False, hoverinfo="skip"
            ))

    # Overview markers (all intervals)
    if len(intervals_df):
        for _, row in intervals_df.iterrows():
            times_all, labels_all = _collect_opt_times_and_labels(row)
            if not _has_plateau(row):
                keep = [lbl in ("O1", "O3") for lbl in labels_all]
                times_all = [t for t, k in zip(times_all, keep) if k]
                labels_all = [lbl for lbl, k in zip(labels_all, keep) if k]
            if times_all:
                oi = index_of.get(optical_key, 0)
                ori = index_of.get(opticalre_key, 1)
                opt_vals = [float(np.interp(t, overview_time, overview_signals[optical_key])) for t in times_all]
                ref_vals = [float(np.interp(t, overview_time, overview_signals[opticalre_key])) for t in times_all]
                fig.add_trace(go.Scatter(x=[max(t, 0.0) for t in times_all], y=opt_vals, yaxis=f"y{oi+1}",
                                         mode="markers", marker=dict(size=6, symbol="circle"),
                                         showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[max(t, 0.0) for t in times_all], y=ref_vals, yaxis=f"y{ori+1}",
                                         mode="markers", marker=dict(size=6, symbol="diamond"),
                                         showlegend=False, hoverinfo="skip"))

        # Window rectangles in overview
        for idx, row in intervals_df.iterrows():
            ws = row.get("window_start")
            we = row.get("window_end")
            if pd.notna(ws) and pd.notna(we):
                fig.add_vrect(x0=max(ws, 0.0), x1=max(we, 0.0), fillcolor="purple", opacity=0.15, 
                              layer="below", line_width=0, row="all", col="all")

        for _, row in intervals_df.iterrows():
            if _has_all_electrical_times(row):
                times = [row.get("entry_base_t"), row.get("entry_peak_t"), row.get("exit_peak_t"), row.get("exit_base_t")]
                times = [t for t in times if pd.notna(t)]
                if times:
                    ei = index_of.get(electrical_key, 2)
                    vals = [float(np.interp(t, overview_time, overview_signals[electrical_key])) for t in times]
                    fig.add_trace(go.Scatter(x=[max(t, 0.0) for t in times], y=vals, yaxis=f"y{ei+1}",
                                             mode="markers", marker=dict(size=6, symbol="x"),
                                             showlegend=False, hoverinfo="skip"))

    # Detailed markers in current window
    if len(intervals_df):
        for _, row in intervals_df.iterrows():
            times_all, labels_all = _collect_opt_times_and_labels(row)
            if not _has_plateau(row):
                keep = [lbl in ("O1", "O3") for lbl in labels_all]
                times_all = [t for t, k in zip(times_all, keep) if k]
                labels_all = [lbl for lbl, k in zip(labels_all, keep) if k]
            mask = [pd.notna(t) and (w0 <= t <= w1) for t in times_all]
            times_in = [t for t, m in zip(times_all, mask) if m]
            labels_in = [lbl for lbl, m in zip(labels_all, mask) if m]
            if times_in:
                oi = index_of.get(optical_key, 0)
                ori = index_of.get(opticalre_key, 1)
                y_opt = [float(np.interp(t, original_time, raw_signals[optical_key])) for t in times_in]
                y_ref = [float(np.interp(t, original_time, raw_signals[opticalre_key])) for t in times_in]
                fig.add_trace(go.Scatter(x=[max(t, 0.0) for t in times_in], y=y_opt, yaxis=f"y{oi+1}",
                                         mode="markers+text",
                                         text=label_texts(labels_in, long_labels),
                                         textposition="top center",
                                         marker=dict(size=8, symbol="circle"),
                                         showlegend=False, hoverinfo="skip"))
                ref_labels = ["R" + lbl[1:] for lbl in labels_in]
                fig.add_trace(go.Scatter(x=[max(t, 0.0) for t in times_in], y=y_ref, yaxis=f"y{ori+1}",
                                         mode="markers+text",
                                         text=label_texts(ref_labels, long_labels),
                                         textposition="top center",
                                         marker=dict(size=8, symbol="diamond"),
                                         showlegend=False, hoverinfo="skip"))

        for _, row in intervals_df.iterrows():
            if _has_all_electrical_times(row):
                ts = [row.get("entry_base_t"), row.get("entry_peak_t"), row.get("exit_peak_t"), row.get("exit_base_t")]
                labs = ["E1", "E2", "E3", "E4"]
                mask = [pd.notna(t) and (w0 <= t <= w1) for t in ts]
                if any(mask):
                    ei = index_of.get(electrical_key, 2)
                    ts2 = [t for t, m in zip(ts, mask) if m]
                    ys = [float(np.interp(t, original_time, raw_signals[electrical_key])) for t in ts2]
                    txt = [lab for lab, m in zip(labs, mask) if m]
                    fig.add_trace(go.Scatter(x=[max(t, 0.0) for t in ts2], y=ys, yaxis=f"y{ei+1}",
                                             mode="markers+text",
                                             text=label_texts(txt, long_labels),
                                             textposition="bottom center",
                                             marker=dict(size=8, symbol="x"),
                                             showlegend=False, hoverinfo="skip"))

    dragmode = "select" if mode == "delete" else "select" if mode == "window" else "zoom"
    fig.update_layout(
        template="plotly_white",
        dragmode=dragmode,
        selectdirection="h",
        xaxis=dict(
            domain=[0, 1],
            range=[w0, w1],
            rangeslider=dict(visible=True, thickness=0.12),
            title=f"Time (s) — Window {max(w0,0.0):.3f} to {max(w1,0.0):.3f}"
        ),
        yaxis=dict(title=f"{display_keys[0]} (optical)",   domain=[0.70, 1.00], fixedrange=False, range=y_ranges[0]),
        yaxis2=dict(title=f"{display_keys[1]} (opticalR)", domain=[0.35, 0.65], fixedrange=False, range=y_ranges[1]),
        yaxis3=dict(title=f"{display_keys[2]} (electrical)", domain=[0.00, 0.30], fixedrange=False, range=y_ranges[2]),
        margin=dict(t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        uirevision="keep"
    )

    # Add-mode step text INSIDE the figure: just below electrical, above range slider
    if mode == "add":
        fig.update_layout(annotations=[dict(
            text=_add_mode_text(click_state),
            x=0.5, xref="paper",
            y=0.03, yref="paper",
            showarrow=False,
            align="center",
            font=dict(size=12, color="#1f3a8a"),
            bgcolor="#eef2ff",
            bordercolor="#c7d2fe",
            borderwidth=1,
            borderpad=4
        )])

    # Draggable guide shapes in Edit mode
    if mode == "edit" and selected_row is not None and 0 <= selected_row < len(intervals_df):
        row = intervals_df.iloc[selected_row].to_dict()
        fig.update_layout(shapes=_guide_shapes_for_event(row, y_ranges))

    return fig

# =============================================================================
# Side panel toggle callbacks
# =============================================================================
@app.callback(
    Output("side-visible", "data"),
    Input("toggle-side", "n_clicks"),
    State("side-visible", "data"),
    prevent_initial_call=True
)
def toggle_side_panel(n, visible):
    return not bool(visible)

@app.callback(
    Output("side-panel", "style"),
    Input("side-visible", "data")
)
def apply_side_visibility(visible):
    base = {"flex": "0 0 520px", "paddingLeft": "14px", "borderLeft": "1px solid #eee"}
    if not visible:
        base["display"] = "none"
    return base

# =============================================================================
# Mode & interactions
# =============================================================================
@app.callback(Output("current-mode", "data"), Input("interaction-mode", "value"))
def set_mode(mode):
    return mode

@app.callback(
    Output("main-plot", "figure"),
    Output("current-window", "data"),
    Output("stored-zoom", "data"),
    Output("intervals-store", "data", allow_duplicate=True),
    Output("msg", "children", allow_duplicate=True),
    Input("main-plot", "relayoutData"),
    State("intervals-store", "data"),
    State("click-state", "data"),
    State("current-window", "data"),
    State("stored-zoom", "data"),
    State("current-mode", "data"),
    State("long-labels", "value"),
    State("selected-row-idx", "data"),
    State("fit-mode", "value"),
    prevent_initial_call=True
)
def handle_plot_interaction(relayoutData, intervals_records, click_state, current_window, stored_zoom, mode, long_labels, selected_row_idx, fit_vals):
    if not current_window or len(current_window) != 2:
        current_window = [default_start, default_end]
    w0, w1 = current_window
    zoom = stored_zoom or {"y1": None, "y2": None, "y3": None}
    msg = dash.no_update
    fit_on = ("on" in (fit_vals or []))
    x_changed = False  # track horizontal pan/zoom

    def set_window(x0, x1):
        nonlocal w0, w1, x_changed
        w0, w1 = clamp_window_pair(float(x0), float(x1))  # always cap to 30s
        x_changed = True

    # Delete via selection box
    if mode == "delete" and relayoutData and "selections" in relayoutData:
        selections = relayoutData["selections"]
        if selections:
            sel = selections[-1]
            if "x0" in sel and "x1" in sel:
                sel_start = min(float(sel["x0"]), float(sel["x1"]))
                sel_end = max(float(sel["x0"]), float(sel["x1"]))
                intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
                original = len(intervals_df)

                def overlaps(row):
                    times = [row.get("event_start"), row.get("event_plateau"), row.get("event_end"),
                             row.get("entry_base_t"), row.get("entry_peak_t"),
                             row.get("exit_peak_t"), row.get("exit_base_t"),
                             row.get("window_start"), row.get("window_end")]
                    times = [t for t in times if pd.notna(t)]
                    return any(sel_start <= t <= sel_end for t in times)

                mask = ~intervals_df.apply(overlaps, axis=1)
                intervals_df = intervals_df[mask].reset_index(drop=True)
                removed = original - len(intervals_df)
                msg = f"Removed {removed} event(s) from {sel_start:.3f}s–{sel_end:.3f}s" if removed else "No events found in selected range"
                intervals_records = intervals_df.to_dict("records")

    # Add window via selection box
    if mode == "window" and relayoutData and "selections" in relayoutData:
        selections = relayoutData["selections"]
        if selections:
            sel = selections[-1]
            if "x0" in sel and "x1" in sel:
                ws = min(float(sel["x0"]), float(sel["x1"]))
                we = max(float(sel["x0"]), float(sel["x1"]))
                
                # Create new window entry
                new_row = {col: np.nan for col in base_columns}
                new_row["window_start"] = ws
                new_row["window_end"] = we
                intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
                new_row["event_id"] = len(intervals_df) + 1
                
                intervals_df = pd.concat([intervals_df, pd.DataFrame([new_row])], ignore_index=True)
                msg = f"Window added: {ws:.3f}s to {we:.3f}s"
                intervals_records = intervals_df.to_dict("records")

    # Zoom/pan & y-axis zooms
    if relayoutData:
        if "xaxis.range[0]" in relayoutData and "xaxis.range[1]" in relayoutData:
            set_window(relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"])
        elif "xaxis.range" in relayoutData and isinstance(relayoutData["xaxis.range"], (list, tuple)) and len(relayoutData["xaxis.range"]) == 2:
            set_window(relayoutData["xaxis.range"][0], relayoutData["xaxis.range"][1])
        elif "xaxis.autorange" in relayoutData:
            w0, w1 = default_start, default_end
            x_changed = True

        if "yaxis.range[0]" in relayoutData and "yaxis.range[1]" in relayoutData:
            zoom["y1"] = [relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"]]
        if "yaxis2.range[0]" in relayoutData and "yaxis2.range[1]" in relayoutData:
            zoom["y2"] = [relayoutData["yaxis2.range[0]"], relayoutData["yaxis2.range[1]"]]
        if "yaxis3.range[0]" in relayoutData and "yaxis3.range[1]" in relayoutData:
            zoom["y3"] = [relayoutData["yaxis3.range[0]"], relayoutData["yaxis3.range[1]"]]
        if "yaxis.autorange" in relayoutData:
            zoom["y1"] = None
        if "yaxis2.autorange" in relayoutData:
            zoom["y2"] = None
        if "yaxis3.autorange" in relayoutData:
            zoom["y3"] = None

        # Shape drags (Edit mode)
        if mode == "edit":
            keys = [k for k in relayoutData.keys() if k.startswith("shapes[") and (k.endswith(".x0") or k.endswith(".x1"))]
            if keys and selected_row_idx is not None:
                shape_x = {}
                for k in keys:
                    idx = int(k.split("[")[1].split("]")[0])
                    shape_x.setdefault(idx, {})
                    if k.endswith(".x0"):
                        shape_x[idx]["x0"] = float(relayoutData[k])
                    elif k.endswith(".x1"):
                        shape_x[idx]["x1"] = float(relayoutData[k])

                def _x_of(s):
                    if "x0" in s and "x1" in s:
                        return float((s["x0"] + s["x1"]) / 2.0)
                    return float(s.get("x0") or s.get("x1"))

                intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
                if 0 <= selected_row_idx < len(intervals_df):
                    row = intervals_df.iloc[selected_row_idx].to_dict()
                    shape_keys = []
                    for key in ["event_start", "event_plateau", "event_end"]:
                        if pd.notna(row.get(key)):
                            shape_keys += [key, key]
                    for key in ["entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t"]:
                        if pd.notna(row.get(key)):
                            shape_keys += [key]
                    for s_idx, coord in shape_x.items():
                        if 0 <= s_idx < len(shape_keys):
                            kname = shape_keys[s_idx]
                            new_t = _x_of(coord)
                            intervals_df.at[selected_row_idx, kname] = new_t
                            if kname in ("event_start", "event_plateau", "event_end"):
                                v_opt = float(np.interp(new_t, original_time, raw_signals[optical_key]))
                                v_ref = float(np.interp(new_t, original_time, raw_signals[opticalre_key]))
                                if kname == "event_start":
                                    intervals_df.at[selected_row_idx, "optical_base"] = v_opt
                                    intervals_df.at[selected_row_idx, "opticalre_base"] = v_ref
                                elif kname == "event_plateau":
                                    intervals_df.at[selected_row_idx, "optical_rise"] = v_opt
                                    intervals_df.at[selected_row_idx, "opticalre_rise"] = v_ref
                                elif kname == "event_end":
                                    intervals_df.at[selected_row_idx, "optical_end"] = v_opt
                                    intervals_df.at[selected_row_idx, "opticalre_end"] = v_ref
                            else:
                                v_el = float(np.interp(new_t, original_time, raw_signals[electrical_key]))
                                if kname == "entry_base_t": intervals_df.at[selected_row_idx, "entry_base"] = v_el
                                if kname == "entry_peak_t": intervals_df.at[selected_row_idx, "entry_peak"] = v_el
                                if kname == "exit_peak_t":  intervals_df.at[selected_row_idx, "exit_peak"]  = v_el
                                if kname == "exit_base_t":  intervals_df.at[selected_row_idx, "exit_base"]  = v_el
                    intervals_records = intervals_df.to_dict("records")
                    msg = "Updated event via guides."

    # If fit-mode is ON and X changed, clear Y zooms to recompute ranges
    if fit_on and x_changed:
        zoom = {"y1": None, "y2": None, "y3": None}

    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    long = ("on" in (long_labels or []))
    fig = build_figure(intervals_df, [w0, w1], click_state, zoom, mode, long_labels=long, selected_row=selected_row_idx)
    return fig, [w0, w1], zoom, intervals_records, msg

# -- Per-signal vertical zoom buttons
def _change_y_range(cur, factor=None, auto=False, data_y=None):
    if auto:
        return None
    if cur is None:
        return _yrange_from_data(data_y if data_y is not None else np.array([]))
    y0, y1 = float(cur[0]), float(cur[1])
    c = (y0 + y1) / 2.0
    h = (y1 - y0) / 2.0
    h = max(h * factor, 1e-9)
    return [c - h, c + h]

@app.callback(
    Output("stored-zoom", "data", allow_duplicate=True),
    Output("main-plot", "figure", allow_duplicate=True),
    Input("y1-plus", "n_clicks"), Input("y1-minus", "n_clicks"), Input("y1-auto", "n_clicks"),
    Input("y2-plus", "n_clicks"), Input("y2-minus", "n_clicks"), Input("y2-auto", "n_clicks"),
    Input("y3-plus", "n_clicks"), Input("y3-minus", "n_clicks"), Input("y3-auto", "n_clicks"),
    State("stored-zoom", "data"),
    State("current-window", "data"),
    State("intervals-store", "data"),
    State("current-mode", "data"),
    State("long-labels", "value"),
    State("selected-row-idx", "data"),
    prevent_initial_call=True
)
def per_signal_yzoom(y1p, y1m, y1a, y2p, y2m, y2a, y3p, y3m, y3a,
                     stored_zoom, current_window, intervals_records, mode, long_labels, selected_row_idx):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    zoom = stored_zoom or {"y1": None, "y2": None, "y3": None}
    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)

    # Map trigger to axis & action
    if trig in ("y1-plus", "y2-plus", "y3-plus"):
        action = ("factor", 0.8)
    elif trig in ("y1-minus", "y2-minus", "y3-minus"):
        action = ("factor", 1.25)
    else:
        action = ("auto", True)  # any of the *-auto buttons

    axis = "y1" if trig.startswith("y1") else "y2" if trig.startswith("y2") else "y3"

    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    _, y1d = get_windowed_signal(optical_key,  w0, w1)
    _, y2d = get_windowed_signal(opticalre_key, w0, w1)
    _, y3d = get_windowed_signal(electrical_key, w0, w1)

    if axis == "y1":
        zoom["y1"] = _change_y_range(zoom.get("y1"), factor=action[1] if action[0] == "factor" else None,
                                     auto=(action[0] == "auto"), data_y=y1d)
    elif axis == "y2":
        zoom["y2"] = _change_y_range(zoom.get("y2"), factor=action[1] if action[0] == "factor" else None,
                                     auto=(action[0] == "auto"), data_y=y2d)
    else:
        zoom["y3"] = _change_y_range(zoom.get("y3"), factor=action[1] if action[0] == "factor" else None,
                                     auto=(action[0] == "auto"), data_y=y3d)

    long = ("on" in (long_labels or []))
    fig = build_figure(intervals_df, [w0, w1], None, zoom, mode, long_labels=long, selected_row=selected_row_idx)
    return zoom, fig

# -- Horizontal zoom buttons (anchored to left edge)
@app.callback(
    Output("current-window", "data", allow_duplicate=True),
    Output("main-plot", "figure", allow_duplicate=True),
    Input("x-in", "n_clicks"), Input("x-out", "n_clicks"), Input("x-auto", "n_clicks"),
    State("current-window", "data"),
    State("stored-zoom", "data"),
    State("intervals-store", "data"),
    State("current-mode", "data"),
    State("long-labels", "value"),
    State("fit-mode", "value"),
    State("selected-row-idx", "data"),
    prevent_initial_call=True
)
def horizontal_zoom(n_in, n_out, n_auto, current_window, stored_zoom, intervals_records, mode, long_labels, fit_vals, selected_row_idx):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)
    span = max(w1 - w0, MIN_WIN_SEC)

    if trig == "x-in":
        new_span = max(span * 0.8, MIN_WIN_SEC)
        new_w0 = w0  # anchor at left edge
        new_w1 = new_w0 + new_span
        if new_w1 > data_end:
            new_w1 = data_end
            new_w0 = max(data_start, new_w1 - new_span)
        new_w0, new_w1 = clamp_window_pair(new_w0, new_w1)
    elif trig == "x-out":
        new_span = min(span * 1.25, WIN_SEC)
        new_w0 = w0  # keep left edge fixed
        new_w1 = new_w0 + new_span
        if new_w1 > data_end:
            new_w1 = data_end
            new_w0 = max(data_start, new_w1 - new_span)
        new_w0, new_w1 = clamp_window_pair(new_w0, new_w1)
    else:  # x-auto
        new_w0, new_w1 = default_start, default_end

    long = ("on" in (long_labels or []))
    fit_on = ("on" in (fit_vals or []))
    zoom = stored_zoom if not fit_on else {"y1": None, "y2": None, "y3": None}

    fig = build_figure(intervals_df, [new_w0, new_w1], None, zoom, mode, long_labels=long, selected_row=selected_row_idx)
    return [new_w0, new_w1], fig

# -- Mode switching initializes click-state, and picks a row for edit mode
@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Output("click-state", "data", allow_duplicate=True),
    Output("selected-row-idx", "data", allow_duplicate=True),
    Output("events-table", "selected_rows", allow_duplicate=True),
    Input("interaction-mode", "value"),
    State("click-state", "data"),
    State("current-window", "data"),
    State("intervals-store", "data"),
    State("stored-zoom", "data"),
    State("long-labels", "value"),
    State("selected-row-idx", "data"),
    prevent_initial_call=True
)
def update_mode(mode, click_state, current_window, intervals_records, stored_zoom, long_labels, selected_row_idx):
    if mode == "add":
        click_state = {"stage": "optical",
                       "optical_times": [], "optical_vals": [],
                       "electrical_times": [], "electrical_vals": [],
                       "pending_event_id": len(intervals_records or [])}
    else:
        click_state = {"stage": "idle",
                       "optical_times": [], "optical_vals": [],
                       "electrical_times": [], "electrical_vals": [],
                       "pending_event_id": -1}

    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)

    if mode == "edit" and (selected_row_idx is None or not (0 <= selected_row_idx < len(intervals_df))):
        if len(intervals_df):
            center = (w0 + w1) / 2.0

            def row_center(r):
                ts = [r.get(k) for k in ["event_start", "event_plateau", "event_end",
                                         "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t"]]
                ts = [t for t in ts if pd.notna(t)]
                if not ts:
                    return np.inf
                return float(np.mean([min(ts), max(ts)]))

            distances = intervals_df.apply(lambda r: abs(row_center(r) - center), axis=1)
            selected_row_idx = int(distances.idxmin())
        else:
            selected_row_idx = None

    long = ("on" in (long_labels or []))
    fig = build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx)
    selected_rows_prop = [selected_row_idx] if selected_row_idx is not None else []
    return fig, click_state, selected_row_idx, selected_rows_prop

# -- Handle clicks to add new events (Add mode)
@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Output("click-state", "data"),
    Output("intervals-store", "data"),
    Output("msg", "children", allow_duplicate=True),
    Input("main-plot", "clickData"),
    State("click-state", "data"),
    State("current-window", "data"),
    State("intervals-store", "data"),
    State("current-mode", "data"),
    State("stored-zoom", "data"),
    State("long-labels", "value"),
    State("selected-row-idx", "data"),
    prevent_initial_call=True
)
def handle_click(clickData, click_state, current_window, intervals_records, mode, stored_zoom, long_labels, selected_row_idx):
    if mode != "add" or not clickData:
        return dash.no_update, click_state, intervals_records, dash.no_update

    intervals_records = intervals_records or []
    intervals_df = pd.DataFrame(intervals_records, columns=base_columns) if intervals_records else pd.DataFrame(columns=base_columns)
    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)

    pt = clickData["points"][0]
    t = float(pt["x"]); curve_idx = pt["curveNumber"]

    # Map to first 6 base traces (3 overview, 3 detail)
    if curve_idx in (0, 3):
        label = display_keys[0]
    elif curve_idx in (1, 4):
        label = display_keys[1]
    elif curve_idx in (2, 5):
        label = display_keys[2]
    else:
        long = ("on" in (long_labels or []))
        return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, intervals_df.to_dict("records"), ""

    click_state = click_state or {}
    click_state.setdefault("stage", "optical")
    click_state.setdefault("optical_times", [])
    click_state.setdefault("optical_vals", [])
    click_state.setdefault("electrical_times", [])
    click_state.setdefault("electrical_vals", [])

    msg = ""

    if click_state["stage"] == "optical":
        if label not in (optical_key, opticalre_key):
            msg = f"Please click on {optical_key} or {opticalre_key} for Optical stage."
            long = ("on" in (long_labels or []))
            return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, intervals_df.to_dict("records"), msg

        val_on_optical = float(np.interp(t, original_time, raw_signals[optical_key]))
        click_state["optical_times"].append(t)
        click_state["optical_vals"].append(val_on_optical)
        n = len(click_state["optical_times"])
        msg = f"Optical point {n}/3 recorded"

        if n < 3:
            long = ("on" in (long_labels or []))
            return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, intervals_df.to_dict("records"), msg

        times_sorted = sorted(click_state["optical_times"])
        optical_vals_sorted   = [float(np.interp(tt, original_time, raw_signals[optical_key]))   for tt in times_sorted]
        opticalre_vals_sorted = [float(np.interp(tt, original_time, raw_signals[opticalre_key])) for tt in times_sorted]
        click_state["pending_optical"] = {
            "event_start":   times_sorted[0],
            "event_plateau": times_sorted[1],
            "event_end":     times_sorted[2],
            "optical_base":  optical_vals_sorted[0],
            "optical_rise":  optical_vals_sorted[1],
            "optical_end":   optical_vals_sorted[2],
            "opticalre_base": opticalre_vals_sorted[0],
            "opticalre_rise": opticalre_vals_sorted[1],
            "opticalre_end":  opticalre_vals_sorted[2],
        }
        click_state["stage"] = "electrical"
        msg += "  Optical confirmed. Now select 4 points on Electrical."
        long = ("on" in (long_labels or []))
        return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, intervals_df.to_dict("records"), msg

    if click_state["stage"] == "electrical":
        if label != electrical_key:
            msg = f"Please click on {electrical_key} for Electrical stage."
            long = ("on" in (long_labels or []))
            return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, intervals_df.to_dict("records"), msg

        val = float(np.interp(t, original_time, raw_signals[electrical_key]))
        click_state["electrical_times"].append(t)
        click_state["electrical_vals"].append(val)
        n = len(click_state["electrical_times"])
        msg = f"Electrical point {n}/4 recorded"

        if n < 4:
            long = ("on" in (long_labels or []))
            return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, intervals_df.to_dict("records"), msg

        row = dict(click_state.get("pending_optical", {}))
        e_times = click_state["electrical_times"]
        e_vals  = [float(np.interp(tt, original_time, raw_signals[electrical_key])) for tt in e_times]
        row.update({
            "entry_base": e_vals[0], "entry_peak": e_vals[1], "exit_peak": e_vals[2], "exit_base": e_vals[3],
            "entry_base_t": e_times[0], "entry_peak_t": e_times[1], "exit_peak_t": e_times[2], "exit_base_t": e_times[3],
            "file_name": os.path.basename(abf_file),
            "sensor": np.nan, "analytes": np.nan, "solution": np.nan, "notes": np.nan,
        })
        intervals_df = pd.concat([intervals_df, pd.DataFrame([row], columns=base_columns)], ignore_index=True)

        click_state = {"stage": "optical",
                       "optical_times": [], "optical_vals": [],
                       "electrical_times": [], "electrical_vals": [],
                       "pending_event_id": len(intervals_df)}
        msg += "  Event saved."
        long = ("on" in (long_labels or []))
        return (
            build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx),
            click_state,
            intervals_df[base_columns].to_dict("records"),
            msg
        )

    long = ("on" in (long_labels or []))
    return (
        build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx),
        click_state,
        intervals_df[base_columns].to_dict("records"),
        msg
    )

# -- Undo & discard pending
@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Output("click-state", "data", allow_duplicate=True),
    Output("msg", "children", allow_duplicate=True),
    Input("undo-btn", "n_clicks"),
    State("click-state", "data"),
    State("current-window", "data"),
    State("intervals-store", "data"),
    State("stored-zoom", "data"),
    State("current-mode", "data"),
    State("long-labels", "value"),
    State("selected-row-idx", "data"),
    prevent_initial_call=True
)
def undo_last_point(n, click_state, current_window, intervals_records, stored_zoom, mode, long_labels, selected_row_idx):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update
    msg = "Nothing to undo."
    click_state = click_state or {}
    if click_state.get("stage") == "optical" and click_state.get("optical_times"):
        click_state["optical_times"].pop(); click_state["optical_vals"].pop()
        msg = "Removed last Optical point."
    elif click_state.get("stage") == "electrical" and click_state.get("electrical_times"):
        click_state["electrical_times"].pop(); click_state["electrical_vals"].pop()
        msg = "Removed last Electrical point."
    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)
    long = ("on" in (long_labels or []))
    return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, msg

@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Output("click-state", "data", allow_duplicate=True),
    Output("msg", "children", allow_duplicate=True),
    Input("discard-btn", "n_clicks"),
    State("click-state", "data"),
    State("current-window", "data"),
    State("intervals-store", "data"),
    State("stored-zoom", "data"),
    State("current-mode", "data"),
    State("long-labels", "value"),
    State("selected-row-idx", "data"),
    prevent_initial_call=True
)
def discard_pending(n, click_state, current_window, intervals_records, stored_zoom, mode, long_labels, selected_row_idx):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update
    click_state = {"stage": "optical",
                   "optical_times": [], "optical_vals": [],
                   "electrical_times": [], "electrical_vals": [],
                   "pending_event_id": len(intervals_records or [])}
    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)
    long = ("on" in (long_labels or []))
    return build_figure(intervals_df, [w0, w1], click_state, stored_zoom, mode, long_labels=long, selected_row=selected_row_idx), click_state, "Discarded pending event."

# -- Save to legacy CSV on disk
@app.callback(
    Output("msg", "children", allow_duplicate=True),
    Input("save-btn", "n_clicks"),
    State("intervals-store", "data"),
    prevent_initial_call=True
)
def save(n, intervals_records):
    if not n:
        return dash.no_update
    df_internal = pd.DataFrame(intervals_records or [], columns=base_columns)
    legacy_df = internal_to_legacy_saveframe(df_internal, abf_file)
    legacy_df.to_csv(output_file, index=False)
    return f"Saved (legacy CSV) to: {output_file}"

# -- Table sync (display legacy columns), row navigation, delete, export
@app.callback(
    Output("events-table", "data"),
    Output("intervals-store", "data", allow_duplicate=True),
    Input("intervals-store", "data"),
    prevent_initial_call=True
)
def refresh_table_from_store(internal_rows):
    # Render table in the *legacy* order while keeping store in INTERNAL format
    df_internal = pd.DataFrame(internal_rows or [], columns=base_columns)
    table_rows = internal_to_legacy_saveframe(df_internal, abf_file).to_dict("records")
    return table_rows, df_internal[base_columns].to_dict("records")

@app.callback(
    Output("selected-row-idx", "data"),
    Output("current-window", "data", allow_duplicate=True),
    Output("main-plot", "figure", allow_duplicate=True),
    Output("events-table", "selected_rows", allow_duplicate=True),
    Input("events-table", "active_cell"),
    Input("events-table", "selected_rows"),
    State("events-table", "data"),          # legacy view (not used for plotting)
    State("current-window", "data"),
    State("stored-zoom", "data"),
    State("intervals-store", "data"),       # INTERNAL rows (used for plotting)
    State("current-mode", "data"),
    State("long-labels", "value"),
    State("fit-mode", "value"),
    prevent_initial_call=True
)
def jump_to_row(active_cell, selected_rows, _table_data_unused, current_window, stored_zoom, intervals_records, mode, long_labels, fit_vals):
    row_idx = None
    if selected_rows:
        row_idx = selected_rows[0]
    elif active_cell and active_cell.get("row") is not None:
        row_idx = active_cell.get("row")

    if row_idx is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    intervals_df = pd.DataFrame(intervals_records or [], columns=base_columns)
    if not (0 <= row_idx < len(intervals_df)):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    w0, w1 = current_window if (current_window and len(current_window) == 2) else (default_start, default_end)
    cur_span = max(w1 - w0, MIN_WIN_SEC)
    row = intervals_df.iloc[row_idx]

    def _clamp_pair(a, b, pad=0.07):
        lo, hi = float(min(a, b)), float(max(a, b))
        span = max(hi - lo, MIN_WIN_SEC)
        pad_abs = span * pad
        return clamp_window_pair(max(lo - pad_abs, data_start), min(hi + pad_abs, data_end))

    t_start = row.get("event_start")
    t_end   = row.get("event_end")
    ws = row.get("window_start")
    we = row.get("window_end")
    
    # Prioritize window columns if present
    if pd.notna(ws) and pd.notna(we):
        new_w0, new_w1 = _clamp_pair(ws, we, pad=0.07)
    elif pd.notna(t_start) and pd.notna(t_end):
        new_w0, new_w1 = _clamp_pair(t_start, t_end, pad=0.07)
    else:
        # center on whatever times we have
        ts = [row.get(k) for k in ["event_start", "event_plateau", "event_end",
                                   "entry_base_t", "entry_peak_t", "exit_peak_t", "exit_base_t",
                                   "window_start", "window_end"]]
        ts = [float(t) for t in ts if pd.notna(t)]
        if ts:
            c = (min(ts) + max(ts)) / 2.0
        else:
            c = (w0 + w1) / 2.0
        new_w0, new_w1 = clamp_window_pair(c - cur_span / 2.0, c + cur_span / 2.0)

    long = ("on" in (long_labels or []))
    fig = build_figure(intervals_df, [new_w0, new_w1], None, stored_zoom, mode, long_labels=long, selected_row=row_idx)
    return row_idx, [new_w0, new_w1], fig, [row_idx]

@app.callback(
    Output("events-table", "data", allow_duplicate=True),
    Output("intervals-store", "data", allow_duplicate=True),
    Output("table-msg", "children"),
    Input("delete-row", "n_clicks"),
    State("events-table", "selected_rows"),
    State("intervals-store", "data"),
    prevent_initial_call=True
)
def delete_selected(n, selected_rows, internal_rows):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update
    if not selected_rows:
        return dash.no_update, dash.no_update, "No row selected."
    idx = selected_rows[0]

    df_internal = pd.DataFrame(internal_rows or [], columns=base_columns)
    if idx < 0 or idx >= len(df_internal):
        return dash.no_update, dash.no_update, "Invalid row."

    df_internal = df_internal.drop(index=idx).reset_index(drop=True)
    table_rows = internal_to_legacy_saveframe(df_internal, abf_file).to_dict("records")
    return table_rows, df_internal[base_columns].to_dict("records"), "Deleted selected row."

@app.callback(
    Output("download-csv", "data"),
    Input("export-csv", "n_clicks"),
    State("export-name", "value"),
    State("intervals-store", "data"),
    prevent_initial_call=True
)
def export_csv(n, name, internal_rows):
    if not n:
        return dash.no_update
    df_internal = pd.DataFrame(internal_rows or [], columns=base_columns)
    legacy_df = internal_to_legacy_saveframe(df_internal, abf_file)
    fname = (name or f"events_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}").strip().replace(".csv", "") + ".csv"
    csv_str = legacy_df.to_csv(index=False)
    return dict(content=csv_str, filename=fname, type="text/csv")

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    port = find_free_port()
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception:
        pass
    app.run(debug=False, use_reloader=False, port=port)