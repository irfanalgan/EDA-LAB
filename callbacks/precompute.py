import threading
import time

import dash
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, _PRECOMPUTE_PROGRESS, get_df as _get_df
from utils.helpers import apply_segment_filter
from modules.profiling import compute_profile
from modules.deep_dive import compute_iv_ranking_optimal
from modules.correlation import compute_correlation_matrix
from modules.screening import screen_columns


# ── Precompute yardımcıları ───────────────────────────────────────────────────
_PRECOMPUTE_STEPS = [
    {"key": "screening",    "label": "Ön Eleme (Screening)"},
    {"key": "profiling",    "label": "Profil Analizi"},
    {"key": "iv_ranking",   "label": "IV Ranking"},
    {"key": "correlation",  "label": "Korelasyon Matrisi"},
]

def _precompute_step_row(step_key: str, label: str, status: str, duration: float = None):
    """status: 'done' | 'running' | 'waiting'"""
    if status == "done":
        icon  = html.Span("✓", style={"color": "#10b981", "fontWeight": "700", "marginRight": "10px", "fontSize": "1rem"})
        text_style = {"color": "#c8cdd8"}
        dur_text = f"{duration:.1f}s" if duration is not None else ""
    elif status == "running":
        icon  = html.Span("⟳", style={"color": "#4F8EF7", "fontWeight": "700", "marginRight": "10px", "fontSize": "1rem"})
        text_style = {"color": "#4F8EF7"}
        dur_text = "hesaplanıyor..."
    else:
        icon  = html.Span("○", style={"color": "#3a4a60", "marginRight": "10px", "fontSize": "1rem"})
        text_style = {"color": "#3a4a60"}
        dur_text = "bekliyor"

    return html.Div([
        icon,
        html.Span(label, style={**text_style, "fontSize": "0.9rem", "width": "220px", "display": "inline-block"}),
        html.Span(dur_text, style={"color": "#6b7a99", "fontSize": "0.8rem", "marginLeft": "12px"}),
    ], style={"padding": "6px 0", "display": "flex", "alignItems": "center"})


def _build_modal_body(step_idx: int, durations: dict):
    """step_idx: şu an hangi adım çalışıyor (0-based). -1 = henüz başlamadı."""
    rows = []
    for i, s in enumerate(_PRECOMPUTE_STEPS):
        if step_idx == -1:
            status = "waiting"
        elif i < step_idx:
            status = "done"
        elif i == step_idx:
            status = "running"
        else:
            status = "waiting"
        rows.append(_precompute_step_row(s["key"], s["label"], status, durations.get(s["key"])))

    n_done   = max(step_idx, 0)
    n_total  = len(_PRECOMPUTE_STEPS)
    pct      = int(n_done / n_total * 100)
    bar_fill = f"{pct}%"

    progress = html.Div([
        html.Div(style={
            "height": "6px", "width": bar_fill,
            "backgroundColor": "#4F8EF7", "borderRadius": "3px",
            "transition": "width 0.4s ease",
        }),
    ], style={
        "height": "6px", "backgroundColor": "#1e2a3a",
        "borderRadius": "3px", "marginTop": "20px",
    })

    return html.Div([
        html.Div(rows),
        progress,
        html.Div(f"%{pct} tamamlandı", style={
            "color": "#6b7a99", "fontSize": "0.78rem", "marginTop": "6px", "textAlign": "right"
        }),
    ])


def _build_modal_body_done(durations: dict):
    rows = [_precompute_step_row(s["key"], s["label"], "done", durations.get(s["key"]))
            for s in _PRECOMPUTE_STEPS]
    total = sum(v for v in durations.values() if v)
    return html.Div([
        html.Div(rows),
        html.Div([
            html.Div(style={
                "height": "6px", "width": "100%",
                "backgroundColor": "#10b981", "borderRadius": "3px",
            }),
        ], style={"height": "6px", "backgroundColor": "#1e2a3a", "borderRadius": "3px", "marginTop": "20px"}),
        html.Div(f"Tüm adımlar tamamlandı — {total:.1f}s", style={
            "color": "#10b981", "fontSize": "0.78rem", "marginTop": "6px", "textAlign": "right"
        }),
    ])


def _run_precompute_background(prog_key: str, key: str, target: str,
                               date_col, seg_col, seg_val):
    """Tüm precompute adımlarını background thread'de çalıştırır.
    İlerleme _PRECOMPUTE_PROGRESS[prog_key]'e yazılır, interval callback okur."""
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 0, "durations": {}, "done": False}

    df_orig = _get_df(key)
    if df_orig is None:
        _PRECOMPUTE_PROGRESS[prog_key]["done"] = True
        return

    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    durations = {}

    # ── Adım 0: Screening ────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        passed, screen_report = screen_columns(df_active, target, date_col, seg_col)
        _SERVER_STORE[f"{key}_screen"] = (passed, screen_report)
        durations["screening"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["screening"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 1, "durations": dict(durations), "done": False}

    # ── Adım 1: Profiling ────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        prof = compute_profile(df_active)
        _SERVER_STORE[f"{key}_profile_{seg_col}_{seg_val}"] = prof
        durations["profiling"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["profiling"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 2, "durations": dict(durations), "done": False}

    # ── Adım 2: IV Ranking ───────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        iv_df = compute_iv_ranking_optimal(df_active, target)
        if not iv_df.empty and iv_df["IV"].sum() > 0:
            _SERVER_STORE[f"{key}_iv_{seg_col}_{seg_val}"] = iv_df
        durations["iv_ranking"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["iv_ranking"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 3, "durations": dict(durations), "done": False}

    # ── Adım 3: Korelasyon ───────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        num_cols_corr = [c for c in df_active.select_dtypes(include=[np.number]).columns
                         if c != target][:30]
        corr_df = compute_correlation_matrix(df_active, num_cols_corr)
        _SERVER_STORE[f"{key}_corr_{seg_col}_{seg_val}_precomp"] = (corr_df, num_cols_corr)
        durations["correlation"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["correlation"] = None

    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 4, "durations": dict(durations), "done": True}


# ── Callback: Yapılandırmayı Onayla ──────────────────────────────────────────
_BTN_CLOSE_VISIBLE = {"fontSize": "0.85rem", "padding": "0.4rem 1.0rem"}
_BTN_HIDDEN        = {"display": "none"}

# (close_style, done_style) — tick callback'i için 2 output
_FOOTER_RUNNING = (_BTN_CLOSE_VISIBLE, _BTN_HIDDEN)
_FOOTER_DONE    = (_BTN_HIDDEN, {"fontSize": "0.85rem", "padding": "0.4rem 1.2rem"})

@app.callback(
    Output("store-config", "data"),
    Output("config-status", "children"),
    Output("store-precompute-state", "data"),
    Output("modal-precompute", "is_open"),
    Output("interval-precompute", "disabled"),
    Output("precompute-modal-body", "children"),
    Output("btn-precompute-close", "style"),
    Output("btn-precompute-done", "style"),
    Input("btn-confirm", "n_clicks"),
    State("dd-target-col", "value"),
    State("dd-date-col", "value"),
    State("dd-segment-col", "value"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def confirm_config(n_clicks, target_col, date_col, segment_col, key):
    no_modal = (dash.no_update,) * 6  # modal, interval, body, close-style, done-style
    if not target_col or target_col == "":
        return (dash.no_update, dbc.Alert(
            "Target kolonu zorunludur.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        ), *no_modal)

    config = {
        "target_col": target_col,
        "date_col": date_col or None,
        "segment_col": segment_col or None,
    }

    prog_key = f"{key}_precompute"
    precompute_state = {"prog_key": prog_key}

    parts = [html.Strong("✓ Onaylandı")]
    if date_col:
        parts += [f"  ·  Tarih: {date_col}"]
    if segment_col:
        parts += [f"  ·  Segment: {segment_col}"]

    # Background thread başlat — Dash thread'i bloklamaz
    t = threading.Thread(
        target=_run_precompute_background,
        args=(prog_key, key, target_col, date_col, segment_col or None, None),
        daemon=True,
    )
    t.start()

    return (
        config,
        dbc.Alert(parts, color="success", style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"}),
        precompute_state,
        True,                        # modal aç
        False,                       # interval hemen başlasın
        _build_modal_body(0, {}),    # adım 0 "çalışıyor" göster
        *_FOOTER_RUNNING,
    )


# ── Callback: Precompute Interval ────────────────────────────────────────────
@app.callback(
    Output("precompute-modal-body",  "children", allow_duplicate=True),
    Output("btn-precompute-close",   "style",    allow_duplicate=True),
    Output("btn-precompute-done",    "style",    allow_duplicate=True),
    Output("interval-precompute",    "disabled", allow_duplicate=True),
    Output("modal-precompute",       "is_open",  allow_duplicate=True),
    Input("interval-precompute", "n_intervals"),
    State("store-precompute-state", "data"),
    prevent_initial_call=True,
)
def precompute_tick(n, state):
    """Background thread'deki ilerlemeyi okur, UI'yı günceller."""
    _no = dash.no_update
    if not state or "prog_key" not in state:
        return _no, _no, _no, True, _no

    prog = _PRECOMPUTE_PROGRESS.get(state["prog_key"])
    if prog is None:
        return _no, _no, _no, _no, _no  # thread henüz başlamadı

    step      = prog.get("step", 0)
    durations = prog.get("durations", {})
    done      = prog.get("done", False)

    if done:
        return _build_modal_body_done(durations), *_FOOTER_DONE, True, True

    return _build_modal_body(step, durations), *_FOOTER_RUNNING, False, _no


@app.callback(
    Output("modal-precompute", "is_open", allow_duplicate=True),
    Output("interval-precompute", "disabled", allow_duplicate=True),
    Input("btn-precompute-close", "n_clicks"),
    Input("btn-precompute-done",  "n_clicks"),
    prevent_initial_call=True,
)
def close_precompute_modal(n_close, n_done):
    return False, True
