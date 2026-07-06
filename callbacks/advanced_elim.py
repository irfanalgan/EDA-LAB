"""
Gelişmiş Değişken Eleme — Dash callback'leri.
Background thread + interval polling ile pipeline çalıştırır.
"""

import threading
import uuid
import time

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from modules.advanced_elim import run_pipeline

import logging
log = logging.getLogger(__name__)

# ── Server-side progress & cancel ────────────────────────────────────────────
_ADVELIM_PROGRESS: dict = {}
_ADVELIM_CANCEL: dict = {}
_ADVELIM_THREAD: threading.Thread | None = None


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _coerce_numeric_raw(X):
    """Ham matrisi ağaç modelin fit edebileceği sayısala çevir; kolon=değişken 1-1 korunur."""
    out = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
        else:  # object/kategorik → ordinal kod (NaN → NaN)
            out[c] = s.astype("category").cat.codes.replace(-1, np.nan)
    return pd.DataFrame(out, index=X.index)


_METHOD_LABELS = {
    "gini": "Tekli Gini",
    "zero_gain": "Sıfır Kazanç",
    "rfe": "RFE",
    "perf_curve": "Performans Eğrisi",
}


def _build_progress_ui(prog):
    """Progress dict'ten sonuç alanı HTML'i üret."""
    if not prog:
        return html.Div()

    children = []
    results = prog.get("results", {})
    counts = prog.get("step_counts", {})
    current_method = prog.get("current_method", "")
    step_progress = prog.get("step_progress", 0)
    done = prog.get("done", False)
    error = prog.get("error")

    if error:
        children.append(html.Div(f"Hata: {error}",
                                 style={"color": "#ef4444", "fontSize": "0.75rem"}))
        return html.Div(children)

    for method in ["gini", "zero_gain", "rfe", "perf_curve"]:
        label = _METHOD_LABELS.get(method, method)

        if method in results:
            # Tamamlanmış adım
            before, after = counts.get(method, (0, 0))
            elim_count = before - after if method != "perf_curve" else 0
            result = results[method]

            if method == "perf_curve":
                txt = f"✓ {label}: grafik hazır"
            else:
                txt = f"✓ {label}: {before} → {after} ({elim_count} elendi)"

            row = html.Div(txt, style={"color": "#3fb950", "fontSize": "0.75rem"})
            children.append(row)

            # Elenen değişken detayları (gini için)
            if method == "gini" and result.get("eliminated"):
                details = result.get("details", {})
                detail_items = [f"{v} ({details.get(v, '?')})" for v in result["eliminated"][:10]]
                children.append(html.Div(
                    "  └ elenen: " + ", ".join(detail_items),
                    style={"color": "#7e8fa4", "fontSize": "0.7rem", "paddingLeft": "12px"}
                ))
        elif method == current_method and not done:
            # Çalışan adım
            children.append(html.Div(
                f"⟳ {label}: %{step_progress} tamamlandı...",
                style={"color": "#4F8EF7", "fontSize": "0.75rem"}
            ))

    # Pipeline özeti
    if done and not error:
        all_elim = prog.get("all_eliminated", [])
        first_counts = list(counts.values())
        total_before = first_counts[0][0] if first_counts else 0
        total_after = first_counts[-1][1] if first_counts else 0

        # Perf curve sonuçlarını al — grafik + slider
        perf_result = results.get("perf_curve")
        if perf_result and perf_result.get("curve"):
            children.append(_build_perf_curve_section(perf_result, total_after))

        children.append(html.Hr(style={"borderColor": "#21262d", "margin": "6px 0"}))
        children.append(html.Div(
            f"Pipeline Özeti: {total_before} → {total_after} ({len(all_elim)} değişken elendi)",
            style={"color": "#c8cdd8", "fontSize": "0.75rem", "fontWeight": "600"}
        ))

    return html.Div(children)


def _build_perf_curve_section(perf_result, n_remaining):
    """Performans eğrisi grafik + slider bölümü."""
    curve = perf_result["curve"]
    ranked_vars = perf_result.get("ranked_vars", [])

    x_vals = [p[0] for p in curve]
    y_vals = [p[1] for p in curve]
    metric_label = "Gini" if curve else ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines+markers",
        marker=dict(size=5, color="#4F8EF7"),
        line=dict(width=2, color="#4F8EF7"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.6)",
        margin=dict(l=40, r=20, t=30, b=40),
        height=220,
        xaxis=dict(title="Değişken Sayısı", gridcolor="#21262d"),
        yaxis=dict(title=metric_label, gridcolor="#21262d"),
        font=dict(size=11),
    )

    max_n = x_vals[-1] if x_vals else n_remaining
    default_n = max_n

    return html.Div([
        dcc.Graph(id="advelim-perfcurve-graph", figure=fig,
                  config={"displayModeBar": False},
                  style={"marginTop": "8px"}),
        html.Div([
            html.Span("İlk N değişkeni seç: ", style={"fontSize": "0.73rem", "color": "#7e8fa4"}),
            dcc.Slider(
                id="advelim-perfcurve-slider",
                min=x_vals[0] if x_vals else 1,
                max=max_n,
                step=1,
                value=default_n,
                marks={x_vals[0]: str(x_vals[0]), max_n: str(max_n)} if x_vals else {},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"padding": "0 10px"}),
        html.Div(id="advelim-perfcurve-info",
                 style={"fontSize": "0.73rem", "color": "#c8cdd8", "textAlign": "center"}),
    ])


# ── Background thread runner ─────────────────────────────────────────────────

def _run_background(prog_key, X, y, steps, max_rows, cancel_event):
    """Background thread: pipeline çalıştır, sonuçları progress dict'e yaz."""
    progress_dict = _ADVELIM_PROGRESS.setdefault(prog_key, {})
    run_pipeline(X, y, steps, max_rows=max_rows,
                 cancel_event=cancel_event, progress_dict=progress_dict)


# ── Callback 1: Toggle collapse ─────────────────────────────────────────────

@app.callback(
    Output("collapse-advelim", "is_open"),
    Input("btn-advelim-toggle", "n_clicks"),
    State("collapse-advelim", "is_open"),
    prevent_initial_call=True,
)
def toggle_advelim_collapse(n, is_open):
    return not is_open


# ── Callback 2: WoE uyarı banner ────────────────────────────────────────────

@app.callback(
    Output("advelim-woe-warning", "style"),
    Input("varsummary-data-tab", "active_tab"),
    prevent_initial_call=True,
)
def toggle_woe_warning(vs_tab):
    if vs_tab == "vs-tab-woe":
        return {"display": "block", "fontSize": "0.72rem", "color": "#f59e0b",
                "backgroundColor": "rgba(245,158,11,0.08)", "padding": "4px 8px",
                "borderRadius": "4px", "marginBottom": "0.5rem"}
    return {"display": "none"}


# ── Callback 3: Hesapla ─────────────────────────────────────────────────────

@app.callback(
    Output("store-advelim-results", "data"),
    Output("interval-advelim", "disabled"),
    Output("btn-advelim-compute", "disabled"),
    Output("btn-advelim-cancel", "disabled"),
    Output("btn-advelim-accept", "disabled"),
    Output("advelim-results-area", "children", allow_duplicate=True),
    Input("btn-advelim-compute", "n_clicks"),
    State("store-active-vars", "data"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("varsummary-data-tab", "active_tab"),
    State("advelim-model-type", "value"),
    State("advelim-sample-size", "value"),
    State("chk-advelim-gini", "value"),
    State("advelim-gini-thresh", "value"),
    State("chk-advelim-zerogain", "value"),
    State("chk-advelim-rfe", "value"),
    State("advelim-rfe-min", "value"),
    State("advelim-rfe-step", "value"),
    State("chk-advelim-perfcurve", "value"),
    State("advelim-pc-step", "value"),
    State("advelim-pc-metric", "value"),
    prevent_initial_call=True,
)
def start_advelim(n_clicks, active_vars, key, config, vs_tab,
                  model_type, sample_size,
                  chk_gini, gini_thresh,
                  chk_zerogain,
                  chk_rfe, rfe_min, rfe_step,
                  chk_perfcurve, pc_step, pc_metric):
    _no = dash.no_update

    if not n_clicks or not key or not config or not active_vars:
        return _no, _no, _no, _no, _no, _no

    target = config.get("target_col")
    if not target:
        return _no, _no, _no, _no, _no, _no

    # En az bir yöntem seçili mi?
    steps = []
    mt = model_type or "lgbm"

    if chk_gini:
        steps.append({"method": "gini", "model_type": mt,
                       "params": {"threshold": float(gini_thresh or 0.85)}})
    if chk_zerogain:
        steps.append({"method": "zero_gain", "model_type": mt, "params": {}})
    if chk_rfe:
        steps.append({"method": "rfe", "model_type": mt,
                       "params": {"min_features": int(rfe_min or 10), "step": int(rfe_step or 5)}})
    if chk_perfcurve:
        steps.append({"method": "perf_curve", "model_type": mt,
                       "params": {"step": int(pc_step or 5), "metric": pc_metric or "gini"}})

    if not steps:
        return _no, _no, _no, _no, _no, html.Div(
            "En az bir yöntem seçin.", style={"color": "#f59e0b", "fontSize": "0.75rem"})

    # Veri kaynağı: aktif tab'a göre
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")
    _pfx = f"{key}_ds_{seg_col}_{seg_val}"
    use_woe = (vs_tab == "vs-tab-woe")

    if use_woe:
        train = _SERVER_STORE.get(f"{_pfx}_train_woe")
    else:
        train = _SERVER_STORE.get(f"{_pfx}_train")
    # Target her zaman raw train'den alınır (WoE verisinde target yok)
    raw_train = _SERVER_STORE.get(f"{_pfx}_train")

    if train is None or raw_train is None:
        return _no, _no, _no, _no, _no, html.Div(
            "Veri bulunamadı. Precompute tamamlanmış olmalı.",
            style={"color": "#ef4444", "fontSize": "0.75rem"})

    if target not in raw_train.columns:
        return _no, _no, _no, _no, _no, html.Div(
            f"Target kolonu ({target}) bulunamadı.",
            style={"color": "#ef4444", "fontSize": "0.75rem"})

    # Sadece aktif değişkenler
    cols = [c for c in active_vars if c in train.columns and c != target]
    if len(cols) < 2:
        return _no, _no, _no, _no, _no, html.Div(
            "En az 2 aktif değişken gerekli.",
            style={"color": "#f59e0b", "fontSize": "0.75rem"})

    X = train[cols].reset_index(drop=True)
    if not use_woe:
        X = _coerce_numeric_raw(X)
    y = raw_train[target].reset_index(drop=True)

    # Uyarı: çok fazla değişken
    warn = ""
    if len(cols) > 200:
        warn = f"⚠ {len(cols)} değişken — işlem uzun sürebilir. Önce basit filtreleri uygulayın."

    # Eski thread iptal
    global _ADVELIM_THREAD
    prog_key = str(uuid.uuid4())[:8]
    cancel_event = threading.Event()
    _ADVELIM_CANCEL[prog_key] = cancel_event

    if _ADVELIM_THREAD is not None and _ADVELIM_THREAD.is_alive():
        # Önceki cancel event'i bul ve set et
        for pk, ce in list(_ADVELIM_CANCEL.items()):
            if pk != prog_key:
                ce.set()
        _ADVELIM_THREAD.join(timeout=2)

    max_rows = int(sample_size) if sample_size else None

    _ADVELIM_PROGRESS[prog_key] = {}
    t = threading.Thread(
        target=_run_background,
        args=(prog_key, X, y, steps, max_rows, cancel_event),
        daemon=True,
    )
    t.start()
    _ADVELIM_THREAD = t

    state = {"prog_key": prog_key, "steps": [s["method"] for s in steps]}

    init_msg = [html.Div("Pipeline başlatıldı...",
                         style={"color": "#4F8EF7", "fontSize": "0.75rem"})]
    if warn:
        init_msg.insert(0, html.Div(warn, style={"color": "#f59e0b", "fontSize": "0.73rem",
                                                   "marginBottom": "4px"}))

    return (
        state,           # store-advelim-results → prog_key
        False,           # interval enable
        True,            # Hesapla disabled
        False,           # İptal enabled
        True,            # Kabul Et disabled
        html.Div(init_msg),
    )


# ── Callback 4: Progress tick ───────────────────────────────────────────────

@app.callback(
    Output("advelim-results-area", "children", allow_duplicate=True),
    Output("interval-advelim", "disabled", allow_duplicate=True),
    Output("btn-advelim-compute", "disabled", allow_duplicate=True),
    Output("btn-advelim-cancel", "disabled", allow_duplicate=True),
    Output("btn-advelim-accept", "disabled", allow_duplicate=True),
    Input("interval-advelim", "n_intervals"),
    State("store-advelim-results", "data"),
    prevent_initial_call=True,
)
def advelim_tick(n, state):
    _no = dash.no_update
    if not state or "prog_key" not in state:
        return _no, True, _no, _no, _no

    prog = _ADVELIM_PROGRESS.get(state["prog_key"])
    if not prog:
        return _no, _no, _no, _no, _no

    done = prog.get("done", False)
    has_perf_curve = "perf_curve" in prog.get("results", {})

    ui = _build_progress_ui(prog)

    if done:
        # Pipeline bitti
        return (
            ui,
            True,            # interval disable
            False,           # Hesapla enable
            True,            # İptal disable
            False,           # Kabul Et enable
        )

    return ui, False, True, False, True


# ── Callback 5: Performans Eğrisi Slider info ───────────────────────────────

@app.callback(
    Output("advelim-perfcurve-info", "children"),
    Input("advelim-perfcurve-slider", "value"),
    State("store-advelim-results", "data"),
    prevent_initial_call=True,
)
def update_perfcurve_info(slider_val, state):
    if not state or not slider_val:
        return ""

    prog = _ADVELIM_PROGRESS.get(state.get("prog_key", ""))
    if not prog:
        return ""

    perf = prog.get("results", {}).get("perf_curve", {})
    curve = perf.get("curve", [])
    if not curve:
        return ""

    # Slider'a en yakın noktayı bul
    n_sel = int(slider_val)
    closest = min(curve, key=lambda p: abs(p[0] - n_sel))
    max_point = curve[-1]

    total_vars = perf.get("ranked_vars", [])
    diff = round(max_point[1] - closest[1], 4)

    return f"Metrik@{closest[0]}: {closest[1]:.4f}  |  Metrik@{max_point[0]}: {max_point[1]:.4f}  |  Fark: {diff:.4f}"


# ── Callback 6: Sonuçları Kabul Et ──────────────────────────────────────────

@app.callback(
    Output("store-vs-overrides", "data", allow_duplicate=True),
    Output("store-active-vars", "data", allow_duplicate=True),
    Output("store-active-snapshot", "data", allow_duplicate=True),
    Output("vs-active-count", "children", allow_duplicate=True),
    Output("btn-advelim-accept", "disabled", allow_duplicate=True),
    Input("btn-advelim-accept", "n_clicks"),
    State("store-advelim-results", "data"),
    State("store-vs-overrides", "data"),
    State("store-active-vars", "data"),
    State("advelim-perfcurve-slider", "value"),
    prevent_initial_call=True,
)
def accept_advelim_results(n, state, overrides, active_vars, slider_val):
    _no = dash.no_update
    if not n or not state:
        return _no, _no, _no, _no, _no

    prog = _ADVELIM_PROGRESS.get(state.get("prog_key", ""))
    if not prog or not prog.get("done"):
        return _no, _no, _no, _no, _no

    # Pipeline'ın elediği değişkenler
    all_eliminated = set(prog.get("all_eliminated", []))

    # Performans eğrisi slider'dan ek eleme
    perf = prog.get("results", {}).get("perf_curve", {})
    if perf and slider_val is not None:
        ranked_vars = perf.get("ranked_vars", [])
        n_keep = int(slider_val)
        kept_by_slider = set(ranked_vars[:n_keep])
        eliminated_by_slider = set(ranked_vars[n_keep:])
        all_eliminated |= eliminated_by_slider

    # Overrides güncelle
    overrides = overrides or {"included": [], "excluded": []}
    overrides["adv_excluded"] = sorted(all_eliminated)

    # active_vars güncelle
    active_set = set(active_vars or []) - all_eliminated
    new_active = sorted(active_set)

    # Sayı
    total = len(set(active_vars or [])) + len(all_eliminated) if active_vars else 0
    count_txt = f"{len(new_active)} / {total} değişken seçili"

    return overrides, new_active, new_active, count_txt, True


# ── Callback 7: İptal ───────────────────────────────────────────────────────

@app.callback(
    Output("advelim-results-area", "children", allow_duplicate=True),
    Output("interval-advelim", "disabled", allow_duplicate=True),
    Output("btn-advelim-compute", "disabled", allow_duplicate=True),
    Output("btn-advelim-cancel", "disabled", allow_duplicate=True),
    Input("btn-advelim-cancel", "n_clicks"),
    State("store-advelim-results", "data"),
    prevent_initial_call=True,
)
def cancel_advelim(n, state):
    _no = dash.no_update
    if not n or not state:
        return _no, _no, _no, _no

    prog_key = state.get("prog_key", "")
    ce = _ADVELIM_CANCEL.get(prog_key)
    if ce:
        ce.set()

    return (
        html.Div("İptal edildi.", style={"color": "#f59e0b", "fontSize": "0.75rem"}),
        True,    # interval disable
        False,   # Hesapla enable
        True,    # İptal disable
    )
