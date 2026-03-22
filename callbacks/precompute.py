import logging
import threading
import time

import dash
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from app_instance import app
from server_state import _SERVER_STORE, _PRECOMPUTE_PROGRESS, get_df as _get_df, clear_store
from utils.helpers import apply_segment_filter, get_splits
from modules.profiling import compute_profile
from modules.correlation import compute_correlation_matrix
from modules.screening import screen_columns
from modules.deep_dive import (
    SPECIAL_CODES, is_special_column, format_binning_table,
    build_period_table, _iv_label, _check_monotonicity,
)
from callbacks.var_summary import compute_var_summary_table, compute_var_summary_raw

logger = logging.getLogger(__name__)


# ── Thread yönetimi ──────────────────────────────────────────────────────────
_active_thread: threading.Thread | None = None
_cancel_event = threading.Event()

# ── Precompute yardımcıları ───────────────────────────────────────────────────
_PRECOMPUTE_STEPS = [
    {"key": "screening",    "label": "Ön Eleme (Screening)"},
    {"key": "profiling",    "label": "Profil Analizi"},
    {"key": "iv_ranking",   "label": "IV Ranking"},
    {"key": "correlation",  "label": "Korelasyon Matrisi"},
    {"key": "var_summary",  "label": "Değişken Özeti (WoE)"},
    {"key": "var_summary_raw", "label": "Değişken Özeti (Ham)"},
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
                               date_col, seg_col, seg_val, config: dict = None,
                               cancel_event: threading.Event = None):
    """Tüm precompute adımlarını background thread'de çalıştırır.
    İlerleme _PRECOMPUTE_PROGRESS[prog_key]'e yazılır, interval callback okur.
    cancel_event set edilirse thread erken sonlanır."""
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 0, "durations": {}, "done": False}

    def _cancelled():
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Precompute iptal edildi — prog_key=%s", prog_key)
            return True
        return False

    df_orig = _get_df(key)
    if df_orig is None:
        _PRECOMPUTE_PROGRESS[prog_key]["done"] = True
        return

    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    durations = {}

    cfg = config or {
        "target_col": target, "date_col": date_col, "oot_date": None,
        "segment_col": seg_col, "target_type": "binary",
        "has_test_split": False, "test_size": 20,
    }
    _pfx = f"{key}_ds_{seg_col}_{seg_val}"

    # ── Adım 0: Screening ────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        passed, screen_report = screen_columns(df_active, target, date_col, seg_col)
        _SERVER_STORE[f"{key}_screen"] = (passed, screen_report)
        _SERVER_STORE[f"{key}_screen_base"] = (list(passed), screen_report.copy())
        durations["screening"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["screening"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 1, "durations": dict(durations), "done": False}
    if _cancelled():
        return

    # ── Adım 1: Split + Profiling ────────────────────────────────────────────
    #   Split'i önce yapıyoruz çünkü profiling raw train+test üzerinden çalışacak.
    try:
        t0 = time.perf_counter()
        df_train, df_test, df_oot = get_splits(df_active, cfg)
        _SERVER_STORE[f"{_pfx}_train"] = df_train
        _SERVER_STORE[f"{_pfx}_test"]  = df_test   # None olabilir
        _SERVER_STORE[f"{_pfx}_oot"]   = df_oot    # None olabilir

        # Profiling: raw train+test (OOT hariç)
        df_raw_train_test = pd.concat([df_train, df_test], ignore_index=True) if df_test is not None else df_train.copy()
        prof = compute_profile(df_raw_train_test)
        _SERVER_STORE[f"{key}_profile_{seg_col}_{seg_val}"] = prof
        durations["profiling"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["profiling"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 2, "durations": dict(durations), "done": False}
    if _cancelled():
        return

    # ── Adım 2: WoE / IV ─────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        _target = config["target_col"]
        _max_bins = int(config.get("max_bins", 4))
        special_codes = SPECIAL_CODES

        # Screening'den geçen kolonlar
        _screen = _SERVER_STORE.get(f"{key}_screen")
        passed_cols = _screen[0] if _screen else []
        if not passed_cols:
            passed_cols = [c for c in df_train.columns
                          if c not in (target, date_col, seg_col)]

        # Special kolonları belirle
        special_cols = {c for c in passed_cols if is_special_column(df_train[c])}

        logger.info("WoE/IV — passed_cols: %d, special_cols: %d", len(passed_cols), len(special_cols))
        logger.info("WoE/IV — df_train: %s, df_test: %s, df_oot: %s",
                     df_train.shape if df_train is not None else None,
                     df_test.shape if df_test is not None else None,
                     df_oot.shape if df_oot is not None else None)

        train_woe_df = pd.DataFrame(index=df_train.index)
        test_woe_df  = pd.DataFrame(index=df_test.index) if df_test is not None else None
        oot_woe_df   = pd.DataFrame(index=df_oot.index) if df_oot is not None else None
        woe_tables   = {}
        bins_dict    = {}
        optb_dict    = {}
        failed_cols  = []
        eksik_map    = {}

        for col in passed_cols:
            if _cancelled():
                return
            try:
                _mb = 2 if col in special_cols else _max_bins
                optb = OptimalBinning(
                    name=col, monotonic_trend="auto_asc_desc",
                    max_n_bins=_mb, dtype="numerical", solver="cp",
                    special_codes=special_codes,
                )
                optb.fit(df_train[col].values, df_train[_target].values)

                # Train tablosu
                raw_bt = optb.binning_table.build(show_digits=8)
                train_bt = format_binning_table(raw_bt)
                # IV: raw tablodan index ile oku (OptBinning index="Totals")
                if "Totals" in raw_bt.index:
                    iv_train = round(float(raw_bt.loc["Totals", "IV"]), 4)
                else:
                    iv_train = round(float(pd.to_numeric(raw_bt["IV"], errors="coerce").sum()), 4)

                # Bin edges
                edges = [-np.inf] + list(optb.splits) + [np.inf]
                bins_dict[col] = edges

                # Eksik %
                eksik_map[col] = round(float(df_train[col].isna().mean() * 100), 2)

                # WoE transform
                train_woe_df[col] = optb.transform(
                    df_train[col].values, metric="woe",
                    metric_missing="empirical", metric_special="empirical")

                if df_test is not None:
                    test_woe_df[col] = optb.transform(
                        df_test[col].values, metric="woe",
                        metric_missing="empirical", metric_special="empirical")

                if df_oot is not None:
                    oot_woe_df[col] = optb.transform(
                        df_oot[col].values, metric="woe",
                        metric_missing="empirical", metric_special="empirical")

                # Test/OOT tabloları
                test_bt, oot_bt = None, None
                iv_test, iv_oot = 0.0, 0.0
                mono_test, mono_oot = "—", "—"

                if df_test is not None and len(df_test) > 0:
                    test_bt = build_period_table(df_test, col, _target, edges, train_bt)
                    if test_bt is not None:
                        _non_total = test_bt["Bin"] != "TOPLAM"
                        iv_test = round(float(pd.to_numeric(
                            test_bt.loc[_non_total, "IV"], errors="coerce").sum()), 4)
                        mono_test = _check_monotonicity(test_bt)

                if df_oot is not None and len(df_oot) > 0:
                    oot_bt = build_period_table(df_oot, col, _target, edges, train_bt)
                    if oot_bt is not None:
                        _non_total = oot_bt["Bin"] != "TOPLAM"
                        iv_oot = round(float(pd.to_numeric(
                            oot_bt.loc[_non_total, "IV"], errors="coerce").sum()), 4)
                        mono_oot = _check_monotonicity(oot_bt)

                woe_tables[col] = {
                    "train_table": train_bt.to_dict("records"),
                    "test_table": test_bt.to_dict("records") if test_bt is not None else None,
                    "oot_table": oot_bt.to_dict("records") if oot_bt is not None else None,
                    "iv_train": iv_train, "iv_test": iv_test, "iv_oot": iv_oot,
                    "monoton_test": mono_test, "monoton_oot": mono_oot,
                }
                optb_dict[col] = optb

            except Exception as e:
                logger.warning("WoE başarısız: %s — %s", col, e)
                failed_cols.append(col)

        # Cache'e yaz
        _SERVER_STORE[f"{_pfx}_woe_tables"] = woe_tables
        _SERVER_STORE[f"{_pfx}_bins"] = bins_dict
        _SERVER_STORE[f"{_pfx}_optb"] = optb_dict
        _SERVER_STORE[f"{_pfx}_train_woe"] = train_woe_df
        _SERVER_STORE[f"{_pfx}_test_woe"] = test_woe_df
        _SERVER_STORE[f"{_pfx}_oot_woe"] = oot_woe_df

        # IV DataFrame — woe_tables'dan türetilir (tek kaynak)
        iv_records = []
        for v, entry in woe_tables.items():
            iv_records.append({
                "Değişken": v,
                "IV": entry["iv_train"],
                "Eksik %": eksik_map.get(v, 0.0),
            })
        iv_df = (pd.DataFrame(iv_records)
                 .sort_values("IV", ascending=False)
                 .reset_index(drop=True))
        iv_df["IV"] = iv_df["IV"].round(4)
        iv_df["Güç"] = iv_df["IV"].apply(_iv_label)
        _SERVER_STORE[f"{key}_iv_{seg_col}_{seg_val}"] = iv_df

        durations["iv_ranking"] = round(time.perf_counter() - t0, 1)
    except Exception:
        logger.exception("Adım 2 (WoE/IV) başarısız")
        durations["iv_ranking"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 3, "durations": dict(durations), "done": False}
    if _cancelled():
        return

    # ── Adım 3: Korelasyon ─────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        _train_woe = _SERVER_STORE.get(f"{_pfx}_train_woe")
        if _train_woe is not None and not _train_woe.empty:
            _woe_cols = [c for c in _train_woe.columns if c in woe_tables]
            if len(_woe_cols) >= 2:
                corr = compute_correlation_matrix(_train_woe[_woe_cols], _woe_cols)
                _SERVER_STORE[f"{key}_corr_{seg_col}_{seg_val}"] = corr
        durations["correlation"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["correlation"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 4, "durations": dict(durations), "done": False}
    if _cancelled():
        return

    # ── Adım 4: Değişken Özeti ─────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        summary = compute_var_summary_table(cfg, key, seg_col, seg_val)
        if summary is not None and not summary.empty:
            _SERVER_STORE[f"{key}_varsummary_{seg_col}_{seg_val}"] = summary
        durations["var_summary"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["var_summary"] = None

    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 5, "durations": dict(durations), "done": False}
    if _cancelled():
        return

    # ── Adım 5: Değişken Özeti (Ham) ─────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        summary_raw = compute_var_summary_raw(cfg, key, seg_col, seg_val)
        if summary_raw is not None and not summary_raw.empty:
            _SERVER_STORE[f"{key}_varsummary_raw_{seg_col}_{seg_val}"] = summary_raw
        durations["var_summary_raw"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["var_summary_raw"] = None

    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 6, "durations": dict(durations), "done": True}


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
    Output("store-model-signal", "data", allow_duplicate=True),
    Output("pg-model-output", "children", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Output("store-pending-note", "data", allow_duplicate=True),
    Output("store-profile-loaded", "data", allow_duplicate=True),
    Output("dd-profile", "value", allow_duplicate=True),
    Output("profile-status", "children", allow_duplicate=True),
    Input("btn-confirm", "n_clicks"),
    State("dd-target-col", "value"),
    State("dd-date-col", "value"),
    State("dd-sort-col", "value"),
    State("dd-oot-date", "value"),
    State("dd-segment-col", "value"),
    State("dd-segment-val", "value"),
    State("chk-train-test-split", "value"),
    State("input-test-size", "value"),
    State("input-max-bins", "value"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def confirm_config(n_clicks, target_col, date_col, sort_col, oot_date, segment_col,
                   segment_val, train_test_val, test_size_cfg, max_bins_cfg, key):
    _MODEL_RESET = (None, "", None, None, None, None, "")  # model-signal, pg-output, model-index, note, profile-loaded, dd-profile, profile-status
    no_modal = (dash.no_update,) * 6  # modal, interval, body, close-style, done-style
    if not target_col or target_col == "":
        return (dash.no_update, dbc.Alert(
            "Target kolonu zorunludur.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        ), *no_modal, *_MODEL_RESET)

    if not date_col or date_col == "":
        return (dash.no_update, dbc.Alert(
            "Tarih kolonu zorunludur.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        ), *no_modal, *_MODEL_RESET)

    df_orig = _get_df(key)
    config = {
        "target_col":      target_col,
        "date_col":        date_col or None,
        "sort_col":        sort_col or None,
        "oot_date":        oot_date or None,
        "segment_col":     segment_col or None,
        "target_type":     "binary",
        "has_test_split":  bool(train_test_val),
        "test_size":       int(test_size_cfg or 20),
        "max_bins":        int(max_bins_cfg or 4),
    }

    prog_key = f"{key}_precompute"
    precompute_state = {"prog_key": prog_key}

    parts = [html.Strong("✓ Onaylandı")]
    if date_col:
        parts += [f"  ·  Tarih: {date_col}"]
    if sort_col:
        parts += [f"  ·  Sıralama: {sort_col}"]
    if oot_date:
        parts += [f"  ·  OOT: ≥ {oot_date}"]
        if bool(train_test_val):
            parts += [f"  ·  Test: %{test_size_cfg or 20}"]
    if segment_col:
        parts += [f"  ·  Segment: {segment_col}"]

    # segment_val: kullanıcı seçtiyse onu kullan, yoksa None (tüm veri)
    seg_val = segment_val if segment_val and segment_val != ["Tümü"] and "Tümü" not in (segment_val if isinstance(segment_val, list) else [segment_val]) else None
    config["segment_val"] = seg_val

    # Eski thread çalışıyorsa iptal et
    global _active_thread, _cancel_event
    if _active_thread is not None and _active_thread.is_alive():
        _cancel_event.set()
        _active_thread.join(timeout=2)
        logger.info("Eski precompute thread iptal edildi")

    # Yeni cancel event (clear durumda)
    _cancel_event = threading.Event()

    # Eski cache'leri temizle — sadece ham veri kalsın, geri kalanı yeniden hesaplansın
    clear_store(keep_key=key)

    # Background thread başlat — Dash thread'i bloklamaz
    t = threading.Thread(
        target=_run_precompute_background,
        args=(prog_key, key, target_col, date_col, segment_col or None, seg_val, config, _cancel_event),
        daemon=True,
    )
    t.start()
    _active_thread = t

    return (
        config,
        dbc.Alert(parts, color="success", style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"}),
        precompute_state,
        True,                        # modal aç
        False,                       # interval hemen başlasın
        _build_modal_body(0, {}),    # adım 0 "çalışıyor" göster
        *_FOOTER_RUNNING,
        *_MODEL_RESET,
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
