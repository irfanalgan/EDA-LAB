import threading
import time

import dash
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, _PRECOMPUTE_PROGRESS, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import build_woe_datasets
from modules.profiling import compute_profile
from modules.correlation import compute_correlation_matrix
from modules.screening import screen_columns
from modules.deep_dive import get_woe_detail, _build_binning_table_from_edges


# ── Precompute yardımcıları ───────────────────────────────────────────────────
_PRECOMPUTE_STEPS = [
    {"key": "screening",    "label": "Ön Eleme (Screening)"},
    {"key": "profiling",    "label": "Profil Analizi"},
    {"key": "iv_ranking",   "label": "IV Ranking"},
    {"key": "correlation",  "label": "Korelasyon Matrisi"},
    {"key": "var_summary",  "label": "Değişken Özeti (WoE)"},
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
                               date_col, seg_col, seg_val, config: dict = None):
    """Tüm precompute adımlarını background thread'de çalıştırır.
    İlerleme _PRECOMPUTE_PROGRESS[prog_key]'e yazılır, interval callback okur."""
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 0, "durations": {}, "done": False}

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

    # Zaten hesaplanmışsa skip
    if f"{_pfx}_train" in _SERVER_STORE:
        _PRECOMPUTE_PROGRESS[prog_key] = {"step": 5, "durations": {}, "done": True}
        return

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

    # ── Adım 2: WoE / IV ─────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()

        # Screening'den geçen kolonlar
        screen_data = _SERVER_STORE.get(f"{key}_screen")
        if screen_data:
            var_list = [c for c in screen_data[0] if c != target]
        else:
            var_list = [c for c in df_train.columns if c != target
                        and c != date_col and c != seg_col]

        # Tek döngüde WoE fit (train) + IV + transform (train/test/oot)
        woe_result = build_woe_datasets(df_train, df_test, df_oot,
                                         target, var_list)

        _SERVER_STORE[f"{_pfx}_train_woe"] = woe_result["train_woe"]
        _SERVER_STORE[f"{_pfx}_test_woe"]  = woe_result["test_woe"]
        _SERVER_STORE[f"{_pfx}_oot_woe"]   = woe_result["oot_woe"]
        _SERVER_STORE[f"{key}_iv_{seg_col}_{seg_val}"]  = woe_result["iv_df"]
        _SERVER_STORE[f"{_pfx}_optb"]      = woe_result["optb_dict"]
        _SERVER_STORE[f"{_pfx}_bins"]      = woe_result["bins_dict"]
        _SERVER_STORE[f"{_pfx}_iv_tables"] = woe_result["iv_tables"]
        _SERVER_STORE[f"{_pfx}_failed"]    = woe_result["failed"]

        # ── WoE dağılım tabloları (train/test/oot) — tek seferlik ──────────────
        optb_dict = woe_result["optb_dict"]
        _iv_tables_raw = woe_result["iv_tables"]   # {col: optbinning bt DataFrame}
        iv_df = woe_result["iv_df"]
        woe_tables = {}

        def _format_bt(bt_raw):
            """Optbinning binning_table → uygulama formatına dönüştür."""
            data_rows = bt_raw[bt_raw["Bin"].astype(str).str.len() > 0]
            data_rows = data_rows[~data_rows["Bin"].isin(["Special", "Missing", "Totals"])]
            rows = []
            for _, r in data_rows.iterrows():
                total = int(r["Count"]); bad = int(r["Event"]); good = int(r["Non-event"])
                rows.append({
                    "Bin": str(r["Bin"]), "Toplam": total, "Bad": bad, "Good": good,
                    "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
                    "WOE": round(float(r["WoE"]), 4),
                    "IV Katkı": round(float(r["IV"]), 4),
                })
            for lbl in ["Special", "Missing"]:
                sr = bt_raw[bt_raw["Bin"] == lbl]
                if not sr.empty and int(sr.iloc[0]["Count"]) > 0:
                    r = sr.iloc[0]
                    rows.append({
                        "Bin": "Eksik" if lbl == "Missing" else "Special",
                        "Toplam": int(r["Count"]), "Bad": int(r["Event"]),
                        "Good": int(r["Non-event"]),
                        "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
                        "WOE": round(float(r["WoE"]), 4),
                        "IV Katkı": round(float(r["IV"]), 4),
                    })
            if not rows:
                return pd.DataFrame()
            result = pd.DataFrame(rows)
            t_n = result["Toplam"].sum(); t_b = result["Bad"].sum()
            total_row = pd.DataFrame([{
                "Bin": "TOPLAM", "Toplam": int(t_n), "Bad": int(t_b),
                "Good": int(result["Good"].sum()),
                "Bad Rate %": round(t_b / t_n * 100, 2) if t_n > 0 else 0.0,
                "WOE": "", "IV Katkı": round(float(bt_raw.loc["Totals", "IV"]), 4),
            }])
            return pd.concat([result, total_row], ignore_index=True)

        def _mono_check(bt):
            """Bad Rate % üzerinden monotonluk kontrol et (Eksik/Special/TOPLAM hariç)."""
            m = bt[~bt["Bin"].isin(["TOPLAM", "Eksik", "Special"])]
            nums = [float(w) for w in m["Bad Rate %"].dropna().tolist()
                    if isinstance(w, (int, float))]
            if len(nums) < 2:
                return "–"
            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            if all(d >= 0 for d in diffs):
                return "Artan ↑"
            if all(d <= 0 for d in diffs):
                return "Azalan ↓"
            return "Monoton Değil ✗"

        for var in var_list:
            _optb = optb_dict.get(var)
            if _optb is None:
                continue
            try:
                # Train tablosu: iv_tables'tan doğrudan formatla (tekrar build() yok)
                bt_raw = _iv_tables_raw.get(var)
                if bt_raw is None or bt_raw.empty:
                    continue
                bt_train = _format_bt(bt_raw)
                if bt_train.empty:
                    continue
                iv_train = float(bt_raw.loc["Totals", "IV"])

                entry = {
                    "train_table": bt_train.to_dict("records"),
                    "iv_train": round(iv_train, 4),
                    "monoton": _mono_check(bt_train),
                }

                # Test tablosu
                if df_test is not None and len(df_test) > 0:
                    try:
                        bt_test, iv_test, _, _ = get_woe_detail(
                            df_test, var, target, fitted_optb=_optb, use_edges=True)
                        if not bt_test.empty:
                            entry["test_table"] = bt_test.to_dict("records")
                            entry["iv_test"] = round(iv_test, 4)
                            entry["monoton_test"] = _mono_check(bt_test)
                    except Exception:
                        pass

                # OOT tablosu
                if df_oot is not None and len(df_oot) > 0:
                    try:
                        bt_oot, iv_oot, _, _ = get_woe_detail(
                            df_oot, var, target, fitted_optb=_optb, use_edges=True)
                        if not bt_oot.empty:
                            entry["oot_table"] = bt_oot.to_dict("records")
                            entry["iv_oot"] = round(iv_oot, 4)
                            entry["monoton_oot"] = _mono_check(bt_oot)
                    except Exception:
                        pass

                woe_tables[var] = entry
            except Exception:
                continue
        _SERVER_STORE[f"{_pfx}_woe_tables"] = woe_tables

        durations["iv_ranking"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["iv_ranking"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 3, "durations": dict(durations), "done": False}

    # ── Adım 3: Korelasyon (Train WoE, tüm değişkenler) ─────────────────────
    try:
        t0 = time.perf_counter()
        train_woe = _SERVER_STORE.get(f"{_pfx}_train_woe")
        if train_woe is not None and not train_woe.empty:
            num_cols_corr = list(train_woe.columns)[:30]
            corr_df = compute_correlation_matrix(train_woe, num_cols_corr)
            _SERVER_STORE[f"{key}_corr_{seg_col}_{seg_val}_precomp"] = (corr_df, num_cols_corr)
        durations["correlation"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["correlation"] = None
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 4, "durations": dict(durations), "done": False}

    # ── Adım 4: Değişken Özeti tablosu ───────────────────────────────────────
    try:
        t0 = time.perf_counter()
        from callbacks.var_summary import compute_var_summary_table, compute_var_summary_raw
        compute_var_summary_table(cfg, key, seg_col, seg_val)
        compute_var_summary_raw(cfg, key, seg_col, seg_val)
        durations["var_summary"] = round(time.perf_counter() - t0, 1)
    except Exception:
        durations["var_summary"] = None

    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 5, "durations": dict(durations), "done": True}


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
    State("dd-oot-date", "value"),
    State("dd-segment-col", "value"),
    State("dd-segment-val", "value"),
    State("chk-train-test-split", "value"),
    State("input-test-size", "value"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def confirm_config(n_clicks, target_col, date_col, oot_date, segment_col,
                   segment_val, train_test_val, test_size_cfg, key):
    _MODEL_RESET = (None, "", None, None, None, None, "")  # model-signal, pg-output, model-index, note, profile-loaded, dd-profile, profile-status
    no_modal = (dash.no_update,) * 6  # modal, interval, body, close-style, done-style
    if not target_col or target_col == "":
        return (dash.no_update, dbc.Alert(
            "Target kolonu zorunludur.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        ), *no_modal, *_MODEL_RESET)

    df_orig = _get_df(key)
    config = {
        "target_col":      target_col,
        "date_col":        date_col or None,
        "oot_date":        oot_date or None,
        "segment_col":     segment_col or None,
        "target_type":     "binary",
        "has_test_split":  bool(train_test_val),
        "test_size":       int(test_size_cfg or 20),
    }

    prog_key = f"{key}_precompute"
    precompute_state = {"prog_key": prog_key}

    parts = [html.Strong("✓ Onaylandı")]
    if date_col:
        parts += [f"  ·  Tarih: {date_col}"]
    if oot_date:
        parts += [f"  ·  OOT: ≥ {oot_date}"]
        if bool(train_test_val):
            parts += [f"  ·  Test: %{test_size_cfg or 20}"]
    if segment_col:
        parts += [f"  ·  Segment: {segment_col}"]

    # segment_val: kullanıcı seçtiyse onu kullan, yoksa None (tüm veri)
    seg_val = segment_val if segment_val and segment_val != ["Tümü"] and "Tümü" not in (segment_val if isinstance(segment_val, list) else [segment_val]) else None
    config["segment_val"] = seg_val

    # Eski model sonuçlarını temizle
    _SERVER_STORE.pop(f"{key}_model_results", None)

    # Background thread başlat — Dash thread'i bloklamaz
    t = threading.Thread(
        target=_run_precompute_background,
        args=(prog_key, key, target_col, date_col, segment_col or None, seg_val, config),
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
