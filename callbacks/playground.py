import logging

import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL, MATCH

log = logging.getLogger(__name__)
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import base64
import io
import shap
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import _PLOT_LAYOUT, _AXIS_STYLE


class SmLogitWrapper:
    """sm.Logit sonucunu sklearn predict_proba arayüzüne sarar (pickle uyumlu)."""
    def __init__(self, result):
        self._r      = result
        self.pvalues = result.pvalues
        self.params  = result.params
        non_const    = [k for k in result.params.index if k != "const"]
        self.coef_   = [result.params[non_const].values]

    def predict_proba(self, X):
        X_c = np.column_stack([np.ones(X.shape[0]), X])
        probs = self._r.predict(exog=X_c)
        return np.column_stack([1 - probs, probs])


# ── Playground: Kolon seçeneklerini doldur ────────────────────────────────────
@app.callback(
    Output("pg-x-col",     "options"),
    Output("pg-x-col",     "value"),
    Output("pg-y-col",     "options"),
    Output("pg-y-col",     "value"),
    Output("pg-y2-col",    "options"),
    Output("pg-y2-col",    "value"),
    Output("pg-color-col", "options"),
    Output("pg-color-col", "value"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
)
def populate_pg_cols(config, expert_excluded, key):
    df = _get_df(key)
    if df is None or not config:
        empty = []
        return empty, None, empty, None, empty, "", empty, ""
    # Tüm kolonlar dahil (tarih, target, segment dahil)
    all_cols   = list(df.columns)
    opts       = [{"label": c, "value": c} for c in all_cols]
    y2_opts    = [{"label": "—", "value": ""}] + opts
    color_opts = [{"label": "—", "value": ""}] + opts

    date_col   = config.get("date_col")
    target_col = config.get("target_col")
    numeric_cols = [c for c in all_cols
                    if pd.api.types.is_numeric_dtype(df[c])
                    and c != target_col and c != date_col]

    # Akıllı defaults: X=tarih, Y=ilk numerik değişken, Y2=target
    x_val  = date_col  if date_col  in all_cols else (all_cols[0] if all_cols else None)
    y_val  = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else x_val)
    y2_val = target_col if target_col in all_cols else ""
    return opts, x_val, opts, y_val, y2_opts, y2_val, color_opts, ""


# ── Playground: Target kolonu ve kesim tarihi doldur ──────────────────────────
@app.callback(
    Output("pg-target-col", "options"),
    Output("pg-target-col", "value", allow_duplicate=True),
    Output("pg-split-date", "options"),
    Output("pg-split-date", "value", allow_duplicate=True),
    Input("store-config", "data"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def populate_pg_model_params(config, key):
    empty = ([], None, [], None)
    if not config or not key:
        return empty
    target_col = config.get("target_col")
    date_col   = config.get("date_col")
    df = _get_df(key)

    # Target seçenekleri: numerik kolonlar
    if df is not None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        target_opts = [{"label": c, "value": c} for c in num_cols]
        target_val  = target_col if target_col in num_cols else (num_cols[0] if num_cols else None)
    else:
        target_opts = [{"label": target_col, "value": target_col}] if target_col else []
        target_val  = target_col

    # Kesim tarihi seçenekleri: tarih kolonundan aylık distinct değerler
    split_opts = []
    split_val  = None
    if df is not None and date_col and date_col in df.columns:
        raw_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        distinct  = sorted(raw_dates.dt.to_period("M").unique().astype(str))
        split_opts = [{"label": d, "value": d} for d in distinct]
        # Varsayılan: ortadaki ay (son %30 test)
        split_val  = distinct[int(len(distinct) * 0.7)] if distinct else None

    return target_opts, target_val, split_opts, split_val


# ── Playground: Grafik çiz ────────────────────────────────────────────────────
@app.callback(
    Output("pg-chart-output", "children"),
    Input("btn-pg-chart", "n_clicks"),
    State("pg-x-col",       "value"),
    State("pg-y-col",       "value"),
    State("pg-chart-type",  "value"),
    State("pg-agg",         "value"),
    State("pg-color-col",   "value"),
    State("pg-y2-col",      "value"),
    State("pg-time-unit",   "value"),
    State("store-key",      "data"),
    State("store-config",   "data"),
    prevent_initial_call=True,
)
def _render_pg_chart(n, x_col, y_col, chart_type, agg, color_col,
                     y2_col, time_unit, key, config):
    if not x_col or not key or not config:
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")
    df = apply_segment_filter(df_orig, seg_col, seg_val).copy()
    color = color_col if color_col else None
    y2    = y2_col    if y2_col    else None

    # ── Tarih gruplama: X tarih kolonu ise period'a dönüştür ─────────────────
    _DATE_UNIT_FMT = {"D": "%Y-%m-%d", "M": "%Y-%m", "Q": None, "Y": "%Y"}
    _DATE_UNIT_LABEL = {"D": "Gün", "M": "Ay", "Q": "Çeyrek", "Y": "Yıl"}
    x_is_date = False
    x_display_col = x_col   # gruplanmış kolon adı (df'ye eklenir)
    if x_col in df.columns:
        try:
            parsed = pd.to_datetime(df[x_col], errors="coerce")
            if parsed.notna().mean() > 0.5:          # çoğunluğu tarih
                unit = time_unit or "M"
                if unit == "Q":
                    df["__x_period__"] = parsed.dt.to_period("Q").astype(str)
                else:
                    df["__x_period__"] = parsed.dt.strftime(_DATE_UNIT_FMT[unit])
                x_display_col = "__x_period__"
                x_is_date = True
        except Exception as e:
            log.debug("Tarih period parse edilemedi: %s", e)

    def _agg_series(frame, grp_cols, col):
        is_numeric = pd.api.types.is_numeric_dtype(frame[col]) if col in frame.columns else False
        effective_agg = agg if (agg == "count" or is_numeric) else "count"
        if effective_agg == "count":
            res = frame.groupby(grp_cols, observed=True).size().reset_index(name=col)
        elif effective_agg == "sum":
            res = frame.groupby(grp_cols, observed=True)[col].sum().reset_index()
        else:
            res = frame.groupby(grp_cols, observed=True)[col].mean().reset_index()
        return res

    def _agg_bad_rate(frame, grp_cols, col):
        """Y2 için özel: kolon binary (0/1) ise bad rate (%), değilse mean."""
        is_bin = (pd.api.types.is_numeric_dtype(frame[col])
                  and frame[col].dropna().isin([0, 1]).all())
        if is_bin:
            res = frame.groupby(grp_cols, observed=True)[col].mean().reset_index()
            res[col] = (res[col] * 100).round(2)   # 0-100 ölçeği
            return res, True
        else:
            return _agg_series(frame, grp_cols, col), False

    time_note  = f" — {_DATE_UNIT_LABEL.get(time_unit or 'M', '')} bazında" if x_is_date else ""
    tick_angle = -35 if x_is_date else 0

    def _y2_trace(frame, x_c, col):
        """Y2 için sağ eksende çizgi izi döndür."""
        d, is_br = _agg_bad_rate(frame, [x_c], col)
        label    = f"{col} (Bad Rate %)" if is_br else col
        y_title  = "Bad Rate %" if is_br else col
        trace = go.Scatter(
            x=d[x_c], y=d[col],
            name=label, mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5),
        )
        return trace, y_title

    def _dual_layout(fig, y1_title, y2_title, title_txt):
        fig.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=title_txt, font=dict(color="#E8EAF0", size=13)),
            height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
        )
        fig.update_xaxes(**_AXIS_STYLE, tickangle=tick_angle)
        fig.update_yaxes(**_AXIS_STYLE, title_text=y1_title, secondary_y=False)
        fig.update_yaxes(**_AXIS_STYLE, title_text=y2_title, secondary_y=True)

    try:
        # ── Scatter / Histogram / Box: Y2 desteklenmiyor, tek eksen ──────────
        if chart_type == "scatter":
            fig = px.scatter(df, x=x_display_col, y=y_col, color=color, opacity=0.6,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_display_col, color=color, nbins=40,
                               color_discrete_sequence=px.colors.qualitative.Set2)
        elif chart_type == "box":
            fig = px.box(df, x=color or x_display_col, y=y_col,
                         color_discrete_sequence=px.colors.qualitative.Set2)

        # ── Bar+Line: Y1=bar(sol), Y2=çizgi(sağ) — her zaman ─────────────────
        elif chart_type == "bar_line":
            grp   = [x_display_col] + ([color] if color and not y2 else [])
            y1_df = _agg_series(df, grp, y_col)
            fig   = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=y1_df[x_display_col], y=y1_df[y_col],
                       name=y_col, marker_color="#4F8EF7", opacity=0.85),
                secondary_y=False)
            y2_title = ""
            if y2:
                tr, y2_title = _y2_trace(df, x_display_col, y2)
                fig.add_trace(tr, secondary_y=True)
            title_txt = f"{x_col}{time_note}  ·  {y_col}" + (f"  /  {y2}" if y2 else "")
            _dual_layout(fig, y_col, y2_title, title_txt)
            return dcc.Graph(figure=fig, config={"displayModeBar": False})

        # ── Bar veya Line: Y2 varsa → secondary Y axis, yoksa → tek eksen ────
        elif chart_type in ("bar", "line") and y2:
            y1_df = _agg_series(df, [x_display_col], y_col)
            fig   = make_subplots(specs=[[{"secondary_y": True}]])
            if chart_type == "bar":
                fig.add_trace(
                    go.Bar(x=y1_df[x_display_col], y=y1_df[y_col],
                           name=y_col, marker_color="#4F8EF7", opacity=0.85),
                    secondary_y=False)
            else:
                fig.add_trace(
                    go.Scatter(x=y1_df[x_display_col], y=y1_df[y_col],
                               name=y_col, mode="lines+markers",
                               line=dict(color="#4F8EF7", width=2),
                               marker=dict(size=5)),
                    secondary_y=False)
            tr, y2_title = _y2_trace(df, x_display_col, y2)
            fig.add_trace(tr, secondary_y=True)
            title_txt = f"{x_col}{time_note}  ·  {y_col}  /  {y2}"
            _dual_layout(fig, y_col, y2_title, title_txt)
            return dcc.Graph(figure=fig, config={"displayModeBar": False})

        # ── Bar veya Line: Y2 yok → renk/grup destekli tek eksen ─────────────
        else:
            grp_cols = [x_display_col] + ([color] if color else [])
            agg_df   = _agg_series(df, grp_cols, y_col)
            if chart_type == "bar":
                fig = px.bar(agg_df, x=x_display_col, y=y_col, color=color, barmode="group",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig = px.line(agg_df, x=x_display_col, y=y_col, color=color, markers=True,
                              color_discrete_sequence=px.colors.qualitative.Set2)

        x_label = x_col + (f" ({_DATE_UNIT_LABEL.get(time_unit or 'M','')})" if x_is_date else "")
        title   = x_label + (f"  ×  {y_col}" if chart_type not in ("histogram", "box") else "")
        fig.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=title, font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, tickangle=tick_angle),
            yaxis=dict(**_AXIS_STYLE),
            height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
    except Exception as e:
        log.error("Playground grafik oluşturulamadı: %s", e)
        return html.Div(f"Grafik oluşturulamadı: {e}", className="alert-info-custom")


# ── Playground: Değişken özeti önizleme (store-active-vars yansıması) ─────────

@app.callback(
    Output("pg-var-summary-preview", "children"),
    Input("store-config", "data"),
    Input("store-active-vars", "data"),
    Input("interval-precompute", "disabled"),
    State("store-key", "data"),
    State("store-expert-exclude", "data"),
)
def render_pg_var_summary_preview(config, active_vars, _precompute_done, key, expert_excluded):
    if not key or not config or not config.get("target_col"):
        return html.Div()
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")

    full_summary = _SERVER_STORE.get(f"{key}_varsummary_{seg_col}_{seg_val}")
    iv_df        = _SERVER_STORE.get(f"{key}_iv_{seg_col}_{seg_val}")

    if full_summary is not None:
        disp = full_summary.copy()
        source_note = None
    elif iv_df is not None:
        disp = iv_df[["Değişken", "IV", "Güç", "Eksik %"]].copy()
        source_note = html.Div(
            "Tam özet için 'Değişken Özeti' sekmesini açın.",
            className="form-hint",
            style={"padding": "0.3rem 0.5rem", "marginBottom": "0.4rem"})
    else:
        return html.Div(
            "Özet henüz hesaplanmadı.",
            className="form-hint", style={"padding": "0.5rem 0.75rem"})

    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        disp = disp[disp["Değişken"].isin(set(screen_result[0]))].copy()
    excluded = set(expert_excluded or [])
    disp = disp[~disp["Değişken"].isin(excluded)].copy()

    total_count = len(disp)

    # store-active-vars'a göre filtrele (Değişken Özeti tabındaki tiklere göre)
    if active_vars:
        active_set = set(active_vars)
        disp = disp[disp["Değişken"].isin(active_set)].copy()

    filtered_count = len(disp)

    cond = [
        {"if": {"filter_query": '{Güç} = "Güçlü"',       "column_id": "Güç"},   "color": "#10b981"},
        {"if": {"filter_query": '{Güç} = "Orta"',         "column_id": "Güç"},   "color": "#4F8EF7"},
        {"if": {"filter_query": '{Güç} = "Zayıf"',        "column_id": "Güç"},   "color": "#f59e0b"},
        {"if": {"filter_query": '{Güç} = "Çok Zayıf"',    "column_id": "Güç"},   "color": "#7e8fa4"},
        {"if": {"filter_query": '{Güç} = "Şüpheli"',      "column_id": "Güç"},   "color": "#ef4444"},
        {"if": {"filter_query": '{Öneri} = "✅ Tut"',      "column_id": "Öneri"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"',  "column_id": "Öneri"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',   "column_id": "Öneri"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Öneri} = "⚠️ İncele"',  "column_id": "Sebep"},  "color": "#f59e0b"},
        {"if": {"filter_query": '{Öneri} = "❌ Çıkar"',   "column_id": "Sebep"},  "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durum} = "Kritik Kayma"', "column_id": "PSI Durum"}, "color": "#ef4444"},
        {"if": {"filter_query": '{PSI Durum} = "Hafif Kayma"',  "column_id": "PSI Durum"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{PSI Durum} = "Stabil"',       "column_id": "PSI Durum"}, "color": "#10b981"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]
    tbl = dash_table.DataTable(
        data=disp.to_dict("records"),
        columns=[{"name": c, "id": c} for c in disp.columns],
        sort_action="native",
        filter_action="native",
        page_size=100,
        fixed_rows={"headers": True},
        style_table={"overflowX": "auto", "overflowY": "auto",
                     "maxHeight": "350px"},
        style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                      "fontWeight": "600", "fontSize": "0.7rem",
                      "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
        style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                    "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                    "padding": "4px 8px", "textAlign": "left"},
        style_data_conditional=cond,
    )
    count_note = html.Div(
        f"{filtered_count} / {total_count} değişken seçili",
        className="form-hint",
        style={"padding": "0.3rem 0.5rem", "marginTop": "0.3rem"})
    parts = []
    if source_note:
        parts.append(source_note)
    parts.append(tbl)
    parts.append(count_note)
    return html.Div(parts)


# ── Playground: Dropdown seçeneklerini doldur (store-active-vars'tan) ─────────
@app.callback(
    Output("pg-var-dropdown", "options"),
    Output("pg-source-container", "children"),
    Output("pg-source-count",     "children"),
    Output("pg-model-container",  "children"),
    Input("store-active-vars",    "data"),
    Input("store-config",         "data"),
)
def populate_pg_var_dropdown(active_vars, config):
    empty_div = html.Div()
    if not config or not config.get("target_col"):
        return [], empty_div, "", empty_div
    base = active_vars or []
    opts = [{"label": c, "value": c} for c in base]
    return opts, empty_div, "", empty_div


# ── Playground: Dropdown → Store senkronizasyonu ──────────────────────────────
@app.callback(
    Output("store-pg-model-vars", "data"),
    Output("pg-model-count",      "children"),
    Output("pg-source-search",    "value"),
    Input("pg-var-dropdown", "value"),
    prevent_initial_call=True,
)
def sync_dropdown_to_store(selected):
    selected = selected or []
    count_txt = f"{len(selected)} değişken" if selected else ""
    return selected, count_txt, ""


# ── Playground: Tümünü Ekle ──────────────────────────────────────────────────
@app.callback(
    Output("pg-var-dropdown", "value"),
    Input("btn-pg-add-all", "n_clicks"),
    State("pg-var-dropdown", "options"),
    prevent_initial_call=True,
)
def pg_add_all(_, options):
    if not options:
        return dash.no_update
    return [o["value"] for o in options]


# ── Playground: Tümünü Temizle ───────────────────────────────────────────────
@app.callback(
    Output("pg-var-dropdown", "value", allow_duplicate=True),
    Input("btn-pg-remove-all", "n_clicks"),
    prevent_initial_call=True,
)
def pg_remove_all(_):
    return []


# ── Playground: Katsayı tablosu checkbox → Dropdown deselect ─────────────────
@app.callback(
    Output("pg-var-dropdown", "value", allow_duplicate=True),
    Input({"type": "pg-coef-table", "index": ALL}, "selected_rows"),
    State({"type": "pg-coef-table", "index": ALL}, "data"),
    State({"type": "pg-coef-table", "index": ALL}, "id"),
    State("pg-var-dropdown", "value"),
    prevent_initial_call=True,
)
def update_vars_from_coef_table(all_sel_rows, all_data, all_ids, current_vars):
    triggered = dash.ctx.triggered_id
    if not triggered or not current_vars:
        return dash.no_update

    # Hangi tablo tetiklendi?
    sel_rows = data = None
    for sr, d, tid in zip(all_sel_rows, all_data, all_ids):
        if tid == triggered:
            sel_rows, data = sr, d
            break
    if sel_rows is None or data is None:
        return dash.no_update

    # Seçilmemiş satırların _orig_var değerleri → çıkarılacak değişkenler
    all_indices = set(range(len(data)))
    unchecked = all_indices - set(sel_rows)
    if not unchecked:
        return dash.no_update  # hepsi seçili, değişiklik yok

    remove_vars = {data[i]["_orig_var"] for i in unchecked
                   if "_orig_var" in data[i] and data[i]["_orig_var"] != "__const__"}
    if not remove_vars:
        return dash.no_update

    new_vars = [v for v in current_vars if v not in remove_vars]
    return new_vars


# ── Playground: P > 0.05 kaldır butonu ───────────────────────────────────────
@app.callback(
    Output({"type": "pg-coef-table", "index": MATCH}, "selected_rows"),
    Input({"type": "btn-pg-drop-pv", "index": MATCH}, "n_clicks"),
    State({"type": "pg-coef-table", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def drop_high_pvalue(_, data):
    if not data:
        return dash.no_update
    return [i for i, row in enumerate(data)
            if row.get("P-Value", 0) <= 0.05]


# ── Playground: Model dropdown'ı (sabit — binary classification) ──────────────
@app.callback(
    Output("pg-model-type", "options"),
    Output("pg-model-type", "value", allow_duplicate=True),
    Input("store-config", "data"),
    prevent_initial_call=True,
)
def update_model_type_options(config):
    if not config:
        return dash.no_update, dash.no_update
    options = [
        {"label": "Logistic Regression", "value": "lr"},
        {"label": "LightGBM",            "value": "lgbm"},
        {"label": "XGBoost",             "value": "xgb"},
    ]
    return options, "lr"


# ── Playground: C ve eşik kontrollerini gizle/göster ─────────────────────────
@app.callback(
    Output("pg-col-c-value",       "style"),
    Output("pg-col-threshold",     "style"),
    Output("pg-col-threshold-val", "style"),
    Input("store-config", "data"),
)
def toggle_classification_controls(config):
    return {}, {}, {}




# ── Null strateji seçenekleri ─────────────────────────────────────────────────
_NULL_OPTIONS_LR = [
    {"label": "Medyan",        "value": "median"},
    {"label": "Ortalama",      "value": "mean"},
    {"label": "Mod (En sık)",  "value": "mode"},
    {"label": "0 ile doldur",  "value": "zero"},
    {"label": "Modele alma",   "value": "reject"},
]

_NULL_OPTIONS_TREE = [
    {"label": "Olduğu gibi bırak (null)", "value": "keep"},
    {"label": "Medyan",                    "value": "median"},
    {"label": "Ortalama",                  "value": "mean"},
    {"label": "Mod (En sık)",              "value": "mode"},
    {"label": "0 ile doldur",              "value": "zero"},
    {"label": "Modele alma",               "value": "reject"},
]


def _build_null_review_ui(null_info, default_strategy, is_lr=True):
    """Null içeren değişkenleri accordion ile listeleyen inceleme paneli üretir."""
    options_lr   = _NULL_OPTIONS_LR
    options_tree = _NULL_OPTIONS_TREE
    options = options_lr if is_lr else options_tree
    default = "mean" if is_lr else "keep"
    default_label = "Ortalama" if is_lr else "Olduğu gibi bırak"

    rows = []
    for col, pct in null_info:
        rows.append(
            dbc.Row([
                dbc.Col(html.Span(col, style={"color": "#d1d5db", "fontWeight": "600"}),
                        width=4),
                dbc.Col(html.Span(f"%{pct:.1f}",
                        style={"color": "#f59e0b" if pct < 30 else "#ef4444",
                                "fontWeight": "600"}), width=2),
                dbc.Col(dbc.Select(
                    id={"type": "null-col-strategy", "col": col},
                    className="dark-select",
                    options=options,
                    value=default,
                    style={"maxWidth": "220px"},
                ), width=4),
            ], className="mb-2 align-items-center")
        )

    if is_lr:
        note = html.Div(
            f"LR null kabul etmez — varsayılan: {default_label}. "
            "Değiştirmek isterseniz aşağıdan seçip Uygula'ya basın.",
            style={"color": "#f59e0b", "fontSize": "0.72rem", "marginBottom": "0.6rem",
                   "fontStyle": "italic"})
    else:
        note = html.Div(
            f"Tree modeller null kabul eder — varsayılan: {default_label}. "
            "Değiştirmek isterseniz aşağıdan seçip Uygula'ya basın.",
            style={"color": "#94a3b8", "fontSize": "0.72rem", "marginBottom": "0.6rem",
                   "fontStyle": "italic"})

    accordion_body = html.Div([
        note,
        html.Div([
            dbc.Row([
                dbc.Col(html.Span("Değişken", style={"color": "#8892a4",
                        "fontSize": "0.72rem", "textTransform": "uppercase"}), width=4),
                dbc.Col(html.Span("Null %", style={"color": "#8892a4",
                        "fontSize": "0.72rem", "textTransform": "uppercase"}), width=2),
                dbc.Col(html.Span("Strateji", style={"color": "#8892a4",
                        "fontSize": "0.72rem", "textTransform": "uppercase"}), width=4),
            ], className="mb-2"),
            *rows,
        ], style={"backgroundColor": "#111827", "padding": "1rem",
                  "borderRadius": "8px", "border": "1px solid #1f2a3c"}),
        html.Div([
            dbc.Button("Uygula", id="btn-pg-null-confirm",
                       color="success", size="sm"),
        ], style={"marginTop": "0.75rem"}),
        html.Div(id="pg-null-status"),
    ])

    title = html.Span([
        html.Span(f"{len(null_info)} değişkende null tespit edildi",
                   style={"color": "#e2e8f0"}),
        html.Span(f"  —  varsayılan: {default_label}",
                   style={"color": "#8892a4", "fontSize": "0.78rem"}),
    ])

    return dbc.Accordion([
        dbc.AccordionItem(accordion_body, title=title, item_id="null-review"),
    ], start_collapsed=True, flush=True,
       style={"marginBottom": "0.75rem",
              "--bs-accordion-bg": "#0e1117",
              "--bs-accordion-active-bg": "#0e1117",
              "--bs-accordion-btn-bg": "#161d2e",
              "--bs-accordion-btn-active-color": "#e2e8f0",
              "--bs-accordion-btn-color": "#e2e8f0",
              "--bs-accordion-border-color": "#1f2a3c"})


def _apply_null_strategies(X, col_strategies):
    """Her kolona belirlenen strateji ile null doldurma uygular.
    reject kolonlarını X'ten düşürür, kalan null'ları doldurur.
    Returns: (X_filled, rejected_cols)
    """
    rejected = []
    for col, strat in col_strategies.items():
        if col not in X.columns or not X[col].isna().any():
            continue
        if strat == "reject":
            rejected.append(col)
            continue
        if strat == "keep":
            continue
        if strat == "mean":
            _m = X[col].mode()
            fill = X[col].mean() if pd.api.types.is_numeric_dtype(X[col]) \
                   else (_m.iloc[0] if len(_m) > 0 else 0)
        elif strat == "mode":
            _m = X[col].mode()
            fill = _m.iloc[0] if len(_m) > 0 else 0
        elif strat == "zero":
            fill = 0
        else:  # median
            _m = X[col].mode()
            fill = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) \
                   else (_m.iloc[0] if len(_m) > 0 else 0)
        X[col] = X[col].fillna(fill)
    if rejected:
        X = X.drop(columns=rejected)
    return X, rejected


def _run_model_pipeline(model_vars, key, config, model_type,
                        threshold_method, threshold_val,
                        pending_note, col_strategies, target_sel=None,
                        custom_params=None):
    """Model kurma pipeline'ı — null inceleme sonrası veya direkt çağrılır."""
    _no = dash.no_update
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div("Veri yüklenmemiş.", className="alert-info-custom"), _no

    target  = target_sel or config["target_col"]
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    if target not in df_active.columns:
        return html.Div(f"Target kolonu '{target}' veri setinde bulunamadı.",
                        className="alert-info-custom"), _no
    y_all = pd.to_numeric(df_active[target], errors='coerce')

    _pfx = f"{key}_ds_{seg_col}_{seg_val}"
    if f"{_pfx}_train" in _SERVER_STORE:
        _df_tr  = _SERVER_STORE[f"{_pfx}_train"]
        _df_te  = _SERVER_STORE.get(f"{_pfx}_test")
        _df_oot = _SERVER_STORE.get(f"{_pfx}_oot")
    else:
        _split_cfg = {**config, "test_size": int(config.get("test_size", 20))}
        _df_tr, _df_te, _df_oot = get_splits(df_active, _split_cfg)

    train_mask = df_active.index.isin(_df_tr.index)
    test_mask  = (df_active.index.isin(_df_te.index)
                  if _df_te is not None else np.zeros(len(df_active), dtype=bool))
    oot_mask   = (df_active.index.isin(_df_oot.index)
                  if _df_oot is not None else np.zeros(len(df_active), dtype=bool))

    n_tr, n_te, n_oot = train_mask.sum(), test_mask.sum(), oot_mask.sum()
    split_parts = [f"Train: {n_tr:,}"]
    if n_te  > 0: split_parts.append(f"Test: {n_te:,}")
    if n_oot > 0: split_parts.append(f"OOT: {n_oot:,}")
    split_info = "  /  ".join(split_parts)

    if n_tr < 30:
        return html.Div(f"Yetersiz train verisi: {n_tr:,} satır.",
                        className="alert-info-custom"), _no

    _DEFAULTS = {
        "lr":    {},
        "lgbm":  dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                      random_state=42, n_jobs=-1, verbose=-1),
        "xgb":   dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                      random_state=42, n_jobs=-1, eval_metric="auc"),
    }
    algo = model_type or "lr"
    if custom_params and algo in ("lgbm", "xgb"):
        _MODEL_PARAMS = {algo: {**_DEFAULTS[algo], **custom_params}}
    else:
        _MODEL_PARAMS = _DEFAULTS

    def _fit_and_render(X_df, disp_names, label, accent):
        """Modeli kur, (compact_html, results_dict, model_obj, scaler_obj) döndür."""
        X = X_df.copy()
        # Null doldurma — sadece ham model için (WoE'de missing bin'e düşer)
        if label == "Ham":
            X, _rejected = _apply_null_strategies(X, col_strategies)
            if len(X.columns) == 0:
                return (html.Div("Tüm değişkenler null nedeniyle çıkarıldı.",
                        className="alert-info-custom"), None, None, None)
        X_tr = X.iloc[train_mask].reset_index(drop=True)
        X_te = X.iloc[test_mask].reset_index(drop=True)
        y_tr = y_all.iloc[train_mask].reset_index(drop=True)
        y_te = y_all.iloc[test_mask].reset_index(drop=True)
        has_test = len(X_te) > 0
        has_oot  = oot_mask.any()
        if has_oot:
            X_oot = X.iloc[oot_mask].reset_index(drop=True)
            y_oot = y_all.iloc[oot_mask].reset_index(drop=True)
        else:
            X_oot = None
            y_oot = None
        if len(X_tr) == 0:
            return html.Div("Split sonrası boş küme oluştu.", className="alert-info-custom"), None, None, None

        is_tree = algo in ("lgbm", "xgb", "rf")
        _use_sm_logit = (algo == "lr")
        _is_woe = (label == "WoE")
        _skip_scale = is_tree or (_use_sm_logit and _is_woe)
        if not _skip_scale:
            scaler = StandardScaler()
            X_tr_s  = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_te_s  = pd.DataFrame(scaler.transform(X_te), columns=X_tr.columns, index=X_te.index) if has_test else pd.DataFrame()
            X_oot_s = pd.DataFrame(scaler.transform(X_oot), columns=X_tr.columns, index=X_oot.index) if has_oot and X_oot is not None else None
        else:
            X_tr_s  = X_tr
            X_te_s  = X_te  if has_test else pd.DataFrame()
            X_oot_s = X_oot if has_oot  else None

        lr_summary_text = None
        try:
            if _use_sm_logit:
                X_tr_const = sm.add_constant(X_tr_s, has_constant="add")
                sm_res = sm.Logit(y_tr, X_tr_const).fit(disp=0, method="bfgs")
                mdl = SmLogitWrapper(sm_res)
                try:
                    lr_summary_text = sm_res.summary().as_text()
                except Exception as e:
                    log.warning("LR summary alınamadı: %s", e)
                    lr_summary_text = None
            elif algo == "lgbm":
                _lgbm_p = dict(_MODEL_PARAMS["lgbm"])
                if "scale_pos_weight" not in _lgbm_p or _lgbm_p["scale_pos_weight"] in (None, 1):
                    _lgbm_p.pop("scale_pos_weight", None)
                mdl = lgb.LGBMClassifier(**_lgbm_p)
            elif algo == "xgb":
                _xgb_p = dict(_MODEL_PARAMS["xgb"])
                if "scale_pos_weight" not in _xgb_p or _xgb_p["scale_pos_weight"] in (None, 1):
                    _xgb_p["scale_pos_weight"] = float(
                        (y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
                mdl = xgb.XGBClassifier(**_xgb_p)
            else:
                raise ValueError(f"Bilinmeyen model tipi: {algo}")
            if not _use_sm_logit:
                mdl.fit(X_tr_s, y_tr)
        except Exception as e:
            log.exception("Model kurulamadı")
            return html.Div(f"Model kurulamadı: {e}", className="alert-info-custom"), None, None, None

        # ── Binary classification çıkışı ──────────────────────────────────────
        tr_prob  = mdl.predict_proba(X_tr_s)[:, 1]
        te_prob  = mdl.predict_proba(X_te_s)[:, 1]  if has_test else None
        oot_prob = mdl.predict_proba(X_oot_s)[:, 1] if has_oot and X_oot_s is not None else None

        # ── Eşik belirleme ────────────────────────────────────────────────────
        thr_method = threshold_method or "fixed"
        _ref_prob  = te_prob if te_prob is not None else tr_prob
        _ref_y     = y_te   if te_prob is not None else y_tr
        if thr_method == "f1":
            _thrs = np.linspace(0.01, 0.99, 99)
            _f1s  = [f1_score(_ref_y, (_ref_prob >= t).astype(int), zero_division=0)
                     for t in _thrs]
            opt_thr = float(_thrs[int(np.argmax(_f1s))])
            thr_label = f"F1 Maks. eşiği: {opt_thr:.2f}"
        elif thr_method == "ks":
            _fpr, _tpr, _thrs_roc = roc_curve(_ref_y, _ref_prob)
            opt_thr = float(_thrs_roc[int(np.argmax(_tpr - _fpr))])
            opt_thr = min(max(opt_thr, 0.01), 0.99)
            thr_label = f"KS noktası eşiği: {opt_thr:.2f}"
        elif thr_method == "custom":
            opt_thr = float(threshold_val or 0.5)
            thr_label = f"Özel eşik: {opt_thr:.2f}"
        else:
            opt_thr   = 0.5
            thr_label = "Eşik: 0.50 (sabit)"

        def _metrics(y_true, y_prob_arr, thr=opt_thr):
            y_pred_arr = (y_prob_arr >= thr).astype(int)
            auc_  = roc_auc_score(y_true, y_prob_arr)
            gini_ = 2 * auc_ - 1
            fpr_, tpr_, _ = roc_curve(y_true, y_prob_arr)
            ks_   = float(np.max(tpr_ - fpr_))
            f1_   = f1_score(y_true, y_pred_arr, zero_division=0)
            prec_ = precision_score(y_true, y_pred_arr, zero_division=0)
            rec_  = recall_score(y_true, y_pred_arr, zero_division=0)
            cm__  = confusion_matrix(y_true, y_pred_arr)
            return dict(auc=auc_, gini=gini_, ks=ks_, f1=f1_,
                        prec=prec_, rec=rec_, cm=cm__,
                        fpr=fpr_, tpr=tpr_, n=len(y_true))

        tr_m  = _metrics(y_tr,  tr_prob)
        te_m  = _metrics(y_te,  te_prob)  if te_prob  is not None else None
        oot_m = _metrics(y_oot, oot_prob) if oot_prob is not None else None

        # ── Önem tablosu verisini oluştur ─────────────────────────────────────
        if not is_tree:
            has_pvalues = _use_sm_logit and hasattr(mdl, "pvalues")
            coef_rows = []
            if has_pvalues:
                const_coef = float(mdl.params.get("const", 0.0))
                const_pv   = float(mdl.pvalues.get("const", np.nan))
                coef_rows.append({"Değişken": "const",
                                  "_orig_var": "__const__",
                                  "Katsayı": round(const_coef, 4),
                                  "P-Value": round(const_pv, 4)})
            # pvalues index'i const + x1,x2... olabilir (numpy input)
            # sıralı eşleştir: pvalues[1:] ↔ X.columns
            _pv_vals = list(mdl.pvalues.values)[1:] if has_pvalues else []
            for i, (c, v) in enumerate(zip(X.columns, mdl.coef_[0])):
                orig = c
                for mv in model_vars:
                    if c == mv or c.startswith(mv + "_"):
                        orig = mv
                        break
                row = {"Değişken": disp_names.get(c, c),
                       "_orig_var": orig,
                       "Katsayı":  round(float(v), 4)}
                if has_pvalues and i < len(_pv_vals):
                    row["P-Value"] = round(float(_pv_vals[i]), 4)
                coef_rows.append(row)
            # const üstte, geri kalan abs sıralı
            const_part = [r for r in coef_rows if r["_orig_var"] == "__const__"]
            var_part = sorted([r for r in coef_rows if r["_orig_var"] != "__const__"],
                              key=lambda r: abs(r["Katsayı"]), reverse=True)
            coef_rows = const_part + var_part
            # downstream için temiz (const dahil, _orig_var hariç)
            imp_records = [{k: v for k, v in r.items() if k != "_orig_var"} for r in coef_rows]
            var_rows_for_table = coef_rows  # _orig_var dahil — DataTable'da kullanılacak
            importance_type = "coef"
        else:
            raw_imp = mdl.feature_importances_
            total   = raw_imp.sum() or 1.0
            coef_rows = []
            for c, v in zip(X.columns, raw_imp):
                orig = c
                for mv in model_vars:
                    if c == mv or c.startswith(mv + "_"):
                        orig = mv
                        break
                coef_rows.append({
                    "Değişken": disp_names.get(c, c),
                    "_orig_var": orig,
                    "Önem (%)": round(float(v / total * 100), 2),
                    "Önem (ham)": round(float(v), 4),
                })
            coef_rows.sort(key=lambda r: r["Önem (%)"], reverse=True)
            imp_records = [{k: v for k, v in r.items() if k != "_orig_var"} for r in coef_rows]
            var_rows_for_table = coef_rows
            has_pvalues = False
            importance_type = "feature_importance"

        # ── SHAP Beeswarm → base64 PNG ────────────────────────────────────────
        shap_img_b64 = None
        if is_tree:
            try:
                _X_shap_df = X_te if has_test and len(X_te) > 0 else (X_oot if has_oot and X_oot is not None else X_tr)
                _shap_n    = len(_X_shap_df)
                _X_shap    = _X_shap_df.values
                explainer = shap.TreeExplainer(mdl)
                shap_vals = explainer.shap_values(_X_shap)
                if isinstance(shap_vals, list):
                    shap_arr = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
                else:
                    shap_arr = shap_vals

                feat_names_shap = [disp_names.get(c, c) for c in X.columns]
                top_n = min(20, shap_arr.shape[1])

                _BG = "#0e1117"
                _FG = "#c8cdd8"

                plt.close("all")
                shap.summary_plot(
                    shap_arr, _X_shap,
                    feature_names=feat_names_shap,
                    max_display=top_n, show=False,
                    plot_size=(9, max(4, top_n * 0.38)),
                )
                fig_mpl = plt.gcf()
                fig_mpl.patch.set_facecolor(_BG)
                ax_mpl  = fig_mpl.axes[0]
                ax_mpl.set_facecolor(_BG)
                ax_mpl.tick_params(colors=_FG, labelsize=9)
                ax_mpl.xaxis.label.set_color(_FG)
                ax_mpl.spines["bottom"].set_color("#2d3a4f")
                ax_mpl.spines["top"].set_visible(False)
                ax_mpl.spines["right"].set_visible(False)
                ax_mpl.spines["left"].set_visible(False)
                ax_mpl.axvline(0, color="#4a5568", linewidth=0.8, zorder=0)
                for cax in fig_mpl.axes[1:]:
                    cax.set_facecolor(_BG)
                    cax.tick_params(colors=_FG, labelsize=8)
                    cax.yaxis.label.set_color(_FG)

                buf = io.BytesIO()
                fig_mpl.savefig(buf, format="png", bbox_inches="tight",
                                facecolor=_BG, dpi=130)
                plt.close("all")
                buf.seek(0)
                shap_img_b64 = base64.b64encode(buf.read()).decode()
            except Exception as e:
                log.warning("SHAP grafiği oluşturulamadı: %s", e)
                shap_img_b64 = None

        # ── Sonuçları serialize et (Sonuç sekmesi için) ───────────────────────
        def _m_dict(m):
            if m is None:
                return None
            return {k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in m.items() if k not in ("cm", "fpr", "tpr")}

        def _cm_list(m):
            if m is None:
                return None
            return m["cm"].tolist()

        def _roc_dict(m):
            if m is None:
                return None
            return {"fpr": m["fpr"].tolist(), "tpr": m["tpr"].tolist()}

        results_dict = {
            "metrics": {
                "train": _m_dict(tr_m),
                "test":  _m_dict(te_m),
                "oot":   _m_dict(oot_m),
            },
            "confusion_matrices": {
                "train": _cm_list(tr_m),
                "test":  _cm_list(te_m),
                "oot":   _cm_list(oot_m),
            },
            "roc_data": {
                "train": _roc_dict(tr_m),
                "test":  _roc_dict(te_m),
                "oot":   _roc_dict(oot_m),
            },
            "probabilities": {
                "train": tr_prob.tolist(),
                "test":  te_prob.tolist() if te_prob is not None else None,
                "oot":   oot_prob.tolist() if oot_prob is not None else None,
            },
            "y_true": {
                "train": y_tr.tolist(),
                "test":  y_te.tolist() if has_test else None,
                "oot":   y_oot.tolist() if has_oot else None,
            },
            "importance_table": imp_records,
            "importance_type": importance_type,
            "lr_summary_text": lr_summary_text,
            "shap_img_b64": shap_img_b64,
            "accent": accent,
            "label": label,
            "thr_label": thr_label,
            "opt_thr": opt_thr,
        }

        # ── Compact HTML (Playground için sadeleştirilmiş) ────────────────────
        def _gc(g): return "#10b981" if g >= 0.4 else "#f59e0b" if g >= 0.2 else "#ef4444"

        gini_cards = []
        for m, title in [(tr_m, "Train"), (te_m, "Test"), (oot_m, "OOT")]:
            if m is None:
                continue
            gc = _gc(m["gini"])
            gini_cards.append(dbc.Col(html.Div([
                html.Div(f"{m['gini']:.4f}", style={"color": gc, "fontSize": "1.25rem",
                                                     "fontWeight": "700"}),
                html.Div(f"{title} Gini", style={"color": "#8892a4", "fontSize": "0.72rem"}),
            ], className="metric-card"), width=3))

        # Katsayı / importance tablosu (tüm değişkenler — checkbox'lu)
        _label_key = "woe" if label == "WoE" else "raw"
        _tbl_style_mini = dict(
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                          "fontWeight": "600", "fontSize": "0.7rem",
                          "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
            style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                        "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                        "padding": "5px 8px", "textAlign": "left"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}
            ],
        )
        mini_cols = [{"name": k, "id": k}
                     for k in var_rows_for_table[0].keys()
                     if k != "_orig_var"] if var_rows_for_table else []
        mini_tbl_title = "Katsayı Tablosu (özet)" if importance_type == "coef" else "Feature Importance (özet)"

        # selected_rows: const hariç tüm satırlar (const tiksiz görünür)
        _sel_rows = [i for i, r in enumerate(var_rows_for_table)
                     if r.get("_orig_var") != "__const__"]

        # P > 0.05 kaldır butonu (sadece LR + p-value varsa)
        _pv_btn_el = html.Div()
        if importance_type == "coef" and has_pvalues:
            _n_high_pv = sum(1 for r in var_rows_for_table
                             if r.get("P-Value", 0) > 0.05
                             and r.get("_orig_var") != "__const__")
            if _n_high_pv > 0:
                _pv_btn_el = html.Div(
                    html.Button(
                        f"P > 0.05 kaldır ({_n_high_pv})",
                        id={"type": "btn-pg-drop-pv", "index": _label_key},
                        className="btn-pv-drop",
                        n_clicks=0),
                    style={"marginBottom": "0.4rem"})

        compact_html = html.Div([
            html.Div(f"{split_info}   ·   {thr_label}",
                     style={"color": "#7e8fa4", "fontSize": "0.72rem",
                            "marginBottom": "0.6rem", "fontStyle": "italic"}),
            dbc.Row(gini_cards, className="g-2 mb-3"),
            html.P(mini_tbl_title, className="section-title",
                   style={"marginBottom": "0.4rem"}),
            _pv_btn_el,
            dash_table.DataTable(
                id={"type": "pg-coef-table", "index": _label_key},
                data=var_rows_for_table,
                columns=mini_cols,
                row_selectable="multi",
                selected_rows=_sel_rows,
                **_tbl_style_mini,
            ),
            html.Div([
                html.Span("ℹ ", style={"color": "#4F8EF7"}),
                html.Span("Detaylı sonuçlar için "),
                html.Strong("Sonuç", style={"color": "#4F8EF7"}),
                html.Span(" sekmesine bakın."),
            ], style={"color": "#6b7a99", "fontSize": "0.78rem", "marginTop": "0.75rem",
                      "padding": "0.5rem 0.75rem", "backgroundColor": "#0d1520",
                      "borderRadius": "6px", "border": "1px solid #1e2a3a"}),
        ])

        _scaler_obj = scaler if not _skip_scale else None
        return compact_html, results_dict, mdl, _scaler_obj

    # ── Ham veriyi null stratejilerine göre doldur ─────────────────────────────
    raw_cols = [v for v in model_vars if v in df_active.columns]
    df_raw_filled = df_active[raw_cols].copy()
    df_raw_filled, _rejected_cols = _apply_null_strategies(df_raw_filled, col_strategies)
    _raw_model_vars = [v for v in raw_cols if v not in _rejected_cols]

    # ── Ham model ─────────────────────────────────────────────────────────────
    X_raw    = pd.get_dummies(df_raw_filled[_raw_model_vars].copy(), drop_first=True)
    raw_disp = {c: c for c in X_raw.columns}
    raw_html, raw_results, raw_mdl, raw_scaler = _fit_and_render(X_raw, raw_disp, "Ham", "#4F8EF7")

    # ── WoE model — cache'ten oku ──────────────────────────────────────────────
    _train_woe = _SERVER_STORE.get(f"{_pfx}_train_woe")
    _test_woe  = _SERVER_STORE.get(f"{_pfx}_test_woe")
    _oot_woe   = _SERVER_STORE.get(f"{_pfx}_oot_woe")
    _opt_dict  = _SERVER_STORE.get(f"{_pfx}_optb", {})

    if _train_woe is None:
        return html.Div("WoE verileri henüz hesaplanmamış. Lütfen önce yapılandırmayı onaylayın.",
                        className="alert-info-custom"), dash.no_update

    woe_parts = [_train_woe]
    if _test_woe is not None:
        woe_parts.append(_test_woe)
    if _oot_woe is not None:
        woe_parts.append(_oot_woe)
    woe_df_enc = pd.concat(woe_parts, axis=0).reindex(df_active.index)

    woe_feat_cols = [v for v in model_vars if v in woe_df_enc.columns]
    woe_html = None
    woe_results = None
    if woe_feat_cols:
        X_woe     = woe_df_enc[woe_feat_cols].copy()
        woe_disp  = {v: v for v in model_vars}
        woe_html, woe_results, woe_mdl, woe_scaler = _fit_and_render(X_woe, woe_disp, "WoE", "#a78bfa")
        failed_woe = [v for v in model_vars if v not in woe_df_enc.columns]
        note_txt  = f"★ WoE — {len(woe_feat_cols)}/{len(model_vars)} değişken encode edildi"
        if failed_woe:
            note_txt += f"  |  encode edilemeyen: {', '.join(failed_woe)}"
        woe_note  = html.Div(note_txt,
            style={"color": "#a78bfa", "fontSize": "0.73rem", "marginBottom": "0.4rem"})
        woe_content = html.Div([woe_note, woe_html])
    else:
        woe_content = html.Div(
            f"WoE encode edilebilen değişken bulunamadı. "
            f"({len(model_vars)} değişken denendi: {', '.join(model_vars[:5])}{'...' if len(model_vars)>5 else ''})",
            className="alert-info-custom")

    # ── WoE dağılım verisi (bin tablosu + monotonluk) — cache'ten oku ───────
    woe_dist = None
    has_test_split = test_mask.any()
    has_oot_split  = oot_mask.any()
    if woe_feat_cols:
        _all_woe_tables = _SERVER_STORE.get(f"{_pfx}_woe_tables", {})
        woe_dist = {v: _all_woe_tables[v] for v in woe_feat_cols
                    if v in _all_woe_tables}

    # ── Korelasyon — Train üzerinden (WoE + ham ayrı) ────────────────────────
    woe_corr = None
    raw_corr = None
    if woe_feat_cols and len(woe_feat_cols) > 1:
        _woe_train = woe_df_enc.loc[train_mask, woe_feat_cols]
        _woe_corr_df = _woe_train.corr().round(4)
        woe_corr = _woe_corr_df.to_dict()
    _raw_corr_cols = [v for v in _raw_model_vars if v in df_raw_filled.columns]
    if len(_raw_corr_cols) > 1:
        _raw_train = df_raw_filled.loc[train_mask, _raw_corr_cols].select_dtypes(include="number")
        if len(_raw_train.columns) > 1:
            raw_corr = _raw_train.corr().round(4).to_dict()

    # ── PSI — TEK KAYNAK: var_summary cache'den oku ────────────────────────
    psi_data = None
    _psi_map = _SERVER_STORE.get(f"{_pfx}_psi_map", {})
    if _psi_map and woe_feat_cols:
        psi_data = [{"Değişken": vc, "PSI (OOT)": _psi_map[vc]}
                    for vc in woe_feat_cols if vc in _psi_map]

    raw_psi_data = None
    _raw_psi_map = _SERVER_STORE.get(f"{_pfx}_raw_psi_map", {})
    if _raw_psi_map and _raw_model_vars:
        raw_psi_data = [{"Değişken": rv, "PSI (OOT)": _raw_psi_map[rv]}
                        for rv in _raw_model_vars if rv in _raw_psi_map]

    # ── VIF — Train / Test / OOT ayrı, WoE üzerinden (tek çıktı) ─────────
    def _calc_vif(X_df):
        from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
        _X = X_df.values.astype(float)
        vif_list = []
        for j in range(_X.shape[1]):
            try:
                v = float(_vif(_X, j))
            except Exception as e:
                log.debug("VIF hesaplanamadı (kolon %d): %s", j, e)
                v = None
            vif_list.append(round(v, 2) if v is not None else None)
        return vif_list

    vif_data = None
    if woe_feat_cols and len(woe_feat_cols) >= 2:
        try:
            _woe_tr = woe_df_enc.loc[train_mask, woe_feat_cols].dropna()
            vif_tr = _calc_vif(_woe_tr)

            vif_te = None
            if test_mask.any():
                _woe_te = woe_df_enc.loc[test_mask, woe_feat_cols].dropna()
                if len(_woe_te) >= 2:
                    vif_te = _calc_vif(_woe_te)

            vif_oot = None
            if oot_mask.any():
                _woe_oot = woe_df_enc.loc[oot_mask, woe_feat_cols].dropna()
                if len(_woe_oot) >= 2:
                    vif_oot = _calc_vif(_woe_oot)

            vif_data = []
            for i, vc in enumerate(woe_feat_cols):
                row = {"Değişken": vc, "Train VIF": vif_tr[i]}
                if vif_te is not None:
                    row["Test VIF"] = vif_te[i]
                if vif_oot is not None:
                    row["OOT VIF"] = vif_oot[i]
                vif_data.append(row)
        except Exception as e:
            log.warning("WoE VIF tablosu oluşturulamadı: %s", e)
            vif_data = None

    # ── Raw VIF — Train / Test / OOT, ham değerler üzerinden ─────────────────
    raw_vif_data = None
    _num_raw_cols = list(df_raw_filled[_raw_model_vars].select_dtypes(include="number").columns)
    if len(_num_raw_cols) >= 2:
        try:
            _raw_tr = df_raw_filled.loc[train_mask, _num_raw_cols].dropna()
            _rvif_tr = _calc_vif(_raw_tr)

            _rvif_te = None
            if test_mask.any():
                _raw_te = df_raw_filled.loc[test_mask, _num_raw_cols].dropna()
                if len(_raw_te) >= 2:
                    _rvif_te = _calc_vif(_raw_te)

            _rvif_oot = None
            if oot_mask.any():
                _raw_oot = df_raw_filled.loc[oot_mask, _num_raw_cols].dropna()
                if len(_raw_oot) >= 2:
                    _rvif_oot = _calc_vif(_raw_oot)

            raw_vif_data = []
            for i, rc in enumerate(_num_raw_cols):
                row = {"Değişken": rc, "Train VIF": _rvif_tr[i]}
                if _rvif_te is not None:
                    row["Test VIF"] = _rvif_te[i]
                if _rvif_oot is not None:
                    row["OOT VIF"] = _rvif_oot[i]
                raw_vif_data.append(row)
        except Exception as e:
            log.warning("Raw VIF tablosu oluşturulamadı: %s", e)
            raw_vif_data = None

    # ── Describe — cache'den oku (precompute'da hesaplandı) ──────────────────
    describe_data = None
    _cached_profile = _SERVER_STORE.get(f"{key}_profile_{seg_col}_{seg_val}")
    if _cached_profile is not None:
        try:
            _desc_cols = [v for v in model_vars if v in _cached_profile["Kolon"].values]
            if _desc_cols:
                _desc_df = _cached_profile[_cached_profile["Kolon"].isin(_desc_cols)]
                describe_data = _desc_df.to_dict("records")
        except Exception as e:
            log.warning("Describe verisi oluşturulamadı: %s", e)
            describe_data = None

    # ── Sonuçları cache'e yaz ─────────────────────────────────────────────────
    cache_key = f"{key}_model_results"
    _first_results = raw_results or woe_results
    _thr_label = _first_results["thr_label"] if _first_results else ""
    _opt_thr   = _first_results["opt_thr"] if _first_results else 0.5
    _SERVER_STORE[cache_key] = {
        "algo": algo,
        "model_vars": list(model_vars),
        "split_info": split_info,
        "thr_label": _thr_label,
        "opt_thr": _opt_thr,
        "corr": woe_corr,
        "raw_corr": raw_corr,
        "woe_dist": woe_dist,
        "psi_data": psi_data,
        "raw_psi_data": raw_psi_data,
        "vif_data": vif_data,
        "raw_vif_data": raw_vif_data,
        "describe_data": describe_data,
        "model_note": pending_note or "",
        "tabs": {
            "raw": raw_results,
            "woe": woe_results,
        },
        "_models": {"raw": raw_mdl, "woe": woe_mdl if woe_feat_cols else None},
        "_scalers": {"raw": raw_scaler, "woe": woe_scaler if woe_feat_cols else None},
        "_opt_dict": {k: v for k, v in _opt_dict.items() if k in model_vars},
        "_split_masks": {
            "train": train_mask.tolist(),
            "test": test_mask.tolist(),
            "oot": oot_mask.tolist(),
        },
        "_target": target,
        "_seg_col": seg_col,
        "_seg_val": seg_val,
        "_date_col": config.get("date_col"),
    }

    pg_tabs = dbc.Tabs([
        dbc.Tab(raw_html,    label="Ham Değerler",        tab_id="res-raw",
                className="tab-content-area"),
        dbc.Tab(woe_content, label="WoE Dönüştürülmüş",  tab_id="res-woe",
                className="tab-content-area"),
    ], active_tab="res-raw")

    return pg_tabs, cache_key


# ── Playground: Değişken seçimi / model tipi değişince null inceleme paneli ──
@app.callback(
    Output("pg-null-review-panel", "children"),
    Output("store-pg-null-strategies", "data", allow_duplicate=True),
    Input("store-pg-model-vars", "data"),
    Input("pg-model-type",       "value"),
    State("store-key",           "data"),
    State("store-config",        "data"),
    prevent_initial_call=True,
)
def update_null_review_panel(model_vars, model_type, key, config):
    if not model_vars or not key or not config:
        return html.Div(), {}
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div(), {}

    seg_col   = config.get("segment_col")
    seg_val   = config.get("segment_val")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    raw_cols = [v for v in model_vars if v in df_active.columns]
    if not raw_cols:
        return html.Div(), {}

    null_pct = (df_active[raw_cols].isnull().sum() / len(df_active) * 100)
    null_cols_info = [(col, pct) for col, pct in null_pct.items() if pct > 0]

    if not null_cols_info:
        return html.Div(), {}

    null_cols_info.sort(key=lambda x: -x[1])
    is_lr = (model_type or "lr") == "lr"
    # Model tipi değişince store'u sıfırla — kullanıcı tekrar "Uygula" bassın
    return _build_null_review_ui(null_cols_info, "median", is_lr=is_lr), {}


# ── Playground: Null strateji "Uygula" → store'a yaz ────────────────────────
@app.callback(
    Output("store-pg-null-strategies", "data"),
    Output("pg-null-status", "children"),
    Input("btn-pg-null-confirm", "n_clicks"),
    State({"type": "null-col-strategy", "col": ALL}, "value"),
    State({"type": "null-col-strategy", "col": ALL}, "id"),
    prevent_initial_call=True,
)
def apply_null_strategies_to_store(_, strategy_values, strategy_ids):
    col_strategies = {}
    missing_selection = []
    for sid, val in zip(strategy_ids, strategy_values):
        if not val:
            missing_selection.append(sid["col"])
        else:
            col_strategies[sid["col"]] = val

    if missing_selection:
        warn_msg = html.Div([
            html.Span("⚠ ", style={"color": "#ef4444"}),
            html.Span("Strateji seçilmemiş değişkenler: ",
                       style={"color": "#e2e8f0", "fontWeight": "600"}),
            html.Span(", ".join(missing_selection), style={"color": "#f59e0b"}),
        ], style={"padding": "0.6rem 0.8rem", "backgroundColor": "#1a1020",
                  "borderRadius": "6px", "border": "1px solid #3a1e2a",
                  "marginBottom": "0.75rem"})
        return dash.no_update, warn_msg

    # Onay mesajı — per-column detay
    _strat_labels = {"median": "Medyan", "mean": "Ortalama", "mode": "Mod",
                     "zero": "0", "reject": "Modele alma", "keep": "Null bırak"}
    detail_items = []
    for col, strat in col_strategies.items():
        lbl = _strat_labels.get(strat, strat)
        clr = "#ef4444" if strat == "reject" else "#94a3b8"
        detail_items.append(
            html.Div([
                html.Span(col, style={"color": "#d1d5db", "fontWeight": "600"}),
                html.Span(f" → {lbl}", style={"color": clr}),
            ], style={"fontSize": "0.75rem"})
        )

    confirm_msg = html.Div([
        html.Div([
            html.Span("✓ ", style={"color": "#10b981"}),
            html.Span("Null stratejileri kaydedildi",
                       style={"color": "#e2e8f0", "fontWeight": "600"}),
        ], style={"marginBottom": "0.4rem"}),
        html.Div(detail_items, style={"display": "flex", "flexWrap": "wrap",
                                       "gap": "0.4rem 1.2rem"}),
    ], style={"padding": "0.6rem 0.8rem", "backgroundColor": "#0d1520",
              "borderRadius": "6px", "border": "1px solid #1e3a2a",
              "marginBottom": "0.75rem"})

    return col_strategies, confirm_msg


# ── Playground: Model Kur — store'daki stratejilerle model kur ───────────────
@app.callback(
    Output("pg-model-output", "children", allow_duplicate=True),
    Output("store-model-signal", "data", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Input("btn-pg-build", "n_clicks"),
    State("store-pg-model-vars",       "data"),
    State("chk-use-woe",               "value"),
    State("pg-test-size",              "value"),
    State("pg-null-strategy",          "value"),
    State("store-pg-null-strategies",  "data"),
    State("pg-model-type",             "value"),
    State("pg-threshold-method",       "value"),
    State("pg-threshold-val",          "value"),
    State("pg-target-col",             "value"),
    State("pg-split-method",           "value"),
    State("pg-split-date",             "value"),
    State("store-key",                 "data"),
    State("store-config",              "data"),
    State("store-pending-note",        "data"),
    # — LightGBM parametreleri —
    State("pg-lgbm-n-estimators",      "value"),
    State("pg-lgbm-learning-rate",     "value"),
    State("pg-lgbm-num-leaves",        "value"),
    State("pg-lgbm-max-depth",         "value"),
    State("pg-lgbm-min-child-samples", "value"),
    State("pg-lgbm-reg-alpha",         "value"),
    State("pg-lgbm-reg-lambda",        "value"),
    State("pg-lgbm-subsample",         "value"),
    State("pg-lgbm-colsample-bytree",  "value"),
    State("pg-lgbm-scale-pos-weight",  "value"),
    # — XGBoost parametreleri —
    State("pg-xgb-n-estimators",       "value"),
    State("pg-xgb-learning-rate",      "value"),
    State("pg-xgb-max-depth",          "value"),
    State("pg-xgb-min-child-weight",   "value"),
    State("pg-xgb-subsample",          "value"),
    State("pg-xgb-colsample-bytree",   "value"),
    State("pg-xgb-reg-alpha",          "value"),
    State("pg-xgb-reg-lambda",         "value"),
    State("pg-xgb-scale-pos-weight",   "value"),
    prevent_initial_call=True,
)
def build_pg_model(_, model_vars, use_woe, test_size_pct, default_null_strategy,
                   per_col_strategies, model_type,
                   threshold_method, threshold_val,
                   target_sel, split_method, split_date,
                   key, config, pending_note,
                   # lgbm params
                   lgbm_n_est, lgbm_lr, lgbm_leaves, lgbm_depth,
                   lgbm_min_child, lgbm_alpha, lgbm_lam,
                   lgbm_subsample, lgbm_colsample, lgbm_spw,
                   # xgb params
                   xgb_n_est, xgb_lr, xgb_depth,
                   xgb_min_child_w, xgb_subsample, xgb_colsample,
                   xgb_alpha, xgb_lam, xgb_spw):
    _no = dash.no_update
    if not model_vars or not key or not config:
        return html.Div("Model listesi boş veya konfigürasyon eksik.",
                        className="alert-info-custom"), _no, _no

    algo = model_type or "lr"
    is_lr = algo == "lr"
    col_strategies = per_col_strategies or {}

    # UI'daki parametre input'larından custom_params oluştur
    custom_params = None
    if algo == "lgbm":
        custom_params = {
            k: v for k, v in {
                "n_estimators": int(lgbm_n_est) if lgbm_n_est else None,
                "learning_rate": float(lgbm_lr) if lgbm_lr else None,
                "num_leaves": int(lgbm_leaves) if lgbm_leaves else None,
                "max_depth": int(lgbm_depth) if lgbm_depth is not None else None,
                "min_child_samples": int(lgbm_min_child) if lgbm_min_child else None,
                "reg_alpha": float(lgbm_alpha) if lgbm_alpha is not None else None,
                "reg_lambda": float(lgbm_lam) if lgbm_lam is not None else None,
                "subsample": float(lgbm_subsample) if lgbm_subsample else None,
                "subsample_freq": 1 if lgbm_subsample and float(lgbm_subsample) < 1.0 else 0,
                "colsample_bytree": float(lgbm_colsample) if lgbm_colsample else None,
                "scale_pos_weight": float(lgbm_spw) if lgbm_spw else None,
            }.items() if v is not None
        }
    elif algo == "xgb":
        custom_params = {
            k: v for k, v in {
                "n_estimators": int(xgb_n_est) if xgb_n_est else None,
                "learning_rate": float(xgb_lr) if xgb_lr else None,
                "max_depth": int(xgb_depth) if xgb_depth else None,
                "min_child_weight": int(xgb_min_child_w) if xgb_min_child_w else None,
                "subsample": float(xgb_subsample) if xgb_subsample else None,
                "colsample_bytree": float(xgb_colsample) if xgb_colsample else None,
                "reg_alpha": float(xgb_alpha) if xgb_alpha is not None else None,
                "reg_lambda": float(xgb_lam) if xgb_lam is not None else None,
                "scale_pos_weight": float(xgb_spw) if xgb_spw else None,
            }.items() if v is not None
        }

    # Null var mı kontrol et
    df_orig = _get_df(key)
    _has_nulls = False
    if df_orig is not None:
        seg_col   = config.get("segment_col")
        seg_val   = config.get("segment_val")
        df_active = apply_segment_filter(df_orig, seg_col, seg_val)
        raw_cols  = [v for v in model_vars if v in df_active.columns]
        _has_nulls = any(df_active[c].isna().any() for c in raw_cols)

    if _has_nulls and not col_strategies:
        # Store boş — default stratejileri otomatik ata
        _default = "mean" if is_lr else "keep"
        for c in raw_cols:
            if df_active[c].isna().any():
                col_strategies[c] = _default

    result = _run_model_pipeline(
        model_vars, key, config, model_type,
        threshold_method, threshold_val,
        pending_note, col_strategies=col_strategies,
        target_sel=target_sel,
        custom_params=custom_params,
    )
    return result[0], result[1], None


# ── Model Parametreleri: accordion göster/gizle ─────────────────────────────
@app.callback(
    Output("pg-param-panel", "style"),
    Output("pg-params-lgbm", "style"),
    Output("pg-params-xgb",  "style"),
    Input("pg-model-type", "value"),
)
def toggle_param_panel(model_type):
    algo = model_type or "lr"
    if algo == "lr":
        return {"display": "none"}, {"display": "none"}, {"display": "none"}
    lgbm_vis = {} if algo == "lgbm" else {"display": "none"}
    xgb_vis  = {} if algo == "xgb"  else {"display": "none"}
    return {}, lgbm_vis, xgb_vis


# ── Model Parametreleri: Varsayılanlara Dön ──────────────────────────────────
@app.callback(
    # LightGBM outputs
    Output("pg-lgbm-n-estimators",     "value", allow_duplicate=True),
    Output("pg-lgbm-learning-rate",    "value", allow_duplicate=True),
    Output("pg-lgbm-num-leaves",       "value", allow_duplicate=True),
    Output("pg-lgbm-max-depth",        "value", allow_duplicate=True),
    Output("pg-lgbm-min-child-samples","value", allow_duplicate=True),
    Output("pg-lgbm-reg-alpha",        "value", allow_duplicate=True),
    Output("pg-lgbm-reg-lambda",       "value", allow_duplicate=True),
    Output("pg-lgbm-subsample",        "value", allow_duplicate=True),
    Output("pg-lgbm-colsample-bytree", "value", allow_duplicate=True),
    Output("pg-lgbm-scale-pos-weight", "value", allow_duplicate=True),
    # XGBoost outputs
    Output("pg-xgb-n-estimators",      "value", allow_duplicate=True),
    Output("pg-xgb-learning-rate",     "value", allow_duplicate=True),
    Output("pg-xgb-max-depth",         "value", allow_duplicate=True),
    Output("pg-xgb-min-child-weight",  "value", allow_duplicate=True),
    Output("pg-xgb-subsample",         "value", allow_duplicate=True),
    Output("pg-xgb-colsample-bytree",  "value", allow_duplicate=True),
    Output("pg-xgb-reg-alpha",         "value", allow_duplicate=True),
    Output("pg-xgb-reg-lambda",        "value", allow_duplicate=True),
    Output("pg-xgb-scale-pos-weight",  "value", allow_duplicate=True),
    Input("btn-pg-params-reset",       "n_clicks"),
    prevent_initial_call=True,
)
def reset_params_to_defaults(_):
    return (
        # LightGBM defaults
        200, 0.05, 31, -1, 20, 0, 0, 1.0, 1.0, 1,
        # XGBoost defaults
        200, 0.05, 6, 1, 1.0, 1.0, 0, 1, 1,
    )


# ── Optuna: Çalıştır ────────────────────────────────────────────────────────
from modules.optuna_tuner import run_optuna as _run_optuna

_OPTUNA_PROGRESS: dict = {}
_OPTUNA_CANCEL:   dict = {}


def _optuna_thread(key, X_tr, y_tr, X_te, y_te, model_type, n_trials, cancel_event):
    progress = _OPTUNA_PROGRESS.setdefault(key, {})
    try:
        _run_optuna(
            X_tr, y_tr, X_te, y_te,
            model_type=model_type,
            n_trials=n_trials,
            cancel_event=cancel_event,
            progress_dict=progress,
        )
    except Exception as exc:
        log.exception("Optuna pipeline hatası")
        progress["done"] = True
        progress["error"] = str(exc)


@app.callback(
    Output("interval-pg-optuna",  "disabled"),
    Output("pg-optuna-progress",  "children"),
    Output("pg-optuna-result",    "children"),
    Input("btn-pg-optuna",        "n_clicks"),
    State("pg-model-type",        "value"),
    State("pg-optuna-trials",     "value"),
    State("store-pg-model-vars",  "data"),
    State("store-key",            "data"),
    State("store-config",         "data"),
    State("pg-target-col",        "value"),
    State("store-pg-null-strategies", "data"),
    prevent_initial_call=True,
)
def start_optuna(_, model_type, n_trials, model_vars, key, config,
                 target_sel, col_strategies):
    if not model_vars or not key or not config:
        return True, html.Div("Model değişkenleri seçilmemiş.",
                              style={"color": "#ef4444"}), ""
    algo = model_type or "lr"
    if algo == "lr":
        return True, html.Div("Optuna sadece LightGBM ve XGBoost için kullanılabilir.",
                              style={"color": "#ef4444"}), ""

    n_trials = int(n_trials or 50)
    target = target_sel or config["target_col"]
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")

    df_orig = _get_df(key)
    if df_orig is None:
        return True, html.Div("Veri bulunamadı.", style={"color": "#ef4444"}), ""

    df_active = apply_segment_filter(df_orig, seg_col, seg_val)
    y_all = pd.to_numeric(df_active[target], errors="coerce")

    _pfx = f"{key}_ds_{seg_col}_{seg_val}"
    if f"{_pfx}_train" in _SERVER_STORE:
        _df_tr = _SERVER_STORE[f"{_pfx}_train"]
        _df_te = _SERVER_STORE.get(f"{_pfx}_test")
    else:
        _split_cfg = {**config, "test_size": int(config.get("test_size", 20))}
        _df_tr, _df_te, _ = get_splits(df_active, _split_cfg)

    train_mask = df_active.index.isin(_df_tr.index)
    test_mask = (df_active.index.isin(_df_te.index)
                 if _df_te is not None else np.zeros(len(df_active), dtype=bool))

    raw_cols = [v for v in model_vars if v in df_active.columns]
    X = df_active[raw_cols]

    # Null stratejileri uygula (ham veri için)
    _strats = col_strategies or {}
    if _strats:
        X, _ = _apply_null_strategies(X.copy(), _strats)

    X_tr = X.iloc[train_mask].reset_index(drop=True)
    X_te = X.iloc[test_mask].reset_index(drop=True)
    y_tr = y_all.iloc[train_mask].reset_index(drop=True)
    y_te = y_all.iloc[test_mask].reset_index(drop=True)

    if len(X_te) == 0:
        return True, html.Div("Test verisi yok — Optuna çalıştırılamaz.",
                              style={"color": "#ef4444"}), ""

    # Önceki çalışmayı iptal et
    old_cancel = _OPTUNA_CANCEL.get(key)
    if old_cancel:
        old_cancel.set()

    cancel_event = threading.Event()
    _OPTUNA_CANCEL[key] = cancel_event
    _OPTUNA_PROGRESS[key] = {"trial": 0, "total": n_trials, "done": False}

    t = threading.Thread(
        target=_optuna_thread,
        args=(key, X_tr, y_tr, X_te, y_te, algo, n_trials, cancel_event),
        daemon=True,
    )
    t.start()

    return (
        False,  # interval enabled
        html.Div([
            html.Span("⏳ Optuna başlatılıyor...",
                       style={"color": "#f59e0b", "fontSize": "0.82rem"}),
        ]),
        "",  # result area empty
    )


# ── Optuna: Progress tick ───────────────────────────────────────────────────
@app.callback(
    Output("pg-optuna-progress", "children", allow_duplicate=True),
    Output("pg-optuna-result",   "children", allow_duplicate=True),
    Output("interval-pg-optuna", "disabled", allow_duplicate=True),
    Output("store-pg-optuna-result", "data"),
    Input("interval-pg-optuna",  "n_intervals"),
    State("store-key",           "data"),
    prevent_initial_call=True,
)
def optuna_progress_tick(_, key):
    prog = _OPTUNA_PROGRESS.get(key or "", {})
    if not prog:
        return dash.no_update, dash.no_update, True, dash.no_update

    trial = prog.get("trial", 0)
    total = prog.get("total", 1)
    best  = prog.get("best_score")
    done  = prog.get("done", False)

    pct = int(trial / max(total, 1) * 100)
    best_txt = f"  (en iyi: {best:.4f})" if best is not None else ""

    progress_bar = html.Div([
        html.Div(
            style={"width": f"{pct}%", "height": "6px",
                   "backgroundColor": "#4F8EF7", "borderRadius": "3px",
                   "transition": "width 0.3s ease"},
            className="optuna-bar-fill",
        ),
    ], style={"backgroundColor": "#1a1f2e", "borderRadius": "3px",
              "height": "6px", "marginBottom": "0.3rem"})

    progress_ui = html.Div([
        progress_bar,
        html.Span(f"{trial}/{total} trial{best_txt}",
                   style={"color": "#94a3b8", "fontSize": "0.78rem"}),
    ])

    if not done:
        return progress_ui, dash.no_update, False, dash.no_update

    # Tamamlandı veya iptal edildi
    error = prog.get("error")
    if error:
        return (
            html.Div(f"Hata: {error}", style={"color": "#ef4444", "fontSize": "0.82rem"}),
            "", True, dash.no_update,
        )

    result = prog.get("result", {})
    if not result or not result.get("best_params"):
        return (
            html.Div("Optuna tamamlandı ancak sonuç üretilemedi.",
                      style={"color": "#f59e0b", "fontSize": "0.82rem"}),
            "", True, dash.no_update,
        )

    bp = result["best_params"]
    tr_g = result.get("best_train_gini")
    te_g = result.get("best_test_gini")
    gap  = result.get("best_gap")
    n_done = result.get("n_trials_completed", 0)

    # Sonuç kartı
    param_items = [
        html.Span(f"{k}: {v}", style={"color": "#d1d5db", "fontSize": "0.75rem",
                                       "marginRight": "0.8rem"})
        for k, v in bp.items()
        if k not in ("random_state", "n_jobs", "verbose", "verbosity",
                      "eval_metric", "subsample_freq")
    ]

    result_ui = html.Div([
        html.Div([
            html.Span("✓ ", style={"color": "#10b981"}),
            html.Span(f"Tamamlandı — {n_done} trial",
                       style={"color": "#e2e8f0", "fontWeight": "600",
                              "fontSize": "0.82rem"}),
        ], style={"marginBottom": "0.4rem"}),
        html.Div([
            html.Span(f"Test Gini: {te_g:.4f}" if te_g else "",
                       style={"color": "#4F8EF7", "fontWeight": "600",
                              "marginRight": "1rem", "fontSize": "0.82rem"}),
            html.Span(f"Train Gini: {tr_g:.4f}" if tr_g else "",
                       style={"color": "#94a3b8", "marginRight": "1rem",
                              "fontSize": "0.82rem"}),
            html.Span(f"Fark: {gap:.4f}" if gap is not None else "",
                       style={"color": "#f59e0b" if gap and gap > 0.05 else "#10b981",
                              "fontSize": "0.82rem"}),
        ], style={"marginBottom": "0.4rem"}),
        html.Div(param_items, style={"display": "flex", "flexWrap": "wrap",
                                      "marginBottom": "0.5rem"}),
        dbc.Button("Parametreleri Kabul Et", id="btn-pg-optuna-accept",
                   color="success", size="sm", outline=True),
    ], style={"padding": "0.6rem 0.8rem", "backgroundColor": "#0d1520",
              "borderRadius": "6px", "border": "1px solid #1e3a2a"})

    done_progress = html.Div([
        progress_bar,
        html.Span(f"{n_done}/{total} trial tamamlandı",
                   style={"color": "#10b981", "fontSize": "0.78rem"}),
    ])

    return done_progress, result_ui, True, result


# ── Optuna: Parametreleri Kabul Et ──────────────────────────────────────────
@app.callback(
    # LightGBM outputs
    Output("pg-lgbm-n-estimators",     "value"),
    Output("pg-lgbm-learning-rate",    "value"),
    Output("pg-lgbm-num-leaves",       "value"),
    Output("pg-lgbm-max-depth",        "value"),
    Output("pg-lgbm-min-child-samples","value"),
    Output("pg-lgbm-reg-alpha",        "value"),
    Output("pg-lgbm-reg-lambda",       "value"),
    Output("pg-lgbm-subsample",        "value"),
    Output("pg-lgbm-colsample-bytree", "value"),
    Output("pg-lgbm-scale-pos-weight", "value"),
    # XGBoost outputs
    Output("pg-xgb-n-estimators",      "value"),
    Output("pg-xgb-learning-rate",     "value"),
    Output("pg-xgb-max-depth",         "value"),
    Output("pg-xgb-min-child-weight",  "value"),
    Output("pg-xgb-subsample",         "value"),
    Output("pg-xgb-colsample-bytree",  "value"),
    Output("pg-xgb-reg-alpha",         "value"),
    Output("pg-xgb-reg-lambda",        "value"),
    Output("pg-xgb-scale-pos-weight",  "value"),
    Input("btn-pg-optuna-accept",      "n_clicks"),
    State("store-pg-optuna-result",    "data"),
    State("pg-model-type",             "value"),
    prevent_initial_call=True,
)
def accept_optuna_params(_, result, model_type):
    _nu = dash.no_update
    lgbm_defaults = [_nu] * 10
    xgb_defaults  = [_nu] * 9

    if not result or not result.get("best_params"):
        return *lgbm_defaults, *xgb_defaults

    bp = result["best_params"]
    algo = model_type or "lr"

    if algo == "lgbm":
        lgbm_vals = [
            bp.get("n_estimators", _nu),
            bp.get("learning_rate", _nu),
            bp.get("num_leaves", _nu),
            bp.get("max_depth", _nu),
            bp.get("min_child_samples", _nu),
            bp.get("reg_alpha", _nu),
            bp.get("reg_lambda", _nu),
            bp.get("subsample", _nu),
            bp.get("colsample_bytree", _nu),
            bp.get("scale_pos_weight", _nu),
        ]
        return *lgbm_vals, *xgb_defaults
    elif algo == "xgb":
        xgb_vals = [
            bp.get("n_estimators", _nu),
            bp.get("learning_rate", _nu),
            bp.get("max_depth", _nu),
            bp.get("min_child_weight", _nu),
            bp.get("subsample", _nu),
            bp.get("colsample_bytree", _nu),
            bp.get("reg_alpha", _nu),
            bp.get("reg_lambda", _nu),
            bp.get("scale_pos_weight", _nu),
        ]
        return *lgbm_defaults, *xgb_vals

    return *lgbm_defaults, *xgb_defaults


# ── Optuna: İptal ───────────────────────────────────────────────────────────
@app.callback(
    Output("pg-optuna-progress", "children", allow_duplicate=True),
    Output("pg-optuna-result",   "children", allow_duplicate=True),
    Input("btn-pg-optuna-cancel", "n_clicks"),
    State("store-key", "data"),
    prevent_initial_call=True,
)
def cancel_optuna(_, key):
    prog = _OPTUNA_PROGRESS.get(key or "", {})
    is_done = prog.get("done", False)

    if is_done:
        # Zaten bitmiş — sonuçları temizle
        _OPTUNA_PROGRESS.pop(key, None)
        _OPTUNA_CANCEL.pop(key, None)
        return "", ""

    # Çalışıyor — iptal et
    cancel_ev = _OPTUNA_CANCEL.get(key)
    if cancel_ev:
        cancel_ev.set()
    return (
        html.Div("İptal ediliyor...",
                  style={"color": "#f59e0b", "fontSize": "0.82rem"}),
        dash.no_update,
    )
