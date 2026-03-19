import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import base64
import io
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import _PLOT_LAYOUT, _AXIS_STYLE, _build_woe_dataset


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
    Output("pg-target-col", "value"),
    Output("pg-split-date", "options"),
    Output("pg-split-date", "value"),
    Input("store-config", "data"),
    State("store-key", "data"),
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
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def _render_pg_chart(n, x_col, y_col, chart_type, agg, color_col,
                     y2_col, time_unit, key, config, seg_val, seg_col_input):
    if not x_col or not key or not config:
        return html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()
    seg_col = config.get("segment_col") or (seg_col_input or None)
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
        except Exception:
            pass

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
        return html.Div(f"Grafik oluşturulamadı: {e}", className="alert-info-custom")


# ── Playground: Değişken özeti önizleme ───────────────────────────────────────
@app.callback(
    Output("pg-var-summary-preview", "children"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    Input("main-tabs", "active_tab"),
    State("store-key", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
)
def render_pg_var_summary_preview(config, expert_excluded, active_tab, key, seg_val, seg_col_input):
    if not key or not config or not config.get("target_col"):
        return html.Div()
    seg_col = config.get("segment_col") or (seg_col_input or None)

    # Önce tam özet cache'ine bak (Değişken Özeti → Hesapla sonrası dolu olur)
    full_summary = _SERVER_STORE.get(f"{key}_summary_{seg_col}_{seg_val}")
    iv_df        = _SERVER_STORE.get(f"{key}_iv_{seg_col}_{seg_val}")

    if full_summary is not None:
        disp = full_summary.copy()
        source_note = None
    elif iv_df is not None:
        # Sadece IV varsa — sınırlı görünüm
        disp = iv_df[["Değişken", "IV", "Güç", "Eksik %"]].copy()
        source_note = html.Div(
            "Tam özet için 'Değişken Özeti' sekmesinde 'Hesapla' butonuna basın "
            "(PSI · Korelasyon · VIF · Öneri sütunları eklenecek).",
            className="form-hint",
            style={"padding": "0.3rem 0.5rem", "marginBottom": "0.4rem"})
    else:
        return html.Div(
            "Özet henüz hesaplanmadı. Önce 'Target & IV' veya 'Değişken Özeti' sekmesini açın.",
            className="form-hint", style={"padding": "0.5rem 0.75rem"})

    excluded = set(expert_excluded or [])
    disp = disp[~disp["Değişken"].isin(excluded)].copy()

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
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                      "fontWeight": "600", "fontSize": "0.7rem",
                      "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
        style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                    "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                    "padding": "4px 8px", "textAlign": "left"},
        style_data_conditional=cond,
    )
    return html.Div([source_note, tbl] if source_note else [tbl])


# ── Playground: Dropdown seçeneklerini doldur ─────────────────────────────────
@app.callback(
    Output("pg-var-dropdown", "options"),
    Output("pg-source-container", "children"),
    Output("pg-source-count",     "children"),
    Output("pg-model-container",  "children"),
    Input("store-config",         "data"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
)
def populate_pg_var_dropdown(config, expert_excluded, key):
    df = _get_df(key)
    empty_div = html.Div()
    if df is None or not config or not config.get("target_col"):
        return [], empty_div, "", empty_div
    excluded = set(expert_excluded or [])
    screen_result = _SERVER_STORE.get(f"{key}_screen")
    if screen_result:
        base = [c for c in screen_result[0]
                if c != config["target_col"] and c not in excluded]
    else:
        cfg = {c for c in [config.get("target_col"), config.get("date_col"),
                            config.get("segment_col")] if c}
        base = [c for c in df.columns if c not in cfg and c not in excluded]
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


# ── Playground: Model dropdown'ı target_type'a göre güncelle ─────────────────
@app.callback(
    Output("pg-model-type", "options"),
    Output("pg-model-type", "value"),
    Input("store-config", "data"),
)
def update_model_type_options(config):
    if not config:
        return dash.no_update, dash.no_update
    target_type = config.get("target_type", "binary")
    if target_type == "continuous":
        options = [
            {"label": "Ridge Regression",       "value": "ridge"},
            {"label": "LightGBM Regressor",     "value": "lgbm"},
            {"label": "XGBoost Regressor",      "value": "xgb"},
            {"label": "Random Forest Regressor","value": "rf"},
        ]
        default = "ridge"
    else:
        options = [
            {"label": "Logistic Regression", "value": "lr"},
            {"label": "LightGBM",            "value": "lgbm"},
            {"label": "XGBoost",             "value": "xgb"},
            {"label": "Random Forest",       "value": "rf"},
        ]
        default = "lr"
    return options, default


# ── Playground: C ve eşik kontrollerini gizle/göster ─────────────────────────
@app.callback(
    Output("pg-col-c-value",       "style"),
    Output("pg-col-threshold",     "style"),
    Output("pg-col-threshold-val", "style"),
    Input("store-config", "data"),
)
def toggle_classification_controls(config):
    target_type = (config or {}).get("target_type", "binary")
    if target_type == "continuous":
        hidden = {"display": "none"}
        return hidden, hidden, hidden
    return {}, {}, {}




# ── Playground: Model kur ─────────────────────────────────────────────────────
@app.callback(
    Output("pg-model-output", "children"),
    Input("btn-pg-build", "n_clicks"),
    State("store-pg-model-vars", "data"),
    State("chk-use-woe",         "value"),
    State("pg-test-size",        "value"),
    State("pg-c-value",            "value"),
    State("pg-model-type",         "value"),
    State("pg-threshold-method",   "value"),
    State("pg-threshold-val",      "value"),
    State("pg-target-col",         "value"),
    State("pg-split-method",     "value"),
    State("pg-split-date",       "value"),
    State("store-key",           "data"),
    State("store-config",        "data"),
    State("dd-segment-val",      "value"),
    State("dd-segment-col",      "value"),
    prevent_initial_call=True,
)
def build_pg_model(_, model_vars, use_woe, test_size_pct, c_val, model_type,
                   threshold_method, threshold_val,
                   target_sel, split_method, split_date,
                   key, config, seg_val, seg_col_input):
    if not model_vars or not key or not config:
        return html.Div("Model listesi boş veya konfigürasyon eksik.",
                        className="alert-info-custom")
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div("Veri yüklenmemiş.", className="alert-info-custom")

    target      = target_sel or config["target_col"]
    seg_col     = config.get("segment_col") or (seg_col_input or None)
    target_type = config.get("target_type", "binary")
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val).reset_index(drop=True)
    C           = float(c_val or 1.0)
    is_regression = (target_type == "continuous")

    if target not in df_active.columns:
        return html.Div(f"Target kolonu '{target}' veri setinde bulunamadı.",
                        className="alert-info-custom")
    y_all = pd.to_numeric(df_active[target], errors='coerce')

    # ── Train / Test / OOT split ───────────────────────────────────────────────
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
                        className="alert-info-custom")

    # ── Base parametreler ─────────────────────────────────────────────────────
    _MODEL_PARAMS = {
        "lr":    {},
        "ridge": {},
        "lgbm":  dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                      random_state=42, n_jobs=-1, verbose=-1),
        "xgb":   dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                      random_state=42, n_jobs=-1,
                      eval_metric="rmse" if is_regression else "auc"),
        "rf":    dict(n_estimators=200, max_depth=10, min_samples_leaf=5,
                      max_features="sqrt", random_state=42, n_jobs=-1),
    }
    algo = model_type or ("ridge" if is_regression else "lr")

    def _fit_and_render(X_df, disp_names, label, accent):
        """Modeli kur, train+test+oot sonuçlarını döndür."""
        X = X_df.copy()
        for col in X.columns:
            if X[col].isna().any():
                fill = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) \
                       else X[col].mode().iloc[0]
                X[col] = X[col].fillna(fill)
        X_tr = X.iloc[train_mask]
        X_te = X.iloc[test_mask]
        y_tr = y_all.iloc[train_mask]
        y_te = y_all.iloc[test_mask]
        has_test = len(X_te) > 0
        has_oot  = oot_mask.any()
        if has_oot:
            X_oot = X.iloc[oot_mask]
            y_oot = y_all.iloc[oot_mask]
        else:
            X_oot = None
            y_oot = None
        if len(X_tr) == 0:
            return html.Div("Split sonrası boş küme oluştu.", className="alert-info-custom")

        # Scaling: tree modelleri ve sm.Logit hariç
        # sm.Logit + WoE: değerler zaten dar aralıkta, scaling yapılmaz
        # Ham (non-WoE) LR'de BFGS overflow riski var → sadece orada scale
        is_tree = algo in ("lgbm", "xgb", "rf")
        _use_sm_logit = (algo == "lr" and not is_regression)
        _is_woe = all(c.endswith("_woe") for c in X_tr.columns)
        _skip_scale = is_tree or (_use_sm_logit and _is_woe)
        if not _skip_scale:
            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_te_s  = scaler.transform(X_te)  if has_test else np.empty((0, X_tr.shape[1]))
            X_oot_s = scaler.transform(X_oot) if has_oot  else None
        else:
            X_tr_s  = X_tr.values if hasattr(X_tr, 'values') else X_tr
            X_te_s  = X_te.values  if has_test else np.empty((0, X_tr.shape[1]))
            X_oot_s = X_oot.values if has_oot  else None

        # statsmodels Logit için sklearn-uyumlu ince sarmalayıcı
        class _SmLogitWrapper:
            """sm.Logit sonucunu sklearn predict_proba arayüzüne sarar."""
            def __init__(self, result):
                self._r      = result
                self.pvalues = result.pvalues   # const dahil
                self.params  = result.params    # const dahil
                non_const    = [k for k in result.params.index if k != "const"]
                self.coef_   = [result.params[non_const].values]

            def predict_proba(self, X):
                X_c = np.column_stack([np.ones(X.shape[0]), X])
                probs = self._r.predict(exog=X_c)
                return np.column_stack([1 - probs, probs])

        try:
            if is_regression:
                if algo == "ridge":
                    mdl = Ridge(alpha=1.0 / C)
                elif algo == "lgbm":
                    mdl = lgb.LGBMRegressor(**_MODEL_PARAMS["lgbm"])
                elif algo == "xgb":
                    mdl = xgb.XGBRegressor(**_MODEL_PARAMS["xgb"])
                else:  # rf
                    mdl = RandomForestRegressor(**_MODEL_PARAMS["rf"])
            else:
                if _use_sm_logit:
                    # statsmodels Logit — scale edilmiş features + const, BFGS
                    X_tr_const = sm.add_constant(X_tr_s, has_constant="add")
                    sm_res = sm.Logit(y_tr, X_tr_const).fit(disp=0, method="bfgs")
                    mdl = _SmLogitWrapper(sm_res)
                elif algo == "lgbm":
                    mdl = lgb.LGBMClassifier(**_MODEL_PARAMS["lgbm"])
                elif algo == "xgb":
                    pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
                    mdl = xgb.XGBClassifier(**_MODEL_PARAMS["xgb"],
                                            scale_pos_weight=pos_w)
                else:  # rf
                    mdl = RandomForestClassifier(**_MODEL_PARAMS["rf"])
            if not _use_sm_logit:
                mdl.fit(X_tr_s, y_tr)
        except Exception as e:
            return html.Div(f"Model kurulamadı: {e}", className="alert-info-custom")

        # ── Regression çıkışı ─────────────────────────────────────────────────
        if is_regression:
            tr_pred = mdl.predict(X_tr_s)
            te_pred = mdl.predict(X_te_s) if has_test else None

            def _reg_metrics(y_true, y_pred):
                r2   = r2_score(y_true, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae  = float(mean_absolute_error(y_true, y_pred))
                return dict(r2=r2, rmse=rmse, mae=mae, n=len(y_true))

            tr_rm = _reg_metrics(y_tr, tr_pred)
            te_rm = _reg_metrics(y_te, te_pred) if has_test and te_pred is not None else None

            def mc_r(v, l, c="#4F8EF7", w=2):
                return dbc.Col(html.Div([
                    html.Div(str(v), className="metric-value",
                             style={"color": c, "fontSize": "1.25rem"}),
                    html.Div(l, className="metric-label"),
                ], className="metric-card"), width=w)

            def _reg_row(m, title, bg):
                r2c = "#10b981" if m["r2"] >= 0.6 else "#f59e0b" if m["r2"] >= 0.3 else "#ef4444"
                return html.Div([
                    html.Div(title, style={"color": "#a8b2c2", "fontSize": "0.72rem",
                                           "fontWeight": "600", "letterSpacing": "0.06em",
                                           "textTransform": "uppercase",
                                           "marginBottom": "0.4rem", "paddingLeft": "0.25rem"}),
                    dbc.Row([
                        mc_r(f"{m['r2']:.4f}",   "R²",   r2c),
                        mc_r(f"{m['rmse']:.4f}",  "RMSE", "#f59e0b"),
                        mc_r(f"{m['mae']:.4f}",   "MAE",  "#a78bfa"),
                        mc_r(f"{m['n']:,}",        "N",    "#556070"),
                    ], className="g-2"),
                ], style={"backgroundColor": bg, "borderRadius": "6px",
                          "padding": "0.6rem 0.5rem", "marginBottom": "0.5rem"})

            reg_rows = [_reg_row(tr_rm, "Train", "#0d1520")]
            if te_rm:
                reg_rows.append(_reg_row(te_rm, "Test", "#0e1624"))
            if has_oot and X_oot_s is not None:
                oot_pred_r = mdl.predict(X_oot_s)
                oot_rm     = _reg_metrics(y_oot, oot_pred_r)
                reg_rows.append(_reg_row(oot_rm, "OOT", "#131c30"))
            metric_panel = html.Div(reg_rows, className="mb-3")

            # Scatter: actual vs predicted (test if exists, else oot)
            _scatter_y   = y_te   if (has_test and te_pred is not None) else (y_oot   if has_oot else y_tr)
            _scatter_p   = te_pred if (has_test and te_pred is not None) else (mdl.predict(X_oot_s) if has_oot else tr_pred)
            _scatter_lbl = "Test"  if (has_test and te_pred is not None) else ("OOT" if has_oot else "Train")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=_scatter_y.values, y=_scatter_p, mode="markers",
                marker=dict(color=accent, size=4, opacity=0.5),
                name=_scatter_lbl,
            ))
            mn = min(float(_scatter_y.min()), float(_scatter_p.min()))
            mx = max(float(_scatter_y.max()), float(_scatter_p.max()))
            fig_scatter.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color="#4a5568", dash="dash", width=1),
                showlegend=False,
            ))
            fig_scatter.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text=f"Actual vs Predicted — {_scatter_lbl}",
                           font=dict(color="#E8EAF0", size=13)),
                xaxis=dict(**_AXIS_STYLE, title="Gerçek"),
                yaxis=dict(**_AXIS_STYLE, title="Tahmin"),
                height=320,
            )

            # Residuals histogram
            residuals = _scatter_y.values - _scatter_p
            fig_resid = go.Figure(go.Histogram(
                x=residuals, nbinsx=40,
                marker_color=accent, opacity=0.8,
                hovertemplate="Artık: %{x:.3f}<br>Sayı: %{y}<extra></extra>",
            ))
            fig_resid.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text=f"Artık (Residual) Dağılımı — {_scatter_lbl}",
                           font=dict(color="#E8EAF0", size=13)),
                xaxis=dict(**_AXIS_STYLE, title="Artık"),
                yaxis=dict(**_AXIS_STYLE, title="Frekans"),
                height=260,
            )

            # Feature importance / coefficients
            _tbl_style_r = dict(
                sort_action="native", page_size=15,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                              "fontWeight": "600", "fontSize": "0.7rem",
                              "border": "1px solid #2d3a4f",
                              "textTransform": "uppercase"},
                style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                            "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                            "padding": "5px 8px", "textAlign": "left"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}
                ],
            )
            is_tree_r = algo in ("lgbm", "xgb", "rf")
            if not is_tree_r:
                coef_rows = [{"Değişken": disp_names.get(c, c),
                              "Katsayı": round(float(v), 4)}
                             for c, v in zip(X.columns, mdl.coef_)]
                imp_df = pd.DataFrame(coef_rows).sort_values("Katsayı", key=abs, ascending=False)
                tbl_title = "Katsayı Tablosu (Ridge)"
            else:
                raw_imp = mdl.feature_importances_
                total   = raw_imp.sum() or 1.0
                imp_df  = pd.DataFrame([
                    {"Değişken": disp_names.get(c, c),
                     "Önem (%)": round(float(v / total * 100), 2)}
                    for c, v in zip(X.columns, raw_imp)
                ]).sort_values("Önem (%)", ascending=False)
                tbl_title = "Feature Importance"

            imp_table = dash_table.DataTable(
                data=imp_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in imp_df.columns],
                **_tbl_style_r,
            )

            # SHAP for tree regressors
            shap_section = None
            if is_tree_r:
                try:
                    _X_shap_r  = (X_te if has_test and len(X_te) > 0
                                  else (X_oot if has_oot and X_oot is not None else X_tr))
                    explainer  = shap.TreeExplainer(mdl)
                    shap_vals  = explainer.shap_values(_X_shap_r.values)
                    shap_names = [disp_names.get(c, c) for c in X.columns]
                    fig_shap, ax_shap = plt.subplots(figsize=(8, max(4, len(shap_names) * 0.3)))
                    fig_shap.patch.set_facecolor("#0e1117")
                    ax_shap.set_facecolor("#0e1117")
                    shap.summary_plot(shap_vals, _X_shap_r.values, feature_names=shap_names,
                                      show=False, plot_size=None)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                                facecolor="#0e1117")
                    plt.close(fig_shap)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode()
                    shap_section = html.Div([
                        html.P("SHAP Beeswarm", className="section-title",
                               style={"marginTop": "1.5rem"}),
                        html.Img(src=f"data:image/png;base64,{img_b64}",
                                 style={"maxWidth": "100%", "borderRadius": "6px"}),
                    ])
                except Exception:
                    pass

            return html.Div([
                html.Div(f"{label}  ·  {split_info}",
                         style={"color": "#7e8fa4", "fontSize": "0.78rem",
                                "marginBottom": "0.75rem"}),
                metric_panel,
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_scatter,
                                     config={"displayModeBar": False}), width=6),
                    dbc.Col(dcc.Graph(figure=fig_resid,
                                     config={"displayModeBar": False}), width=6),
                ], className="mb-3"),
                html.P(tbl_title, className="section-title"),
                imp_table,
                shap_section or html.Div(),
            ])

        # ── Multiclass çıkışı ─────────────────────────────────────────────────
        is_multiclass = (target_type == "multiclass")
        if is_multiclass:
            tr_pred_cls  = mdl.predict(X_tr_s)
            te_pred_cls  = mdl.predict(X_te_s) if has_test else None
            oot_pred_cls = mdl.predict(X_oot_s) if has_oot and X_oot_s is not None else None

            def _mc_metrics(y_t, y_p):
                return dict(
                    acc = accuracy_score(y_t, y_p),
                    f1m = f1_score(y_t, y_p, average="macro",    zero_division=0),
                    f1w = f1_score(y_t, y_p, average="weighted", zero_division=0),
                    n   = len(y_t),
                )
            tr_mmc  = _mc_metrics(y_tr,  tr_pred_cls)
            te_mmc  = _mc_metrics(y_te,  te_pred_cls)  if has_test and te_pred_cls is not None  else None
            oot_mmc = _mc_metrics(y_oot, oot_pred_cls) if has_oot  and oot_pred_cls is not None else None

            def mc_m(v, l, c="#4F8EF7", w=2):
                return dbc.Col(html.Div([
                    html.Div(str(v), className="metric-value",
                             style={"color": c, "fontSize": "1.25rem"}),
                    html.Div(l, className="metric-label"),
                ], className="metric-card"), width=w)

            def _mc_row(m, title, bg):
                acc_c = "#10b981" if m["acc"] >= 0.8 else "#f59e0b" if m["acc"] >= 0.6 else "#ef4444"
                return html.Div([
                    html.Div(title, style={"color": "#a8b2c2", "fontSize": "0.72rem",
                                           "fontWeight": "600", "letterSpacing": "0.06em",
                                           "textTransform": "uppercase",
                                           "marginBottom": "0.4rem", "paddingLeft": "0.25rem"}),
                    dbc.Row([
                        mc_m(f"{m['acc']:.4f}",  "Accuracy",    acc_c),
                        mc_m(f"{m['f1m']:.4f}",  "F1 Macro",    "#4F8EF7"),
                        mc_m(f"{m['f1w']:.4f}",  "F1 Weighted", "#a78bfa"),
                        mc_m(f"{m['n']:,}",       "N",           "#556070"),
                    ], className="g-2"),
                ], style={"backgroundColor": bg, "borderRadius": "6px",
                          "padding": "0.6rem 0.5rem", "marginBottom": "0.5rem"})

            mc_rows = [_mc_row(tr_mmc, "Train", "#0d1520")]
            if te_mmc:
                mc_rows.append(_mc_row(te_mmc,  "Test", "#0e1624"))
            if oot_mmc:
                mc_rows.append(_mc_row(oot_mmc, "OOT",  "#131c30"))
            metric_panel = html.Div(mc_rows, className="mb-3")

            # Feature importance / coefficients
            is_tree_mc = algo in ("lgbm", "xgb", "rf")
            if not is_tree_mc:
                coef_rows_mc = []
                for i, cls in enumerate(mdl.classes_):
                    coef_arr = mdl.coef_[i] if len(mdl.coef_) > 1 else mdl.coef_[0]
                    for c, v in zip(X.columns, coef_arr):
                        coef_rows_mc.append({"Sınıf": str(int(float(cls))),
                                             "Değişken": disp_names.get(c, c),
                                             "Katsayı": round(float(v), 4)})
                imp_df_mc = pd.DataFrame(coef_rows_mc)
                tbl_title_mc = "Katsayı Tablosu (Sınıf Bazlı)"
            else:
                raw_imp_mc = mdl.feature_importances_
                total_mc   = raw_imp_mc.sum() or 1.0
                imp_df_mc  = pd.DataFrame([
                    {"Değişken": disp_names.get(c, c),
                     "Önem (%)": round(float(v / total_mc * 100), 2)}
                    for c, v in zip(X.columns, raw_imp_mc)
                ]).sort_values("Önem (%)", ascending=False)
                tbl_title_mc = "Feature Importance"

            _tbl_style_mc = dict(
                sort_action="native", page_size=15,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                              "fontWeight": "600", "fontSize": "0.7rem",
                              "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
                style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                            "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                            "padding": "5px 8px", "textAlign": "left"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}],
            )

            return html.Div([
                html.Div(f"{label}  ·  {split_info}",
                         style={"color": "#7e8fa4", "fontSize": "0.78rem",
                                "marginBottom": "0.75rem"}),
                metric_panel,
                html.P(tbl_title_mc, className="section-title"),
                dash_table.DataTable(
                    data=imp_df_mc.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in imp_df_mc.columns],
                    **_tbl_style_mc,
                ),
            ])

        # ── Binary classification çıkışı ──────────────────────────────────────
        tr_prob  = mdl.predict_proba(X_tr_s)[:, 1]
        te_prob  = mdl.predict_proba(X_te_s)[:, 1]  if has_test else None
        oot_prob = mdl.predict_proba(X_oot_s)[:, 1] if has_oot and X_oot_s is not None else None

        # ── Eşik belirleme (test üzerinden, yoksa train) ──────────────────────
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

        # ── Metrik rengi ──────────────────────────────────────────────────────
        def _gc(g): return "#10b981" if g >= 0.4 else "#f59e0b" if g >= 0.2 else "#ef4444"

        # ── Metrik kartı ──────────────────────────────────────────────────────
        def mc(v, l, c="#4F8EF7", w=2):
            return dbc.Col(html.Div([
                html.Div(str(v), className="metric-value",
                         style={"color": c, "fontSize": "1.25rem"}),
                html.Div(l, className="metric-label"),
            ], className="metric-card"), width=w)

        def _metric_row(m, title, bg):
            gc = _gc(m["gini"])
            return html.Div([
                html.Div(title, style={"color": "#a8b2c2", "fontSize": "0.72rem",
                                       "fontWeight": "600", "letterSpacing": "0.06em",
                                       "textTransform": "uppercase",
                                       "marginBottom": "0.4rem", "paddingLeft": "0.25rem"}),
                dbc.Row([
                    mc(f"{m['gini']:.4f}", "Gini",      gc),
                    mc(f"{m['auc']:.4f}",  "AUC",       gc),
                    mc(f"{m['ks']:.4f}",   "KS",        "#4F8EF7"),
                    mc(f"{m['f1']:.4f}",   "F1",        "#a78bfa"),
                    mc(f"{m['prec']:.4f}", "Precision", "#a78bfa"),
                    mc(f"{m['rec']:.4f}",  "Recall",    "#a78bfa"),
                    mc(f"{m['n']:,}",      "N",         "#556070"),
                ], className="g-2"),
            ], style={"backgroundColor": bg, "borderRadius": "6px",
                      "padding": "0.6rem 0.5rem", "marginBottom": "0.5rem"})

        bin_rows = [_metric_row(tr_m, "Train", "#0d1520")]
        if te_m:
            bin_rows.append(_metric_row(te_m,  "Test", "#0e1624"))
        if oot_m:
            bin_rows.append(_metric_row(oot_m, "OOT",  "#131c30"))
        metric_panel = html.Div(bin_rows, className="mb-3")

        # ── ROC (train + test + oot) ───────────────────────────────────────────
        accent_tr = "#556070"
        fig_roc = go.Figure()
        roc_traces = [(tr_m, accent_tr, f"Train (AUC={tr_m['auc']:.3f})")]
        if te_m:
            roc_traces.append((te_m,  accent,     f"Test  (AUC={te_m['auc']:.3f})"))
        if oot_m:
            roc_traces.append((oot_m, "#a78bfa",  f"OOT   (AUC={oot_m['auc']:.3f})"))
        for m, col_, nm in roc_traces:
            fig_roc.add_trace(go.Scatter(
                x=m["fpr"], y=m["tpr"], mode="lines",
                line=dict(color=col_, width=2), name=nm,
                fill="tozeroy" if col_ == accent else None,
                fillcolor=f"rgba({','.join(str(int(accent.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.06)",
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#4a5568", dash="dash", width=1), showlegend=False))
        fig_roc.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text="ROC Eğrisi", font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, title="FPR"),
            yaxis=dict(**_AXIS_STYLE, title="TPR"),
            height=300, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
        )

        # ── Confusion matrix (test if available, else oot, else train) ────────
        _cm_src = te_m if te_m else (oot_m if oot_m else tr_m)
        _cm_lbl = "Test" if te_m else ("OOT" if oot_m else "Train")
        tn, fp, fn, tp_ = _cm_src["cm"].ravel()
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp_]],
            x=["Pred: 0", "Pred: 1"], y=["Actual: 0", "Actual: 1"],
            text=[[str(tn), str(fp)], [str(fn), str(tp_)]],
            texttemplate="%{text}",
            colorscale=[[0, "#0e1117"], [1, accent]], showscale=False,
        ))
        fig_cm.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=f"Confusion Matrix — {_cm_lbl} ({thr_label})",
                       font=dict(color="#E8EAF0", size=13)),
            height=260, xaxis=dict(side="top"),
        )

        # ── Önem tablosu: LR → katsayı; tree → feature importance ───────────
        _tbl_style = dict(
            sort_action="native", page_size=15,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                          "fontWeight": "600", "fontSize": "0.7rem",
                          "border": "1px solid #2d3a4f",
                          "textTransform": "uppercase"},
            style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                        "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                        "padding": "5px 8px", "textAlign": "left"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}
            ],
        )
        if not is_tree:
            # Logistic Regression: katsayı + p-value (statsmodels), const dahil
            has_pvalues = _use_sm_logit and hasattr(mdl, "pvalues")
            # const satırı
            if has_pvalues:
                const_coef = float(mdl.params.get("const", 0.0))
                const_pv   = float(mdl.pvalues.get("const", np.nan))
                coef_rows = [{"Değişken": "const",
                              "Katsayı": round(const_coef, 4),
                              "P-Value": round(const_pv, 4)}]
            else:
                coef_rows = []
            # değişken satırları
            for c, v in zip(X.columns, mdl.coef_[0]):
                row = {"Değişken": disp_names.get(c, c),
                       "Katsayı":  round(float(v), 4)}
                if has_pvalues:
                    row["P-Value"] = round(float(mdl.pvalues.get(c, np.nan)), 4)
                coef_rows.append(row)
            imp_df = pd.DataFrame(coef_rows)
            # const sabit üstte, geri kalan abs(katsayı) sıralı
            if has_pvalues:
                const_part = imp_df.iloc[:1]
                var_part   = imp_df.iloc[1:].sort_values("Katsayı", key=abs, ascending=False)
                imp_df = pd.concat([const_part, var_part], ignore_index=True)
            else:
                imp_df = imp_df.sort_values("Katsayı", key=abs, ascending=False)

            # P-Value renk kodlaması
            pv_cond = []
            if has_pvalues:
                pv_cond = [
                    {"if": {"filter_query": "{P-Value} < 0.05", "column_id": "P-Value"},
                     "color": "#10b981", "fontWeight": "600"},
                    {"if": {"filter_query": "{P-Value} >= 0.05 && {P-Value} < 0.10",
                            "column_id": "P-Value"},
                     "color": "#f59e0b", "fontWeight": "600"},
                    {"if": {"filter_query": "{P-Value} >= 0.10", "column_id": "P-Value"},
                     "color": "#ef4444"},
                ]
            imp_df_cond = pv_cond + [{"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}]
            tbl_title = "Katsayı Tablosu"
            imp_table = dash_table.DataTable(
                data=imp_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in imp_df.columns],
                style_data_conditional=imp_df_cond,
                **{k: v for k, v in _tbl_style.items() if k != "style_data_conditional"},
            )
        else:
            # Tree modeller: feature importance (gain / split)
            raw_imp = mdl.feature_importances_
            total   = raw_imp.sum() or 1.0
            imp_rows = [
                {"Değişken": disp_names.get(c, c),
                 "Önem (%)": round(float(v / total * 100), 2),
                 "Önem (ham)": round(float(v), 4)}
                for c, v in zip(X.columns, raw_imp)
            ]
            imp_df = pd.DataFrame(imp_rows).sort_values("Önem (%)", ascending=False)
            # Renk: top %25 yeşil, orta sarı, alt kırmızı
            top_thr = float(imp_df["Önem (%)"].quantile(0.75))
            mid_thr = float(imp_df["Önem (%)"].quantile(0.25))
            imp_df_cond = [
                {"if": {"filter_query": f"{{Önem (%)}} >= {top_thr:.2f}",
                        "column_id": "Önem (%)"},
                 "color": "#10b981", "fontWeight": "700"},
                {"if": {"filter_query": f"{{Önem (%)}} >= {mid_thr:.2f} && {{Önem (%)}} < {top_thr:.2f}",
                        "column_id": "Önem (%)"},
                 "color": "#f59e0b"},
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
            ]
            imp_label = {"lgbm": "gain (normalize)", "xgb": "gain (normalize)",
                         "rf": "mean decrease impurity"}.get(algo, "")
            tbl_title = f"Feature Importance  ·  {imp_label}"
            imp_table = dash_table.DataTable(
                data=imp_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in imp_df.columns],
                style_data_conditional=imp_df_cond,
                **{k: v for k, v in _tbl_style.items() if k != "style_data_conditional"},
            )

        # ── SHAP Beeswarm — shap.summary_plot → PNG embed ────────────────────
        shap_section = None
        if is_tree:
            try:
                # Use test for SHAP if available, else oot, else train
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
                    max_display=top_n,
                    show=False,
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
                # colorbar dark
                for cax in fig_mpl.axes[1:]:
                    cax.set_facecolor(_BG)
                    cax.tick_params(colors=_FG, labelsize=8)
                    cax.yaxis.label.set_color(_FG)

                buf = io.BytesIO()
                fig_mpl.savefig(buf, format="png", bbox_inches="tight",
                                facecolor=_BG, dpi=130)
                plt.close("all")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()

                shap_section = html.Div([
                    html.Hr(style={"borderColor": "#1f2a3c", "margin": "0.8rem 0"}),
                    html.P(
                        f"SHAP Beeswarm  ·  n={_shap_n}  ·  top {top_n} değişken",
                        className="section-title",
                        style={"marginBottom": "0.5rem"},
                    ),
                    html.Img(
                        src=f"data:image/png;base64,{img_b64}",
                        style={"width": "100%", "maxWidth": "820px",
                               "display": "block", "margin": "0 auto"},
                    ),
                ])
            except Exception as _shap_err:
                shap_section = html.Div(
                    f"SHAP hesaplanamadı: {_shap_err}",
                    className="alert-info-custom",
                    style={"marginTop": "0.5rem"},
                )

        return html.Div([
            html.Div(f"{split_info}   ·   {thr_label}",
                     style={"color": "#7e8fa4", "fontSize": "0.72rem",
                            "marginBottom": "0.6rem", "fontStyle": "italic"}),
            metric_panel,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc, config={"displayModeBar": False}), width=5),
                dbc.Col([
                    html.P(tbl_title, className="section-title",
                           style={"marginBottom": "0.4rem"}),
                    imp_table,
                ], width=4),
                dbc.Col(dcc.Graph(figure=fig_cm, config={"displayModeBar": False}), width=3),
            ]),
            shap_section or html.Div(),
        ])

    # ── Ham model ─────────────────────────────────────────────────────────────
    raw_cols = [v for v in model_vars if v in df_active.columns]
    X_raw    = pd.get_dummies(df_active[raw_cols].copy(), drop_first=True)
    raw_disp = {c: c for c in X_raw.columns}
    raw_tab  = _fit_and_render(X_raw, raw_disp, "Ham", "#4F8EF7")

    # ── WoE model ─────────────────────────────────────────────────────────────
    woe_cache_key = f"{key}_woe_{seg_col}_{seg_val}"
    if woe_cache_key not in _SERVER_STORE:
        woe_df_enc, _ = _build_woe_dataset(df_active, target, model_vars)
        _SERVER_STORE[woe_cache_key] = (woe_df_enc, _)
    else:
        woe_df_enc, _ = _SERVER_STORE[woe_cache_key]
        new_vars = [v for v in model_vars if f"{v}_woe" not in woe_df_enc.columns]
        if new_vars:
            extra, ef = _build_woe_dataset(df_active, target, new_vars)
            woe_df_enc = pd.concat([woe_df_enc, extra], axis=1)
            _SERVER_STORE[woe_cache_key] = (woe_df_enc, _ + ef)

    woe_feat_cols = [f"{v}_woe" for v in model_vars if f"{v}_woe" in woe_df_enc.columns]
    if woe_feat_cols:
        X_woe     = woe_df_enc[woe_feat_cols].copy()
        woe_disp  = {f"{v}_woe": v for v in model_vars}
        woe_tab   = _fit_and_render(X_woe, woe_disp, "WoE", "#a78bfa")
        failed_woe = [v for v in model_vars if f"{v}_woe" not in woe_df_enc.columns]
        note_txt  = f"★ WoE — {len(woe_feat_cols)}/{len(model_vars)} değişken encode edildi"
        if failed_woe:
            note_txt += f"  |  encode edilemeyen: {', '.join(failed_woe)}"
        woe_note  = html.Div(note_txt,
            style={"color": "#a78bfa", "fontSize": "0.73rem", "marginBottom": "0.4rem"})
        woe_content = html.Div([woe_note, woe_tab])
    else:
        failed_woe = model_vars
        woe_content = html.Div(
            f"WoE encode edilebilen değişken bulunamadı. "
            f"({len(model_vars)} değişken denendi: {', '.join(model_vars[:5])}{'...' if len(model_vars)>5 else ''})",
            className="alert-info-custom")

    return dbc.Tabs([
        dbc.Tab(raw_tab,     label="Ham Değerler",        tab_id="res-raw",
                className="tab-content-area"),
        dbc.Tab(woe_content, label="WoE Dönüştürülmüş",  tab_id="res-woe",
                className="tab-content-area"),
    ], active_tab="res-raw")
