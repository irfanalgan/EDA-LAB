import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score, recall_score)
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
)
def render_pg_var_summary_preview(config, expert_excluded, active_tab, key):
    if not key or not config or not config.get("target_col"):
        return html.Div()
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")

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
        {"label": "Random Forest",       "value": "rf"},
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




# ── Playground: Model kur ─────────────────────────────────────────────────────
@app.callback(
    Output("pg-model-output", "children", allow_duplicate=True),
    Output("store-model-signal", "data", allow_duplicate=True),
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
    State("store-pending-note",  "data"),
    prevent_initial_call=True,
)
def build_pg_model(_, model_vars, use_woe, test_size_pct, c_val, model_type,
                   threshold_method, threshold_val,
                   target_sel, split_method, split_date,
                   key, config, pending_note):
    _no = dash.no_update
    if not model_vars or not key or not config:
        return html.Div("Model listesi boş veya konfigürasyon eksik.",
                        className="alert-info-custom"), _no
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div("Veri yüklenmemiş.", className="alert-info-custom"), _no

    target      = target_sel or config["target_col"]
    seg_col     = config.get("segment_col")
    seg_val     = config.get("segment_val")
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val).reset_index(drop=True)
    C           = float(c_val or 1.0)

    if target not in df_active.columns:
        return html.Div(f"Target kolonu '{target}' veri setinde bulunamadı.",
                        className="alert-info-custom"), _no
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
                        className="alert-info-custom"), _no

    # ── Base parametreler ─────────────────────────────────────────────────────
    _MODEL_PARAMS = {
        "lr":    {},
        "lgbm":  dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                      random_state=42, n_jobs=-1, verbose=-1),
        "xgb":   dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                      random_state=42, n_jobs=-1, eval_metric="auc"),
        "rf":    dict(n_estimators=200, max_depth=10, min_samples_leaf=5,
                      max_features="sqrt", random_state=42, n_jobs=-1),
    }
    algo = model_type or "lr"

    def _fit_and_render(X_df, disp_names, label, accent):
        """Modeli kur, (compact_html, results_dict, model_obj, scaler_obj) döndür."""
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
            return html.Div("Split sonrası boş küme oluştu.", className="alert-info-custom"), None, None, None

        is_tree = algo in ("lgbm", "xgb", "rf")
        _use_sm_logit = (algo == "lr")
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

        lr_summary_text = None
        try:
            if _use_sm_logit:
                X_tr_const = sm.add_constant(X_tr_s, has_constant="add")
                sm_res = sm.Logit(y_tr, X_tr_const).fit(disp=0, method="bfgs")
                mdl = SmLogitWrapper(sm_res)
                try:
                    lr_summary_text = sm_res.summary().as_text()
                except Exception:
                    lr_summary_text = None
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
            if has_pvalues:
                const_coef = float(mdl.params.get("const", 0.0))
                const_pv   = float(mdl.pvalues.get("const", np.nan))
                coef_rows = [{"Değişken": "const",
                              "Katsayı": round(const_coef, 4),
                              "P-Value": round(const_pv, 4)}]
            else:
                coef_rows = []
            # pvalues index'i const + x1,x2... olabilir (numpy input)
            # sıralı eşleştir: pvalues[1:] ↔ X.columns
            _pv_vals = list(mdl.pvalues.values)[1:] if has_pvalues else []
            for i, (c, v) in enumerate(zip(X.columns, mdl.coef_[0])):
                row = {"Değişken": disp_names.get(c, c),
                       "Katsayı":  round(float(v), 4)}
                if has_pvalues and i < len(_pv_vals):
                    row["P-Value"] = round(float(_pv_vals[i]), 4)
                coef_rows.append(row)
            imp_records = coef_rows
            # const üstte, geri kalan abs sıralı
            imp_df = pd.DataFrame(coef_rows)
            if has_pvalues and len(imp_df) > 1:
                const_part = imp_df.iloc[:1]
                var_part   = imp_df.iloc[1:].sort_values("Katsayı", key=abs, ascending=False)
                imp_df = pd.concat([const_part, var_part], ignore_index=True)
            else:
                imp_df = imp_df.sort_values("Katsayı", key=abs, ascending=False)
            imp_records = imp_df.to_dict("records")
            importance_type = "coef"
        else:
            raw_imp = mdl.feature_importances_
            total   = raw_imp.sum() or 1.0
            imp_rows = [
                {"Değişken": disp_names.get(c, c),
                 "Önem (%)": round(float(v / total * 100), 2),
                 "Önem (ham)": round(float(v), 4)}
                for c, v in zip(X.columns, raw_imp)
            ]
            imp_df = pd.DataFrame(imp_rows).sort_values("Önem (%)", ascending=False)
            imp_records = imp_df.to_dict("records")
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
            except Exception:
                shap_img_b64 = None

        # ── VIF hesapla ──────────────────────────────────────────────────────
        vif_data = None
        if X_tr.shape[1] >= 2:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
                _X_vif = X_tr.values.astype(float)
                vif_data = []
                for j in range(_X_vif.shape[1]):
                    try:
                        v = float(_vif(_X_vif, j))
                    except Exception:
                        v = None
                    vif_data.append({"Değişken": disp_names.get(X_tr.columns[j], X_tr.columns[j]),
                                     "VIF": round(v, 2) if v is not None else None})
            except Exception:
                vif_data = None

        # ── PSI hesapla (Train vs Test / OOT) ─────────────────────────────────
        def _calc_psi(base, comp, n_bins=10):
            """Population Stability Index."""
            eps = 1e-4
            mn, mx = float(base.min()), float(base.max())
            if mn == mx:
                return 0.0
            bins = np.linspace(mn, mx, n_bins + 1)
            bins[0]  = -np.inf
            bins[-1] =  np.inf
            b_pct = np.histogram(base, bins=bins)[0] / len(base)
            c_pct = np.histogram(comp, bins=bins)[0] / len(comp)
            b_pct = np.where(b_pct < eps, eps, b_pct)
            c_pct = np.where(c_pct < eps, eps, c_pct)
            return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))

        psi_data = None
        if has_test or has_oot:
            psi_data = {}
            for col in X_tr.columns:
                d_name = disp_names.get(col, col)
                row = {"Değişken": d_name}
                if has_test:
                    row["PSI (Test)"]  = round(_calc_psi(X_tr[col].values, X_te[col].values), 4)
                if has_oot and X_oot is not None:
                    row["PSI (OOT)"]   = round(_calc_psi(X_tr[col].values, X_oot[col].values), 4)
                psi_data[d_name] = row
            psi_data = list(psi_data.values())

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
            "vif_data": vif_data,
            "psi_data": psi_data,
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

        # Top-5 importance mini tablo
        top5 = imp_records[:6] if imp_records[0].get("Değişken") == "const" else imp_records[:5]
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
        mini_cols = [{"name": k, "id": k} for k in top5[0].keys()] if top5 else []
        mini_tbl_title = "Katsayı Tablosu (özet)" if importance_type == "coef" else "Feature Importance (özet)"

        compact_html = html.Div([
            html.Div(f"{split_info}   ·   {thr_label}",
                     style={"color": "#7e8fa4", "fontSize": "0.72rem",
                            "marginBottom": "0.6rem", "fontStyle": "italic"}),
            dbc.Row(gini_cards, className="g-2 mb-3"),
            html.P(mini_tbl_title, className="section-title",
                   style={"marginBottom": "0.4rem"}),
            dash_table.DataTable(data=top5, columns=mini_cols, **_tbl_style_mini),
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

    # ── Ham model ─────────────────────────────────────────────────────────────
    raw_cols = [v for v in model_vars if v in df_active.columns]
    X_raw    = pd.get_dummies(df_active[raw_cols].copy(), drop_first=True)
    raw_disp = {c: c for c in X_raw.columns}
    raw_html, raw_results, raw_mdl, raw_scaler = _fit_and_render(X_raw, raw_disp, "Ham", "#4F8EF7")

    # ── WoE model ─────────────────────────────────────────────────────────────
    woe_cache_key = f"{key}_woe_{seg_col}_{seg_val}"
    if woe_cache_key not in _SERVER_STORE:
        woe_df_enc, _failed, _opt_dict = _build_woe_dataset(df_active, target, model_vars)
        _SERVER_STORE[woe_cache_key] = (woe_df_enc, _failed, _opt_dict)
    else:
        stored = _SERVER_STORE[woe_cache_key]
        # Eski 2-tuple format uyumu — opt_dict'i yeniden hesapla
        if len(stored) == 2:
            woe_df_enc, _failed = stored
            _, _, _opt_dict = _build_woe_dataset(df_active, target, model_vars)
            _SERVER_STORE[woe_cache_key] = (woe_df_enc, _failed, _opt_dict)
        else:
            woe_df_enc, _failed, _opt_dict = stored
        new_vars = [v for v in model_vars if f"{v}_woe" not in woe_df_enc.columns]
        if new_vars:
            extra, ef, eo = _build_woe_dataset(df_active, target, new_vars)
            woe_df_enc = pd.concat([woe_df_enc, extra], axis=1)
            _opt_dict = {**_opt_dict, **eo}
            _SERVER_STORE[woe_cache_key] = (woe_df_enc, _failed + ef, _opt_dict)

    woe_feat_cols = [f"{v}_woe" for v in model_vars if f"{v}_woe" in woe_df_enc.columns]
    woe_html = None
    woe_results = None
    if woe_feat_cols:
        X_woe     = woe_df_enc[woe_feat_cols].copy()
        woe_disp  = {f"{v}_woe": v for v in model_vars}
        woe_html, woe_results, woe_mdl, woe_scaler = _fit_and_render(X_woe, woe_disp, "WoE", "#a78bfa")
        failed_woe = [v for v in model_vars if f"{v}_woe" not in woe_df_enc.columns]
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

    # ── WoE dağılım verisi (bin tablosu + monotonluk) ──────────────────────
    woe_dist = None
    has_test_split = test_mask.any()
    has_oot_split  = oot_mask.any()
    if woe_feat_cols:
        from modules.deep_dive import get_woe_detail
        woe_dist = {}
        # Train split
        df_train_split = df_active[train_mask].reset_index(drop=True)
        # Karşı taraf: Test varsa Test, yoksa OOT
        if has_test_split:
            df_comp_split = df_active[test_mask].reset_index(drop=True)
            comp_label = "Test"
        elif has_oot_split:
            df_comp_split = df_active[oot_mask].reset_index(drop=True)
            comp_label = "OOT"
        else:
            df_comp_split = None
            comp_label = None

        for wc in woe_feat_cols:
            var_name = wc.replace("_woe", "")
            try:
                bt_train, iv_train, _ = get_woe_detail(df_train_split, var_name, target)
                if bt_train.empty:
                    continue
                # Monotonluk kontrolü: TOPLAM hariç WOE sütununa bak
                woe_vals = bt_train[bt_train["Bin"] != "TOPLAM"]["WOE"].tolist()
                woe_nums = [w for w in woe_vals if isinstance(w, (int, float))]
                if len(woe_nums) >= 2:
                    diffs = [woe_nums[i+1] - woe_nums[i] for i in range(len(woe_nums)-1)]
                    if all(d >= 0 for d in diffs):
                        monoton = "Artan ↑"
                    elif all(d <= 0 for d in diffs):
                        monoton = "Azalan ↓"
                    else:
                        monoton = "Monoton Değil ✗"
                else:
                    monoton = "–"

                entry = {
                    "train_table": bt_train.to_dict("records"),
                    "iv_train": round(iv_train, 4),
                    "monoton": monoton,
                }

                # Karşı taraf tablosu
                if df_comp_split is not None:
                    try:
                        bt_comp, iv_comp, _ = get_woe_detail(df_comp_split, var_name, target)
                        if not bt_comp.empty:
                            entry["comp_table"] = bt_comp.to_dict("records")
                            entry["iv_comp"] = round(iv_comp, 4)
                            entry["comp_label"] = comp_label
                    except Exception:
                        pass

                woe_dist[var_name] = entry
            except Exception:
                continue

    # ── Korelasyon matrisleri ───────────────────────────────────────────────
    raw_corr = X_raw[raw_cols].corr().round(4).to_dict() if len(raw_cols) > 1 else None
    woe_corr = None
    if woe_feat_cols and len(woe_feat_cols) > 1:
        _woe_corr_df = woe_df_enc[woe_feat_cols].corr().round(4)
        _woe_corr_df.index   = [c.replace("_woe", "") for c in _woe_corr_df.index]
        _woe_corr_df.columns = [c.replace("_woe", "") for c in _woe_corr_df.columns]
        woe_corr = _woe_corr_df.to_dict()

    # ── Describe verisi (model değişkenleri) ────────────────────────────────
    from modules.profiling import compute_profile
    _desc_cols = [v for v in model_vars if v in df_active.columns]
    describe_data = None
    if _desc_cols:
        try:
            _desc_df = compute_profile(df_active[_desc_cols])
            describe_data = _desc_df.to_dict("records")
        except Exception:
            describe_data = None

    # ── Sonuçları cache'e yaz ─────────────────────────────────────────────────
    cache_key = f"{key}_model_results"
    _thr_label = raw_results["thr_label"] if raw_results else ""
    _opt_thr   = raw_results["opt_thr"] if raw_results else 0.5
    _SERVER_STORE[cache_key] = {
        "algo": algo,
        "model_vars": list(model_vars),
        "split_info": split_info,
        "thr_label": _thr_label,
        "opt_thr": _opt_thr,
        "corr": {"raw": raw_corr, "woe": woe_corr},
        "woe_dist": woe_dist,
        "describe_data": describe_data,
        "model_note": pending_note or "",
        "tabs": {
            "raw": raw_results,
            "woe": woe_results,
        },
        # SQL & Pickle için ek objeler
        "_models": {"raw": raw_mdl, "woe": woe_mdl if woe_feat_cols else None},
        "_scalers": {"raw": raw_scaler, "woe": woe_scaler if woe_feat_cols else None},
        "_opt_dict": _opt_dict,  # {col_name: OptimalBinning} — WoE pickle
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
