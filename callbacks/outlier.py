import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _tab_info, _PLOT_LAYOUT, _AXIS_STYLE, _TABLE_STYLE


# ── Callback: Outlier Analizi Sekmesi — Layout ────────────────────────────────
@app.callback(
    Output("tab-outlier", "children"),
    Input("store-config", "data"),
    State("store-key", "data"),
)
def render_outlier_tab(config, key):
    df = _get_df(key)
    if df is None or not config or not config.get("target_col"):
        return html.Div("Önce veri yükleyin ve yapılandırın.", className="alert-info-custom")

    cfg_cols = {c for c in [config.get("target_col"), config.get("date_col"),
                             config.get("segment_col")] if c}
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in cfg_cols]
    if not num_cols:
        return html.Div("Sayısal değişken bulunamadı.", className="alert-info-custom")

    col_opts = [{"label": c, "value": c} for c in num_cols]

    layout = html.Div([
        _tab_info("Outlier Analizi", "IQR · Z-Score",
                  "Sayısal değişkenlerde aykırı değerleri IQR veya Z-score yöntemiyle tespit eder. "
                  "IQR 1.5× tipik sınır, 3.0× yalnızca aşırı aykırı değerler içindir. "
                  "Z-score > 3σ standart tercih. Hangi müşterinin kaç farklı değişkende aykırı "
                  "çıktığını 'Müşteri Bazında Outlier Detayı' tablosundan görebilirsiniz.",
                  "#f59e0b"),
        # ── Kontroller — Satır 1 ──────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Label("Müşteri No Kolonu", className="form-label"),
                html.Div("Zorunlu — müşteri bazında analiz", className="form-hint"),
                dbc.Select(id="out-id-col", className="dark-select",
                           options=[{"label": c, "value": c} for c in df.columns],
                           value=df.columns[0]),
            ], width=4),
            dbc.Col([
                dbc.Label("Yöntem", className="form-label"),
                html.Div("\u00a0", className="form-hint"),
                dbc.Select(id="out-method", className="dark-select",
                           options=[
                               {"label": "IQR (Çeyrekler Açıklığı)", "value": "iqr"},
                               {"label": "Z-Score (Standart Sapma)", "value": "zscore"},
                           ], value="iqr"),
            ], width=3),
            dbc.Col([
                dbc.Label("IQR Çarpanı", className="form-label"),
                html.Div("\u00a0", className="form-hint"),
                dbc.Select(id="out-iqr-k", className="dark-select",
                           options=[
                               {"label": "1.5  (normal)",  "value": "1.5"},
                               {"label": "3.0  (aşırı)",   "value": "3.0"},
                           ], value="1.5"),
            ], width=2, id="out-iqr-col"),
            dbc.Col([
                dbc.Label("Z-Score Eşiği", className="form-label"),
                html.Div("\u00a0", className="form-hint"),
                dbc.Select(id="out-z-k", className="dark-select",
                           options=[
                               {"label": "2.0  (±2σ)", "value": "2.0"},
                               {"label": "2.5  (±2.5σ)", "value": "2.5"},
                               {"label": "3.0  (±3σ)", "value": "3.0"},
                           ], value="3.0"),
            ], width=2, id="out-z-col", style={"display": "none"}),
        ], className="mb-2"),
        # ── Kontroller — Satır 2 ──────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Label("Görselleştir", className="form-label"),
                dbc.Select(id="out-var-sel", className="dark-select",
                           options=col_opts, value=num_cols[0]),
            ], width=4),
            dbc.Col(
                dbc.Button("Tara", id="btn-outlier-run", color="primary", size="sm",
                           style={"width": "80px"}),
                width=2,
                className="d-flex align-items-center",
                style={"paddingTop": "1.55rem"},
            ),
        ], className="mb-3"),

        # ── Çıktı alanı ───────────────────────────────────────────────────────
        html.Div(id="outlier-output"),
    ])
    return layout


@app.callback(
    Output("out-iqr-col", "style"),
    Output("out-z-col",   "style"),
    Input("out-method",   "value"),
)
def toggle_outlier_params(method):
    show, hide = {}, {"display": "none"}
    return (show, hide) if method == "iqr" else (hide, show)


@app.callback(
    Output("outlier-output", "children"),
    Input("btn-outlier-run", "n_clicks"),
    State("out-id-col",   "value"),
    State("out-method",   "value"),
    State("out-iqr-k",    "value"),
    State("out-z-k",      "value"),
    State("out-var-sel",  "value"),
    State("store-key",    "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    State("dd-segment-col", "value"),
    prevent_initial_call=True,
)
def run_outlier_analysis(_, id_col, method, iqr_k, z_k, vis_var,
                         key, config, seg_val, seg_col_input):
    df_orig = _get_df(key)
    if df_orig is None or not config:
        return html.Div("Veri yok.", className="alert-info-custom")

    seg_col = config.get("segment_col") or (seg_col_input or None)
    df = apply_segment_filter(df_orig, seg_col, seg_val).copy()

    cfg_cols = {c for c in [config.get("target_col"), config.get("date_col"),
                             config.get("segment_col")] if c}
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in cfg_cols]

    k = float(iqr_k if method == "iqr" else z_k)

    # ── Her değişken için outlier maskesi hesapla ──────────────────────────────
    def _outlier_mask(series):
        s = series.dropna()
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
        else:
            mu, sigma = s.mean(), s.std()
            lo, hi = mu - k * sigma, mu + k * sigma
        mask = (series < lo) | (series > hi)
        return mask, lo, hi

    summary_rows = []
    # outlier_flags: her satır için kaç değişkende outlier olduğu
    outlier_count = pd.Series(0, index=df.index)

    for col in num_cols:
        mask, lo, hi = _outlier_mask(df[col])
        n_out = int(mask.sum())
        n_tot = int(df[col].notna().sum())
        pct   = round(n_out / n_tot * 100, 2) if n_tot > 0 else 0.0
        summary_rows.append({
            "Değişken":    col,
            "N Outlier":   n_out,
            "% Outlier":   pct,
            "Alt Sınır":   round(lo, 4),
            "Üst Sınır":   round(hi, 4),
            "Min":         round(float(df[col].min()), 4),
            "Max":         round(float(df[col].max()), 4),
        })
        outlier_count += mask.fillna(False).astype(int)

    summary_df = pd.DataFrame(summary_rows).sort_values("% Outlier", ascending=False)

    # ── Müşteri outlier sayısı tablosu ────────────────────────────────────────
    if not id_col or id_col not in df.columns:
        id_col = df.columns[0]

    cust_df = pd.DataFrame({
        id_col:                    df[id_col].values,
        "Outlier Değişken Sayısı": outlier_count.values,
    })
    tgt = config.get("target_col")
    if tgt and tgt in df.columns:
        cust_df["Target"] = df[tgt].values
    cust_df = (cust_df[cust_df["Outlier Değişken Sayısı"] > 0]
               .sort_values("Outlier Değişken Sayısı", ascending=False)
               .drop_duplicates(subset=[id_col]))

    # Dağılım: kaç müşteri kaç değişkende outlier
    _vc = outlier_count.value_counts().reset_index()
    _vc.columns = ["Outlier Değişken Sayısı", "Müşteri Sayısı"]
    dist_df = _vc[_vc["Outlier Değişken Sayısı"] > 0].sort_values("Outlier Değişken Sayısı")

    # ── Özet tablo renk koşulları ──────────────────────────────────────────────
    pct_hi = float(summary_df["% Outlier"].quantile(0.75)) if len(summary_df) else 5.0
    pct_mi = float(summary_df["% Outlier"].quantile(0.5))  if len(summary_df) else 2.0
    sum_cond = [
        {"if": {"filter_query": f"{{% Outlier}} >= {pct_hi:.2f}", "column_id": "% Outlier"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": f"{{% Outlier}} >= {pct_mi:.2f} && {{% Outlier}} < {pct_hi:.2f}",
                "column_id": "% Outlier"},
         "color": "#f59e0b"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]
    cust_cond = [
        {"if": {"filter_query": "{Outlier Değişken Sayısı} >= 5", "column_id": "Outlier Değişken Sayısı"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": "{Outlier Değişken Sayısı} >= 2 && {Outlier Değişken Sayısı} < 5",
                "column_id": "Outlier Değişken Sayısı"},
         "color": "#f59e0b"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    method_lbl = f"IQR × {k}" if method == "iqr" else f"Z-score > {k}"
    n_any = int((outlier_count > 0).sum())
    pct_any = round(n_any / len(df) * 100, 1) if len(df) else 0

    # ── Seçili değişken grafikleri ─────────────────────────────────────────────
    charts = html.Div()
    if vis_var and vis_var in df.columns:
        mask_v, lo_v, hi_v = _outlier_mask(df[vis_var])
        s = df[vis_var].dropna()

        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df[vis_var], name=vis_var,
            marker=dict(color="#4F8EF7", outliercolor="#ef4444",
                        line=dict(outliercolor="#ef4444", outlierwidth=2)),
            line=dict(color="#4F8EF7"),
            fillcolor="rgba(79,142,247,0.15)",
            boxmean=True,
        ))
        fig_box.add_hline(y=lo_v, line=dict(color="#f59e0b", dash="dash", width=1))
        fig_box.add_hline(y=hi_v, line=dict(color="#f59e0b", dash="dash", width=1))
        fig_box.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=f"{vis_var} — Box Plot  ({method_lbl})",
                       font=dict(color="#E8EAF0", size=13)),
            yaxis=dict(**_AXIS_STYLE),
            height=340, showlegend=False,
        )

        # Histogram — outlier bölgeler shaded
        fig_hist = go.Figure()
        bins_arr = np.histogram_bin_edges(s, bins="auto")
        fig_hist.add_trace(go.Histogram(
            x=df[vis_var], nbinsx=min(60, len(bins_arr)),
            marker=dict(color="#4F8EF7", opacity=0.7),
            name="Dağılım",
        ))
        # Outlier bölgesi — sol ve sağ shaded rect
        y_max_est = len(df) * 0.25  # yaklaşık üst sınır (layout sonra otomatik ayarlar)
        for x0, x1, label in [
            (float(s.min()), lo_v, f"< {lo_v:.3g}"),
            (hi_v, float(s.max()), f"> {hi_v:.3g}"),
        ]:
            if x0 < x1:
                fig_hist.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="rgba(239,68,68,0.12)",
                    line=dict(color="#ef4444", width=1, dash="dot"),
                    annotation_text=label,
                    annotation_position="top left",
                    annotation=dict(font=dict(color="#ef4444", size=9)),
                )
        fig_hist.add_vline(x=lo_v, line=dict(color="#f59e0b", dash="dash", width=1))
        fig_hist.add_vline(x=hi_v, line=dict(color="#f59e0b", dash="dash", width=1))
        fig_hist.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=f"{vis_var} — Dağılım  ·  Outlier: {int(mask_v.sum())} ({round(mask_v.mean()*100,1)}%)",
                       font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE),
            yaxis=dict(**_AXIS_STYLE),
            height=340, showlegend=False,
            bargap=0.05,
        )

        charts = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_box,  config={"displayModeBar": False}), width=5),
            dbc.Col(dcc.Graph(figure=fig_hist, config={"displayModeBar": False}), width=7),
        ], className="mb-3")

    return html.Div([
        # Özet banner
        html.Div([
            html.Span(f"Yöntem: {method_lbl}", style={"color": "#a8b2c2", "fontSize": "0.78rem"}),
            html.Span("  ·  ", style={"color": "#4a5568"}),
            html.Span(f"Taranan değişken: {len(num_cols)}",
                      style={"color": "#a8b2c2", "fontSize": "0.78rem"}),
            html.Span("  ·  ", style={"color": "#4a5568"}),
            html.Span(f"En az 1 outlier olan satır: {n_any:,} ({pct_any}%)",
                      style={"color": "#f59e0b", "fontSize": "0.78rem", "fontWeight": "600"}),
        ], className="mb-3"),

        # Grafikler
        charts,

        # Değişken özet tablosu
        dbc.Row([
            dbc.Col([
                html.P("Değişken Bazında Outlier Özeti", className="section-title"),
                dash_table.DataTable(
                    data=summary_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in summary_df.columns],
                    sort_action="native", page_size=15,
                    style_data_conditional=sum_cond,
                    style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                                  "border": "1px solid #2d3a4f", "fontSize": "0.78rem"},
                    **{k2: v2 for k2, v2 in _TABLE_STYLE.items()
                       if k2 != "style_data_conditional"},
                ),
            ], width=8),
            dbc.Col([
                html.P("Outlier Dağılımı (Kaç Değişkende?)", className="section-title"),
                dash_table.DataTable(
                    data=dist_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in dist_df.columns],
                    sort_action="native",
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
                    ],
                    **{k2: v2 for k2, v2 in _TABLE_STYLE.items()
                       if k2 != "style_data_conditional"},
                ),
            ], width=4),
        ], className="mb-3"),

        # Müşteri detay tablosu
        html.P(f"Müşteri Bazında Outlier Detayı  ·  {id_col}", className="section-title"),
        html.Div(f"{len(cust_df):,} müşteri en az 1 değişkende outlier.",
                 className="form-hint", style={"marginBottom": "0.4rem"}),
        dash_table.DataTable(
            data=cust_df.head(1000).to_dict("records"),
            columns=[{"name": c, "id": c} for c in cust_df.columns],
            sort_action="native", page_size=20, filter_action="native",
            style_data_conditional=cust_cond,
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                          "border": "1px solid #2d3a4f", "fontSize": "0.78rem"},
            **{k2: v2 for k2, v2 in _TABLE_STYLE.items()
               if k2 != "style_data_conditional"},
        ),
    ])
