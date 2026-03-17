from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _PLOT_LAYOUT, _AXIS_STYLE, _make_r_badge, _safe_pair_scatter, _make_pair_scatter
from modules.correlation import get_numeric_cols, compute_correlation_matrix, find_high_corr_pairs, compute_vif


# ── Callback: Korelasyon ──────────────────────────────────────────────────────
@app.callback(
    Output("corr-content", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    Input("corr-threshold", "value"),
    Input("corr-max-cols", "value"),
    Input("store-expert-exclude", "data"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def render_correlation_content(config, seg_val, threshold, max_cols_str, expert_excluded, key, seg_col_input):
    if not key or not config or not config.get("target_col"):
        return html.Div()

    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div()

    threshold  = float(threshold or 0.75)
    max_cols   = int(max_cols_str or 20)
    target     = config["target_col"]
    seg_col    = config.get("segment_col") or (seg_col_input or None)
    df_active  = apply_segment_filter(df_orig, seg_col, seg_val)
    expert_excluded_set = set(expert_excluded or [])

    # ── Cache: Korelasyon Matrisi ─────────────────────────────────────────────
    cache_key = f"{key}_corr_{seg_col}_{seg_val}_{max_cols}"
    if cache_key in _SERVER_STORE:
        corr_df, cols = _SERVER_STORE[cache_key]
        cols = [c for c in cols if c not in expert_excluded_set]
        corr_df = corr_df.loc[cols, cols] if cols else corr_df
    else:
        excl = [c for c in [target, config.get("date_col")] if c]
        cols = get_numeric_cols(df_active, exclude=excl, max_cols=max_cols)
        screen_result = _SERVER_STORE.get(f"{key}_screen")
        if screen_result:
            passed_set = set(screen_result[0])
            cols = [c for c in cols if c in passed_set]
        cols = [c for c in cols if c not in expert_excluded_set]
        if len(cols) < 2:
            return html.Div("Yeterli sayıda numerik kolon bulunamadı (en az 2 gerekli).",
                            className="alert-info-custom")
        corr_df = compute_correlation_matrix(df_active, cols)
        _SERVER_STORE[cache_key] = (corr_df, cols)

    # ── VIF: sadece IV ≥ 0.10 olan değişkenler ───────────────────────────────
    iv_cache_key = f"{key}_iv_{seg_col}_{seg_val}"
    vif_cols = cols
    iv_filtered = False
    if iv_cache_key in _SERVER_STORE:
        iv_df_cached = _SERVER_STORE[iv_cache_key]
        iv_high = set(iv_df_cached[iv_df_cached["IV"] >= 0.10]["Değişken"].tolist())
        filtered = [c for c in cols if c in iv_high]
        if len(filtered) >= 2:
            vif_cols = filtered
            iv_filtered = True

    vif_cache_key = f"{key}_vif_{seg_col}_{seg_val}_{max_cols}"
    if vif_cache_key in _SERVER_STORE:
        vif_df = _SERVER_STORE[vif_cache_key]
        # iv_filtered durumunu cache'den türet
        iv_filtered = "En Benzer" in vif_df.columns if not vif_df.empty else iv_filtered
    else:
        vif_df = compute_vif(df_active, vif_cols) if len(vif_cols) >= 2 else pd.DataFrame()

        # "En Benzer" kolonu: hangi değişkenle en yüksek korelasyon
        if not vif_df.empty:
            try:
                sub = corr_df.loc[
                    [v for v in vif_cols if v in corr_df.index],
                    [v for v in vif_cols if v in corr_df.columns],
                ]
                en_benzer = []
                for var in vif_df["Değişken"]:
                    if var not in sub.columns:
                        en_benzer.append("—")
                        continue
                    row = sub[var].drop(var, errors="ignore")
                    if row.empty:
                        en_benzer.append("—")
                        continue
                    top = row.abs().idxmax()
                    en_benzer.append(f"{top}  (r = {row[top]:+.3f})")
                vif_df = vif_df.copy()
                vif_df.insert(2, "En Benzer", en_benzer)
            except Exception:
                pass

        _SERVER_STORE[vif_cache_key] = vif_df

    # ── 1. Heatmap ────────────────────────────────────────────────────────────
    show_text = len(cols) <= 18
    # Dark-theme uyumlu colorscale: 0 → plot arka planı, negatif → kırmızı, pozitif → mavi
    _dark_corr_scale = [
        [0.00, "#b91c1c"],  # -1.0  koyu kırmızı
        [0.35, "#ef4444"],  # -0.3  açık kırmızı
        [0.45, "#2d3a4f"],  # -0.1  arka plana yakın
        [0.50, "#0E1117"],  # 0.0   plot arka planı
        [0.55, "#1e3a5f"],  # +0.1  arka plana yakın
        [0.65, "#4F8EF7"],  # +0.3  açık mavi
        [1.00, "#1d4ed8"],  # +1.0  koyu mavi
    ]
    fig_heat = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=cols, y=cols,
        colorscale=_dark_corr_scale,
        zmid=0, zmin=-1, zmax=1,
        text=corr_df.round(2).values if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=9, color="#E8EAF0"),
        colorbar=dict(
            title=dict(text="r", font=dict(color="#8892a4", size=11)),
            thickness=12, len=0.8,
            tickfont=dict(color="#8892a4", size=10),
        ),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.4f}<extra></extra>",
    ))
    cell_px = max(18, min(40, 600 // max(len(cols), 1)))
    fig_heat.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Korelasyon Matrisi  ({len(cols)} değişken)",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#8892a4"),
                   showgrid=False, linecolor="#232d3f"),
        yaxis=dict(tickfont=dict(size=9, color="#8892a4"),
                   showgrid=False, linecolor="#232d3f", autorange="reversed"),
        height=max(400, len(cols) * cell_px + 100),
    )
    fig_heat.update_layout(margin=dict(l=120, r=40, t=50, b=120))

    # ── 2. Tüm çiftler tablosu (|r| büyükten küçüğe) ─────────────────────────
    all_pairs = find_high_corr_pairs(corr_df, threshold=0.60)

    # ── 3. Korelasyon Çifti dropdownları ──────────────────────────────────────
    var_opts = [{"label": c, "value": c} for c in cols]
    default2 = cols[1] if len(cols) > 1 else cols[0]
    try:
        init_r = float(df_active[[cols[0], default2]].corr().iloc[0, 1])
    except Exception:
        init_r = float("nan")

    # ── 4. VIF Tablosu ────────────────────────────────────────────────────────
    vif_section = html.Div()
    if vif_df is not None and not vif_df.empty:
        vif_cond = [
            {"if": {"filter_query": '{Uyarı} = "✗ Yüksek"', "column_id": "Uyarı"},
             "color": "#ef4444", "fontWeight": "700"},
            {"if": {"filter_query": '{Uyarı} = "⚠ Orta"',   "column_id": "Uyarı"},
             "color": "#f59e0b", "fontWeight": "600"},
            {"if": {"filter_query": '{Uyarı} = "✓ Normal"',  "column_id": "Uyarı"},
             "color": "#10b981"},
            {"if": {"filter_query": "{VIF} >= 10", "column_id": "VIF"},
             "color": "#ef4444", "fontWeight": "700"},
            {"if": {"filter_query": "{VIF} >= 5 && {VIF} < 10", "column_id": "VIF"},
             "color": "#f59e0b"},
            {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
        ]
        vif_tsv = vif_df.to_csv(sep="\t", index=False)
        iv_note = (
            html.Div("IV ≥ 0.10 olan değişkenler üzerinden hesaplandı",
                     className="form-hint", style={"marginBottom": "0.4rem"})
            if iv_filtered else html.Div()
        )
        vif_section = html.Div([
            html.Div(
                dcc.Clipboard(target_id="vif-tsv", title="Kopyala",
                              style={"cursor": "pointer", "fontSize": "0.72rem",
                                     "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                     "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                     "backgroundColor": "#1a2035", "float": "right",
                                     "marginBottom": "0.4rem"}),
                style={"overflow": "hidden"},
            ),
            html.Pre(vif_tsv, id="vif-tsv", style={"display": "none"}),
            iv_note,
            dash_table.DataTable(
                data=vif_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in vif_df.columns],
                sort_action="native",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                              "fontWeight": "700", "fontSize": "0.72rem",
                              "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
                style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                            "fontSize": "0.82rem", "border": "1px solid #232d3f"},
                style_data_conditional=vif_cond,
                style_cell={"padding": "0.4rem 0.65rem"},
                style_cell_conditional=[
                    {"if": {"column_id": "VIF"}, "textAlign": "right"},
                    {"if": {"column_id": "En Benzer"},
                     "color": "#a8b2c2", "fontSize": "0.78rem"},
                ],
            ),
        ])
    elif iv_filtered is False and not (vif_df is not None and not vif_df.empty):
        vif_section = html.Div(
            "Target & IV sekmesini açarak IV hesaplatın — VIF, IV ≥ 0.10 değişkenler için otomatik filtrelenir.",
            className="form-hint",
        )

    # ── Legend notu ───────────────────────────────────────────────────────────
    legend = html.Div([
        html.Span("Korelasyon: ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
        html.Span("Mavi = Pozitif  · ", style={"color": "#4F8EF7", "fontSize": "0.73rem"}),
        html.Span("Kırmızı = Negatif  · ", style={"color": "#ef4444", "fontSize": "0.73rem"}),
        html.Span("VIF Eşikleri: ", style={"color": "#7e8fa4", "fontSize": "0.73rem",
                                           "marginLeft": "1rem"}),
        html.Span("< 5 Normal  · ", style={"color": "#10b981", "fontSize": "0.73rem"}),
        html.Span("5–10 Orta  · ", style={"color": "#f59e0b", "fontSize": "0.73rem"}),
        html.Span("> 10 Yüksek", style={"color": "#ef4444", "fontSize": "0.73rem"}),
    ], style={"marginTop": "0.5rem", "marginBottom": "1.5rem"})

    # ── Pairs tablosu koşullu renk ────────────────────────────────────────────
    pairs_cond = [
        {"if": {"filter_query": "{|Korelasyon|} >= 0.9",  "column_id": "|Korelasyon|"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": "{|Korelasyon|} >= 0.75 && {|Korelasyon|} < 0.9",
                "column_id": "|Korelasyon|"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": "{|Korelasyon|} >= 0.5 && {|Korelasyon|} < 0.75",
                "column_id": "|Korelasyon|"},
         "color": "#4F8EF7"},
        {"if": {"filter_query": "{Korelasyon} < 0", "column_id": "Korelasyon"},
         "color": "#ef4444"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]
    pairs_tsv = all_pairs.to_csv(sep="\t", index=False)

    return html.Div([
        html.P("Korelasyon Matrisi", className="section-title"),
        dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
        legend,

        # ── Çiftler tablosu + VIF ──────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.P("Korelasyon Çiftleri", className="section-title"),
                html.Div(
                    dcc.Clipboard(target_id="pairs-tsv", title="Kopyala",
                                  style={"cursor": "pointer", "fontSize": "0.72rem",
                                         "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                         "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                         "backgroundColor": "#1a2035", "float": "right",
                                         "marginBottom": "0.4rem"}),
                    style={"overflow": "hidden"},
                ),
                html.Pre(pairs_tsv, id="pairs-tsv", style={"display": "none"}),
                dash_table.DataTable(
                    data=all_pairs.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in all_pairs.columns],
                    sort_action="native",
                    page_size=15,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                                  "fontWeight": "700", "fontSize": "0.72rem",
                                  "border": "1px solid #2d3a4f",
                                  "textTransform": "uppercase"},
                    style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                                "fontSize": "0.82rem", "border": "1px solid #232d3f"},
                    style_data_conditional=pairs_cond,
                    style_cell={"padding": "0.4rem 0.65rem"},
                    style_cell_conditional=[
                        {"if": {"column_id": ["Korelasyon", "|Korelasyon|"]},
                         "textAlign": "right"},
                    ],
                ),
                html.Div([
                    html.Span("|r| renk kodları — ",
                              style={"color": "#7e8fa4", "fontSize": "0.72rem"}),
                    html.Span("≥ 0.90 ", style={"color": "#ef4444", "fontSize": "0.72rem", "fontWeight": "700"}),
                    html.Span("≥ 0.75 ", style={"color": "#f59e0b", "fontSize": "0.72rem", "fontWeight": "600"}),
                    html.Span("≥ 0.50 ", style={"color": "#4F8EF7", "fontSize": "0.72rem"}),
                    html.Span("  ·  Negatif r kırmızı",
                              style={"color": "#ef4444", "fontSize": "0.72rem",
                                     "marginLeft": "0.5rem"}),
                ], style={"marginTop": "0.4rem"}),
            ], width=6),
            dbc.Col([
                html.P("VIF — Çoklu Doğrusallık", className="section-title"),
                vif_section,
            ], width=6),
        ], className="mb-4"),

        # ── İkili Analiz ───────────────────────────────────────────────────────
        html.P("İkili Değişken Analizi", className="section-title"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Değişken 1", className="form-label"),
                dbc.Select(id="corr-var1", options=var_opts, value=cols[0],
                           className="dark-select"),
            ], width=5),
            dbc.Col([
                dbc.Label("Değişken 2", className="form-label"),
                dbc.Select(id="corr-var2", options=var_opts, value=default2,
                           className="dark-select"),
            ], width=5),
            dbc.Col([
                dbc.Label("\u00a0", className="form-label"),
                html.Div(id="corr-r-badge", children=_make_r_badge(init_r)),
            ], width=2, style={"display": "flex", "alignItems": "flex-end"}),
        ], className="mb-3", align="end"),
        dcc.Loading(
            html.Div(id="pair-scatter",
                     children=_safe_pair_scatter(df_active, cols[0], default2, target)),
            type="dot", color="#4F8EF7", delay_show=250,
        ),
    ])


# ── Callback: Çift Scatter ────────────────────────────────────────────────────
@app.callback(
    Output("pair-scatter", "children"),
    Output("corr-r-badge", "children"),
    Input("corr-var1", "value"),
    Input("corr-var2", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("dd-segment-val", "value"),
    prevent_initial_call=True,
)
def render_pair_scatter(var1, var2, key, config, seg_val):
    if not var1 or not var2 or not key or not config:
        return html.Div(), html.Div()
    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div(), html.Div()
    seg_col   = config.get("segment_col")
    target    = config["target_col"]
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    is_num1 = pd.api.types.is_numeric_dtype(df_active[var1]) if var1 in df_active.columns else False
    is_num2 = pd.api.types.is_numeric_dtype(df_active[var2]) if var2 in df_active.columns else False

    if is_num1 and is_num2:
        try:
            r = float(df_active[[var1, var2]].corr().iloc[0, 1])
        except Exception:
            r = float("nan")
        r_badge = _make_r_badge(r)
    else:
        r_badge = html.Div(
            html.Div("Kategorik × Sayısal",
                     style={"fontSize": "0.72rem", "color": "#7e8fa4",
                            "fontWeight": "600", "letterSpacing": "0.06em",
                            "textTransform": "uppercase"}),
            className="metric-card",
            style={"padding": "0.55rem 1rem", "minWidth": "120px",
                   "textAlign": "center"},
        )

    try:
        pair_chart = _make_pair_scatter(df_active, var1, var2, target)
    except Exception as exc:
        pair_chart = html.Div(
            f"Grafik oluşturulamadı: {exc}",
            style={"color": "#ef4444", "padding": "1rem", "fontSize": "0.8rem"},
        )
    return pair_chart, r_badge
