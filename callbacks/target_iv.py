from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _tab_info, _PLOT_LAYOUT, _AXIS_STYLE
from modules.target_analysis import compute_target_stats, compute_target_over_time
from modules.deep_dive import compute_iv_ranking_optimal


# ── Callback: Target & IV Sekmesi ─────────────────────────────────────────────
@app.callback(
    Output("tab-target-iv", "children"),
    Input("store-config", "data"),
    Input("dd-segment-val", "value"),
    State("store-key", "data"),
    State("dd-segment-col", "value"),
)
def update_target_iv(config, seg_val, key, seg_col_input):
    df_orig = _get_df(key)
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()

    target    = config["target_col"]
    date_col  = config.get("date_col")
    seg_col   = config.get("segment_col") or (seg_col_input or None)
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    # ── IV cache: aynı key+segment için yeniden hesaplama yapma ──────────────
    cache_key = f"{key}_iv_{seg_col}_{seg_val}"
    if cache_key in _SERVER_STORE:
        iv_df = _SERVER_STORE[cache_key]
    else:
        iv_df = compute_iv_ranking_optimal(df_active, target)
        _SERVER_STORE[cache_key] = iv_df

    stats = compute_target_stats(df_active, target)

    # ── 1. Target İstatistik Kartları ─────────────────────────────────────────
    def tcard(value, label, color="#4F8EF7"):
        return dbc.Col(html.Div([
            html.Div(str(value), className="metric-value", style={"color": color}),
            html.Div(label, className="metric-label"),
        ], className="metric-card"), width=2)

    imbalance_color = "#ef4444" if stats["bad_rate"] < 5 or stats["bad_rate"] > 50 else "#f59e0b" if stats["bad_rate"] < 15 else "#10b981"

    stats_row = dbc.Row([
        tcard(f"{stats['valid']:,}",         "Geçerli Kayıt"),
        tcard(f"{stats['bad']:,}",           "Bad (1)",   "#ef4444"),
        tcard(f"{stats['good']:,}",          "Good (0)",  "#10b981"),
        tcard(f"%{stats['bad_rate']:.2f}",   "Bad Rate",  imbalance_color),
        tcard(f"{stats['ratio']:.1f}x",      "Good/Bad Oran"),
        tcard(f"{stats['missing']:,}",       "Target Eksik", "#f59e0b" if stats["missing"] > 0 else "#556070"),
    ], className="g-3 mb-4")

    # ── 2. Bad Rate Over Time (date_col varsa) ────────────────────────────────
    time_chart = html.Div()
    if date_col and date_col in df_active.columns:
        time_df = compute_target_over_time(df_active, target, date_col)
        if len(time_df) > 1:
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=time_df["Tarih"], y=time_df["total_count"],
                name="Toplam", marker_color="#232d4f", yaxis="y2", opacity=0.6,
            ))
            fig_time.add_trace(go.Scatter(
                x=time_df["Tarih"], y=time_df["bad_rate"],
                name="Bad Rate %", mode="lines+markers",
                line=dict(color="#ef4444", width=2),
                marker=dict(size=5),
            ))
            fig_time.update_layout(
                **_PLOT_LAYOUT,
                title=dict(text="Bad Rate Zaman Serisi", font=dict(color="#E8EAF0", size=13)),
                xaxis={**_AXIS_STYLE},
                yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
                yaxis2=dict(title="Kayıt Sayısı", overlaying="y", side="right",
                            showgrid=False),
                legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
                hovermode="x unified",
                height=320,
            )
            time_chart = html.Div([
                html.P("Bad Rate Trendi", className="section-title"),
                dcc.Graph(figure=fig_time, config={"displayModeBar": False}),
            ], style={"marginBottom": "2rem"})

    # ── 3. IV Ranking ─────────────────────────────────────────────────────────

    iv_color_map = {
        "Çok Zayıf": "#4a5568",
        "Zayıf":     "#f59e0b",
        "Orta":      "#4F8EF7",
        "Güçlü":     "#10b981",
        "Şüpheli":   "#ef4444",
    }

    # IV Bar chart (top 25)
    top_iv = iv_df.head(25).iloc[::-1]  # ters sıra — en yüksek üstte
    bar_colors = [iv_color_map.get(g, "#4F8EF7") for g in top_iv["Güç"]]

    fig_iv = go.Figure(go.Bar(
        x=top_iv["IV"],
        y=top_iv["Değişken"],
        orientation="h",
        marker_color=bar_colors,
        text=top_iv["IV"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
        textfont=dict(size=10, color="#8892a4"),
        hovertemplate="<b>%{y}</b><br>IV: %{x:.4f}<extra></extra>",
    ))
    fig_iv.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text="IV Liderlik Tablosu (Top 25)", font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(**_AXIS_STYLE, title="Information Value"),
        yaxis=dict(**_AXIS_STYLE, tickfont=dict(size=10)),
        height=max(400, len(top_iv) * 26),
        showlegend=False,
    )
    # Eşik çizgileri
    for thresh, label, color in [(0.02, "Zayıf", "#4a5568"), (0.10, "Orta", "#f59e0b"),
                                  (0.30, "Güçlü", "#10b981"), (0.50, "Şüpheli", "#ef4444")]:
        fig_iv.add_vline(x=thresh, line_dash="dot", line_color=color, opacity=0.5,
                         annotation_text=label, annotation_font_color=color,
                         annotation_font_size=9)

    # IV Tablo
    iv_cond = [
        {"if": {"filter_query": '{Güç} = "Güçlü"',    "column_id": "Güç"}, "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Güç} = "Orta"',     "column_id": "Güç"}, "color": "#4F8EF7", "fontWeight": "600"},
        {"if": {"filter_query": '{Güç} = "Zayıf"',    "column_id": "Güç"}, "color": "#f59e0b"},
        {"if": {"filter_query": '{Güç} = "Şüpheli"',  "column_id": "Güç"}, "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Güç} = "Çok Zayıf"',"column_id": "Güç"}, "color": "#7e8fa4"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    iv_display = iv_df

    iv_tsv = iv_display.to_csv(sep="\t", index=False)
    iv_table = html.Div([
        html.Div(
            dcc.Clipboard(target_id="iv-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
            style={"overflow": "hidden"},
        ),
        html.Pre(iv_tsv, id="iv-tsv", style={"display": "none"}),
        dash_table.DataTable(
            data=iv_display.to_dict("records"),
            columns=[{"name": c, "id": c} for c in iv_display.columns],
            sort_action="native",
            filter_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#111827", "color": "#a8b2c2",
                "fontWeight": "700", "fontSize": "0.73rem",
                "border": "1px solid #2d3a4f",
                "textTransform": "uppercase", "letterSpacing": "0.06em",
            },
            style_data={
                "backgroundColor": "#161C27", "color": "#c8cdd8",
                "fontSize": "0.83rem", "border": "1px solid #232d3f",
            },
            style_data_conditional=iv_cond,
            style_cell={"padding": "0.45rem 0.75rem"},
            style_cell_conditional=[
                {"if": {"column_id": ["IV", "Bin Sayısı", "Eksik %"]}, "textAlign": "right"},
            ],
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8", "border": "1px solid #2d3a4f"},
            css=[{"selector": ".dash-filter input", "rule": "color: #c8cdd8 !important;"}],
        ),
    ])

    return html.Div([
        _tab_info("Target & IV", "Bad Rate · Zaman Trendi · Information Value",
                  "Target değişkeninin genel dağılımını, zaman içindeki seyrini ve her değişkenin "
                  "IV (Information Value) sıralamasını gösterir. IV < 0.02 anlamsız, 0.02–0.10 zayıf, "
                  "0.10–0.30 orta, > 0.30 güçlü ayırıcı güç.",
                  "#10b981"),
        html.P("Target Dağılımı", className="section-title"),
        stats_row,
        time_chart,
        dbc.Row([
            dbc.Col([
                html.P("IV Sıralaması", className="section-title"),
                dcc.Graph(figure=fig_iv, config={"displayModeBar": False}),
            ], width=6),
            dbc.Col([
                html.P("IV Tablosu", className="section-title"),
                iv_table,
            ], width=6),
        ]),
        html.Div([
            html.Span("IV Eşikleri: ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("< 0.02 Çok Zayıf  · ", style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("0.02–0.10 Zayıf  · ", style={"color": "#f59e0b", "fontSize": "0.73rem"}),
            html.Span("0.10–0.30 Orta  · ", style={"color": "#4F8EF7", "fontSize": "0.73rem"}),
            html.Span("0.30–0.50 Güçlü  · ", style={"color": "#10b981", "fontSize": "0.73rem"}),
            html.Span("> 0.50 Şüpheli", style={"color": "#ef4444", "fontSize": "0.73rem"}),
        ], style={"marginTop": "0.75rem"}),
    ])
