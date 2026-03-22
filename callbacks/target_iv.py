import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import _tab_info, _PLOT_LAYOUT, _AXIS_STYLE
from modules.target_analysis import compute_target_stats, compute_target_over_time


def _tcard(value, label, color="#4F8EF7"):
    return dbc.Col(html.Div([
        html.Div(str(value), className="metric-value", style={"color": color}),
        html.Div(label, className="metric-label"),
    ], className="metric-card"), width=2)


def _binary_stats_row(df, target):
    stats = compute_target_stats(df, target)
    imbalance_color = (
        "#ef4444" if stats["bad_rate"] < 5 or stats["bad_rate"] > 50
        else "#f59e0b" if stats["bad_rate"] < 15 else "#10b981"
    )
    return dbc.Row([
        _tcard(f"{stats['valid']:,}",        "Geçerli Kayıt"),
        _tcard(f"{stats['bad']:,}",          "Bad (1)",   "#ef4444"),
        _tcard(f"{stats['good']:,}",         "Good (0)",  "#10b981"),
        _tcard(f"%{stats['bad_rate']:.2f}",  "Bad Rate",  imbalance_color),
        _tcard(f"{stats['ratio']:.1f}x",     "Good/Bad Oran"),
        _tcard(f"{stats['missing']:,}",      "Target Eksik",
               "#f59e0b" if stats["missing"] > 0 else "#556070"),
    ], className="g-3 mb-2")


def _render_trend_chart(df_plot, target, date_col, period_label):
    """Verilen df_plot için binary bad-rate trend grafiği HTML döndürür."""
    time_df = compute_target_over_time(df_plot, target, date_col)
    if len(time_df) < 2:
        return html.Div(f"Yeterli veri yok ({period_label}).",
                        style={"color": "#7e8fa4", "fontSize": "0.8rem"})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=time_df["Tarih"], y=time_df["total_count"],
        name="Toplam", marker_color="#232d4f", yaxis="y2", opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=time_df["Tarih"], y=time_df["bad_rate"],
        name="Bad Rate %", mode="lines+markers",
        line=dict(color="#ef4444", width=2), marker=dict(size=5),
    ))
    fig.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Bad Rate Trendi — {period_label}",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis={**_AXIS_STYLE},
        yaxis=dict(**_AXIS_STYLE, title="Bad Rate %", ticksuffix="%"),
        yaxis2=dict(title="Kayıt Sayısı", overlaying="y",
                    side="right", showgrid=False),
        legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
        hovermode="x unified", height=320,
    )
    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ], style={"marginBottom": "2rem"})


# ── Callback: Target & IV Sekmesi ─────────────────────────────────────────────
@app.callback(
    Output("tab-target-iv", "children"),
    Input("store-config", "data"),
    Input("store-expert-exclude", "data"),
    Input("interval-precompute", "disabled"),
    State("store-key", "data"),
)
def update_target_iv(config, expert_excluded, _precompute_done, key):
    df_orig = _get_df(key)
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()

    target      = config["target_col"]
    date_col    = config.get("date_col")
    oot_date    = config.get("oot_date")
    seg_col     = config.get("segment_col")
    seg_val     = config.get("segment_val")
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val)
    excluded_set = set(expert_excluded or [])

    # IV için sadece train verisi kullan
    df_train, df_test, df_oot = get_splits(df_active, config)

    # ── Trend periyot dropdown (date_col varsa) ────────────────────────────────
    def _build_trend_dropdown():
        if not date_col or date_col not in df_active.columns:
            return html.Div(), []
        opts = [{"label": "Tüm Veri", "value": "all"}]
        if oot_date:
            opts.append({"label": "Train (OOT öncesi)", "value": "train"})
            if df_test is not None and len(df_test) > 0:
                opts.append({"label": "Test", "value": "test"})
            if df_oot is not None and len(df_oot) > 0:
                opts.append({"label": "OOT", "value": "oot"})
        row = dbc.Row([
            dbc.Col([
                dbc.Label("Trend Periyodu", className="form-label",
                          style={"fontSize": "0.78rem", "marginBottom": "3px"}),
                dbc.Select(
                    id="dd-trend-period",
                    options=opts,
                    value="all",
                    className="dark-select",
                    style={"maxWidth": "220px", "fontSize": "0.82rem"},
                ),
            ], width="auto"),
        ], className="mb-3 align-items-end")
        return row, [o["value"] for o in opts]

    trend_dropdown, _trend_opts = _build_trend_dropdown()

    # IV — sadece train üzerinden
    # IV — sadece cache'den oku (hesaplama kaldırıldı — yeniden yazılacak)
    cache_key = f"{key}_iv_{seg_col}_{seg_val}"
    iv_df = _SERVER_STORE.get(cache_key)
    if iv_df is None:
        iv_df = pd.DataFrame(columns=["Değişken", "IV", "Eksik %", "Güç"])

    # Elenen değişkenleri IV tablosundan çıkar
    if excluded_set and "Değişken" in iv_df.columns:
        iv_df = iv_df[~iv_df["Değişken"].isin(excluded_set)].reset_index(drop=True)

    # Train stats badge
    tr_stats  = compute_target_stats(df_train, target)
    train_note = html.Div(
        f"IV Train verisi: n={len(df_train):,}  ·  Bad Rate %{tr_stats['bad_rate']:.2f}"
        + (f"  ·  OOT ≥ {oot_date}" if oot_date else "  ·  Rastgele split"),
        style={"color": "#a78bfa", "fontSize": "0.73rem",
               "marginBottom": "0.75rem"},
    )

    # İlk gösterim: "Tüm Veri" varsayılan seçenek
    stats_row = html.Div(
        _binary_stats_row(df_active, target),
        id="trend-stats-row",
    )

    time_chart = html.Div()
    if date_col and date_col in df_active.columns:
        _trend_store_data = {
            "key": key, "target": target, "date_col": date_col,
            "seg_col": seg_col, "seg_val": seg_val,
            "oot_date": oot_date,
            "has_test_split": config.get("has_test_split", False),
            "test_size": config.get("test_size", 20),
        }
        time_chart = html.Div([
            trend_dropdown,
            dcc.Store(id="store-trend-config", data=_trend_store_data),
            html.Div(
                _render_trend_chart(df_active, target, date_col, "Tüm Veri"),
                id="trend-chart-container",
            ),
        ])

    iv_label_note = html.Div(
        "ℹ IV sadece Train verisi üzerinden hesaplanmıştır.",
        style={"color": "#7e8fa4", "fontSize": "0.72rem",
               "marginBottom": "0.5rem"},
    ) if oot_date else html.Div()

    iv_color_map = {
        "Çok Zayıf": "#4a5568", "Zayıf": "#f59e0b",
        "Orta": "#4F8EF7",      "Güçlü": "#10b981", "Şüpheli": "#ef4444",
    }
    top_iv     = iv_df.head(25).iloc[::-1]
    bar_colors = [iv_color_map.get(g, "#4F8EF7") for g in top_iv["Güç"]]
    fig_iv = go.Figure(go.Bar(
        x=top_iv["IV"], y=top_iv["Değişken"], orientation="h",
        marker_color=bar_colors,
        text=top_iv["IV"].apply(lambda x: f"{x:.4f}"),
        textposition="outside", textfont=dict(size=10, color="#8892a4"),
        hovertemplate="<b>%{y}</b><br>IV: %{x:.4f}<extra></extra>",
    ))
    fig_iv.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text="IV Liderlik Tablosu (Top 25)",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(**_AXIS_STYLE, title="Information Value"),
        yaxis=dict(**_AXIS_STYLE, tickfont=dict(size=10)),
        height=max(400, len(top_iv) * 26), showlegend=False,
    )
    for thresh, lbl, clr in [(0.02, "Zayıf", "#4a5568"), (0.10, "Orta", "#f59e0b"),
                              (0.30, "Güçlü", "#10b981"), (0.50, "Şüpheli", "#ef4444")]:
        fig_iv.add_vline(x=thresh, line_dash="dot", line_color=clr, opacity=0.5,
                         annotation_text=lbl, annotation_font_color=clr,
                         annotation_font_size=9)

    iv_cond = [
        {"if": {"filter_query": '{Güç} = "Güçlü"',    "column_id": "Güç"},
         "color": "#10b981", "fontWeight": "700"},
        {"if": {"filter_query": '{Güç} = "Orta"',     "column_id": "Güç"},
         "color": "#4F8EF7", "fontWeight": "600"},
        {"if": {"filter_query": '{Güç} = "Zayıf"',    "column_id": "Güç"},
         "color": "#f59e0b"},
        {"if": {"filter_query": '{Güç} = "Şüpheli"',  "column_id": "Güç"},
         "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": '{Güç} = "Çok Zayıf"',"column_id": "Güç"},
         "color": "#7e8fa4"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]
    iv_tsv   = iv_df.to_csv(sep="\t", index=False)
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
            data=iv_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in iv_df.columns],
            sort_action="native", filter_action="native", page_size=20,
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
                {"if": {"column_id": ["IV", "Bin Sayısı", "Eksik %"]},
                 "textAlign": "right"},
            ],
            style_filter={"backgroundColor": "#0e1117", "color": "#c8cdd8",
                          "border": "1px solid #2d3a4f"},
            css=[{"selector": ".dash-filter input",
                  "rule": "color: #c8cdd8 !important;"}],
        ),
    ])

    return html.Div([
        _tab_info("Target & IV", "Bad Rate · Zaman Trendi · Information Value",
                  "Target değişkeninin genel dağılımını, zaman içindeki seyrini ve her "
                  "değişkenin IV (Information Value) sıralamasını gösterir. "
                  "IV < 0.02 anlamsız, 0.02–0.10 zayıf, 0.10–0.30 orta, > 0.30 güçlü.",
                  "#10b981"),
        html.P("Target Dağılımı", className="section-title"),
        stats_row,
        time_chart,
        html.P("IV Sıralaması (Train)", className="section-title"),
        train_note,
        iv_label_note,
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_iv, config={"displayModeBar": False}),
            ], width=6),
            dbc.Col([
                html.P("IV Tablosu", className="section-title"),
                iv_table,
            ], width=6),
        ]),
        html.Div([
            html.Span("IV Eşikleri: ",
                      style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("< 0.02 Çok Zayıf  · ",
                      style={"color": "#7e8fa4", "fontSize": "0.73rem"}),
            html.Span("0.02–0.10 Zayıf  · ",
                      style={"color": "#f59e0b", "fontSize": "0.73rem"}),
            html.Span("0.10–0.30 Orta  · ",
                      style={"color": "#4F8EF7", "fontSize": "0.73rem"}),
            html.Span("0.30–0.50 Güçlü  · ",
                      style={"color": "#10b981", "fontSize": "0.73rem"}),
            html.Span("> 0.50 Şüpheli",
                      style={"color": "#ef4444", "fontSize": "0.73rem"}),
        ], style={"marginTop": "0.75rem"}),
    ])


# ── Callback: Trend grafiği + stats — periyot seçimine göre güncelle ─────────
@app.callback(
    Output("trend-chart-container", "children"),
    Output("trend-stats-row", "children"),
    Input("dd-trend-period", "value"),
    State("store-trend-config", "data"),
    prevent_initial_call=True,
)
def update_trend_chart(period_val, trend_cfg):
    _no = dash.no_update
    if not trend_cfg:
        return html.Div(), _no

    key         = trend_cfg["key"]
    target      = trend_cfg["target"]
    date_col    = trend_cfg["date_col"]
    seg_col     = trend_cfg.get("seg_col")
    seg_val     = trend_cfg.get("seg_val")
    oot_date    = trend_cfg.get("oot_date")

    df_orig = _get_df(key)
    if df_orig is None:
        return html.Div(), _no

    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    period = period_val or "all"
    if period != "all" and oot_date:
        df_train, df_test, df_oot = get_splits(df_active, trend_cfg)
        if period == "train":
            df_plot, period_label = df_train, "Train"
        elif period == "test" and df_test is not None:
            df_plot, period_label = df_test, "Test"
        elif period == "oot" and df_oot is not None:
            df_plot, period_label = df_oot, "OOT"
        else:
            df_plot, period_label = df_active, "Tüm Veri"
    else:
        df_plot, period_label = df_active, "Tüm Veri"

    new_stats = _binary_stats_row(df_plot, target)

    return _render_trend_chart(df_plot, target, date_col, period_label), new_stats
