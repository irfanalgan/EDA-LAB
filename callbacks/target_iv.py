import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from utils.helpers import apply_segment_filter, get_splits
from utils.chart_helpers import _tab_info, _PLOT_LAYOUT, _AXIS_STYLE
from modules.target_analysis import compute_target_stats, compute_target_over_time
from modules.deep_dive import compute_iv_ranking_optimal


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


def _render_trend_chart(df_plot, target, date_col, target_type, period_label):
    """Verilen df_plot için trend grafiği HTML döndürür."""
    if target_type == "binary":
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
    else:
        line_color = "#4F8EF7" if target_type == "continuous" else "#a78bfa"
        return _time_chart_generic(df_plot, target, date_col,
                                   f"Ort. {target}", line_color)


def _time_chart_generic(df_active, target, date_col, y_label, line_color):
    """Zaman içinde target ortalaması/oranı — her target tipi için kullanılır."""
    tmp = df_active[[date_col, target]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Veri aralığına göre otomatik resample frekansı seç
    span_days = (tmp.index.max() - tmp.index.min()).days if len(tmp) > 1 else 0
    if span_days <= 90:
        freq = "W"
    elif span_days <= 730:
        freq = "ME"
    elif span_days <= 1825:
        freq = "QE"
    else:
        freq = "YE"

    agg = tmp.resample(freq)[target].agg(
        mean_val="mean", count_val="count"
    ).reset_index().rename(columns={date_col: "Tarih"})
    agg = agg[agg["count_val"] > 0]  # boş periyotları çıkar
    if len(agg) < 2:
        return html.Div()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["Tarih"], y=agg["count_val"],
        name="Kayıt Sayısı", marker_color="#232d4f", yaxis="y2", opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=agg["Tarih"], y=agg["mean_val"],
        name=y_label, mode="lines+markers",
        line=dict(color=line_color, width=2), marker=dict(size=5),
    ))
    fig.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"{y_label} — Zaman Serisi", font=dict(color="#E8EAF0", size=13)),
        xaxis={**_AXIS_STYLE},
        yaxis=dict(**_AXIS_STYLE, title=y_label),
        yaxis2=dict(title="Kayıt Sayısı", overlaying="y", side="right", showgrid=False),
        legend=dict(bgcolor="#161C27", bordercolor="#232d3f"),
        hovermode="x unified", height=320,
    )
    return html.Div([
        html.P(f"{y_label} Trendi", className="section-title"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ], style={"marginBottom": "2rem"})


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

    target      = config["target_col"]
    date_col    = config.get("date_col")
    oot_date    = config.get("oot_date")
    seg_col     = config.get("segment_col") or (seg_col_input or None)
    target_type = config.get("target_type", "binary")
    df_active   = apply_segment_filter(df_orig, seg_col, seg_val)

    # IV için sadece train verisi kullan
    df_train, df_test, df_oot = get_splits(df_active, config)

    s = pd.to_numeric(df_active[target], errors="coerce")
    n_total   = len(s)
    n_missing = int(s.isna().sum())
    n_valid   = n_total - n_missing

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

    # ── Binary ────────────────────────────────────────────────────────────────
    if target_type == "binary":
        # IV — sadece train üzerinden
        oot_key   = config.get("oot_date") or "none"
        cache_key = f"{key}_iv_{seg_col}_{seg_val}_{oot_key}"
        if cache_key in _SERVER_STORE:
            iv_df = _SERVER_STORE[cache_key]
        else:
            iv_df = compute_iv_ranking_optimal(df_train, target)
            _SERVER_STORE[cache_key] = iv_df

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
                "target_type": target_type,
                "seg_col": seg_col, "seg_val": seg_val,
                "oot_date": oot_date,
                "has_test_split": config.get("has_test_split", False),
                "test_size": config.get("test_size", 20),
            }
            time_chart = html.Div([
                trend_dropdown,
                dcc.Store(id="store-trend-config", data=_trend_store_data),
                html.Div(
                    _render_trend_chart(df_active, target, date_col,
                                        target_type, "Tüm Veri"),
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

    # ── Continuous ────────────────────────────────────────────────────────────
    if target_type == "continuous":
        stats_row = dbc.Row([
            _tcard(f"{n_valid:,}",           "Geçerli Kayıt"),
            _tcard(f"{s.mean():.4f}",        "Ortalama",  "#4F8EF7"),
            _tcard(f"{s.median():.4f}",      "Medyan",    "#a78bfa"),
            _tcard(f"{s.std():.4f}",         "Std",       "#f59e0b"),
            _tcard(f"{s.min():.4f}",         "Min",       "#556070"),
            _tcard(f"{s.max():.4f}",         "Max",       "#556070"),
            _tcard(f"{n_missing:,}",         "Eksik",
                   "#f59e0b" if n_missing > 0 else "#556070"),
        ], className="g-3 mb-4")

        fig_hist = go.Figure(go.Histogram(
            x=s.dropna(), nbinsx=50,
            marker_color="#4F8EF7", opacity=0.8,
            hovertemplate="Değer: %{x}<br>Sayı: %{y}<extra></extra>",
        ))
        fig_hist.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=f"Target Dağılımı — {target}",
                       font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, title=target),
            yaxis=dict(**_AXIS_STYLE, title="Frekans"),
            height=300,
        )

        time_chart = html.Div()
        if date_col and date_col in df_active.columns:
            _trend_store_data = {
                "key": key, "target": target, "date_col": date_col,
                "target_type": target_type,
                "seg_col": seg_col, "seg_val": seg_val,
                "oot_date": oot_date,
                "has_test_split": config.get("has_test_split", False),
                "test_size": config.get("test_size", 20),
            }
            time_chart = html.Div([
                trend_dropdown,
                dcc.Store(id="store-trend-config", data=_trend_store_data),
                html.Div(
                    _render_trend_chart(df_active, target, date_col,
                                        target_type, "Tüm Veri"),
                    id="trend-chart-container",
                ),
            ])

        iv_note = html.Div(
            "ℹ️  WoE / Information Value yalnızca binary (0/1) target için hesaplanır. "
            "Bu target continuous olarak algılandı — Değişken Özeti sekmesinde "
            "Mutual Information kullanılmaktadır.",
            style={"color": "#a78bfa", "fontSize": "0.8rem",
                   "padding": "0.6rem 1rem", "border": "1px solid #3b2d6b",
                   "borderRadius": "6px", "backgroundColor": "#1a1535",
                   "marginTop": "1rem"},
        )

        return html.Div([
            _tab_info("Target Dağılımı",
                      "Sürekli Değişken · Zaman Trendi",
                      "Target sayısal sürekli bir değişken (ör. gün sayısı, oran). "
                      "Dağılım, zaman trendi ve temel istatistikler gösterilmektedir.",
                      "#4F8EF7"),
            html.P("Target İstatistikleri", className="section-title"),
            stats_row,
            dcc.Graph(figure=fig_hist, config={"displayModeBar": False},
                      style={"marginBottom": "2rem"}),
            time_chart,
            iv_note,
        ])

    # ── Multiclass ────────────────────────────────────────────────────────────
    if target_type == "multiclass":
        vc = s.dropna().astype(int).value_counts().sort_index().reset_index()
        vc.columns = ["Sınıf", "Sayı"]
        vc["Oran %"] = (vc["Sayı"] / vc["Sayı"].sum() * 100).round(2)

        stats_row = dbc.Row([
            _tcard(f"{n_valid:,}",       "Geçerli Kayıt"),
            _tcard(f"{len(vc)}",         "Sınıf Sayısı", "#4F8EF7"),
            _tcard(f"{int(s.mode()[0])}","Dominant Sınıf", "#10b981"),
            _tcard(f"{vc['Oran %'].max():.1f}%",
                   "Dominant Oran",
                   "#ef4444" if vc["Oran %"].max() > 80 else "#f59e0b"),
            _tcard(f"{n_missing:,}", "Eksik",
                   "#f59e0b" if n_missing > 0 else "#556070"),
        ], className="g-3 mb-4")

        fig_bar = go.Figure(go.Bar(
            x=vc["Sınıf"].astype(str), y=vc["Sayı"],
            marker_color="#4F8EF7", opacity=0.85,
            text=vc["Oran %"].apply(lambda x: f"%{x:.1f}"),
            textposition="outside",
            hovertemplate="Sınıf: %{x}<br>Sayı: %{y}<extra></extra>",
        ))
        fig_bar.update_layout(
            **_PLOT_LAYOUT,
            title=dict(text=f"Sınıf Dağılımı — {target}",
                       font=dict(color="#E8EAF0", size=13)),
            xaxis=dict(**_AXIS_STYLE, title="Sınıf"),
            yaxis=dict(**_AXIS_STYLE, title="Kayıt Sayısı"),
            height=300,
        )

        time_chart = html.Div()
        if date_col and date_col in df_active.columns:
            _trend_store_data = {
                "key": key, "target": target, "date_col": date_col,
                "target_type": target_type,
                "seg_col": seg_col, "seg_val": seg_val,
                "oot_date": oot_date,
                "has_test_split": config.get("has_test_split", False),
                "test_size": config.get("test_size", 20),
            }
            time_chart = html.Div([
                trend_dropdown,
                dcc.Store(id="store-trend-config", data=_trend_store_data),
                html.Div(
                    _render_trend_chart(df_active, target, date_col,
                                        target_type, "Tüm Veri"),
                    id="trend-chart-container",
                ),
            ])

        iv_note = html.Div(
            "ℹ️  WoE / Information Value yalnızca binary (0/1) target için hesaplanır. "
            "Bu target multiclass olarak algılandı — Değişken Özeti sekmesinde "
            "Mutual Information kullanılmaktadır.",
            style={"color": "#a78bfa", "fontSize": "0.8rem",
                   "padding": "0.6rem 1rem", "border": "1px solid #3b2d6b",
                   "borderRadius": "6px", "backgroundColor": "#1a1535",
                   "marginTop": "1rem"},
        )

        return html.Div([
            _tab_info("Target Dağılımı",
                      "Multiclass · Sınıf Dağılımı · Zaman Trendi",
                      "Target çok sınıflı bir değişken. Sınıf dağılımı ve zaman trendi "
                      "gösterilmektedir.",
                      "#a78bfa"),
            html.P("Target İstatistikleri", className="section-title"),
            stats_row,
            dcc.Graph(figure=fig_bar, config={"displayModeBar": False},
                      style={"marginBottom": "2rem"}),
            time_chart,
            iv_note,
        ])

    # ── Categorical ───────────────────────────────────────────────────────────
    vc = df_active[target].dropna().astype(str).value_counts().head(20).reset_index()
    vc.columns = ["Değer", "Sayı"]
    vc["Oran %"] = (vc["Sayı"] / vc["Sayı"].sum() * 100).round(2)

    stats_row = dbc.Row([
        _tcard(f"{n_valid:,}",    "Geçerli Kayıt"),
        _tcard(f"{len(vc)}",      "Unique Değer (Top 20)", "#4F8EF7"),
        _tcard(f"{n_missing:,}",  "Eksik",
               "#f59e0b" if n_missing > 0 else "#556070"),
    ], className="g-3 mb-4")

    fig_cat = go.Figure(go.Bar(
        x=vc["Değer"], y=vc["Sayı"],
        marker_color="#4F8EF7", opacity=0.85,
        text=vc["Oran %"].apply(lambda x: f"%{x:.1f}"),
        textposition="outside",
    ))
    fig_cat.update_layout(
        **_PLOT_LAYOUT,
        title=dict(text=f"Değer Dağılımı — {target}",
                   font=dict(color="#E8EAF0", size=13)),
        xaxis=dict(**_AXIS_STYLE, title="Değer"),
        yaxis=dict(**_AXIS_STYLE, title="Sayı"),
        height=300,
    )

    return html.Div([
        _tab_info("Target Dağılımı", "Kategorik Target · Değer Dağılımı",
                  "Target kategorik bir değişken. En sık 20 değerin dağılımı gösterilmektedir.",
                  "#f59e0b"),
        html.P("Target İstatistikleri", className="section-title"),
        stats_row,
        dcc.Graph(figure=fig_cat, config={"displayModeBar": False}),
        html.Div(
            "ℹ️  IV/WoE ve model metrikleri kategorik target için hesaplanmaz.",
            style={"color": "#a78bfa", "fontSize": "0.8rem", "marginTop": "1rem"},
        ),
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
    target_type = trend_cfg["target_type"]
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

    # Stats kartları — yalnızca binary target için; seçili periyodun verisi
    if target_type == "binary":
        new_stats = _binary_stats_row(df_plot, target)
    else:
        new_stats = _no

    return _render_trend_chart(df_plot, target, date_col, target_type, period_label), new_stats
