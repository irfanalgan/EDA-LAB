from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from app_instance import app
from server_state import get_df as _get_df
from utils.helpers import apply_segment_filter
from utils.chart_helpers import _tab_info
from modules.profiling import compute_profile, profile_summary


# ── Callback: Profiling Sekmesi ───────────────────────────────────────────────
@app.callback(
    Output("tab-profiling", "children"),
    Input("store-config", "data"),
    State("store-key", "data"),
)
def update_profiling(config, key):
    df_orig = _get_df(key)
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()

    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")
    df_active = apply_segment_filter(df_orig, seg_col, seg_val)

    profile = compute_profile(df_active)   # local kopya üzerinde çalışır
    summary = profile_summary(profile, len(df_active))

    # ── Özet kartları ─────────────────────────────────────────────────────────
    def scard(value, label, color="#4F8EF7"):
        return dbc.Col(html.Div([
            html.Div(str(value), className="metric-value", style={"fontSize": "1.25rem", "color": color}),
            html.Div(label, className="metric-label"),
        ], className="metric-card"), width=2)

    summary_row = dbc.Row([
        scard(summary["total_cols"],      "Toplam Kolon"),
        scard(summary["numeric_cols"],    "Nümerik"),
        scard(summary["categorical_cols"],"Kategorik"),
        scard(summary["full_cols"],       "Tam Dolu",      "#10b981"),
        scard(summary["mid_missing"],     "Orta Eksik\n(5–50%)", "#f59e0b"),
        scard(summary["high_missing"],    "Yüksek Eksik\n(>50%)", "#ef4444"),
    ], className="g-3 mb-4")

    # ── Koşullu renklendirme ──────────────────────────────────────────────────
    cond_style = [
        {"if": {"filter_query": "{Eksik %} > 50", "column_id": "Eksik %"},
         "backgroundColor": "rgba(239,68,68,0.15)", "color": "#ef4444", "fontWeight": "700"},
        {"if": {"filter_query": "{Eksik %} > 5 && {Eksik %} <= 50", "column_id": "Eksik %"},
         "backgroundColor": "rgba(245,158,11,0.12)", "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": "{Eksik %} = 0", "column_id": "Eksik %"},
         "color": "#10b981"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
    ]

    profile_tsv = profile.to_csv(sep="\t", index=False)
    profile_table = html.Div([
        html.Div(
            dcc.Clipboard(target_id="profile-tsv", title="Kopyala",
                          style={"cursor": "pointer", "fontSize": "0.72rem",
                                 "color": "#a8b2c2", "padding": "0.2rem 0.55rem",
                                 "border": "1px solid #2d3a4f", "borderRadius": "4px",
                                 "backgroundColor": "#1a2035", "float": "right",
                                 "marginBottom": "0.4rem", "letterSpacing": "0.04em"}),
            style={"overflow": "hidden"},
        ),
        html.Pre(profile_tsv, id="profile-tsv", style={"display": "none"}),
        dash_table.DataTable(
            data=profile.to_dict("records"),
            columns=[{"name": c, "id": c} for c in profile.columns],
            sort_action="native",
            filter_action="native",
            page_size=25,
            page_action="native",
            fixed_columns={"headers": True, "data": 1},
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_header={
                "backgroundColor": "#111827",
                "color": "#a8b2c2",
                "fontWeight": "700",
                "fontSize": "0.73rem",
                "border": "1px solid #2d3a4f",
                "textTransform": "uppercase",
                "letterSpacing": "0.06em",
            },
            style_data={
                "backgroundColor": "#161C27",
                "color": "#c8cdd8",
                "fontSize": "0.83rem",
                "border": "1px solid #232d3f",
            },
            style_data_conditional=cond_style,
            style_cell={"padding": "0.45rem 0.75rem", "textAlign": "left"},
            style_cell_conditional=[
                {"if": {"column_id": ["Dolu Sayı", "Eksik Sayı", "Eksik %",
                                      "Tekil Değer", "En Sık %",
                                      "Ortalama", "Std", "Min",
                                      "P1", "P5", "P10", "P25", "Medyan",
                                      "P75", "P90", "P95", "P99", "Max"]},
                 "textAlign": "right"},
            ],
            style_filter={
                "backgroundColor": "#0e1117",
                "color": "#c8cdd8",
                "border": "1px solid #2d3a4f",
            },
            css=[{"selector": ".dash-filter input", "rule": "color: #c8cdd8 !important;"}],
        ),
    ])

    return html.Div([
        _tab_info("Describe", "Kolon Kalite Analizi",
                  "Her değişken için eksik oran, kardinalite, veri tipi ve ön eleme sonucunu gösterir. "
                  "Kırmızı satırlar eksik > %50, sarı eksik %5–50, yeşil tam dolu. Eksik veya sabit "
                  "değişkenler otomatik olarak ön elemeden geçer.",
                  "#a78bfa"),
        html.P("Describe", className="section-title"),
        summary_row,
        profile_table,
        html.P(
            f"Renk kodları — Kırmızı: Eksik > %50  ·  Sarı: Eksik %5–50  ·  Yeşil: Tam Dolu",
            style={"fontSize": "0.73rem", "color": "#7e8fa4", "marginTop": "0.6rem"},
        ),
    ])
