"""İzleme — HHI tab callback'leri."""

from dash import html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import calc_hhi_from_summary, aggregate_summaries

_TH = {"backgroundColor": "#1a2332", "color": "#c8cdd8",
       "fontWeight": "bold", "fontSize": "0.75rem"}
_TD = {"backgroundColor": "#0e1117", "color": "#c8cdd8",
       "fontSize": "0.75rem", "border": "1px solid #2d3a4f",
       "padding": "4px 8px"}
_TD_ODD = {"if": {"row_index": "odd"}, "backgroundColor": "#141b27"}

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, color="#c8cdd8"),
    margin=dict(l=50, r=20, t=30, b=40),
    height=320,
)

_NO_DATA = html.P("Henüz veri yok.",
                   style={"color": "#7e8fa4", "fontSize": "0.85rem",
                          "textAlign": "center", "padding": "2rem 0"})


def _hhi_label(val):
    if val < 0.06:
        return "Dengeli dağılım"
    if val < 0.10:
        return "Orta konsantrasyon"
    return "Yüksek konsantrasyon"


def _render_hhi(rating_counts, title=""):
    hhi, rows = calc_hhi_from_summary(rating_counts)
    if not rows:
        return _NO_DATA

    data = []
    for r in rows:
        if r["count"] == 0:
            continue
        data.append({
            "Rating": r["rating"],
            "Adet": r["count"],
            "Pay (%)": f"{r['share']:.2%}",
            "HHI Katkı": f"{r['hhi_contrib']:.6f}",
        })

    label = _hhi_label(hhi)
    color = "#10b981" if hhi < 0.06 else ("#f59e0b" if hhi < 0.10 else "#ef4444")

    return html.Div([
        html.H6(title or "HHI",
                style={"color": "#c8cdd8", "fontSize": "0.9rem",
                       "marginBottom": "0.5rem"}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("HHI", style={"color": "#7e8fa4", "fontSize": "0.72rem",
                                     "marginBottom": "2px"}),
                html.H5(f"{hhi:.4f}", style={"color": color, "marginBottom": "0"}),
                html.Small(label, style={"color": color, "fontSize": "0.72rem"}),
            ]), className="bg-dark border-secondary"), width=4),
        ], className="g-2 mb-3"),
        dash_table.DataTable(
            columns=[{"name": c, "id": c}
                     for c in ["Rating", "Adet", "Pay (%)", "HHI Katkı"]],
            data=data,
            style_header=_TH, style_cell=_TD,
            style_data_conditional=[_TD_ODD],
            page_size=30,
            style_table={"overflowX": "auto"},
        ),
    ])


# ── Callback 1: Populate ───────────────────────────────────────────────────
@app.callback(
    Output("mon-hhi-trend-dd", "options"),
    Output("mon-hhi-trend-dd", "value"),
    Output("mon-hhi-trend-chart", "children"),
    Output("mon-hhi-cum-content", "children"),
    Input("store-mon-summaries-signal", "data"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_hhi_populate(signal, key):
    if not signal or not key:
        return [], None, _NO_DATA, _NO_DATA

    summaries = _MON_STORE.get(key + "_period_summaries", [])
    if not summaries:
        return [], None, _NO_DATA, _NO_DATA

    # HHI anlık metrik — tüm dönemler
    options = [{"label": s["period_label"], "value": s["period_label"]}
               for s in summaries]
    default = summaries[-1]["period_label"]

    # Trend chart
    labels, hhi_vals = [], []
    for s in summaries:
        labels.append(s["period_label"])
        hhi, _ = calc_hhi_from_summary(s["rating_counts"])
        hhi_vals.append(hhi)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=hhi_vals, mode="lines+markers",
                             name="HHI", line=dict(color="#a78bfa", width=2),
                             marker=dict(size=6)))
    fig.add_hline(y=0.06, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Orta (0.06)")
    fig.add_hline(y=0.10, line_dash="dash", line_color="#ef4444",
                  annotation_text="Yüksek (0.10)")
    fig.update_layout(**_CHART_LAYOUT, title="HHI Trendi", yaxis_title="HHI")
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Kümülatif
    cum = aggregate_summaries(summaries)
    cum_content = _render_hhi(cum["rating_counts"], "Kümülatif HHI") if cum else _NO_DATA

    return options, default, chart, cum_content


# ── Callback 2: Dönem seçimi ───────────────────────────────────────────────
@app.callback(
    Output("mon-hhi-trend-detail", "children"),
    Input("mon-hhi-trend-dd", "value"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_hhi_select_period(period_label, key):
    if not period_label or not key:
        return _NO_DATA
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    selected = next((s for s in summaries if s["period_label"] == period_label), None)
    if not selected:
        return _NO_DATA
    return _render_hhi(selected["rating_counts"], f"Dönem: {period_label}")
