"""İzleme — Temerrüt Oranı tab callback'leri."""

from dash import html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import aggregate_summaries

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, color="#c8cdd8"),
    margin=dict(l=50, r=20, t=30, b=40),
    height=320,
)

_NO_DATA = html.P("Henüz veri yok (olgun dönem gerekli).",
                   style={"color": "#7e8fa4", "fontSize": "0.85rem",
                          "textAlign": "center", "padding": "2rem 0"})


def _render_bad_rate(n_total, n_bad, bad_rate, title=""):
    return html.Div([
        html.H6(title or "Temerrüt Oranı",
                style={"color": "#c8cdd8", "fontSize": "0.9rem",
                       "marginBottom": "0.5rem"}),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Toplam Gözlem", style={"color": "#7e8fa4",
                       "fontSize": "0.72rem", "marginBottom": "2px"}),
                html.H5(f"{n_total:,}", style={"color": "#c8cdd8",
                         "marginBottom": "0"}),
            ]), className="bg-dark border-secondary"), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Default", style={"color": "#7e8fa4",
                       "fontSize": "0.72rem", "marginBottom": "2px"}),
                html.H5(f"{n_bad:,}", style={"color": "#ef4444",
                         "marginBottom": "0"}),
            ]), className="bg-dark border-secondary"), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Temerrüt Oranı", style={"color": "#7e8fa4",
                       "fontSize": "0.72rem", "marginBottom": "2px"}),
                html.H5(f"{bad_rate:.2%}", style={"color": "#f59e0b",
                         "marginBottom": "0"}),
            ]), className="bg-dark border-secondary"), width=4),
        ], className="g-2"),
    ])


# ── Callback 1: Populate ───────────────────────────────────────────────────
@app.callback(
    Output("mon-badrate-trend-dd", "options"),
    Output("mon-badrate-trend-dd", "value"),
    Output("mon-badrate-trend-chart", "children"),
    Output("mon-badrate-cum-content", "children"),
    Input("store-mon-summaries-signal", "data"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_badrate_populate(signal, key):
    if not signal or not key:
        return [], None, _NO_DATA, _NO_DATA

    summaries = _MON_STORE.get(key + "_period_summaries", [])
    mature = [s for s in summaries if s.get("is_mature", False)]
    if not mature:
        return [], None, _NO_DATA, _NO_DATA

    options = [{"label": s["period_label"], "value": s["period_label"]}
               for s in mature]
    default = mature[-1]["period_label"]

    # Trend chart
    labels = [s["period_label"] for s in mature]
    rates = [s["bad_rate"] * 100 for s in mature]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=rates, mode="lines+markers",
                             name="Temerrüt Oranı (%)",
                             line=dict(color="#ef4444", width=2),
                             marker=dict(size=6)))
    fig.update_layout(**_CHART_LAYOUT, title="Temerrüt Oranı Trendi",
                      yaxis_title="Temerrüt Oranı (%)")
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Kümülatif
    cum = aggregate_summaries(mature)
    if cum:
        cum_content = _render_bad_rate(
            cum["n_total"], cum["n_bad"], cum["bad_rate"],
            "Kümülatif Temerrüt Oranı")
    else:
        cum_content = _NO_DATA

    return options, default, chart, cum_content


# ── Callback 2: Dönem seçimi ───────────────────────────────────────────────
@app.callback(
    Output("mon-badrate-trend-detail", "children"),
    Input("mon-badrate-trend-dd", "value"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_badrate_select_period(period_label, key):
    if not period_label or not key:
        return _NO_DATA
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    selected = next((s for s in summaries if s["period_label"] == period_label), None)
    if not selected:
        return _NO_DATA
    return _render_bad_rate(
        selected["n_total"], selected["n_bad"], selected["bad_rate"],
        f"Dönem: {period_label}")
