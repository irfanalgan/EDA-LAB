"""İzleme — Temerrüt Oranı tab callback'leri."""

from dash import html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import aggregate_summaries

_TH = {"backgroundColor": "#1a2332", "color": "#c8cdd8",
       "fontWeight": "600", "fontSize": "0.7rem", "padding": "6px 8px",
       "borderBottom": "2px solid #3b82f6", "textAlign": "center"}
_TD = {"backgroundColor": "#0e1117", "color": "#c8cdd8",
       "fontSize": "0.72rem", "border": "1px solid #1e293b",
       "padding": "4px 8px", "textAlign": "center"}
_TD_ODD = {"if": {"row_index": "odd"}, "backgroundColor": "#141b27"}
_TD_TOTAL = {"if": {"filter_query": '{Rating} = "Toplam"'},
             "backgroundColor": "#1a2332", "fontWeight": "bold",
             "borderTop": "2px solid #3b82f6"}

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


def _render_bad_rate(n_total, n_bad, bad_rate, rating_counts=None,
                     rating_defaults=None, title=""):
    children = [
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
    ]

    # Rating bazlı detaylı tablo
    if rating_counts and rating_defaults:
        n_ratings = len(rating_counts)
        total_good_all = sum(c - d for c, d in zip(rating_counts, rating_defaults))
        total_bad_all = sum(rating_defaults)
        data = []
        cum_good, cum_bad = 0, 0
        for i in range(n_ratings):
            count = rating_counts[i]
            default = rating_defaults[i]
            if count == 0:
                continue
            good = count - default
            cum_good += good
            cum_bad += default
            pct_cum_good = cum_good / total_good_all * 100 if total_good_all > 0 else 0
            pct_cum_bad = cum_bad / total_bad_all * 100 if total_bad_all > 0 else 0
            data.append({
                "Rating": i + 1,
                "Good": good,
                "Bad": default,
                "Toplam": count,
                "Temerrüt Oranı": f"{default / count:.2%}",
                "Küm Good": cum_good,
                "Küm Bad": cum_bad,
                "%Küm Good": f"{pct_cum_good:.2f}%",
                "%Küm Bad": f"{pct_cum_bad:.2f}%",
            })
        # Toplam satır
        if data:
            tot_good = total_good_all
            tot_bad = total_bad_all
            tot_total = tot_good + tot_bad
            data.append({
                "Rating": "Toplam",
                "Good": tot_good,
                "Bad": tot_bad,
                "Toplam": tot_total,
                "Temerrüt Oranı": f"{tot_bad / tot_total:.2%}" if tot_total else "0%",
                "Küm Good": tot_good,
                "Küm Bad": tot_bad,
                "%Küm Good": "100.00%",
                "%Küm Bad": "100.00%",
            })
        cols = ["Rating", "Good", "Bad", "Toplam", "Temerrüt Oranı",
                "Küm Good", "Küm Bad", "%Küm Good", "%Küm Bad"]
        children.append(
            html.Div(
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in cols],
                    data=data,
                    style_header=_TH, style_cell=_TD,
                    style_data_conditional=[_TD_ODD, _TD_TOTAL],
                    page_size=30,
                    style_table={"overflowX": "auto"},
                ),
                style={"marginTop": "1rem"},
            )
        )

    return html.Div(children)


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
            cum["rating_counts"], cum["rating_defaults"],
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
        selected["rating_counts"], selected["rating_defaults"],
        f"Dönem: {period_label}")
