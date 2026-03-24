"""İzleme — Gini/KS tab callback'leri."""

from dash import html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import (
    calc_ks_from_summary, calc_gini_from_summary, aggregate_summaries,
)

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

_NO_DATA = html.P("Henüz veri yok (olgun dönem gerekli).",
                   style={"color": "#7e8fa4", "fontSize": "0.85rem",
                          "textAlign": "center", "padding": "2rem 0"})


def _build_metric_cards(ks, gini, ar):
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("KS", style={"color": "#7e8fa4", "fontSize": "0.72rem",
                                "marginBottom": "2px"}),
            html.H5(f"{ks:.2f}%", style={"color": "#4F8EF7", "marginBottom": "0"}),
        ]), className="bg-dark border-secondary"), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Gini", style={"color": "#7e8fa4", "fontSize": "0.72rem",
                                  "marginBottom": "2px"}),
            html.H5(f"{gini:.2f}%", style={"color": "#10b981", "marginBottom": "0"}),
        ]), className="bg-dark border-secondary"), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("AR", style={"color": "#7e8fa4", "fontSize": "0.72rem",
                                "marginBottom": "2px"}),
            html.H5(f"{ar:.2f}%", style={"color": "#a78bfa", "marginBottom": "0"}),
        ]), className="bg-dark border-secondary"), width=4),
    ], className="g-2 mb-3")


def _build_ks_table(ks_val, rows):
    if not rows:
        return html.Div()
    data = []
    tot_good, tot_bad, tot_total = 0, 0, 0
    for r in rows:
        if r["total"] == 0:
            continue
        tot_good += r["good"]
        tot_bad += r["bad"]
        tot_total += r["total"]
        data.append({
            "Rating": r["rating"],
            "Good": r["good"],
            "Bad": r["bad"],
            "Toplam": r["total"],
            "Temerrüt Oranı": f"{r['bad_rate']:.2%}",
            "Küm Good": r["cum_good"],
            "Küm Bad": r["cum_bad"],
            "%Küm Good": f"{r['pct_cum_good']:.2f}%",
            "%Küm Bad": f"{r['pct_cum_bad']:.2f}%",
            "Fark": f"{r['diff']:.2f}%",
        })
    # Toplam satır
    if data:
        last = data[-1]
        data.append({
            "Rating": "Toplam",
            "Good": tot_good,
            "Bad": tot_bad,
            "Toplam": tot_total,
            "Temerrüt Oranı": f"{tot_bad / tot_total:.2%}" if tot_total else "0%",
            "Küm Good": last["Küm Good"],
            "Küm Bad": last["Küm Bad"],
            "%Küm Good": "100.00%",
            "%Küm Bad": "100.00%",
            "Fark": "",
        })
    cols = ["Rating", "Good", "Bad", "Toplam", "Temerrüt Oranı",
            "Küm Good", "Küm Bad", "%Küm Good", "%Küm Bad", "Fark"]
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=data,
        style_header=_TH, style_cell=_TD,
        style_data_conditional=[_TD_ODD],
        page_size=30,
        style_table={"overflowX": "auto"},
    )


def _build_gini_table(gini_val, details):
    """Excel formatında Gini tablosu (kötüden iyiye sıralı)."""
    gini_rows = details.get("rows", [])
    if not gini_rows:
        return html.Div()
    data = []
    for r in gini_rows:
        if r["total"] == 0:
            continue
        data.append({
            "Rating": r["rating"],
            "Good": r["good"],
            "Bad": r["bad"],
            "Total": r["total"],
            "Bad Rate": f"{r['bad_rate']:.2%}",
            "Cum Good": r["cum_good"],
            "Cum Bad": r["cum_bad"],
            "Cum Total": r["cum_total"],
            "% Cum Good": f"{r['pct_cum_good']:.2f}%",
            "% Cum Bad": f"{r['pct_cum_bad']:.2f}%",
            "% Cum Total": f"{r['pct_cum_total']:.2f}%",
            "Gini Area": f"{r['gini_area']:.4f}",
            "Random": f"{r['random']:.2f}%",
            "Perfect Curve": f"{r['perfect_curve']:.2f}%",
        })
    # Toplam satır
    if data:
        last = data[-1]
        data.append({
            "Rating": "Toplam",
            "Good": "", "Bad": "", "Total": "", "Bad Rate": "",
            "Cum Good": last["Cum Good"],
            "Cum Bad": last["Cum Bad"],
            "Cum Total": last["Cum Total"],
            "% Cum Good": "100.00%",
            "% Cum Bad": "100.00%",
            "% Cum Total": "100.00%",
            "Gini Area": f"{sum(r['gini_area'] for r in gini_rows):.4f}",
            "Random": "", "Perfect Curve": "",
        })
    cols = ["Rating", "Good", "Bad", "Total", "Bad Rate",
            "Cum Good", "Cum Bad", "Cum Total",
            "% Cum Good", "% Cum Bad", "% Cum Total",
            "Gini Area", "Random", "Perfect Curve"]
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=data,
        style_header=_TH, style_cell=_TD,
        style_data_conditional=[_TD_ODD],
        page_size=30,
        style_table={"overflowX": "auto"},
    )


def _render_disc(rating_counts, rating_defaults, title_prefix=""):
    ks, ks_rows = calc_ks_from_summary(rating_counts, rating_defaults)
    gini, ar, _, details = calc_gini_from_summary(rating_counts, rating_defaults)
    return html.Div([
        html.H6(f"{title_prefix}Gini/KS Metrikleri",
                style={"color": "#c8cdd8", "fontSize": "0.9rem",
                       "marginBottom": "0.5rem"}),
        _build_metric_cards(ks, gini * 100, ar * 100),
        html.H6("KS Tablosu", style={"color": "#c8cdd8", "fontSize": "0.85rem",
                                      "marginTop": "1rem", "marginBottom": "0.5rem"}),
        _build_ks_table(ks, ks_rows),
        html.H6(f"Gini Tablosu — Gini: {gini * 100:.2f}%",
                style={"color": "#c8cdd8", "fontSize": "0.85rem",
                       "marginTop": "1.5rem", "marginBottom": "0.5rem"}),
        _build_gini_table(gini, details),
    ])


# ── Callback 1: Populate ───────────────────────────────────────────────────
@app.callback(
    Output("mon-disc-trend-dd", "options"),
    Output("mon-disc-trend-dd", "value"),
    Output("mon-disc-trend-chart", "children"),
    Output("mon-disc-cum-content", "children"),
    Input("store-mon-summaries-signal", "data"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_disc_populate(signal, key):
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
    labels, ks_vals, gini_vals = [], [], []
    for s in mature:
        labels.append(s["period_label"])
        ks, _ = calc_ks_from_summary(s["rating_counts"], s["rating_defaults"])
        gini, _, _, _ = calc_gini_from_summary(s["rating_counts"], s["rating_defaults"])
        ks_vals.append(ks)
        gini_vals.append(gini * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=ks_vals, mode="lines+markers",
                             name="KS (%)", line=dict(color="#4F8EF7", width=2),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=labels, y=gini_vals, mode="lines+markers",
                             name="Gini (%)", line=dict(color="#10b981", width=2),
                             marker=dict(size=6)))
    fig.update_layout(**_CHART_LAYOUT, title="Gini/KS Trendi",
                      yaxis_title="%")
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Kümülatif
    cum = aggregate_summaries(mature)
    if cum:
        cum_content = _render_disc(cum["rating_counts"], cum["rating_defaults"],
                                   "Kümülatif ")
    else:
        cum_content = _NO_DATA

    return options, default, chart, cum_content


# ── Callback 2: Dönem seçimi ───────────────────────────────────────────────
@app.callback(
    Output("mon-disc-trend-detail", "children"),
    Input("mon-disc-trend-dd", "value"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_disc_select_period(period_label, key):
    if not period_label or not key:
        return _NO_DATA
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    selected = next((s for s in summaries if s["period_label"] == period_label), None)
    if not selected:
        return _NO_DATA
    return _render_disc(selected["rating_counts"], selected["rating_defaults"],
                        f"Dönem: {period_label} — ")
