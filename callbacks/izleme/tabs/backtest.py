"""İzleme — Backtesting tab callback'leri."""

from dash import html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import calc_backtesting_table, aggregate_summaries

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


def _render_backtest(rating_counts, rating_defaults, title=""):
    rows = calc_backtesting_table(rating_counts, rating_defaults)
    if not rows:
        return _NO_DATA

    data = []
    for r in rows:
        if r["rating"] != "Grand_Total" and r["count"] == 0:
            continue
        data.append({
            "Rating": r["rating"],
            "Adet": r["count"],
            "Yoğunlaşma": f"{r['concentration']:.2%}",
            "Default": r["default"],
            "DR": f"{r['dr']:.4f}",
            "MIDPD": f"{r['midpd']:.4f}",
            "Conservatism": "TRUE" if r["conservatism"] else "FALSE",
            "Üst Sınır": f"{r['upper_limit']:.4f}",
            "Alt Sınır": f"{r['lower_limit']:.4f}",
            "Üst Flag": r["upper_flag"],
            "Alt Flag": r["lower_flag"],
            "Monotonicity": "TRUE" if r.get("monotonicity", True) else "FALSE",
        })

    cols = ["Rating", "Adet", "Yoğunlaşma", "Default", "DR", "MIDPD",
            "Conservatism", "Monotonicity", "Üst Sınır", "Alt Sınır",
            "Üst Flag", "Alt Flag"]

    return html.Div([
        html.H6(title or "Backtesting — Binomial Test",
                style={"color": "#c8cdd8", "fontSize": "0.9rem",
                       "marginBottom": "0.5rem"}),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in cols],
            data=data,
            style_header=_TH, style_cell={**_TD, "minWidth": "70px"},
            style_data_conditional=[
                _TD_ODD,
                {"if": {"filter_query": '{Üst Flag} = "Exceeding the Range"',
                        "column_id": "Üst Flag"},
                 "color": "#ef4444", "fontWeight": "bold"},
                {"if": {"filter_query": '{Alt Flag} = "Below the Range"',
                        "column_id": "Alt Flag"},
                 "color": "#3b82f6", "fontWeight": "bold"},
                {"if": {"filter_query": '{Conservatism} = "FALSE"',
                        "column_id": "Conservatism"},
                 "color": "#f59e0b"},
                {"if": {"filter_query": '{Monotonicity} = "FALSE"',
                        "column_id": "Monotonicity"},
                 "color": "#f59e0b"},
            ],
            page_size=30,
            style_table={"overflowX": "auto"},
        ),
    ])


# ── Callback 1: Populate ───────────────────────────────────────────────────
@app.callback(
    Output("mon-backtesting-trend-dd", "options"),
    Output("mon-backtesting-trend-dd", "value"),
    Output("mon-backtesting-trend-chart", "children"),
    Output("mon-backtesting-cum-content", "children"),
    Input("store-mon-summaries-signal", "data"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_backtest_populate(signal, key):
    if not signal or not key:
        return [], None, _NO_DATA, _NO_DATA

    summaries = _MON_STORE.get(key + "_period_summaries", [])
    mature = [s for s in summaries if s.get("is_mature", False)]
    if not mature:
        return [], None, _NO_DATA, _NO_DATA

    options = [{"label": s["period_label"], "value": s["period_label"]}
               for s in mature]
    default = mature[-1]["period_label"]

    # Trend chart — dönemler × exceeding oranı
    labels, exceed_pcts = [], []
    for s in mature:
        labels.append(s["period_label"])
        rows = calc_backtesting_table(s["rating_counts"], s["rating_defaults"])
        # Grand_Total hariç, count > 0 olanlar arasında exceeding oranı
        active = [r for r in rows if r["rating"] != "Grand_Total" and r["count"] > 0]
        n_exceed = sum(1 for r in active if r["upper_flag"] == "Exceeding the Range")
        exceed_pcts.append(n_exceed / len(active) * 100 if active else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=exceed_pcts, mode="lines+markers",
                             name="Exceeding %",
                             line=dict(color="#ef4444", width=2),
                             marker=dict(size=6)))
    fig.update_layout(**_CHART_LAYOUT, title="Backtesting — Aşım Oranı Trendi",
                      yaxis_title="Aşım Oranı (%)")
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Kümülatif
    cum = aggregate_summaries(mature)
    if cum:
        cum_content = _render_backtest(
            cum["rating_counts"], cum["rating_defaults"],
            "Kümülatif Backtesting")
    else:
        cum_content = _NO_DATA

    return options, default, chart, cum_content


# ── Callback 2: Dönem seçimi ───────────────────────────────────────────────
@app.callback(
    Output("mon-backtesting-trend-detail", "children"),
    Input("mon-backtesting-trend-dd", "value"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_backtest_select_period(period_label, key):
    if not period_label or not key:
        return _NO_DATA
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    selected = next((s for s in summaries if s["period_label"] == period_label), None)
    if not selected:
        return _NO_DATA
    return _render_backtest(
        selected["rating_counts"], selected["rating_defaults"],
        f"Dönem: {period_label}")
