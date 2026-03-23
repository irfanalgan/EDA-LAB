"""İzleme — Göç Matrisi tab callback'leri."""

from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import aggregate_summaries, N_RATINGS

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, color="#c8cdd8"),
    margin=dict(l=50, r=20, t=30, b=40),
    height=320,
)

_NO_DATA = html.P("Henüz veri yok (ID kolonu gerekli).",
                   style={"color": "#7e8fa4", "fontSize": "0.85rem",
                          "textAlign": "center", "padding": "2rem 0"})


def _stability_ratio(matrix):
    """Köşegen toplamı / genel toplam."""
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0
    diag = sum(matrix[i][i] for i in range(min(len(matrix), N_RATINGS)))
    return diag / total


def _render_migration(matrix, matched_count, title=""):
    if matrix is None:
        return _NO_DATA

    arr = np.array(matrix)
    # Aktif rating'leri bul (satır veya sütun toplamı > 0)
    row_sums = arr.sum(axis=1)
    col_sums = arr.sum(axis=0)
    active = [i for i in range(N_RATINGS) if row_sums[i] > 0 or col_sums[i] > 0]

    if not active:
        return _NO_DATA

    labels = [str(i + 1) for i in active]
    sub = arr[np.ix_(active, active)]

    # Yüzdeye çevir (satır bazlı)
    row_totals = sub.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    pct = sub / row_totals * 100

    # Anotasyon metni: count (pct%)
    text = []
    for i in range(len(active)):
        row_text = []
        for j in range(len(active)):
            row_text.append(f"{int(sub[i, j])}<br>({pct[i, j]:.1f}%)")
        text.append(row_text)

    stab = _stability_ratio(matrix)

    fig = go.Figure(data=go.Heatmap(
        z=pct.tolist(),
        x=labels, y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#c8cdd8"),
        margin=dict(l=50, r=20, t=40, b=50),
        height=max(400, len(active) * 30 + 100),
        title=title or "Göç Matrisi",
        xaxis_title="İzleme Rating",
        yaxis_title="Referans Rating",
        yaxis=dict(autorange="reversed"),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Kararlılık Oranı", style={"color": "#7e8fa4",
                       "fontSize": "0.72rem", "marginBottom": "2px"}),
                html.H5(f"{stab:.2%}", style={"color": "#10b981",
                         "marginBottom": "0"}),
            ]), className="bg-dark border-secondary"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Eşleşen Müşteri", style={"color": "#7e8fa4",
                       "fontSize": "0.72rem", "marginBottom": "2px"}),
                html.H5(f"{matched_count:,}", style={"color": "#c8cdd8",
                         "marginBottom": "0"}),
            ]), className="bg-dark border-secondary"), width=3),
        ], className="g-2 mb-3"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


# ── Callback 1: Populate ───────────────────────────────────────────────────
@app.callback(
    Output("mon-migration-trend-dd", "options"),
    Output("mon-migration-trend-dd", "value"),
    Output("mon-migration-trend-chart", "children"),
    Output("mon-migration-cum-content", "children"),
    Input("store-mon-summaries-signal", "data"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_migration_populate(signal, key):
    if not signal or not key:
        return [], None, _NO_DATA, _NO_DATA

    summaries = _MON_STORE.get(key + "_period_summaries", [])
    # Göç matrisi olan dönemler
    with_mig = [s for s in summaries if s.get("migration_matrix") is not None]
    if not with_mig:
        return [], None, _NO_DATA, _NO_DATA

    options = [{"label": s["period_label"], "value": s["period_label"]}
               for s in with_mig]
    default = with_mig[-1]["period_label"]

    # Trend chart — kararlılık oranı
    labels = [s["period_label"] for s in with_mig]
    stab_vals = [_stability_ratio(s["migration_matrix"]) * 100 for s in with_mig]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=stab_vals, mode="lines+markers",
                             name="Kararlılık (%)",
                             line=dict(color="#10b981", width=2),
                             marker=dict(size=6)))
    fig.update_layout(**_CHART_LAYOUT, title="Kararlılık Oranı Trendi",
                      yaxis_title="Kararlılık (%)")
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Kümülatif
    cum = aggregate_summaries(with_mig)
    if cum and cum.get("migration_matrix"):
        cum_content = _render_migration(
            cum["migration_matrix"], cum["migration_matched_count"],
            "Kümülatif Göç Matrisi")
    else:
        cum_content = _NO_DATA

    return options, default, chart, cum_content


# ── Callback 2: Dönem seçimi ───────────────────────────────────────────────
@app.callback(
    Output("mon-migration-trend-detail", "children"),
    Input("mon-migration-trend-dd", "value"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_migration_select_period(period_label, key):
    if not period_label or not key:
        return _NO_DATA
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    selected = next((s for s in summaries if s["period_label"] == period_label), None)
    if not selected or selected.get("migration_matrix") is None:
        return _NO_DATA
    return _render_migration(
        selected["migration_matrix"],
        selected.get("migration_matched_count", 0),
        f"Dönem: {period_label}")
