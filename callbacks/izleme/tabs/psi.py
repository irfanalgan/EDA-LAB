"""İzleme — PSI tab callback'leri (Değişken PSI + Rating PSI)."""

from dash import html, dcc, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app_instance import app
from server_state import _MON_STORE
from callbacks.izleme.compute import (
    calc_var_psi, calc_rating_psi, aggregate_summaries,
)

# ── Ortak stil sabitleri ────────────────────────────────────────────────────
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

_NO_DATA = html.P("Henüz veri yok.", style={"color": "#7e8fa4",
                   "fontSize": "0.85rem", "textAlign": "center",
                   "padding": "2rem 0"})


def _psi_label(val):
    if val < 0.10:
        return "Düşük"
    if val < 0.25:
        return "Orta"
    return "Yüksek"


def _build_var_psi_table(ref_summary, mon_var_psi):
    """Değişken bazlı detaylı bin-level PSI tabloları üret."""
    ref_var_psi = ref_summary.get("var_psi", {})
    tables = []

    for var in sorted(ref_var_psi.keys()):
        if var not in mon_var_psi:
            continue
        ref_v = ref_var_psi[var]
        mon_v = mon_var_psi[var]
        psi_val, bin_rows = calc_var_psi(ref_v, mon_v)
        if not bin_rows:
            continue

        has_woe = bool(ref_v.get("woe_values"))

        # Toplam hesapla
        ref_total = sum(r["ref_count"] for r in bin_rows)
        mon_total = sum(r["mon_count"] for r in bin_rows)
        ref_bad_total = sum(r["ref_bad"] for r in bin_rows)
        mon_bad_total = sum(r["mon_bad"] for r in bin_rows)
        ref_good_total = ref_total - ref_bad_total
        mon_good_total = mon_total - mon_bad_total

        data = []
        for r in bin_rows:
            ref_good = r["ref_count"] - r["ref_bad"]
            mon_good = r["mon_count"] - r["mon_bad"]
            ref_bad_rate = r["ref_bad"] / r["ref_count"] if r["ref_count"] > 0 else 0
            mon_bad_rate = r["mon_bad"] / r["mon_count"] if r["mon_count"] > 0 else 0
            ref_good_pct = ref_good / ref_good_total * 100 if ref_good_total > 0 else 0
            ref_bad_pct = r["ref_bad"] / ref_bad_total * 100 if ref_bad_total > 0 else 0
            ref_t_pct = r["ref_count"] / ref_total * 100 if ref_total > 0 else 0
            mon_good_pct = mon_good / mon_good_total * 100 if mon_good_total > 0 else 0
            mon_bad_pct = r["mon_bad"] / mon_bad_total * 100 if mon_bad_total > 0 else 0
            mon_t_pct = r["mon_count"] / mon_total * 100 if mon_total > 0 else 0

            # Bin label
            lo = r.get("edge_lo")
            hi = r.get("edge_hi")
            if lo is not None and hi is not None:
                lo_s = "-∞" if lo == float("-inf") else f"{lo:.4g}"
                hi_s = "∞" if hi == float("inf") else f"{hi:.4g}"
                bin_label = f"[{lo_s}, {hi_s})"
            else:
                bin_label = str(r.get("bin_idx", ""))

            row = {
                "Bins": bin_label,
                "REF Good": ref_good,
                "REF Bad": r["ref_bad"],
                "REF Total": r["ref_count"],
                "REF Bad Rate": f"{ref_bad_rate:.2%}",
                "REF Good%": f"{ref_good_pct:.2f}%",
                "REF Bad%": f"{ref_bad_pct:.2f}%",
                "REF T%": f"{ref_t_pct:.2f}%",
            }
            if has_woe:
                row["REF WoE"] = f"{r.get('woe', 0):.4f}"
                row["REF IV"] = f"{r.get('ref_iv_contrib', 0):.4f}"

            row.update({
                "MON Good": mon_good,
                "MON Bad": r["mon_bad"],
                "MON Total": r["mon_count"],
                "MON Bad Rate": f"{mon_bad_rate:.2%}",
                "MON Good%": f"{mon_good_pct:.2f}%",
                "MON Bad%": f"{mon_bad_pct:.2f}%",
                "MON T%": f"{mon_t_pct:.2f}%",
            })
            if has_woe:
                row["MON WoE"] = f"{r.get('woe', 0):.4f}"
                row["MON IV"] = f"{r.get('mon_iv_contrib', 0):.4f}"

            row["Band PSI"] = f"{r['psi_contrib']:.4f}"
            data.append(row)

        # Toplam satır
        ref_br = ref_bad_total / ref_total if ref_total > 0 else 0
        mon_br = mon_bad_total / mon_total if mon_total > 0 else 0
        total_row = {
            "Bins": "TOPLAM",
            "REF Good": ref_good_total,
            "REF Bad": ref_bad_total,
            "REF Total": ref_total,
            "REF Bad Rate": f"{ref_br:.2%}",
            "REF Good%": "100.00%",
            "REF Bad%": "100.00%",
            "REF T%": "100.00%",
        }
        if has_woe:
            ref_iv = sum(r.get("ref_iv_contrib", 0) for r in bin_rows)
            total_row["REF WoE"] = ""
            total_row["REF IV"] = f"{ref_iv:.4f}"

        total_row.update({
            "MON Good": mon_good_total,
            "MON Bad": mon_bad_total,
            "MON Total": mon_total,
            "MON Bad Rate": f"{mon_br:.2%}",
            "MON Good%": "100.00%",
            "MON Bad%": "100.00%",
            "MON T%": "100.00%",
        })
        if has_woe:
            mon_iv = sum(r.get("mon_iv_contrib", 0) for r in bin_rows)
            total_row["MON WoE"] = ""
            total_row["MON IV"] = f"{mon_iv:.4f}"

        total_row["Band PSI"] = f"{psi_val:.4f}"
        data.append(total_row)

        # Kolon sırası
        cols = ["Bins",
                "REF Good", "REF Bad", "REF Total", "REF Bad Rate",
                "REF Good%", "REF Bad%", "REF T%"]
        if has_woe:
            cols += ["REF WoE", "REF IV"]
        cols += ["MON Good", "MON Bad", "MON Total", "MON Bad Rate",
                 "MON Good%", "MON Bad%", "MON T%"]
        if has_woe:
            cols += ["MON WoE", "MON IV"]
        cols += ["Band PSI"]

        psi_color = "#ef4444" if psi_val >= 0.25 else (
            "#f59e0b" if psi_val >= 0.10 else "#10b981")
        tables.append(html.Div([
            html.H6(f"{var} — PSI: {psi_val:.4f} ({_psi_label(psi_val)})",
                     style={"color": psi_color, "fontSize": "0.85rem",
                            "marginTop": "1rem", "marginBottom": "0.3rem"}),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in cols],
                data=data,
                style_header=_TH,
                style_cell={**_TD, "minWidth": "60px"},
                style_data_conditional=[_TD_ODD],
                page_size=30,
                style_table={"overflowX": "auto"},
            ),
        ]))

    if not tables:
        return html.P("Değişken PSI verisi yok.",
                       style={"color": "#7e8fa4", "fontSize": "0.82rem"})

    return html.Div(tables)


def _build_rating_psi_table(ref_summary, mon_summary):
    """Rating PSI tablosu üret."""
    psi_val, bin_rows = calc_rating_psi(
        ref_summary["rating_counts"], mon_summary["rating_counts"])
    if not bin_rows:
        return html.Div()

    data = []
    for r in bin_rows:
        data.append({
            "Rating": r["rating"],
            "REF Adet": r["ref_count"],
            "REF %": f"{r['ref_pct']:.2%}",
            "MON Adet": r["mon_count"],
            "MON %": f"{r['mon_pct']:.2%}",
            "PSI Katkı": f"{r['psi_contrib']:.4f}",
        })
    data.append({
        "Rating": "TOPLAM",
        "REF Adet": sum(r["ref_count"] for r in bin_rows),
        "REF %": "100%",
        "MON Adet": sum(r["mon_count"] for r in bin_rows),
        "MON %": "100%",
        "PSI Katkı": f"{psi_val:.4f}",
    })

    cols = ["Rating", "REF Adet", "REF %", "MON Adet", "MON %", "PSI Katkı"]
    return html.Div([
        html.H6(f"Rating PSI: {psi_val:.4f} — {_psi_label(psi_val)}",
                style={"color": "#c8cdd8", "fontSize": "0.85rem",
                       "marginTop": "1rem", "marginBottom": "0.5rem"}),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in cols],
            data=data,
            style_header=_TH, style_cell=_TD,
            style_data_conditional=[_TD_ODD],
            page_size=30,
            style_table={"overflowX": "auto"},
        ),
    ])


# ── Callback 1: Populate (signal → dropdown + chart + cum) ─────────────────
@app.callback(
    Output("mon-psi-trend-dd", "options"),
    Output("mon-psi-trend-dd", "value"),
    Output("mon-psi-trend-chart", "children"),
    Output("mon-psi-cum-content", "children"),
    Input("store-mon-summaries-signal", "data"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_psi_populate(signal, key):
    if not signal or not key:
        return [], None, _NO_DATA, _NO_DATA

    ref_summary = _MON_STORE.get(key + "_ref_summary")
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    if not ref_summary or not summaries:
        return [], None, _NO_DATA, _NO_DATA

    # Dropdown seçenekleri
    options = [{"label": s["period_label"], "value": s["period_label"]}
               for s in summaries]
    default = summaries[-1]["period_label"]

    # Trend chart — Rating PSI + per-variable PSI
    ref_var = ref_summary.get("var_psi", {})
    labels = [s["period_label"] for s in summaries]

    # Rating PSI trendi
    rating_psis = []
    for s in summaries:
        rpsi, _ = calc_rating_psi(ref_summary["rating_counts"], s["rating_counts"])
        rating_psis.append(rpsi)

    # Per-variable PSI trendi
    var_names = sorted(ref_var.keys())
    var_psi_series = {v: [] for v in var_names}
    for s in summaries:
        for var in var_names:
            if var in s.get("var_psi", {}):
                pv, _ = calc_var_psi(ref_var[var], s["var_psi"][var])
                var_psi_series[var].append(pv)
            else:
                var_psi_series[var].append(0)

    # Renk paleti (değişkenler için)
    _VAR_COLORS = [
        "#60a5fa", "#34d399", "#c084fc", "#fb923c", "#f472b6",
        "#38bdf8", "#a3e635", "#e879f9", "#fbbf24", "#22d3ee",
    ]

    fig = go.Figure()
    # Rating PSI — kalın ana çizgi
    fig.add_trace(go.Scatter(
        x=labels, y=rating_psis, mode="lines+markers",
        name="Rating PSI", line=dict(color="#ef4444", width=3),
        marker=dict(size=7)))
    # Per-variable PSI — ince çizgiler
    for i, var in enumerate(var_names):
        color = _VAR_COLORS[i % len(_VAR_COLORS)]
        fig.add_trace(go.Scatter(
            x=labels, y=var_psi_series[var], mode="lines+markers",
            name=var, line=dict(color=color, width=1.5),
            marker=dict(size=4), opacity=0.7))
    fig.add_hline(y=0.10, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Orta (0.10)")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444",
                  annotation_text="Yüksek (0.25)")
    fig.update_layout(**_CHART_LAYOUT, title="PSI Trendi",
                      yaxis_title="PSI",
                      legend=dict(font=dict(size=9)))
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Kümülatif
    cum = aggregate_summaries(summaries)
    if cum:
        cum_content = html.Div([
            html.H6("Kümülatif Değişken PSI",
                     style={"color": "#c8cdd8", "fontSize": "0.9rem",
                            "marginBottom": "0.5rem"}),
            _build_var_psi_table(ref_summary, cum.get("var_psi", {})),
            _build_rating_psi_table(ref_summary, cum),
        ])
    else:
        cum_content = _NO_DATA

    return options, default, chart, cum_content


# ── Callback 2: Dönem seçimi → trend detail ────────────────────────────────
@app.callback(
    Output("mon-psi-trend-detail", "children"),
    Input("mon-psi-trend-dd", "value"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_psi_select_period(period_label, key):
    if not period_label or not key:
        return _NO_DATA

    ref_summary = _MON_STORE.get(key + "_ref_summary")
    summaries = _MON_STORE.get(key + "_period_summaries", [])
    if not ref_summary or not summaries:
        return _NO_DATA

    # Seçilen dönemi bul
    selected = next((s for s in summaries if s["period_label"] == period_label), None)
    if not selected:
        return _NO_DATA

    return html.Div([
        html.H6(f"Dönem: {period_label}",
                style={"color": "#c8cdd8", "fontSize": "0.9rem",
                       "marginBottom": "0.5rem"}),
        _build_var_psi_table(ref_summary, selected.get("var_psi", {})),
        _build_rating_psi_table(ref_summary, selected),
    ])
