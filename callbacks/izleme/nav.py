"""Geliştirme / İzleme üst seviye container toggle + İzleme sidebar toggle."""

import dash
from dash import Input, Output
from app_instance import app

# ── Sidebar geçiş sabitleri (layout/izleme.py ile aynı) ─────────────────────
_T = "max-width 0.3s ease-in-out, flex 0.3s ease-in-out, opacity 0.25s ease-in-out"
_COL_SIDEBAR_OPEN   = {"padding": "0", "position": "relative", "overflow": "visible",
                        "flex": "0 0 25%",  "maxWidth": "25%",  "opacity": "1",
                        "transition": _T}
_COL_SIDEBAR_CLOSED = {"padding": "0", "position": "relative", "overflow": "visible",
                        "flex": "0 0 36px", "maxWidth": "36px", "opacity": "1",
                        "transition": _T}
_COL_MAIN_OPEN      = {"padding": "0", "flex": "0 0 75%",               "maxWidth": "75%",
                        "transition": _T}
_COL_MAIN_CLOSED    = {"padding": "0", "flex": "0 0 calc(100% - 36px)", "maxWidth": "calc(100% - 36px)",
                        "transition": _T}
_SIDEBAR_OPEN_STYLE   = {"transition": "opacity 0.25s ease-in-out, max-width 0.3s ease-in-out"}
_SIDEBAR_CLOSED_STYLE = {"maxWidth": "0", "overflow": "hidden", "opacity": "0",
                          "padding": "0", "minHeight": "0",
                          "transition": "opacity 0.2s ease-in-out, max-width 0.3s ease-in-out"}


# ── Üst seviye Geliştirme / İzleme toggle ────────────────────────────────────
@app.callback(
    Output("container-gelistirme", "style"),
    Output("container-izleme", "style"),
    Output("btn-nav-gelistirme", "className"),
    Output("btn-nav-izleme", "className"),
    Input("btn-nav-gelistirme", "n_clicks"),
    Input("btn-nav-izleme", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_top_section(n_gel, n_izl):
    ctx = dash.callback_context.triggered_id
    if ctx == "btn-nav-izleme":
        return (
            {"display": "none"},
            {"display": "block"},
            "top-nav-link",
            "top-nav-link active",
        )
    # default: Geliştirme aktif
    return (
        {"display": "block"},
        {"display": "none"},
        "top-nav-link active",
        "top-nav-link",
    )


# ── İzleme sidebar toggle ───────────────────────────────────────────────────
@app.callback(
    Output("col-mon-sidebar", "style"),
    Output("col-mon-main", "style"),
    Output("mon-sidebar", "style"),
    Output("mon-btn-sidebar-toggle", "children"),
    Input("mon-btn-sidebar-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_mon_sidebar(n):
    if n and n % 2 == 1:
        return _COL_SIDEBAR_CLOSED, _COL_MAIN_CLOSED, _SIDEBAR_CLOSED_STYLE, "›"
    return _COL_SIDEBAR_OPEN, _COL_MAIN_OPEN, _SIDEBAR_OPEN_STYLE, "‹"
