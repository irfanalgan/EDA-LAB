import dash
from dash import Input, Output

from app_instance import app
from layout import _STICKY_TABS

_TAB_IDS = [tid for tid, _ in _STICKY_TABS]


@app.callback(
    Output("main-tabs", "active_tab"),
    [Input(f"sticky-tab-{tid}", "n_clicks") for tid in _TAB_IDS],
    prevent_initial_call=True,
)
def sticky_nav_switch(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return btn_id.replace("sticky-tab-", "")


@app.callback(
    [Output(f"sticky-tab-{tid}", "className") for tid in _TAB_IDS],
    Input("main-tabs", "active_tab"),
)
def update_sticky_active(active_tab):
    return [
        "sn-link sn-active" if tid == active_tab else "sn-link"
        for tid in _TAB_IDS
    ]
