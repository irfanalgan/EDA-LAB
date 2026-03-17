from dash import Input, Output, State
from app_instance import app

_SHOW = {"display": "block", "position": "absolute", "inset": "0",
         "zIndex": "500", "backgroundColor": "#0E1117",
         "overflowY": "auto", "padding": "0"}
_HIDE = {"display": "none"}


@app.callback(
    Output("help-overlay", "style"),
    Input("btn-help-open",  "n_clicks"),
    Input("btn-help-close", "n_clicks"),
    State("help-overlay", "style"),
    prevent_initial_call=True,
)
def toggle_help(open_clicks, close_clicks, current_style):
    from dash import callback_context
    triggered = callback_context.triggered_id
    if triggered == "btn-help-open":
        return _SHOW
    return _HIDE
