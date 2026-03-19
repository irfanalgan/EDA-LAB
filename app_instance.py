import dash
import dash_bootstrap_components as dbc

import urllib.request

def _cdn_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False

_extra_css = []
_font_url = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap"
_icon_url = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
if _cdn_reachable(_font_url):
    _extra_css.append(_font_url)
if _cdn_reachable(_icon_url):
    _extra_css.append(_icon_url)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY] + _extra_css,
    title="EDA LAB",
    suppress_callback_exceptions=True,
)
