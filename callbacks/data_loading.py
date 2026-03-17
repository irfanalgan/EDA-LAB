import uuid
import base64
import io

import dash
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from data.loader import get_data_from_sql


# ── Callback: Kaynak Toggle ───────────────────────────────────────────────────
@app.callback(
    Output("source-sql-div", "style"),
    Output("source-csv-div", "style"),
    Input("radio-source", "value"),
)
def toggle_source(source):
    if source == "csv":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


# ── Callback: CSV Dosya Adı Göster ────────────────────────────────────────────
@app.callback(
    Output("csv-filename-display", "children"),
    Input("upload-csv", "filename"),
    prevent_initial_call=True,
)
def show_csv_filename(filename):
    if filename:
        return f"📄 {filename}"
    return ""


# ── Callback: CSV Yükle ───────────────────────────────────────────────────────
@app.callback(
    Output("store-key",   "data",     allow_duplicate=True),
    Output("load-status", "children", allow_duplicate=True),
    Input("btn-load-csv", "n_clicks"),
    State("upload-csv",   "contents"),
    State("upload-csv",   "filename"),
    State("csv-separator", "value"),
    prevent_initial_call=True,
)
def load_csv(n_clicks, contents, filename, sep):
    if not contents or not filename:
        return dash.no_update, dbc.Alert(
            "Önce bir CSV dosyası seçin.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    if not filename.lower().endswith(".csv"):
        return dash.no_update, dbc.Alert(
            "Yalnızca .csv uzantılı dosyalar desteklenir.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")),
                         sep=sep, low_memory=False)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        return key, dbc.Alert(
            [html.Strong(f"{len(df):,} satır"),
             f"  ·  {df.shape[1]} kolon  ·  {filename}"],
            color="success",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    except Exception as e:
        return dash.no_update, dbc.Alert(
            f"Okuma hatası: {e}", color="danger",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )


# ── Callback: Veriyi Yükle (SQL) ──────────────────────────────────────────────
@app.callback(
    Output("store-key", "data"),
    Output("load-status", "children"),
    Input("btn-load", "n_clicks"),
    State("input-table", "value"),
    prevent_initial_call=True,
)
def load_data(n_clicks, table_name):
    if not table_name or not table_name.strip():
        return dash.no_update, dbc.Alert(
            "Lütfen bir tablo adı girin.", color="warning",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    try:
        df = get_data_from_sql(table_name.strip())
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        return key, dbc.Alert(
            [
                html.Strong(f"{len(df):,} satır"),
                f"  ·  {df.shape[1]} kolon yüklendi.",
            ],
            color="success",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )
    except Exception as e:
        return dash.no_update, dbc.Alert(
            str(e), color="danger",
            style={"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"},
        )


# ── Callback: Kolon Yapılandırması bölümünü aç, dropdown seçeneklerini doldur ─
@app.callback(
    Output("collapse-config", "is_open"),
    Output("dd-target-col", "options"),
    Output("dd-date-col", "options"),
    Output("dd-segment-col", "options"),
    Input("store-key", "data"),
)
def open_config_section(key):
    df = _get_df(key)
    if df is None:
        return False, [], [], []

    all_opts = (
        [{"label": "Kolon seçiniz...", "value": "", "disabled": True}]
        + [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in df.columns]
    )

    # Tarih kolonları — datetime veya isimde date/tarih geçenler başa alınır
    date_keywords = {"date", "tarih", "dt", "time", "zaman"}
    date_cols = sorted(
        df.columns,
        key=lambda c: (
            0 if pd.api.types.is_datetime64_any_dtype(df[c])
            else 1 if any(k in c.lower() for k in date_keywords)
            else 2
        ),
    )
    date_opts = (
        [{"label": "—", "value": ""}]
        + [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in date_cols]
    )

    # Segment kolonları — object veya düşük kardinalite
    seg_cols = [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= 50
    ]
    seg_opts = (
        [{"label": "—", "value": ""}]
        + [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in seg_cols]
    )

    return True, all_opts, date_opts, seg_opts
