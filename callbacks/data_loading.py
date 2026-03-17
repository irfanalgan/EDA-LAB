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
from utils.helpers import coerce_numeric_columns


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
        df, converted = coerce_numeric_columns(df)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        conv_note = f"  ·  {len(converted)} kolon otomatik numerik dönüştürüldü" if converted else ""
        return key, dbc.Alert(
            [html.Strong(f"{len(df):,} satır"),
             f"  ·  {df.shape[1]} kolon  ·  {filename}{conv_note}"],
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
        df, converted = coerce_numeric_columns(df)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        conv_note = f"  ·  {len(converted)} kolon otomatik numerik dönüştürüldü" if converted else ""
        return key, dbc.Alert(
            [
                html.Strong(f"{len(df):,} satır"),
                f"  ·  {df.shape[1]} kolon yüklendi.{conv_note}",
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


# ── Callback: OOT Tarih Dropdown'ını Doldur ───────────────────────────────────
@app.callback(
    Output("collapse-oot-date", "is_open"),
    Output("dd-oot-date", "options"),
    Output("dd-oot-date", "value"),
    Input("dd-date-col", "value"),
    State("store-key", "data"),
)
def populate_oot_date(date_col, key):
    """Tarih kolonu seçilince OOT dropdown'ını distinct aylarla doldur."""
    if not date_col or not key:
        return False, [{"label": "— opsiyonel —", "value": ""}], ""
    df = _get_df(key)
    if df is None or date_col not in df.columns:
        return False, [{"label": "— opsiyonel —", "value": ""}], ""
    raw_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    distinct = sorted(raw_dates.dt.to_period("M").unique().astype(str))
    if not distinct:
        return False, [{"label": "— opsiyonel —", "value": ""}], ""
    opts = [{"label": "— opsiyonel —", "value": ""}] + [{"label": d, "value": d} for d in distinct]
    # Varsayılan: ~%80 noktası (train büyük, OOT küçük)
    default_idx = max(0, int(len(distinct) * 0.8))
    default = distinct[default_idx]
    return True, opts, default


# ── Callback: Train/Test Collapse (config paneli) ─────────────────────────────
@app.callback(
    Output("collapse-test-size-cfg", "is_open"),
    Input("chk-train-test-split", "value"),
)
def toggle_test_size_cfg(val):
    return bool(val)
