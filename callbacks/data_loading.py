import uuid
import base64
import io

import dash
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from data.loader import get_data_from_sql, get_data_from_sql_multi, get_config_defaults
from utils.helpers import coerce_numeric_columns


# ── Callback: SQL bağlantı alanlarını config.toml'dan doldur ─────────────────
@app.callback(
    Output("input-sql-server",   "value"),
    Output("input-sql-database", "value"),
    Output("dd-sql-driver",      "value"),
    Input("radio-source", "value"),
)
def fill_sql_defaults(_):
    cfg = get_config_defaults()
    return cfg["server"], cfg["database"], cfg["driver"]


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


# ── Callback: SQL — tablo satırı ekle/kaldır ─────────────────────────────────
@app.callback(
    Output("store-sql-table-count", "data"),
    Output("sql-table-row-2",       "style"),
    Output("sql-table-row-3",       "style"),
    Output("div-sql-join-key",      "style"),
    Input("btn-add-sql-table",  "n_clicks"),
    Input("btn-remove-sql-2",   "n_clicks"),
    Input("btn-remove-sql-3",   "n_clicks"),
    State("store-sql-table-count", "data"),
    prevent_initial_call=True,
)
def manage_sql_tables(add, rem2, rem3, count):
    ctx = dash.callback_context.triggered_id
    if ctx == "btn-add-sql-table":
        count = min(count + 1, 3)
    elif ctx == "btn-remove-sql-2":
        count = max(count - 1, 1)
    elif ctx == "btn-remove-sql-3":
        count = max(count - 1, 2)

    show  = {"display": "block"}
    hide  = {"display": "none"}
    row2  = show if count >= 2 else hide
    row3  = show if count >= 3 else hide
    jk    = show if count >= 2 else hide
    return count, row2, row3, jk


# ── Callback: CSV — dosya satırı ekle/kaldır ─────────────────────────────────
@app.callback(
    Output("store-csv-file-count", "data"),
    Output("csv-file-row-2",       "style"),
    Output("csv-file-row-3",       "style"),
    Output("div-csv-join-key",     "style"),
    Input("btn-add-csv-file",  "n_clicks"),
    Input("btn-remove-csv-2",  "n_clicks"),
    Input("btn-remove-csv-3",  "n_clicks"),
    State("store-csv-file-count", "data"),
    prevent_initial_call=True,
)
def manage_csv_files(add, rem2, rem3, count):
    ctx = dash.callback_context.triggered_id
    if ctx == "btn-add-csv-file":
        count = min(count + 1, 3)
    elif ctx == "btn-remove-csv-2":
        count = max(count - 1, 1)
    elif ctx == "btn-remove-csv-3":
        count = max(count - 1, 2)

    show = {"display": "block"}
    hide = {"display": "none"}
    row2 = show if count >= 2 else hide
    row3 = show if count >= 3 else hide
    jk   = show if count >= 2 else hide
    return count, row2, row3, jk


# ── Callback: CSV Dosya Adı Göster ────────────────────────────────────────────
@app.callback(
    Output("csv-filename-display",   "children"),
    Output("csv-filename-display-2", "children"),
    Output("csv-filename-display-3", "children"),
    Input("upload-csv",   "filename"),
    Input("upload-csv-2", "filename"),
    Input("upload-csv-3", "filename"),
    prevent_initial_call=True,
)
def show_csv_filenames(fn1, fn2, fn3):
    def fmt(fn):
        return fn if fn else ""
    return fmt(fn1), fmt(fn2), fmt(fn3)


# ── Yardımcı: CSV içeriğini oku ───────────────────────────────────────────────
def _read_csv_content(contents, filename, sep):
    if not contents or not filename:
        return None, None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")),
                     sep=sep, low_memory=False)
    return df, filename


def _join_dataframes(dfs: list[pd.DataFrame], join_key: list[str]) -> pd.DataFrame:
    """Master + ek DataFrame'leri join_key üzerinde axis=1 birleştirir."""
    result = dfs[0]
    jk = [k.strip() for k in join_key if k.strip()]
    if jk:
        result = result.set_index(jk)
    for df in dfs[1:]:
        if jk:
            df = df.set_index(jk)
        drop_cols = [c for c in df.columns if c in result.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        result = pd.concat([result, df], axis=1)
    if jk:
        result = result.reset_index()
    return result


# ── Callback: CSV Yükle ───────────────────────────────────────────────────────
@app.callback(
    Output("store-key",   "data",     allow_duplicate=True),
    Output("load-status", "children", allow_duplicate=True),
    Input("btn-load-csv", "n_clicks"),
    State("upload-csv",   "contents"), State("upload-csv",   "filename"),
    State("upload-csv-2", "contents"), State("upload-csv-2", "filename"),
    State("upload-csv-3", "contents"), State("upload-csv-3", "filename"),
    State("csv-separator",        "value"),
    State("input-csv-join-key",   "value"),
    State("store-csv-file-count", "data"),
    prevent_initial_call=True,
)
def load_csv(n_clicks,
             c1, fn1, c2, fn2, c3, fn3,
             sep, join_key_raw, file_count):
    if not c1 or not fn1:
        return dash.no_update, _warn("Önce bir CSV dosyası seçin.")

    sep = sep or ","
    join_key = [k.strip() for k in (join_key_raw or "").split(",") if k.strip()]

    try:
        dfs = []
        filenames = []
        for contents, filename in [(c1, fn1), (c2, fn2), (c3, fn3)]:
            if contents and filename:
                df, fn = _read_csv_content(contents, filename, sep)
                dfs.append(df)
                filenames.append(fn)

        if len(dfs) > 1 and join_key:
            result = _join_dataframes(dfs, join_key)
        else:
            result = dfs[0]

        result, converted = coerce_numeric_columns(result)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = result
        _SERVER_STORE[f"{key}_quality"] = {"converted": converted}

        files_str = " + ".join(filenames)
        conv_note = f"  ·  {len(converted)} kolon numerik dönüştürüldü" if converted else ""
        join_note = f"  ·  {len(dfs)} dosya birleştirildi" if len(dfs) > 1 else ""
        return key, _ok(f"{len(result):,} satır  ·  {result.shape[1]} kolon  ·  {files_str}{join_note}{conv_note}")

    except Exception as e:
        return dash.no_update, _err(f"Okuma hatası: {e}")


# ── Callback: Veriyi Yükle (SQL) ──────────────────────────────────────────────
@app.callback(
    Output("store-key", "data"),
    Output("load-status", "children"),
    Input("btn-load", "n_clicks"),
    State("input-table-1",        "value"),
    State("input-table-2",        "value"),
    State("input-table-3",        "value"),
    State("input-sql-join-key",   "value"),
    State("store-sql-table-count","data"),
    State("input-sql-server",     "value"),
    State("input-sql-database",   "value"),
    State("dd-sql-driver",        "value"),
    prevent_initial_call=True,
)
def load_data(n_clicks,
              t1, t2, t3,
              join_key_raw, table_count,
              server, database, driver):
    if not t1 or not t1.strip():
        return dash.no_update, _warn("Lütfen bir tablo adı girin.")

    join_key = [k.strip() for k in (join_key_raw or "").split(",") if k.strip()]
    tables   = [t for t in [t1, t2, t3] if t and t.strip()]

    try:
        if len(tables) > 1 and join_key:
            df = get_data_from_sql_multi(
                tables=tables,
                join_key=join_key,
                server=server, database=database, driver=driver,
            )
            join_note = f"  ·  {len(tables)} tablo birleştirildi"
        else:
            df = get_data_from_sql(
                tables[0],
                server=server, database=database, driver=driver,
            )
            join_note = ""

        df, converted = coerce_numeric_columns(df)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        _SERVER_STORE[f"{key}_quality"] = {"converted": converted}
        conv_note = f"  ·  {len(converted)} kolon numerik dönüştürüldü" if converted else ""
        return key, _ok(f"{len(df):,} satır  ·  {df.shape[1]} kolon{join_note}{conv_note}")

    except Exception as e:
        return dash.no_update, _err(str(e))


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
    default_idx = max(0, int(len(distinct) * 0.8))
    return True, opts, distinct[default_idx]


# ── Callback: Train/Test Collapse ─────────────────────────────────────────────
@app.callback(
    Output("collapse-test-size-cfg", "is_open"),
    Input("chk-train-test-split", "value"),
)
def toggle_test_size_cfg(val):
    return bool(val)


# ── Yardımcı alert fonksiyonları ──────────────────────────────────────────────
_ALERT_STYLE = {"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"}

def _ok(msg):
    return dbc.Alert(msg, color="success", style=_ALERT_STYLE)

def _warn(msg):
    return dbc.Alert(msg, color="warning", style=_ALERT_STYLE)

def _err(msg):
    return dbc.Alert(msg, color="danger", style=_ALERT_STYLE)
