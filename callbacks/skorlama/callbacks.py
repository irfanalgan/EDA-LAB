"""Skorlama callback'leri."""

import logging
import uuid
import base64
import io
import pickle

import dash
from dash import html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from app_instance import app

log = logging.getLogger(__name__)

# ── Sunucu tarafli gecici depolama (dcc.Store JSON'a sigmaz) ──────────────────
_SKOR_STORE: dict = {}

# ── Ortak DataTable stili ─────────────────────────────────────────────────────
_TBL = dict(
    sort_action="native", page_size=20,
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "#161d2e", "color": "#a8b2c2",
                  "fontWeight": "600", "fontSize": "0.7rem",
                  "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
    style_cell={"backgroundColor": "#111827", "color": "#d1d5db",
                "fontSize": "0.78rem", "border": "1px solid #1f2a3c",
                "padding": "5px 8px", "textAlign": "left"},
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}
    ],
)

_A = {"padding": "0.4rem 0.75rem", "fontSize": "0.78rem"}

# Ortak output listesi — veri yukleme callback'leri icin
_DATA_OUTPUTS = [
    Output("skorlama-load-status", "children"),
    Output("skorlama-dd-exclude-cols", "options"),
    Output("skorlama-dd-exclude-cols", "value"),
    Output("skorlama-store-session-key", "data"),
    Output("skorlama-validation-msg", "children"),
    Output("skorlama-preview-table", "children"),
    Output("skorlama-score-status", "children"),
    Output("skorlama-btn-push-sql", "disabled"),
]

# CSV callback icin allow_duplicate versiyonu
_DATA_OUTPUTS_DUP = [
    Output("skorlama-load-status", "children", allow_duplicate=True),
    Output("skorlama-dd-exclude-cols", "options", allow_duplicate=True),
    Output("skorlama-dd-exclude-cols", "value", allow_duplicate=True),
    Output("skorlama-store-session-key", "data", allow_duplicate=True),
    Output("skorlama-validation-msg", "children", allow_duplicate=True),
    Output("skorlama-preview-table", "children", allow_duplicate=True),
    Output("skorlama-score-status", "children", allow_duplicate=True),
    Output("skorlama-btn-push-sql", "disabled", allow_duplicate=True),
]


def _store_df(df, existing_key=None):
    """DataFrame'i _SKOR_STORE'a kaydet, UI degerlerini dondur."""
    key = existing_key or str(uuid.uuid4())
    # Eski verileri temizle
    for suffix in ("_df", "_bundle", "_opt", "_scored"):
        _SKOR_STORE.pop(key + suffix, None)
    _SKOR_STORE[key + "_df"] = df

    opts = [{"label": c, "value": c} for c in df.columns]
    status = dbc.Alert(
        f"{len(df):,} satir x {len(df.columns)} kolon yuklendi.",
        color="success", style=_A,
    )
    return (status, opts, [], key, "", "", "", True)


# ── CB0: Kaynak toggle (SQL / CSV) ───────────────────────────────────────────
@app.callback(
    Output("skorlama-source-sql-div", "style"),
    Output("skorlama-source-csv-div", "style"),
    Input("skorlama-radio-source", "value"),
)
def toggle_source(source):
    if source == "csv":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


# ── CB1a: SQL'den veri yukle ──────────────────────────────────────────────────
@app.callback(
    *_DATA_OUTPUTS,
    Input("skorlama-btn-load-sql", "n_clicks"),
    State("skorlama-sql-src-server", "value"),
    State("skorlama-sql-src-database", "value"),
    State("skorlama-sql-src-driver", "value"),
    State("skorlama-sql-src-table", "value"),
    State("skorlama-chk-top1000", "value"),
    State("skorlama-store-session-key", "data"),
    prevent_initial_call=True,
)
def load_sql(n, server, database, driver, table, top1000, existing_key):
    if not table or not table.strip():
        return (dbc.Alert("Tablo adi giriniz.", color="warning", style=_A),
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update)
    try:
        from data.loader import get_data_from_sql
        top_n = 1000 if top1000 and "top1000" in top1000 else None
        df = get_data_from_sql(table.strip(), server=server, database=database,
                               driver=driver, top_n=top_n)
        return _store_df(df, existing_key)
    except Exception as e:
        log.error("[skorlama] SQL yuklemede hata: %s", e)
        return (dbc.Alert(f"Hata: {e}", color="danger", style=_A),
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update)


# ── CB1b: CSV'den veri yukle ──────────────────────────────────────────────────
@app.callback(
    *_DATA_OUTPUTS_DUP,
    Input("skorlama-btn-load-csv", "n_clicks"),
    State("skorlama-upload-data", "contents"),
    State("skorlama-upload-data", "filename"),
    State("skorlama-csv-separator", "value"),
    State("skorlama-store-session-key", "data"),
    prevent_initial_call=True,
)
def load_csv(n, contents, filename, sep, existing_key):
    if not contents:
        return (dbc.Alert("Once bir CSV dosyasi secin.", color="warning", style=_A),
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update)
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")),
                         sep=sep, low_memory=False)
        return _store_df(df, existing_key)
    except Exception as e:
        log.error("[skorlama] CSV yuklemede hata: %s", e)
        return (dbc.Alert(f"Hata: {e}", color="danger", style=_A),
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update)


# ── CB2: Model pickle yukle ──────────────────────────────────────────────────
@app.callback(
    Output("skorlama-model-status", "children"),
    Output("skorlama-validation-msg", "children", allow_duplicate=True),
    Input("skorlama-upload-model", "contents"),
    State("skorlama-upload-model", "filename"),
    State("skorlama-store-session-key", "data"),
    State("skorlama-dd-exclude-cols", "value"),
    prevent_initial_call=True,
)
def upload_model(contents, filename, key, exclude_cols):
    if not contents:
        return no_update, no_update
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        bundle = pickle.loads(decoded)
    except Exception as e:
        return dbc.Alert(f"Pickle okunamadi: {e}", color="danger", style=_A), no_update

    if not isinstance(bundle, dict) or "model" not in bundle or "model_vars" not in bundle:
        return dbc.Alert("Gecersiz pickle formati. 'model' ve 'model_vars' anahtarlari gerekli.",
                         color="danger", style=_A), no_update

    if key:
        _SKOR_STORE[key + "_bundle"] = bundle
    else:
        return dbc.Alert("Once veri yukleyin.", color="warning", style=_A), no_update

    algo = bundle.get("algo", "?")
    tab = bundle.get("tab", "?")
    n_vars = len(bundle["model_vars"])
    thr = bundle.get("opt_thr", 0.5)

    status = dbc.Alert(
        f"{algo.upper()} | {tab.upper()} | {n_vars} degisken | esik={thr:.4f}",
        color="info", style=_A,
    )

    validation = _build_validation(key, exclude_cols)
    return status, validation


# ── CB3: OPT pickle yukle ────────────────────────────────────────────────────
@app.callback(
    Output("skorlama-opt-status", "children"),
    Input("skorlama-upload-opt", "contents"),
    State("skorlama-upload-opt", "filename"),
    State("skorlama-store-session-key", "data"),
    prevent_initial_call=True,
)
def upload_opt(contents, filename, key):
    if not contents:
        return no_update
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        opt_dict = pickle.loads(decoded)
    except Exception as e:
        return dbc.Alert(f"OPT pickle okunamadi: {e}", color="danger", style=_A)

    if not isinstance(opt_dict, dict):
        return dbc.Alert("OPT pickle dict formatinda olmali.", color="danger", style=_A)

    if key:
        # Model bundle varsa, sadece model değişkenlerini tut
        bundle = _SKOR_STORE.get(key + "_bundle")
        if bundle:
            model_vars = bundle.get("model_vars", [])
            if model_vars:
                opt_dict = {k: v for k, v in opt_dict.items() if k in model_vars}
        _SKOR_STORE[key + "_opt"] = opt_dict

    return dbc.Alert(f"OPT yuklendi — {len(opt_dict)} degisken", color="info", style=_A)


# ── CB4: Kolon secimi degistiginde validasyon ─────────────────────────────────
@app.callback(
    Output("skorlama-validation-msg", "children", allow_duplicate=True),
    Input("skorlama-dd-exclude-cols", "value"),
    State("skorlama-store-session-key", "data"),
    prevent_initial_call=True,
)
def validate_on_exclude(exclude_cols, key):
    return _build_validation(key, exclude_cols)


def _build_validation(key, exclude_cols):
    """Model degiskenleri vs veri kolonlari karsilastirmasi."""
    if not key:
        return ""
    df = _SKOR_STORE.get(key + "_df")
    bundle = _SKOR_STORE.get(key + "_bundle")
    if df is None or bundle is None:
        return ""

    model_vars = bundle["model_vars"]
    available = set(df.columns) - set(exclude_cols or [])
    found = [v for v in model_vars if v in available]
    missing = [v for v in model_vars if v not in available]

    parts = []
    parts.append(html.Span(
        f"{len(found)}/{len(model_vars)} model degiskeni bulundu",
        style={"color": "#10B981" if not missing else "#F59E0B"},
    ))
    if missing:
        parts.append(html.Br())
        parts.append(html.Span(
            f"Eksik: {', '.join(missing[:10])}"
            + (f" (+{len(missing)-10} daha)" if len(missing) > 10 else ""),
            style={"color": "#EF4444", "fontSize": "0.75rem"},
        ))
    return html.Div(parts)


# ── CB5: Skorla ──────────────────────────────────────────────────────────────
@app.callback(
    Output("skorlama-score-status", "children", allow_duplicate=True),
    Output("skorlama-preview-table", "children", allow_duplicate=True),
    Output("skorlama-btn-push-sql", "disabled", allow_duplicate=True),
    Input("skorlama-btn-score", "n_clicks"),
    State("skorlama-store-session-key", "data"),
    State("skorlama-dd-exclude-cols", "value"),
    prevent_initial_call=True,
)
def score(n_clicks, key, exclude_cols):
    if not key:
        return dbc.Alert("Once veri yukleyin.", color="warning", style=_A), no_update, True

    df = _SKOR_STORE.get(key + "_df")
    bundle = _SKOR_STORE.get(key + "_bundle")
    if df is None:
        return dbc.Alert("Veri bulunamadi.", color="warning", style=_A), no_update, True
    if bundle is None:
        return dbc.Alert("Model pickle yukleyin.", color="warning", style=_A), no_update, True

    model = bundle["model"]
    scaler = bundle.get("scaler")
    model_vars = bundle["model_vars"]
    tab = bundle.get("tab", "raw")
    opt_dict = _SKOR_STORE.get(key + "_opt")

    # Eksik degisken kontrolu
    available = set(df.columns) - set(exclude_cols or [])
    missing = [v for v in model_vars if v not in available]
    if missing:
        return (
            dbc.Alert(f"Eksik degiskenler: {', '.join(missing[:15])}", color="danger", style=_A),
            no_update, True,
        )

    try:
        X = df[model_vars].copy()

        # WoE donusumu
        if tab == "woe" and opt_dict:
            for c in X.columns:
                if c in opt_dict:
                    test_var = X[c].values
                    X[c] = opt_dict[c].transform(
                        test_var,
                        metric="woe",
                        metric_missing="empirical",
                        metric_special="empirical",
                    )

        # Scaler
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

        # Predict
        probas = model.predict_proba(X)[:, 1]

        # Sonuc: orijinal tum kolonlar + PREDICT_PROBA saga eklenir
        result_df = df.copy()
        result_df["PREDICT_PROBA"] = probas
        _SKOR_STORE[key + "_scored"] = result_df

        # Preview (ilk 50 satir)
        preview = result_df.head(50)
        table = dash_table.DataTable(
            id="skorlama-datatable-preview",
            columns=[{"name": c, "id": c} for c in preview.columns],
            data=preview.to_dict("records"),
            **_TBL,
        )
        status = dbc.Alert(
            f"Skorlama tamamlandi — {len(result_df):,} satir, "
            f"PREDICT_PROBA eklendi.",
            color="success", style=_A,
        )
        return status, table, False

    except Exception as e:
        log.error("[skorlama] Skorlamada hata: %s", e)
        return dbc.Alert(f"Hata: {e}", color="danger", style=_A), no_update, True


# ── CB6: SQL'e yaz ───────────────────────────────────────────────────────────
@app.callback(
    Output("skorlama-sql-status", "children"),
    Input("skorlama-btn-push-sql", "n_clicks"),
    State("skorlama-store-session-key", "data"),
    State("skorlama-sql-server", "value"),
    State("skorlama-sql-database", "value"),
    State("skorlama-sql-driver", "value"),
    State("skorlama-sql-table-name", "value"),
    prevent_initial_call=True,
)
def push_to_sql(n_clicks, key, server, database, driver, table_name):
    if not key or key + "_scored" not in _SKOR_STORE:
        return dbc.Alert("Once skorlama yapin.", color="warning", style=_A)
    if not table_name or not table_name.strip():
        return dbc.Alert("Tablo adi giriniz.", color="warning", style=_A)
    if not server or not database:
        return dbc.Alert("Server ve Database bilgisi gerekli.", color="warning", style=_A)

    result_df = _SKOR_STORE[key + "_scored"]
    try:
        from sqlalchemy import create_engine
        drv = driver.replace(" ", "+") if driver else "ODBC+Driver+18+for+SQL+Server"
        conn_str = (f"mssql+pyodbc://{server}/{database}"
                    f"?trusted_connection=yes&driver={drv}&TrustServerCertificate=yes")
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            result_df.to_sql(
                name=table_name.strip(),
                con=conn,
                if_exists="replace",
                index=False,
            )
        msg = (f"{len(result_df):,} satir x {len(result_df.columns)} kolon → "
               f"{table_name.strip()} tablosuna yazildi.")
        # Bellek temizligi — tek seferlik is, veriyle baska isimiz yok
        for suffix in ("_df", "_bundle", "_opt", "_scored"):
            _SKOR_STORE.pop(key + suffix, None)
        log.info("[skorlama] Session %s temizlendi — bellek serbest.", key[:8])
        return dbc.Alert(msg, color="success", style=_A)
    except Exception as e:
        log.error("[skorlama] SQL yaziminda hata: %s", e)
        return dbc.Alert(f"SQL Hatasi: {e}", color="danger", style=_A)
