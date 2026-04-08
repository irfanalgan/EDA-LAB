"""Toplu Skorlama callback'leri — chunk chunk skorlama + SQL yazma."""

import threading
import logging
import uuid
import time
import base64
import io
import pickle
import math

import dash
import pandas as pd
import numpy as np
from dash import html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from app_instance import app
from server_state import _PRECOMPUTE_PROGRESS

log = logging.getLogger(__name__)

# ── Server-side store ────────────────────────────────────────────────────────
_BATCH_STORE: dict = {}

_A = {"padding": "0.4rem 0.75rem", "fontSize": "0.78rem"}


# ── Yardimci fonksiyonlar ────────────────────────────────────────────────────
def _build_alchemy_conn(server, database, driver):
    drv = driver.replace(" ", "+") if driver else "ODBC+Driver+18+for+SQL+Server"
    return f"mssql+pyodbc://{server}/{database}?trusted_connection=yes&driver={drv}&TrustServerCertificate=yes"


def _score_chunk(df, model, scaler, model_vars, tab, opt_dict):
    """Tek chunk'i skorla → PREDICT_PROBA kolonlu DataFrame dondur."""
    X = df[model_vars].copy()
    if tab == "woe" and opt_dict:
        for c in X.columns:
            if c in opt_dict:
                X[c] = opt_dict[c].transform(
                    X[c].values,
                    metric="woe",
                    metric_missing="empirical",
                    metric_special="empirical",
                )
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    result = df.copy()
    result["PREDICT_PROBA"] = model.predict_proba(X)[:, 1]
    return result


def _write_chunk(df, engine, table_name, if_exists):
    """DataFrame'i SQL'e yaz."""
    with engine.connect() as conn:
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False)
        conn.commit()


def _build_validation(columns, bundle, exclude_cols):
    """Model degiskenleri vs veri kolonlari karsilastirmasi."""
    if not bundle:
        return ""
    model_vars = bundle["model_vars"]
    available = set(columns) - set(exclude_cols or [])
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


def _build_estimate(total_rows, n_chunks, n_cols):
    """Veri buyuklugune gore tahmini sure hesapla."""
    # Kaba tahmin: chunk basina ~3-5sn (okuma + skorlama + yazma)
    # Kolon sayisi arttikca skorlama yavaslar
    sec_per_chunk = 3 + (n_cols / 100)  # 100 kolon → ~4sn, 300 kolon → ~6sn
    total_sec = n_chunks * sec_per_chunk

    if total_sec < 60:
        time_str = f"~{int(total_sec)} saniye"
    elif total_sec < 3600:
        time_str = f"~{int(total_sec / 60)} dakika"
    else:
        h = int(total_sec / 3600)
        m = int((total_sec % 3600) / 60)
        time_str = f"~{h} saat {m} dakika"

    return html.Div([
        html.I(className="bi bi-clock", style={"marginRight": "0.4rem"}),
        html.Span(f"Tahmini sure: {time_str}"),
        html.Span(f"  ({total_rows:,} satir, {n_chunks} parca)",
                  style={"color": "#6b7a99", "marginLeft": "0.4rem"}),
    ], style={"color": "#7e8fa4", "fontSize": "0.78rem"})


# ── CB-B0: Tab toggle ───────────────────────────────────────────────────────
@app.callback(
    Output("skorlama-tab-tekli", "style"),
    Output("skorlama-tab-toplu", "style"),
    Output("skorlama-btn-tab-tekli", "className"),
    Output("skorlama-btn-tab-toplu", "className"),
    Input("skorlama-btn-tab-tekli", "n_clicks"),
    Input("skorlama-btn-tab-toplu", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_skorlama_tabs(n_tekli, n_toplu):
    ctx = dash.callback_context.triggered_id
    _show = {"padding": "2rem 3rem", "maxWidth": "1100px", "margin": "0 auto"}
    _hide = {"display": "none"}
    if ctx == "skorlama-btn-tab-toplu":
        return _hide, _show, "top-nav-link", "top-nav-link active"
    return _show, _hide, "top-nav-link active", "top-nav-link"


# ── CB-B0b: Rehber modal toggle ──────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-guide-modal", "is_open"),
    Input("skorlama-batch-btn-guide", "n_clicks"),
    State("skorlama-batch-guide-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_guide_modal(n, is_open):
    return not is_open


# ── CB-B1: Kaynak toggle ────────────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-source-sql-div", "style"),
    Output("skorlama-batch-source-files-div", "style"),
    Input("skorlama-batch-radio-source", "value"),
)
def toggle_batch_source(source):
    if source == "files":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


# ── CB-B2: Model pickle yukle ───────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-model-status", "children"),
    Output("skorlama-batch-store-key", "data"),
    Input("skorlama-batch-upload-model", "contents"),
    State("skorlama-batch-upload-model", "filename"),
    State("skorlama-batch-store-key", "data"),
    prevent_initial_call=True,
)
def upload_batch_model(contents, filename, existing_key):
    if not contents:
        return no_update, no_update
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        bundle = pickle.loads(decoded)
    except Exception as e:
        return dbc.Alert(f"Pickle okunamadi: {e}", color="danger", style=_A), no_update

    if not isinstance(bundle, dict) or "model" not in bundle or "model_vars" not in bundle:
        return (dbc.Alert("Gecersiz format. 'model' ve 'model_vars' anahtarlari gerekli.",
                          color="danger", style=_A), no_update)

    key = existing_key or str(uuid.uuid4())
    _BATCH_STORE[key + "_bundle"] = bundle

    algo = bundle.get("algo", "?")
    tab = bundle.get("tab", "?")
    n_vars = len(bundle["model_vars"])
    thr = bundle.get("opt_thr", 0.5)
    status = dbc.Alert(
        f"{algo.upper()} | {tab.upper()} | {n_vars} degisken | esik={thr:.4f}",
        color="info", style=_A,
    )
    return status, key


# ── CB-B3: OPT pickle yukle ─────────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-opt-status", "children"),
    Input("skorlama-batch-upload-opt", "contents"),
    State("skorlama-batch-upload-opt", "filename"),
    State("skorlama-batch-store-key", "data"),
    prevent_initial_call=True,
)
def upload_batch_opt(contents, filename, key):
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
        _BATCH_STORE[key + "_opt"] = opt_dict
    return dbc.Alert(f"OPT yuklendi — {len(opt_dict)} degisken", color="info", style=_A)


# ── CB-B4: Baglan & Onizle ──────────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-preview-status", "children"),
    Output("skorlama-batch-dd-exclude", "options"),
    Output("skorlama-batch-dd-exclude", "value"),
    Output("skorlama-batch-validation-msg", "children"),
    Output("skorlama-batch-exclude-wrap", "style"),
    Output("skorlama-batch-btn-start", "disabled"),
    Output("skorlama-batch-store-key", "data", allow_duplicate=True),
    Output("skorlama-batch-estimate", "children"),
    Input("skorlama-batch-btn-preview", "n_clicks"),
    State("skorlama-batch-store-key", "data"),
    State("skorlama-batch-radio-source", "value"),
    # SQL kaynak
    State("skorlama-batch-sql-src-server", "value"),
    State("skorlama-batch-sql-src-database", "value"),
    State("skorlama-batch-sql-src-driver", "value"),
    State("skorlama-batch-sql-src-table", "value"),
    State("skorlama-batch-chunk-size", "value"),
    # Dosya kaynak
    State("skorlama-batch-upload-files", "contents"),
    State("skorlama-batch-upload-files", "filename"),
    State("skorlama-batch-csv-separator", "value"),
    prevent_initial_call=True,
)
def batch_preview(n, key, source,
                  src_server, src_db, src_driver, src_table, chunk_size,
                  file_contents, file_names, csv_sep):
    key = key or str(uuid.uuid4())
    bundle = _BATCH_STORE.get(key + "_bundle") if key else None
    _no = (no_update, [], [], "", {"display": "none"}, True, key, "")

    if source == "sql":
        if not src_table or not src_table.strip():
            return (dbc.Alert("Kaynak tablo adi giriniz.", color="warning", style=_A),
                    *_no[1:])
        try:
            import pyodbc
            from data.loader import _build_conn_str, _quote_table
            conn_str = _build_conn_str(src_server, src_db, src_driver)
            with pyodbc.connect(conn_str, timeout=15) as conn:
                # Kolon listesi — SELECT TOP 1
                top1 = pd.read_sql(
                    f"SELECT TOP 1 * FROM {_quote_table(src_table.strip())}", conn)
                columns = list(top1.columns)
                # Satir sayisi
                count_df = pd.read_sql(
                    f"SELECT COUNT(*) AS cnt FROM {_quote_table(src_table.strip())}", conn)
                total_rows = int(count_df["cnt"].iloc[0])
        except Exception as e:
            log.error("[batch] SQL baglanti hatasi: %s", e)
            return (dbc.Alert(f"Baglanti hatasi: {e}", color="danger", style=_A),
                    *_no[1:])

        cs = int(chunk_size) if chunk_size else 100000
        n_chunks = math.ceil(total_rows / cs)
        _BATCH_STORE[key + "_columns"] = columns
        _BATCH_STORE[key + "_meta"] = {
            "source": "sql", "total_rows": total_rows, "total_chunks": n_chunks,
            "chunk_size": cs,
        }

        opts = [{"label": c, "value": c} for c in columns]
        validation = _build_validation(columns, bundle, [])
        status = dbc.Alert(
            f"Baglanti OK — {total_rows:,} satir, {cs:,} chunk → {n_chunks} parca",
            color="success", style=_A,
        )
        estimate = _build_estimate(total_rows, n_chunks, len(columns))
        return status, opts, [], validation, {}, False, key, estimate

    elif source == "files":
        if not file_contents or not file_names:
            return (dbc.Alert("Once dosya secin.", color="warning", style=_A),
                    *_no[1:])
        try:
            # Ilk dosyanin header'ini oku
            first_content = file_contents[0]
            _, content_string = first_content.split(",", 1)
            raw = base64.b64decode(content_string)
            fname = file_names[0]
            if fname.lower().endswith((".parquet", ".parq")):
                header_df = pd.read_parquet(io.BytesIO(raw)).head(1)
            else:
                header_df = pd.read_csv(io.BytesIO(raw), sep=csv_sep,
                                        nrows=1, low_memory=False)
            columns = list(header_df.columns)
        except Exception as e:
            log.error("[batch] Dosya okuma hatasi: %s", e)
            return (dbc.Alert(f"Dosya okuma hatasi: {e}", color="danger", style=_A),
                    *_no[1:])

        # Dosya iceriklerini store'a kaydet (thread'de kullanilacak)
        files_data = []
        for fc, fn in zip(file_contents, file_names):
            _, cs_b64 = fc.split(",", 1)
            files_data.append((fn, base64.b64decode(cs_b64)))
        _BATCH_STORE[key + "_files"] = files_data
        _BATCH_STORE[key + "_columns"] = columns
        _BATCH_STORE[key + "_meta"] = {
            "source": "files", "total_chunks": len(file_names),
        }

        opts = [{"label": c, "value": c} for c in columns]
        validation = _build_validation(columns, bundle, [])
        file_list = ", ".join(file_names[:5])
        if len(file_names) > 5:
            file_list += f" (+{len(file_names)-5} daha)"
        status = dbc.Alert(
            f"{len(file_names)} dosya yuklendi — {file_list}",
            color="success", style=_A,
        )
        # Dosya kaynak icin toplam boyut tahmini
        total_bytes = sum(len(raw) for _, raw in files_data)
        est_rows = int(total_bytes / max(len(columns) * 12, 1))  # kaba tahmin
        estimate = _build_estimate(est_rows, len(file_names), len(columns))
        return status, opts, [], validation, {}, False, key, estimate

    return _no


# ── CB-B5: Exclude cols degisince validation ─────────────────────────────────
@app.callback(
    Output("skorlama-batch-validation-msg", "children", allow_duplicate=True),
    Input("skorlama-batch-dd-exclude", "value"),
    State("skorlama-batch-store-key", "data"),
    prevent_initial_call=True,
)
def batch_validate_on_exclude(exclude_cols, key):
    if not key:
        return ""
    columns = _BATCH_STORE.get(key + "_columns", [])
    bundle = _BATCH_STORE.get(key + "_bundle")
    return _build_validation(columns, bundle, exclude_cols)


# ── CB-B6: Toplu Skorla (Baslat) ────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-result-status", "children"),
    Output("skorlama-batch-compute-state", "data"),
    Output("skorlama-batch-interval", "disabled"),
    Output("skorlama-batch-modal", "is_open"),
    Output("skorlama-batch-btn-start", "disabled", allow_duplicate=True),
    Input("skorlama-batch-btn-start", "n_clicks"),
    State("skorlama-batch-store-key", "data"),
    State("skorlama-batch-dd-exclude", "value"),
    State("skorlama-batch-radio-source", "value"),
    # Kaynak SQL
    State("skorlama-batch-sql-src-server", "value"),
    State("skorlama-batch-sql-src-database", "value"),
    State("skorlama-batch-sql-src-driver", "value"),
    State("skorlama-batch-sql-src-table", "value"),
    State("skorlama-batch-chunk-size", "value"),
    # Dosya separator
    State("skorlama-batch-csv-separator", "value"),
    # Hedef SQL
    State("skorlama-batch-tgt-server", "value"),
    State("skorlama-batch-tgt-database", "value"),
    State("skorlama-batch-tgt-driver", "value"),
    State("skorlama-batch-tgt-table", "value"),
    prevent_initial_call=True,
)
def start_batch(n, key, exclude_cols, source,
                src_server, src_db, src_driver, src_table, chunk_size, csv_sep,
                tgt_server, tgt_db, tgt_driver, tgt_table):
    _err = lambda msg: (dbc.Alert(msg, color="warning", style=_A),
                        no_update, True, False, False)

    if not key:
        return _err("Once 'Baglan & Onizle' yapin.")
    bundle = _BATCH_STORE.get(key + "_bundle")
    if not bundle:
        return _err("Model pickle yukleyin.")
    meta = _BATCH_STORE.get(key + "_meta")
    if not meta:
        return _err("Once 'Baglan & Onizle' yapin.")
    if not tgt_table or not tgt_table.strip():
        return _err("Hedef tablo adi giriniz.")
    if not tgt_server or not tgt_db:
        return _err("Hedef server ve database bilgisi gerekli.")

    # Eksik degisken kontrolu
    model_vars = bundle["model_vars"]
    columns = _BATCH_STORE.get(key + "_columns", [])
    available = set(columns) - set(exclude_cols or [])
    missing = [v for v in model_vars if v not in available]
    if missing:
        return _err(f"Eksik degiskenler: {', '.join(missing[:10])}")

    # Config kaydet
    config = {
        "source": source,
        "exclude_cols": list(exclude_cols or []),
        "total_chunks": meta["total_chunks"],
        "target_conn": _build_alchemy_conn(tgt_server, tgt_db, tgt_driver),
        "target_table": tgt_table.strip(),
    }
    if source == "sql":
        from data.loader import _quote_table
        config["src_conn"] = _build_alchemy_conn(src_server, src_db, src_driver)
        config["src_table"] = _quote_table(src_table.strip())
        config["chunk_size"] = int(chunk_size) if chunk_size else 100000
    elif source == "files":
        config["sep"] = csv_sep or ","

    _BATCH_STORE[key + "_config"] = config

    # Progress + thread
    prog_key = f"batch_{key}"
    _PRECOMPUTE_PROGRESS[prog_key] = {
        "step": 0, "total_steps": meta["total_chunks"],
        "msg": "Baslatiliyor...", "done": False, "error": None,
    }
    cancel_event = threading.Event()
    _BATCH_STORE[key + "_cancel"] = cancel_event

    t = threading.Thread(
        target=_run_batch,
        args=(key, prog_key, cancel_event),
        daemon=True,
    )
    t.start()

    compute_state = {"prog_key": prog_key, "key": key}
    return ("", compute_state, False, True, True)


# ── CB-B7: Progress polling ─────────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-result-status", "children", allow_duplicate=True),
    Output("skorlama-batch-interval", "disabled", allow_duplicate=True),
    Output("skorlama-batch-modal", "is_open", allow_duplicate=True),
    Output("skorlama-batch-modal-step", "children"),
    Output("skorlama-batch-modal-progress", "value"),
    Output("skorlama-batch-btn-start", "disabled", allow_duplicate=True),
    Input("skorlama-batch-interval", "n_intervals"),
    State("skorlama-batch-compute-state", "data"),
    prevent_initial_call=True,
)
def poll_batch_progress(n_intervals, compute_state):
    if not compute_state:
        return no_update, True, False, "", 0, False

    prog_key = compute_state.get("prog_key")
    if not prog_key:
        return no_update, True, False, "", 0, False

    progress = _PRECOMPUTE_PROGRESS.get(prog_key, {})
    step = progress.get("step", 0)
    total = progress.get("total_steps", 1)
    done = progress.get("done", False)
    error = progress.get("error")
    msg = progress.get("msg", "")
    pct = int(step / max(total, 1) * 100)

    if error:
        return (
            dbc.Alert([html.Strong("Hata: "), error], color="danger", style=_A),
            True, False, f"Hata: {error}", 0, False,
        )

    if done:
        return (
            dbc.Alert([html.Strong(msg)], color="success", style=_A),
            True, False, "Tamamlandi!", 100, False,
        )

    return (no_update, False, True, msg, pct, True)


# ── Dosya listesi gosterimi ─────────────────────────────────────────────────
@app.callback(
    Output("skorlama-batch-files-list", "children"),
    Input("skorlama-batch-upload-files", "filename"),
    prevent_initial_call=True,
)
def show_file_list(filenames):
    if not filenames:
        return ""
    items = [html.Li(f, style={"fontSize": "0.76rem"}) for f in filenames[:10]]
    if len(filenames) > 10:
        items.append(html.Li(f"+{len(filenames)-10} daha...",
                             style={"fontSize": "0.76rem", "color": "#6b7a99"}))
    return html.Ul(items, style={"margin": "0", "paddingLeft": "1.2rem"})


# ── Background thread ───────────────────────────────────────────────────────
def _run_batch(batch_key, prog_key, cancel_event):
    """Background thread — chunk chunk skorlama + SQL yazma."""
    prog = _PRECOMPUTE_PROGRESS[prog_key]

    bundle = _BATCH_STORE[batch_key + "_bundle"]
    model = bundle["model"]
    scaler = bundle.get("scaler")
    model_vars = bundle["model_vars"]
    tab = bundle.get("tab", "raw")
    opt_dict = _BATCH_STORE.get(batch_key + "_opt")
    config = _BATCH_STORE[batch_key + "_config"]
    total = config["total_chunks"]
    written = 0

    try:
        from sqlalchemy import create_engine
        target_engine = create_engine(
            config["target_conn"],
            connect_args={"fast_executemany": True},
        )

        source = config["source"]

        if source == "sql":
            src_engine = create_engine(
                config["src_conn"],
                connect_args={"fast_executemany": True},
            )
            query = f"SELECT * FROM {config['src_table']}"
            chunks = pd.read_sql(query, src_engine, chunksize=config["chunk_size"])

            for i, chunk_df in enumerate(chunks):
                if cancel_event.is_set():
                    prog["msg"] = f"Iptal edildi ({written} chunk yazildi)"
                    prog["done"] = True
                    return
                prog["step"] = i + 1
                prog["msg"] = f"Chunk {i+1}/{total} skorlaniyor..."

                scored = _score_chunk(chunk_df, model, scaler, model_vars,
                                      tab, opt_dict)
                _write_chunk(scored, target_engine, config["target_table"],
                             "replace" if i == 0 else "append")
                written += 1

        elif source == "files":
            files = _BATCH_STORE.get(batch_key + "_files", [])
            sep = config.get("sep", ",")

            for i, (fname, raw_bytes) in enumerate(files):
                if cancel_event.is_set():
                    prog["msg"] = f"Iptal edildi ({written} dosya yazildi)"
                    prog["done"] = True
                    return
                prog["step"] = i + 1
                prog["msg"] = f"Dosya {i+1}/{total}: {fname}"

                if fname.lower().endswith((".parquet", ".parq")):
                    df = pd.read_parquet(io.BytesIO(raw_bytes))
                else:
                    df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep,
                                     low_memory=False)

                scored = _score_chunk(df, model, scaler, model_vars,
                                      tab, opt_dict)
                _write_chunk(scored, target_engine, config["target_table"],
                             "replace" if i == 0 else "append")
                written += 1

        prog["step"] = total
        prog["msg"] = (f"Tamamlandi — {written} parca skorlandi, "
                       f"hedef: {config['target_table']}")
        prog["done"] = True

    except Exception as e:
        log.error("[batch] Skorlama hatasi: %s", e, exc_info=True)
        prog["error"] = f"{e} ({written} parca yazildi)"
        prog["done"] = True

    finally:
        # Bellek temizligi — skorlama bittikten sonra hicbir sey bellekte kalmaz
        for suffix in ("_bundle", "_opt", "_config", "_files", "_columns", "_meta", "_cancel"):
            _BATCH_STORE.pop(batch_key + suffix, None)
        _PRECOMPUTE_PROGRESS.pop(prog_key, None)
        log.info("[batch] Session %s temizlendi — bellek serbest.", batch_key[:8])
