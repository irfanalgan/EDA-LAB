import uuid
import base64
import io

import dash
from dash import html, dcc, Input, Output, State, clientside_callback, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd

from app_instance import app
from server_state import _SERVER_STORE, get_df as _get_df
from data.loader import get_data_from_sql, get_data_from_sql_multi, get_config_defaults
from utils.helpers import coerce_numeric_columns

_N_SLIDES = 8  # layout'taki slayt sayısı


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
    Output("div-sql-jk-1",          "style"),
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
    jk1   = show if count >= 2 else hide
    return count, row2, row3, jk1


# ── Callback: CSV — dosya satırı ekle/kaldır ─────────────────────────────────
@app.callback(
    Output("store-csv-file-count", "data"),
    Output("csv-file-row-2",       "style"),
    Output("csv-file-row-3",       "style"),
    Output("div-csv-jk-1",         "style"),
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
    jk1  = show if count >= 2 else hide
    return count, row2, row3, jk1


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


def _join_dataframes(dfs: list[pd.DataFrame], join_keys_per_table: list[list[str]],
                     join_hows: list | None = None) -> pd.DataFrame:
    """Master + ek DataFrame'leri her tablonun kendi key'i üzerinde birleştirir.

    join_keys_per_table[0] → master (left_on)
    join_keys_per_table[i] → i. dosya (right_on)
    join_hows[i] → i. dosyanın join tipi ("left" / "inner"), index 0 kullanılmaz
    """
    result = dfs[0]
    left_keys = join_keys_per_table[0] if join_keys_per_table else []

    if not left_keys:
        return result

    # Master dosyada join key kontrolü
    missing = [k for k in left_keys if k not in result.columns]
    if missing:
        raise KeyError(
            f"Dosya 1 içinde join key bulunamadı: {missing}. "
            f"Mevcut kolonlar: {list(result.columns[:20])}"
        )

    for i, df in enumerate(dfs[1:], start=1):
        how = (join_hows[i] if join_hows and i < len(join_hows) else None) or "left"
        right_keys = join_keys_per_table[i] if i < len(join_keys_per_table) and join_keys_per_table[i] else left_keys

        # Ek dosyada join key kontrolü
        missing_r = [k for k in right_keys if k not in df.columns]
        if missing_r:
            raise KeyError(
                f"Dosya {i+1} içinde join key bulunamadı: {missing_r}. "
                f"Mevcut kolonlar: {list(df.columns[:20])}"
            )
        all_keys = set(left_keys) | set(right_keys)
        drop_cols = [c for c in df.columns if c in result.columns and c not in all_keys]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if left_keys == right_keys:
            result = pd.merge(result, df, on=left_keys, how=how)
        else:
            result = pd.merge(result, df, left_on=left_keys, right_on=right_keys, how=how)
            extra = [rk for lk, rk in zip(left_keys, right_keys) if lk != rk and rk in result.columns]
            if extra:
                result = result.drop(columns=extra)
    return result


# ── Callback: CSV Yükle ───────────────────────────────────────────────────────
@app.callback(
    Output("store-key",   "data",     allow_duplicate=True),
    Output("load-status", "children", allow_duplicate=True),
    Output("store-model-signal", "data", allow_duplicate=True),
    Output("store-profile-loaded", "data", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Output("pg-model-output", "children", allow_duplicate=True),
    Output("store-pending-note", "data", allow_duplicate=True),
    Input("btn-load-csv", "n_clicks"),
    State("upload-csv",   "contents"), State("upload-csv",   "filename"),
    State("upload-csv-2", "contents"), State("upload-csv-2", "filename"),
    State("upload-csv-3", "contents"), State("upload-csv-3", "filename"),
    State("csv-separator",        "value"),
    State("input-csv-jk-1",      "value"),
    State("input-csv-jk-2",      "value"),
    State("input-csv-jk-3",      "value"),
    State("radio-csv-join-2",    "value"),
    State("radio-csv-join-3",    "value"),
    State("store-csv-file-count", "data"),
    prevent_initial_call=True,
)
def load_csv(n_clicks,
             c1, fn1, c2, fn2, c3, fn3,
             sep, jk1_raw, jk2_raw, jk3_raw, jt2, jt3, file_count):
    _RESET = (None, None, None, "", None)
    if not c1 or not fn1:
        return dash.no_update, _warn("Önce bir CSV dosyası seçin."), *_RESET

    sep = sep or ","
    join_hows = [None, jt2 or "left", jt3 or "left"]
    jk_per_table = []
    for raw in [jk1_raw, jk2_raw, jk3_raw]:
        keys = [k.strip() for k in (raw or "").split(",") if k.strip()]
        jk_per_table.append(keys)

    master_keys = jk_per_table[0]

    try:
        dfs = []
        filenames = []
        for contents, filename in [(c1, fn1), (c2, fn2), (c3, fn3)]:
            if contents and filename:
                df, fn = _read_csv_content(contents, filename, sep)
                dfs.append(df)
                filenames.append(fn)

        if len(dfs) > 1 and master_keys:
            result = _join_dataframes(dfs, jk_per_table[:len(dfs)],
                                      join_hows=join_hows[:len(dfs)])
        else:
            result = dfs[0]

        result, converted = coerce_numeric_columns(result)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = result
        _SERVER_STORE[f"{key}_quality"] = {"converted": converted}

        files_str = " + ".join(filenames)
        conv_note = f"  ·  {len(converted)} kolon numerik dönüştürüldü" if converted else ""
        join_note = f"  ·  {len(dfs)} dosya birleştirildi" if len(dfs) > 1 else ""
        return key, _ok(f"{len(result):,} satır  ·  {result.shape[1]} kolon  ·  {files_str}{join_note}{conv_note}"), *_RESET

    except Exception as e:
        return dash.no_update, _err(f"Okuma hatası: {e}"), *_RESET


# ── Callback: Veriyi Yükle (SQL) ──────────────────────────────────────────────
@app.callback(
    Output("store-key", "data"),
    Output("load-status", "children"),
    Output("store-model-signal", "data", allow_duplicate=True),
    Output("store-profile-loaded", "data", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Output("pg-model-output", "children", allow_duplicate=True),
    Output("store-pending-note", "data", allow_duplicate=True),
    Input("btn-load", "n_clicks"),
    State("input-table-1",        "value"),
    State("input-table-2",        "value"),
    State("input-table-3",        "value"),
    State("input-sql-jk-1",      "value"),
    State("input-sql-jk-2",      "value"),
    State("input-sql-jk-3",      "value"),
    State("radio-sql-join-2",    "value"),
    State("radio-sql-join-3",    "value"),
    State("store-sql-table-count","data"),
    State("input-sql-server",     "value"),
    State("input-sql-database",   "value"),
    State("dd-sql-driver",        "value"),
    State("chk-sql-top1000",      "value"),
    prevent_initial_call=True,
)
def load_data(n_clicks,
              t1, t2, t3,
              jk1_raw, jk2_raw, jk3_raw, jt2, jt3,
              table_count,
              server, database, driver, top1000_val):
    _RESET = (None, None, None, "", None)  # model-signal, profile, model-index, pg-output, note
    if not t1 or not t1.strip():
        return dash.no_update, _warn("Lütfen bir tablo adı girin."), *_RESET

    top_n = 1000 if "top1000" in (top1000_val or []) else None
    join_hows = [None, jt2 or "left", jt3 or "left"]

    tables = [t for t in [t1, t2, t3] if t and t.strip()]
    jk_raws = [jk1_raw, jk2_raw, jk3_raw]
    # Her tablonun key listesini ayrı ayrı parse et
    jk_per_table = []
    for raw in jk_raws:
        keys = [k.strip() for k in (raw or "").split(",") if k.strip()]
        jk_per_table.append(keys)

    master_keys = jk_per_table[0]  # Tablo 1'in key'leri = left_on

    try:
        if len(tables) > 1 and master_keys:
            df = get_data_from_sql_multi(
                tables=tables,
                join_keys_per_table=jk_per_table[:len(tables)],
                server=server, database=database, driver=driver,
                top_n=top_n,
                join_hows=join_hows[:len(tables)],
            )
            join_note = f"  ·  {len(tables)} tablo birleştirildi"
        else:
            df = get_data_from_sql(
                tables[0],
                server=server, database=database, driver=driver,
                top_n=top_n,
            )
            join_note = ""

        top_note = "  ·  TOP 1000" if top_n else ""
        df, converted = coerce_numeric_columns(df)
        key = str(uuid.uuid4())
        _SERVER_STORE[key] = df
        _SERVER_STORE[f"{key}_quality"] = {"converted": converted}
        conv_note = f"  ·  {len(converted)} kolon numerik dönüştürüldü" if converted else ""
        return key, _ok(f"{len(df):,} satır  ·  {df.shape[1]} kolon{join_note}{conv_note}{top_note}"), *_RESET

    except Exception as e:
        return dash.no_update, _err(str(e)), *_RESET


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

    all_opts = [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in df.columns]

    date_keywords = {"date", "tarih", "dt", "time", "zaman"}
    date_cols = sorted(
        df.columns,
        key=lambda c: (
            0 if pd.api.types.is_datetime64_any_dtype(df[c])
            else 1 if any(k in c.lower() for k in date_keywords)
            else 2
        ),
    )
    date_opts = [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in date_cols]

    seg_cols = [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= 50
    ]
    seg_opts = [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in seg_cols]

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


# ══════════════════════════════════════════════════════════════════════════════
#   LOADING SLIDESHOW — modal aç/kapat + slayt navigasyonu
# ══════════════════════════════════════════════════════════════════════════════

# ── Yükleme başlayınca modalı aç ─────────────────────────────────────────────
@app.callback(
    Output("modal-slideshow",    "is_open", allow_duplicate=True),
    Output("interval-slideshow", "disabled", allow_duplicate=True),
    Output("store-slide-index",  "data", allow_duplicate=True),
    Output("interval-slideshow", "n_intervals", allow_duplicate=True),
    Input("btn-load",     "n_clicks"),
    Input("btn-load-csv", "n_clicks"),
    prevent_initial_call=True,
)
def open_slideshow_on_load(*_):
    return True, False, 0, 0  # open modal, enable interval, reset slide


# ── Veri yüklenince veya hata olunca modalı kapat ───────────────────────────
@app.callback(
    Output("modal-slideshow",    "is_open"),
    Output("interval-slideshow", "disabled"),
    Input("store-key", "data"),
    Input("load-status", "children"),
    Input("btn-slideshow-close", "n_clicks"),
    prevent_initial_call=True,
)
def close_slideshow_on_data(key, status, _close):
    trigger = dash.callback_context.triggered_id
    if trigger == "btn-slideshow-close":
        return False, True
    if trigger == "store-key" and key:
        return False, True
    if trigger == "load-status":
        return False, True
    return dash.no_update, dash.no_update


# ── Interval ile otomatik slayt ilerleme ─────────────────────────────────────
app.clientside_callback(
    f"""
    function(n_intervals, current) {{
        if (n_intervals === undefined || n_intervals === 0) return current || 0;
        return ((current || 0) + 1) % {_N_SLIDES};
    }}
    """,
    Output("store-slide-index", "data"),
    Input("interval-slideshow", "n_intervals"),
    State("store-slide-index", "data"),
    prevent_initial_call=True,
)


# ── Dot tıklama → slayt değiştir + interval sıfırla ─────────────────────────
_dot_inputs = ", ".join(
    f'{{"component_id": "slide-dot-{i}", "property": "n_clicks"}}'
    for i in range(_N_SLIDES)
)
app.clientside_callback(
    f"""
    function({", ".join(f"d{i}" for i in range(_N_SLIDES))}) {{
        var ctx = dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {{
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }}
        var prop_id = ctx.triggered[0].prop_id;
        for (var i = 0; i < {_N_SLIDES}; i++) {{
            if (prop_id === "slide-dot-" + i + ".n_clicks") {{
                return [i, 0];
            }}
        }}
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    }}
    """,
    Output("store-slide-index", "data", allow_duplicate=True),
    Output("interval-slideshow", "n_intervals", allow_duplicate=True),
    [Input(f"slide-dot-{i}", "n_clicks") for i in range(_N_SLIDES)],
    prevent_initial_call=True,
)


# ── Slayt index değişince görünürlüğü, dots ve progress güncelle ─────────────
_slide_outputs = [Output(f"slide-{i}", "style") for i in range(_N_SLIDES)]
_dot_outputs = [Output(f"slide-dot-{i}", "className") for i in range(_N_SLIDES)]

app.clientside_callback(
    f"""
    function(idx) {{
        var slides = [];
        var dots = [];
        for (var i = 0; i < {_N_SLIDES}; i++) {{
            slides.push(i === idx ? {{"display": "block"}} : {{"display": "none"}});
            dots.push(i === idx ? "slide-dot dot-active" : "slide-dot");
        }}
        var pct = ((idx + 1) / {_N_SLIDES}) * 100;
        var progressStyle = {{"width": pct + "%"}};
        return slides.concat(dots).concat([progressStyle]);
    }}
    """,
    _slide_outputs + _dot_outputs + [Output("slide-progress-fill", "style")],
    Input("store-slide-index", "data"),
    prevent_initial_call=True,
)


# ── Elapsed time sayacı (her saniye güncelle) ────────────────────────────────
app.clientside_callback(
    """
    function(is_open) {
        if (!is_open) {
            if (window._slideshowTimer) {
                clearInterval(window._slideshowTimer);
                window._slideshowTimer = null;
            }
            return "";
        }
        var start = Date.now();
        var el = document.getElementById("slideshow-elapsed");
        if (window._slideshowTimer) clearInterval(window._slideshowTimer);
        window._slideshowTimer = setInterval(function() {
            var diff = Math.floor((Date.now() - start) / 1000);
            var m = Math.floor(diff / 60);
            var s = diff % 60;
            if (el) el.textContent = m + ":" + (s < 10 ? "0" : "") + s;
        }, 1000);
        return "0:00";
    }
    """,
    Output("slideshow-elapsed", "children"),
    Input("modal-slideshow", "is_open"),
    prevent_initial_call=True,
)
