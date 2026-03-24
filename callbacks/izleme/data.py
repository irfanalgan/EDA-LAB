"""İzleme — veri yükleme, config, metrics, önizleme callback'leri.

Geliştirme tarafındaki data_loading.py ve preview.py mantığının
bağımsız İzleme versiyonu. Tüm ID'ler mon- prefix'lidir.
Veri _MON_STORE'da saklanır — Geliştirme'nin _SERVER_STORE'una dokunmaz.

Veri mimarisi:
  _MON_STORE[key + "_ref"]  → Referans (geliştirme) DataFrame
  _MON_STORE[key + "_mon"]  → İzleme (canlı) DataFrame
  _MON_STORE[key]           → Aktif toggle'a göre gösterilen veri (config/preview için)
"""

import uuid
import base64
import io

import dash
from dash import html, dcc, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd

import pickle as _pickle

from app_instance import app
from server_state import _MON_STORE, _PRECOMPUTE_PROGRESS, get_mon_df, clear_mon_store
from data.loader import get_data_from_sql, get_data_from_sql_multi, get_config_defaults
from utils.helpers import coerce_numeric_columns
from callbacks.izleme.compute import start_mon_compute

# ── Yardımcı: CSV okuma (data_loading.py'den) ───────────────────────────────
def _read_csv_content(contents, filename, sep):
    if not contents or not filename:
        return None, None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")),
                     sep=sep, low_memory=False)
    return df, filename


def _join_dataframes(dfs, join_keys_per_table, join_hows=None):
    result = dfs[0]
    left_keys = join_keys_per_table[0] if join_keys_per_table else []
    if not left_keys:
        return result
    missing = [k for k in left_keys if k not in result.columns]
    if missing:
        raise KeyError(f"Dosya 1 içinde join key bulunamadı: {missing}")
    for i, df in enumerate(dfs[1:], start=1):
        how = (join_hows[i] if join_hows and i < len(join_hows) else None) or "left"
        right_keys = (join_keys_per_table[i]
                      if i < len(join_keys_per_table) and join_keys_per_table[i]
                      else left_keys)
        missing_r = [k for k in right_keys if k not in df.columns]
        if missing_r:
            raise KeyError(f"Dosya {i+1} içinde join key bulunamadı: {missing_r}")
        all_keys = set(left_keys) | set(right_keys)
        drop_cols = [c for c in df.columns if c in result.columns and c not in all_keys]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if left_keys == right_keys:
            result = pd.merge(result, df, on=left_keys, how=how)
        else:
            result = pd.merge(result, df, left_on=left_keys, right_on=right_keys, how=how)
            extra = [rk for lk, rk in zip(left_keys, right_keys)
                     if lk != rk and rk in result.columns]
            if extra:
                result = result.drop(columns=extra)
    return result


# ── Yardımcı: alert fonksiyonları ────────────────────────────────────────────
_ALERT_STYLE = {"padding": "0.4rem 0.75rem", "fontSize": "0.8rem"}

def _ok(msg):
    return dbc.Alert(msg, color="success", style=_ALERT_STYLE)

def _warn(msg):
    return dbc.Alert(msg, color="warning", style=_ALERT_STYLE)

def _err(msg):
    return dbc.Alert(msg, color="danger", style=_ALERT_STYLE)


# ── Yardımcı: kolon uyumu kontrolü ─────────────────────────────────────────
def _check_column_match(key, loaded):
    """Referans ve İzleme verileri yüklendiyse kolon uyumunu kontrol et.
    Returns (toast_open, toast_message)."""
    if not loaded or loaded.get("ref_rows") is None or loaded.get("mon_rows") is None:
        return False, ""
    ref_df = _MON_STORE.get(key + "_ref")
    mon_df = _MON_STORE.get(key + "_mon")
    if ref_df is None or mon_df is None:
        return False, ""
    ref_cols = set(ref_df.columns)
    mon_cols = set(mon_df.columns)
    if ref_cols != mon_cols:
        only_ref = ref_cols - mon_cols
        only_mon = mon_cols - ref_cols
        parts = ["Referans ve İzleme tabloları aynı kolon yapısına sahip değil."]
        if only_ref:
            parts.append(f"Sadece Referans'ta: {', '.join(sorted(only_ref))}")
        if only_mon:
            parts.append(f"Sadece İzleme'de: {', '.join(sorted(only_mon))}")
        return True, " — ".join(parts)
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#   Format Modal — ilk açılışta göster
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-modal-format", "is_open"),
    Input("container-izleme", "style"),
    Input("mon-btn-format-modal-close", "n_clicks"),
    State("mon-modal-format", "is_open"),
    State("store-mon-config", "data"),
    prevent_initial_call=True,
)
def mon_toggle_format_modal(container_style, close_clicks, is_open, config):
    ctx = dash.callback_context.triggered_id
    if ctx == "mon-btn-format-modal-close":
        return False
    # İzleme tabı görünür olduğunda ve henüz config yoksa aç
    if ctx == "container-izleme":
        visible = container_style and container_style.get("display") != "none"
        if visible and (not config or not config.get("target_col")):
            return True
    return is_open


# ══════════════════════════════════════════════════════════════════════════════
#   Referans / İzleme Toggle
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("store-mon-toggle", "data"),
    Output("mon-btn-toggle-ref", "outline"),
    Output("mon-btn-toggle-mon", "outline"),
    Input("mon-btn-toggle-ref", "n_clicks"),
    Input("mon-btn-toggle-mon", "n_clicks"),
    State("store-mon-toggle", "data"),
    prevent_initial_call=True,
)
def mon_switch_toggle(ref_clicks, mon_clicks, current):
    ctx = dash.callback_context.triggered_id
    if ctx == "mon-btn-toggle-ref":
        return "ref", False, True   # ref aktif (outline=False), mon pasif
    elif ctx == "mon-btn-toggle-mon":
        return "mon", True, False   # mon aktif, ref pasif
    return current, current != "ref", current != "mon"


# ══════════════════════════════════════════════════════════════════════════════
#   Durum göstergeleri (Referans: ✓ … / İzleme: ✓ …)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-ref-status", "children"),
    Output("mon-mon-status", "children"),
    Input("store-mon-loaded", "data"),
)
def mon_update_data_status(loaded):
    if not loaded:
        loaded = {}
    ref_rows = loaded.get("ref_rows")
    mon_rows = loaded.get("mon_rows")
    ref_text = (html.Span(f"Referans: ✓ yüklendi ({ref_rows:,} satır)",
                          style={"color": "#10b981"})
                if ref_rows is not None
                else html.Span("Referans: — yüklenmedi", style={"color": "#7e8fa4"}))
    mon_text = (html.Span(f"İzleme: ✓ yüklendi ({mon_rows:,} satır)",
                          style={"color": "#10b981"})
                if mon_rows is not None
                else html.Span("İzleme: — yüklenmedi", style={"color": "#7e8fa4"}))
    return ref_text, mon_text


# ══════════════════════════════════════════════════════════════════════════════
#   SQL defaults
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-input-sql-server",   "value"),
    Output("mon-input-sql-database", "value"),
    Output("mon-dd-sql-driver",      "value"),
    Input("mon-radio-source", "value"),
)
def mon_fill_sql_defaults(_):
    cfg = get_config_defaults()
    return cfg["server"], cfg["database"], cfg["driver"]


# ══════════════════════════════════════════════════════════════════════════════
#   Kaynak toggle (SQL / CSV)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-source-sql-div", "style"),
    Output("mon-source-csv-div", "style"),
    Input("mon-radio-source", "value"),
)
def mon_toggle_source(source):
    if source == "csv":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


# ══════════════════════════════════════════════════════════════════════════════
#   SQL tablo satırı ekle/kaldır
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-store-sql-table-count", "data"),
    Output("mon-sql-table-row-2",       "style"),
    Output("mon-sql-table-row-3",       "style"),
    Output("mon-div-sql-jk-1",          "style"),
    Input("mon-btn-add-sql-table",  "n_clicks"),
    Input("mon-btn-remove-sql-2",   "n_clicks"),
    Input("mon-btn-remove-sql-3",   "n_clicks"),
    State("mon-store-sql-table-count", "data"),
    prevent_initial_call=True,
)
def mon_manage_sql_tables(add, rem2, rem3, count):
    ctx = dash.callback_context.triggered_id
    if ctx == "mon-btn-add-sql-table":
        count = min(count + 1, 3)
    elif ctx == "mon-btn-remove-sql-2":
        count = max(count - 1, 1)
    elif ctx == "mon-btn-remove-sql-3":
        count = max(count - 1, 2)
    show = {"display": "block"}
    hide = {"display": "none"}
    return count, show if count >= 2 else hide, show if count >= 3 else hide, show if count >= 2 else hide


# ══════════════════════════════════════════════════════════════════════════════
#   CSV dosya satırı ekle/kaldır
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-store-csv-file-count", "data"),
    Output("mon-csv-file-row-2",       "style"),
    Output("mon-csv-file-row-3",       "style"),
    Output("mon-div-csv-jk-1",         "style"),
    Input("mon-btn-add-csv-file",  "n_clicks"),
    Input("mon-btn-remove-csv-2",  "n_clicks"),
    Input("mon-btn-remove-csv-3",  "n_clicks"),
    State("mon-store-csv-file-count", "data"),
    prevent_initial_call=True,
)
def mon_manage_csv_files(add, rem2, rem3, count):
    ctx = dash.callback_context.triggered_id
    if ctx == "mon-btn-add-csv-file":
        count = min(count + 1, 3)
    elif ctx == "mon-btn-remove-csv-2":
        count = max(count - 1, 1)
    elif ctx == "mon-btn-remove-csv-3":
        count = max(count - 1, 2)
    show = {"display": "block"}
    hide = {"display": "none"}
    return count, show if count >= 2 else hide, show if count >= 3 else hide, show if count >= 2 else hide


# ══════════════════════════════════════════════════════════════════════════════
#   CSV dosya adı göster
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-csv-filename-display",   "children"),
    Output("mon-csv-filename-display-2", "children"),
    Output("mon-csv-filename-display-3", "children"),
    Input("mon-upload-csv",   "filename"),
    Input("mon-upload-csv-2", "filename"),
    Input("mon-upload-csv-3", "filename"),
    prevent_initial_call=True,
)
def mon_show_csv_filenames(fn1, fn2, fn3):
    return fn1 or "", fn2 or "", fn3 or ""


# ══════════════════════════════════════════════════════════════════════════════
#   CSV Yükle
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("store-mon-key",    "data", allow_duplicate=True),
    Output("store-mon-loaded", "data", allow_duplicate=True),
    Output("mon-load-status",  "children", allow_duplicate=True),
    Output("mon-toast-column-mismatch", "is_open", allow_duplicate=True),
    Output("mon-toast-column-mismatch", "children", allow_duplicate=True),
    Input("mon-btn-load-csv", "n_clicks"),
    State("mon-upload-csv",   "contents"), State("mon-upload-csv",   "filename"),
    State("mon-upload-csv-2", "contents"), State("mon-upload-csv-2", "filename"),
    State("mon-upload-csv-3", "contents"), State("mon-upload-csv-3", "filename"),
    State("mon-csv-separator",        "value"),
    State("mon-input-csv-jk-1",      "value"),
    State("mon-input-csv-jk-2",      "value"),
    State("mon-input-csv-jk-3",      "value"),
    State("mon-radio-csv-join-2",    "value"),
    State("mon-radio-csv-join-3",    "value"),
    State("mon-store-csv-file-count", "data"),
    State("store-mon-toggle",  "data"),
    State("store-mon-key",     "data"),
    State("store-mon-loaded",  "data"),
    prevent_initial_call=True,
)
def mon_load_csv(n_clicks,
                 c1, fn1, c2, fn2, c3, fn3,
                 sep, jk1_raw, jk2_raw, jk3_raw, jt2, jt3, file_count,
                 toggle, existing_key, loaded):
    if not c1 or not fn1:
        return no_update, no_update, _warn("Önce bir CSV dosyası seçin."), False, ""

    sep = sep or ","
    join_hows = [None, jt2 or "left", jt3 or "left"]
    jk_per_table = []
    for raw in [jk1_raw, jk2_raw, jk3_raw]:
        keys = [k.strip() for k in (raw or "").split(",") if k.strip()]
        jk_per_table.append(keys)

    try:
        dfs, filenames = [], []
        for contents, filename in [(c1, fn1), (c2, fn2), (c3, fn3)]:
            if contents and filename:
                df, fn = _read_csv_content(contents, filename, sep)
                dfs.append(df)
                filenames.append(fn)

        if len(dfs) > 1 and jk_per_table[0]:
            result = _join_dataframes(dfs, jk_per_table[:len(dfs)],
                                      join_hows=join_hows[:len(dfs)])
        else:
            result = dfs[0]

        result, converted = coerce_numeric_columns(result)

        # Key yönetimi: ilk yüklemede yeni key, sonrakilerde mevcut key kullan
        key = existing_key or str(uuid.uuid4())
        loaded = dict(loaded or {})

        # Toggle'a göre store key belirle
        suffix = "_ref" if toggle == "ref" else "_mon"
        _MON_STORE[key + suffix] = result

        # Loaded state güncelle
        if toggle == "ref":
            loaded["ref_rows"] = len(result)
        else:
            loaded["mon_rows"] = len(result)

        # Kolon uyumu kontrolü
        toast_open, toast_msg = _check_column_match(key, loaded)

        files_str = " + ".join(filenames)
        conv_note = f"  ·  {len(converted)} kolon numerik dönüştürüldü" if converted else ""
        join_note = f"  ·  {len(dfs)} dosya birleştirildi" if len(dfs) > 1 else ""
        label = "Referans" if toggle == "ref" else "İzleme"
        return (key, loaded,
                _ok(f"[{label}] {len(result):,} satır  ·  {result.shape[1]} kolon  ·  {files_str}{join_note}{conv_note}"),
                toast_open, toast_msg)

    except Exception as e:
        return no_update, no_update, _err(f"Okuma hatası: {e}"), False, ""


# ══════════════════════════════════════════════════════════════════════════════
#   SQL Yükle
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("store-mon-key",   "data"),
    Output("store-mon-loaded", "data"),
    Output("mon-load-status", "children"),
    Output("mon-toast-column-mismatch", "is_open"),
    Output("mon-toast-column-mismatch", "children"),
    Input("mon-btn-load", "n_clicks"),
    State("mon-input-table-1",         "value"),
    State("mon-input-table-2",         "value"),
    State("mon-input-table-3",         "value"),
    State("mon-input-sql-jk-1",       "value"),
    State("mon-input-sql-jk-2",       "value"),
    State("mon-input-sql-jk-3",       "value"),
    State("mon-radio-sql-join-2",     "value"),
    State("mon-radio-sql-join-3",     "value"),
    State("mon-store-sql-table-count", "data"),
    State("mon-input-sql-server",      "value"),
    State("mon-input-sql-database",    "value"),
    State("mon-dd-sql-driver",         "value"),
    State("mon-chk-sql-top1000",       "value"),
    State("store-mon-toggle",  "data"),
    State("store-mon-key",     "data"),
    State("store-mon-loaded",  "data"),
    prevent_initial_call=True,
)
def mon_load_data(n_clicks, t1, t2, t3,
                  jk1_raw, jk2_raw, jk3_raw, jt2, jt3,
                  table_count, server, database, driver, top1000_val,
                  toggle, existing_key, loaded):
    if not t1 or not t1.strip():
        return no_update, no_update, _warn("Lütfen bir tablo adı girin."), False, ""

    top_n = 1000 if "top1000" in (top1000_val or []) else None
    join_hows = [None, jt2 or "left", jt3 or "left"]
    tables = [t for t in [t1, t2, t3] if t and t.strip()]
    jk_per_table = []
    for raw in [jk1_raw, jk2_raw, jk3_raw]:
        keys = [k.strip() for k in (raw or "").split(",") if k.strip()]
        jk_per_table.append(keys)

    try:
        if len(tables) > 1 and jk_per_table[0]:
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
                tables[0], server=server, database=database,
                driver=driver, top_n=top_n,
            )
            join_note = ""

        top_note = "  ·  TOP 1000" if top_n else ""
        df, converted = coerce_numeric_columns(df)

        # Key yönetimi
        key = existing_key or str(uuid.uuid4())
        loaded = dict(loaded or {})

        # Toggle'a göre store key belirle
        suffix = "_ref" if toggle == "ref" else "_mon"
        _MON_STORE[key + suffix] = df

        # Loaded state güncelle
        if toggle == "ref":
            loaded["ref_rows"] = len(df)
        else:
            loaded["mon_rows"] = len(df)

        # Kolon uyumu kontrolü
        toast_open, toast_msg = _check_column_match(key, loaded)

        conv_note = f"  ·  {len(converted)} kolon numerik dönüştürüldü" if converted else ""
        label = "Referans" if toggle == "ref" else "İzleme"
        return (key, loaded,
                _ok(f"[{label}] {len(df):,} satır  ·  {df.shape[1]} kolon{join_note}{conv_note}{top_note}"),
                toast_open, toast_msg)

    except Exception as e:
        return no_update, no_update, _err(str(e)), False, ""


# ══════════════════════════════════════════════════════════════════════════════
#   Kolon Yapılandırması — collapse aç + dropdown doldur
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-collapse-config", "is_open"),
    Output("mon-dd-target-col",   "options"),
    Output("mon-dd-date-col",     "options"),
    Output("mon-dd-pd-col",       "options"),
    Output("mon-dd-id-col",       "options"),
    Input("store-mon-loaded", "data"),
    State("store-mon-key", "data"),
)
def mon_open_config_section(loaded, key):
    # Her iki veri de yüklenmeden config açılmaz
    if not loaded or not key:
        return False, [], [], [], []
    if loaded.get("ref_rows") is None or loaded.get("mon_rows") is None:
        return False, [], [], [], []

    # Kolon uyumu yoksa config açılmaz
    ref_df = _MON_STORE.get(key + "_ref")
    mon_df = _MON_STORE.get(key + "_mon")
    if ref_df is None or mon_df is None:
        return False, [], [], [], []
    if set(ref_df.columns) != set(mon_df.columns):
        return False, [], [], [], []

    # Referans verisinden dropdown seçenekleri oluştur
    df = ref_df
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

    # PD kolonu: sadece sayısal kolonlar
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    pd_opts = [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in num_cols]

    return True, all_opts, date_opts, pd_opts, all_opts


# ══════════════════════════════════════════════════════════════════════════════
#   Yapılandırmayı Onayla
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("store-mon-config",   "data"),
    Output("mon-config-status",  "children"),
    Output("mon-collapse-profile-actions", "is_open"),
    Output("store-mon-compute-state", "data"),
    Output("interval-mon-compute", "disabled"),
    Output("mon-modal-compute", "is_open", allow_duplicate=True),
    Input("mon-btn-confirm", "n_clicks"),
    State("mon-dd-target-col",        "value"),
    State("mon-dd-date-col",          "value"),
    State("mon-dd-pd-col",            "value"),
    State("mon-dd-id-col",            "value"),
    State("mon-input-maturity",       "value"),
    State("mon-radio-period-freq",    "value"),
    State("mon-chk-woe",             "value"),
    State("mon-chk-woe-pre",         "value"),
    State("store-mon-key",            "data"),
    prevent_initial_call=True,
)
def mon_confirm_config(n_clicks, target_col, date_col, pd_col,
                       id_col, maturity_months, period_freq, woe_chk,
                       woe_pre_chk, key):
    if not target_col:
        return no_update, _warn("Target kolonu zorunludur."), False, no_update, no_update, False
    if not date_col:
        return no_update, _warn("Tarih kolonu zorunludur."), False, no_update, no_update, False
    if not pd_col:
        return no_update, _warn("PD kolonu zorunludur."), False, no_update, no_update, False

    woe_enabled = bool(woe_chk and "woe" in woe_chk)
    woe_pretransformed = bool(woe_pre_chk and "pre" in woe_pre_chk)
    maturity_months = int(maturity_months or 12)
    period_freq = period_freq or "M"

    # Model değişkenleri otomatik: target, tarih, PD, ID dışındaki tüm kolonlar
    ref_df = _MON_STORE.get(key + "_ref") if key else None
    exclude = {target_col, date_col, pd_col}
    if id_col:
        exclude.add(id_col)
    model_vars = [c for c in (ref_df.columns if ref_df is not None else [])
                  if c not in exclude]

    config = {
        "target_col":      target_col,
        "date_col":        date_col,
        "pd_col":          pd_col,
        "id_col":          id_col or None,
        "maturity_months": maturity_months,
        "period_freq":     period_freq,
        "woe_enabled":     woe_enabled,
        "woe_pretransformed": woe_pretransformed,
        "model_vars":      model_vars,
    }

    # Background hesaplama başlat
    prog_key = start_mon_compute(key, config)

    parts = [html.Strong("✓ Onaylandı — hesaplama başladı")]
    parts += [f"  ·  PD: {pd_col}"]
    parts += [f"  ·  Tarih: {date_col}"]
    parts += [f"  ·  {len(model_vars)} model değişkeni"]
    if id_col:
        parts += [f"  ·  ID: {id_col}"]

    compute_state = {"prog_key": prog_key, "key": key}
    return (config,
            dbc.Alert(parts, color="info", style=_ALERT_STYLE),
            True,
            compute_state,
            False,    # interval enabled
            True)     # compute modal open


# ══════════════════════════════════════════════════════════════════════════════
#   Config banner
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-config-banner", "children"),
    Input("store-mon-config", "data"),
)
def mon_update_config_banner(config):
    if not config or not config.get("target_col"):
        return html.Div()

    def badge(label, value, color="#4F8EF7"):
        return html.Span([
            html.Span(label, className="banner-badge-label"),
            html.Span(value, className="banner-badge-value"),
        ], className="banner-badge", style={"borderColor": color})

    items = [badge("TARGET", config["target_col"])]
    if config.get("pd_col"):
        items.append(badge("PD", config["pd_col"], "#8b5cf6"))
    if config.get("date_col"):
        items.append(badge("TARİH", config["date_col"], "#10b981"))
    model_vars = config.get("model_vars", [])
    if model_vars:
        items.append(badge("DEĞİŞKEN", str(len(model_vars)), "#06b6d4"))
    if config.get("id_col"):
        items.append(badge("ID", config["id_col"], "#f59e0b"))
    if config.get("woe_enabled"):
        items.append(badge("WoE", "Aktif", "#a78bfa"))
    freq_label = "Aylık" if config.get("period_freq") == "M" else "Çeyreklik"
    items.append(badge("DÖNEM", freq_label, "#7e8fa4"))
    return html.Div(items, className="config-banner")


# ══════════════════════════════════════════════════════════════════════════════
#   Metrics row
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-metrics-row", "children"),
    Input("store-mon-config", "data"),
    State("store-mon-key", "data"),
)
def mon_update_metrics(config, key):
    df_orig = _MON_STORE.get(key + "_mon") if key else None

    if df_orig is None:
        return html.Div(
            "Sol menüden tablo adını girin ve Veriyi Yükle butonuna tıklayın.",
            className="alert-info-custom",
        )
    if not config or not config.get("target_col"):
        return html.Div(
            "Kolon yapılandırmasını tamamlayın ve onaylayın.",
            className="alert-info-custom",
        )

    date_col = config.get("date_col")
    df_active = df_orig
    active_rows = len(df_active)
    target = config["target_col"]

    def card(value, label, accent="#4F8EF7", tooltip=None):
        return dbc.Col(html.Div([
            html.Div(value, className="metric-value", style={"color": accent}),
            html.Div(label, className="metric-label"),
        ], className="metric-card",
           title=tooltip,
           style={"cursor": "help"} if tooltip else {}),
        width=3)

    s_target = pd.to_numeric(df_active[target], errors="coerce")
    target_rate = s_target.mean() * 100
    n_bad  = int(s_target.sum())
    n_good = int((s_target == 0).sum())
    tooltip = f"1 (Bad):  {n_bad:,}\n0 (Good): {n_good:,}\nToplam: {active_rows:,}"

    if date_col and date_col in df_active.columns:
        try:
            dates = pd.to_datetime(df_active[date_col], errors="coerce").dropna()
            d_min = dates.min().strftime("%Y-%m")
            d_max = dates.max().strftime("%Y-%m")
            date_card = card(f"{d_min} – {d_max}", f"Tarih Aralığı  ({date_col})", "#7e8fa4")
        except Exception:
            date_card = card("—", f"Tarih Aralığı  ({date_col})", "#7e8fa4")
    else:
        date_card = card("—", "Tarih Aralığı", "#7e8fa4")

    cards = [
        card(f"{active_rows:,}", "Aktif Satır"),
        card(f"{df_active.shape[1]}", "Kolon Sayısı"),
        date_card,
        card(f"%{target_rate:.2f}", f"Temerrüt Oranı  ({target})", "#ef4444", tooltip=tooltip),
    ]
    return dbc.Row(cards, className="g-3 mb-0")


# ══════════════════════════════════════════════════════════════════════════════
#   Önizleme (DataTable)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-data-preview", "children"),
    Input("store-mon-config", "data"),
    State("store-mon-key", "data"),
)
def mon_update_preview(config, key):
    df_orig = _MON_STORE.get(key + "_mon") if key else None
    if df_orig is None or not config or not config.get("target_col"):
        return html.Div()

    df_active = df_orig
    preview = df_active.head(50)

    return html.Div([
        html.P("Veri Önizleme", className="section-title"),
        dash_table.DataTable(
            data=preview.to_dict("records"),
            columns=[{"name": c, "id": c} for c in preview.columns],
            page_size=20,
            page_action="native",
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1a2035",
                "color": "#E8EAF0",
                "fontWeight": "600",
                "fontSize": "0.78rem",
                "border": "1px solid #2d3a4f",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
            },
            style_data={
                "backgroundColor": "#161C27",
                "color": "#c8cdd8",
                "fontSize": "0.83rem",
                "border": "1px solid #232d3f",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"},
                {"if": {"state": "selected"}, "backgroundColor": "#1a3a6e",
                 "border": "1px solid #4F8EF7"},
            ],
            style_filter={
                "backgroundColor": "#0e1117",
                "color": "#c8cdd8",
                "border": "1px solid #2d3a4f",
            },
            css=[{"selector": ".dash-filter input", "rule": "color: #c8cdd8 !important;"}],
        ),
        html.P(
            f"İlk 50 satır gösteriliyor  ·  Toplam aktif kayıt: {len(df_active):,}",
            style={"fontSize": "0.75rem", "color": "#7e8fa4", "marginTop": "0.5rem"},
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#   WoE checkbox → upload alanını aç/kapat
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-collapse-woe-upload", "is_open"),
    Input("mon-chk-woe", "value"),
)
def mon_toggle_woe_upload(chk_val):
    return bool(chk_val and "woe" in chk_val)


# ══════════════════════════════════════════════════════════════════════════════
#   Opt pickle yükle → _MON_STORE[key + "_opt"]
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-opt-pickle-status", "children"),
    Input("mon-upload-opt-pickle", "contents"),
    State("mon-upload-opt-pickle", "filename"),
    State("store-mon-key", "data"),
    prevent_initial_call=True,
)
def mon_upload_opt_pickle(contents, filename, key):
    if not contents or not filename:
        return ""
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        opt = _pickle.loads(decoded)

        if not key:
            key = str(uuid.uuid4())
        _MON_STORE[key + "_opt"] = opt

        return html.Span(f"✓ {filename} yüklendi", style={"color": "#10b981"})
    except Exception as e:
        return html.Span(f"Hata: {e}", style={"color": "#ef4444"})


# ══════════════════════════════════════════════════════════════════════════════
#   Hesaplama ilerleme polling (interval)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-config-status",          "children", allow_duplicate=True),
    Output("interval-mon-compute",       "disabled",  allow_duplicate=True),
    Output("store-mon-summaries-signal", "data"),
    Output("mon-modal-compute",          "is_open"),
    Output("mon-compute-modal-step",     "children"),
    Output("mon-compute-modal-progress", "value"),
    Input("interval-mon-compute", "n_intervals"),
    State("store-mon-compute-state", "data"),
    prevent_initial_call=True,
)
def mon_poll_compute_progress(n_intervals, compute_state):
    if not compute_state:
        return no_update, True, no_update, False, "", 0

    prog_key = compute_state.get("prog_key")
    if not prog_key:
        return no_update, True, no_update, False, "", 0

    progress = _PRECOMPUTE_PROGRESS.get(prog_key, {})
    step = progress.get("step", 0)
    done = progress.get("done", False)
    error = progress.get("error")

    step_labels = {
        0: "Başlatılıyor...",
        1: "Referans özeti hesaplanıyor...",
        2: "Dönemlere ayrılıyor...",
        3: "Dönem özetleri hesaplanıyor...",
        4: "Olgunlaşma kontrolü...",
    }
    total_steps = 4
    pct = int(step / total_steps * 100)

    if error:
        return (
            dbc.Alert([html.Strong("Hata: "), error],
                      color="danger", style=_ALERT_STYLE),
            True,
            no_update,
            False,  # close compute modal
            f"Hata: {error}",
            0,
        )

    if done:
        key = compute_state.get("key")
        return (
            dbc.Alert([html.Strong("✓ Hesaplama tamamlandı")],
                      color="success", style=_ALERT_STYLE),
            True,
            {"key": key, "ts": n_intervals},
            False,  # close compute modal
            "Tamamlandı!",
            100,
        )

    label = step_labels.get(step, f"Adım {step}...")
    return (
        dbc.Alert([html.Strong(f"Hesaplanıyor ({step}/{total_steps}): "), label],
                  color="info", style=_ALERT_STYLE),
        False,
        no_update,
        True,  # keep compute modal open
        f"Adım {step}/{total_steps}: {label}",
        pct,
    )


# ══════════════════════════════════════════════════════════════════════════════
#   Compute Modal — elapsed timer (clientside)
# ══════════════════════════════════════════════════════════════════════════════
app.clientside_callback(
    """
    function(is_open) {
        if (!is_open) {
            if (window._monComputeTimer) {
                clearInterval(window._monComputeTimer);
                window._monComputeTimer = null;
            }
            return "";
        }
        var start = Date.now();
        var el = document.getElementById("mon-compute-modal-elapsed");
        if (window._monComputeTimer) clearInterval(window._monComputeTimer);
        window._monComputeTimer = setInterval(function() {
            var diff = Math.floor((Date.now() - start) / 1000);
            var m = Math.floor(diff / 60);
            var s = diff % 60;
            if (el) el.textContent = m + ":" + (s < 10 ? "0" : "") + s;
        }, 1000);
        return "0:00";
    }
    """,
    Output("mon-compute-modal-elapsed", "children"),
    Input("mon-modal-compute", "is_open"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#   İzleme Slideshow — veri yüklenirken tab bilgilendirmesi
# ══════════════════════════════════════════════════════════════════════════════
_MON_N_SLIDES = 6


# ── Yükleme başlayınca slideshow aç ────────────────────────────────────────
@app.callback(
    Output("mon-modal-slideshow",    "is_open", allow_duplicate=True),
    Output("mon-interval-slideshow", "disabled", allow_duplicate=True),
    Output("mon-store-slide-index",  "data", allow_duplicate=True),
    Output("mon-interval-slideshow", "n_intervals", allow_duplicate=True),
    Input("mon-btn-load",     "n_clicks"),
    Input("mon-btn-load-csv", "n_clicks"),
    prevent_initial_call=True,
)
def mon_open_slideshow_on_load(*_):
    return True, False, 0, 0


# ── Veri yüklenince veya close butonu ile slideshow kapat ──────────────────
@app.callback(
    Output("mon-modal-slideshow",    "is_open"),
    Output("mon-interval-slideshow", "disabled"),
    Input("store-mon-key", "data"),
    Input("mon-load-status", "children"),
    Input("mon-btn-slideshow-close", "n_clicks"),
    prevent_initial_call=True,
)
def mon_close_slideshow(key, status, _close):
    trigger = dash.callback_context.triggered_id
    if trigger == "mon-btn-slideshow-close":
        return False, True
    if trigger == "store-mon-key" and key:
        return False, True
    if trigger == "mon-load-status":
        return False, True
    return no_update, no_update


# ── Interval ile otomatik slayt ilerleme ───────────────────────────────────
app.clientside_callback(
    f"""
    function(n_intervals, current) {{
        if (n_intervals === undefined || n_intervals === 0) return current || 0;
        return ((current || 0) + 1) % {_MON_N_SLIDES};
    }}
    """,
    Output("mon-store-slide-index", "data"),
    Input("mon-interval-slideshow", "n_intervals"),
    State("mon-store-slide-index", "data"),
    prevent_initial_call=True,
)


# ── Dot tıklama → slayt değiştir + interval sıfırla ──────────────────────
app.clientside_callback(
    f"""
    function({", ".join(f"d{i}" for i in range(_MON_N_SLIDES))}) {{
        var ctx = dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {{
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }}
        var prop_id = ctx.triggered[0].prop_id;
        for (var i = 0; i < {_MON_N_SLIDES}; i++) {{
            if (prop_id === "mon-slide-dot-" + i + ".n_clicks") {{
                return [i, 0];
            }}
        }}
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    }}
    """,
    Output("mon-store-slide-index", "data", allow_duplicate=True),
    Output("mon-interval-slideshow", "n_intervals", allow_duplicate=True),
    [Input(f"mon-slide-dot-{i}", "n_clicks") for i in range(_MON_N_SLIDES)],
    prevent_initial_call=True,
)


# ── Slayt index → görünürlük, dots, progress güncelle ────────────────────
_mon_slide_outputs = [Output(f"mon-slide-{i}", "style") for i in range(_MON_N_SLIDES)]
_mon_dot_outputs = [Output(f"mon-slide-dot-{i}", "className") for i in range(_MON_N_SLIDES)]

app.clientside_callback(
    f"""
    function(idx) {{
        var slides = [];
        var dots = [];
        for (var i = 0; i < {_MON_N_SLIDES}; i++) {{
            slides.push(i === idx ? {{"display": "block"}} : {{"display": "none"}});
            dots.push(i === idx ? "slide-dot dot-active" : "slide-dot");
        }}
        var pct = ((idx + 1) / {_MON_N_SLIDES}) * 100;
        var progressStyle = {{"width": pct + "%"}};
        return slides.concat(dots).concat([progressStyle]);
    }}
    """,
    _mon_slide_outputs + _mon_dot_outputs + [Output("mon-slide-progress-fill", "style")],
    Input("mon-store-slide-index", "data"),
    prevent_initial_call=True,
)


# ── Slideshow elapsed time sayacı ─────────────────────────────────────────
app.clientside_callback(
    """
    function(is_open) {
        if (!is_open) {
            if (window._monSlideshowTimer) {
                clearInterval(window._monSlideshowTimer);
                window._monSlideshowTimer = null;
            }
            return "";
        }
        var start = Date.now();
        var el = document.getElementById("mon-slideshow-elapsed");
        if (window._monSlideshowTimer) clearInterval(window._monSlideshowTimer);
        window._monSlideshowTimer = setInterval(function() {
            var diff = Math.floor((Date.now() - start) / 1000);
            var m = Math.floor(diff / 60);
            var s = diff % 60;
            if (el) el.textContent = m + ":" + (s < 10 ? "0" : "") + s;
        }, 1000);
        return "0:00";
    }
    """,
    Output("mon-slideshow-elapsed", "children"),
    Input("mon-modal-slideshow", "is_open"),
    prevent_initial_call=True,
)
