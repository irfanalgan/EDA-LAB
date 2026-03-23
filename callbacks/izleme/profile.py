"""İzleme — profil kaydet / yükle / sil.

Profiller profiles/izleme/<isim>/ altına kaydedilir.
Geliştirme profilleriyle (profiles/<isim>/) hiçbir ilişkisi yoktur.
"""

import json
import pickle
import shutil
import uuid
from pathlib import Path

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from app_instance import app
from server_state import _MON_STORE, clear_mon_store
from callbacks.izleme.compute import start_mon_compute, start_mon_incremental
from data.loader import _build_conn_str, _quote_table

_MON_PROFILES_DIR = Path(__file__).parent.parent.parent / "profiles" / "izleme"
_MON_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

_ALERT_STYLE = {"padding": "0.4rem 0.75rem", "fontSize": "0.78rem"}


def _check_sql_for_new_data(conn_info, config, last_period, saved_count,
                            saved_columns=None):
    """SQL'e bağlanıp yeni veri var mı kontrol et.

    Returns:
        (status, new_df, detail)
        status: "no_change" | "new_data" | "schema_changed" | "error"
        new_df: Sadece yeni satırları içeren DataFrame (veya None)
        detail: Ek bilgi (schema_changed ise fark mesajı)
    """
    import pyodbc

    server = conn_info.get("server", "")
    database = conn_info.get("database", "")
    driver = conn_info.get("driver", "")
    tables = conn_info.get("tables", [])

    if not server or not database or not tables:
        return "no_change", None, None

    date_col = config.get("date_col", "")
    if not date_col:
        return "no_change", None, None

    try:
        conn_str = _build_conn_str(server, database, driver)
        table_name = _quote_table(tables[0])

        with pyodbc.connect(conn_str) as conn:
            # Toplam count kontrolü
            count_sql = f"SELECT COUNT(*) FROM {table_name}"
            current_count = pd.read_sql(count_sql, conn).iloc[0, 0]

            if current_count <= saved_count:
                return "no_change", None, None

            # Yeni veri var — last_period'dan sonraki satırları çek
            if last_period:
                try:
                    period_end = pd.Period(last_period).end_time.strftime("%Y-%m-%d")
                    where = f"WHERE [{date_col}] > '{period_end}'"
                except Exception:
                    where = ""
            else:
                where = ""

            sql = f"SELECT * FROM {table_name} {where}"
            new_df = pd.read_sql(sql, conn)

            # Multi-table join varsa
            if len(tables) > 1:
                join_keys = conn_info.get("join_keys", [""])
                left_keys = [k.strip() for k in join_keys[0].split(",") if k.strip()]
                if left_keys:
                    for i, tbl in enumerate(tables[1:], start=1):
                        jk_str = join_keys[i] if i < len(join_keys) else ""
                        right_keys = [k.strip() for k in jk_str.split(",") if k.strip()]
                        if not right_keys:
                            right_keys = left_keys
                        tbl_df = pd.read_sql(f"SELECT * FROM {_quote_table(tbl)}", conn)
                        dup_cols = [c for c in tbl_df.columns
                                    if c in new_df.columns and c not in right_keys]
                        tbl_df = tbl_df.drop(columns=dup_cols, errors="ignore")
                        new_df = pd.merge(new_df, tbl_df,
                                          left_on=left_keys, right_on=right_keys,
                                          how="left")

            if len(new_df) == 0:
                return "no_change", None, None

            # ── Kolon yapısı kontrolü ─────────────────────────────────
            if saved_columns:
                saved_set = set(saved_columns)
                current_set = set(new_df.columns.tolist())
                added = current_set - saved_set
                removed = saved_set - current_set
                if added or removed:
                    parts = []
                    if added:
                        parts.append(f"Eklenen: {', '.join(sorted(added))}")
                    if removed:
                        parts.append(f"Kaldırılan: {', '.join(sorted(removed))}")
                    detail = " · ".join(parts)
                    return "schema_changed", None, detail

            return "new_data", new_df, None

    except Exception as e:
        return "error", None, str(e)


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _list_mon_profiles() -> list[dict]:
    profiles = []
    if not _MON_PROFILES_DIR.exists():
        return profiles
    for d in sorted(_MON_PROFILES_DIR.iterdir()):
        meta_file = d / "meta.json"
        if d.is_dir() and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                label = f"{d.name}  ({meta.get('saved_at', '?')})"
                profiles.append({"label": label, "value": d.name})
            except Exception:
                profiles.append({"label": d.name, "value": d.name})
        elif d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except Exception:
                pass
    return profiles


def _save_mon_profile(name: str, key: str, config: dict,
                      connection_info: dict | None = None):
    """Özet tabanlı profil kaydet — ref_df + özetler saklanır."""
    profile_dir = _MON_PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    ref_summary = _MON_STORE.get(key + "_ref_summary")
    period_summaries = _MON_STORE.get(key + "_period_summaries")
    mon_df = _MON_STORE.get(key + "_mon")

    # Son dönem ve toplam count — incremental update kontrolü için
    last_period = ""
    mon_total_count = 0
    if period_summaries:
        last_period = period_summaries[-1]["period_label"]
        mon_total_count = sum(s["n_total"] for s in period_summaries)

    # İzleme verisinin kolon listesi — schema kontrolü için
    mon_columns = []
    if mon_df is not None:
        mon_columns = mon_df.columns.tolist()

    meta = {
        "format_version": 2,
        "config": config,
        "connection": connection_info or {},
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ref_n_total": ref_summary["n_total"] if ref_summary else 0,
        "mon_n_total": mon_total_count,
        "n_periods": len(period_summaries) if period_summaries else 0,
        "last_period": last_period,
        "mon_total_count": mon_total_count,
        "mon_columns": mon_columns,
    }
    (profile_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Özetleri kaydet
    if ref_summary is not None:
        with open(profile_dir / "ref_summary.pkl", "wb") as f:
            pickle.dump(ref_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    if period_summaries is not None:
        with open(profile_dir / "period_summaries.pkl", "wb") as f:
            pickle.dump(period_summaries, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Referans ham verisi (göç matrisi incremental hesaplama için)
    ref_df = _MON_STORE.get(key + "_ref")
    if ref_df is not None:
        with open(profile_dir / "ref_df.pkl", "wb") as f:
            pickle.dump(ref_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Opt pickle (WoE dönüşüm formülü)
    opt = _MON_STORE.get(key + "_opt")
    opt_path = profile_dir / "opt.pkl"
    if opt is not None:
        with open(opt_path, "wb") as f:
            pickle.dump(opt, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif opt_path.exists():
        opt_path.unlink()

    # Eski format dosyalarını temizle (varsa)
    for old_file in ["ref_data.pkl", "mon_data.pkl", "data.pkl"]:
        old_path = profile_dir / old_file
        if old_path.exists():
            old_path.unlink()


def _load_mon_profile(name: str):
    """Profil yükle — özet veya eski format.

    Returns: (new_key, config, opt, is_new_format)
      is_new_format=True → özetler _MON_STORE'a yüklendi, hesaplama gerekmez
      is_new_format=False → ham veriler _MON_STORE'a yüklendi, hesaplama gerekir
    """
    profile_dir = _MON_PROFILES_DIR / name
    meta = json.loads((profile_dir / "meta.json").read_text(encoding="utf-8"))
    config = meta["config"]
    format_version = meta.get("format_version", 1)

    new_key = str(uuid.uuid4())
    clear_mon_store()

    # Opt pickle
    opt_path = profile_dir / "opt.pkl"
    opt = None
    if opt_path.exists():
        with open(opt_path, "rb") as f:
            opt = pickle.load(f)
    if opt is not None:
        _MON_STORE[new_key + "_opt"] = opt

    if format_version >= 2:
        # ── Yeni format: özetleri yükle ──────────────────────────────────
        ref_sum_path = profile_dir / "ref_summary.pkl"
        if ref_sum_path.exists():
            with open(ref_sum_path, "rb") as f:
                _MON_STORE[new_key + "_ref_summary"] = pickle.load(f)

        ps_path = profile_dir / "period_summaries.pkl"
        if ps_path.exists():
            with open(ps_path, "rb") as f:
                _MON_STORE[new_key + "_period_summaries"] = pickle.load(f)

        # Referans ham verisi (göç matrisi incremental hesaplama için)
        ref_df_path = profile_dir / "ref_df.pkl"
        if ref_df_path.exists():
            with open(ref_df_path, "rb") as f:
                _MON_STORE[new_key + "_ref"] = pickle.load(f)

        return new_key, config, opt, True
    else:
        # ── Eski format: ham veri yükle → kullanıcı hesaplatacak ─────────
        ref_path = profile_dir / "ref_data.pkl"
        if ref_path.exists():
            with open(ref_path, "rb") as f:
                _MON_STORE[new_key + "_ref"] = pickle.load(f)

        mon_path = profile_dir / "mon_data.pkl"
        if mon_path.exists():
            with open(mon_path, "rb") as f:
                _MON_STORE[new_key + "_mon"] = pickle.load(f)

        # En eski format (tek data.pkl)
        if not ref_path.exists() and not mon_path.exists():
            old_path = profile_dir / "data.pkl"
            if old_path.exists():
                with open(old_path, "rb") as f:
                    _MON_STORE[new_key + "_mon"] = pickle.load(f)

        return new_key, config, opt, False


def _delete_mon_profile(name: str):
    import os, stat
    profile_dir = _MON_PROFILES_DIR / name
    if profile_dir.exists():
        def _force_remove(func, path, _exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(profile_dir, onerror=_force_remove)


# ══════════════════════════════════════════════════════════════════════════════
#   Profil dropdown — sayfa açılışında doldur
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-dd-profile", "options"),
    Input("mon-dd-profile", "id"),
)
def mon_populate_profile_dropdown(_):
    return _list_mon_profiles()


# ══════════════════════════════════════════════════════════════════════════════
#   Profil Kaydet — modal aç/kapat
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-modal-profile-save", "is_open"),
    Input("mon-btn-profile-save", "n_clicks"),
    Input("mon-btn-profile-save-confirm", "n_clicks"),
    State("mon-modal-profile-save", "is_open"),
    prevent_initial_call=True,
)
def mon_toggle_save_modal(open_clicks, confirm_clicks, is_open):
    ctx = dash.callback_context.triggered_id
    if ctx == "mon-btn-profile-save":
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#   Profil Kaydet (confirm)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-profile-status", "children", allow_duplicate=True),
    Output("mon-dd-profile", "options", allow_duplicate=True),
    Output("mon-dd-profile", "value", allow_duplicate=True),
    Output("mon-modal-profile-save", "is_open", allow_duplicate=True),
    Output("mon-toast-profile-saved", "is_open", allow_duplicate=True),
    Output("mon-toast-profile-saved", "children", allow_duplicate=True),
    Input("mon-btn-profile-save-confirm", "n_clicks"),
    State("mon-input-profile-name", "value"),
    State("store-mon-key", "data"),
    State("store-mon-config", "data"),
    State("mon-input-sql-server", "value"),
    State("mon-input-sql-database", "value"),
    State("mon-dd-sql-driver", "value"),
    State("mon-input-table-1", "value"),
    State("mon-input-table-2", "value"),
    State("mon-input-table-3", "value"),
    State("mon-input-sql-jk-1", "value"),
    State("mon-input-sql-jk-2", "value"),
    State("mon-input-sql-jk-3", "value"),
    prevent_initial_call=True,
)
def mon_save_profile_cb(_, name, key, config,
                        server, database, driver, t1, t2, t3, jk1, jk2, jk3):
    no = dash.no_update
    if not name or not name.strip():
        return (
            dbc.Alert("Profil adı boş olamaz.", color="warning", style=_ALERT_STYLE),
            no, no, no, False, "",
        )
    if not key or not config:
        return (
            dbc.Alert("Önce veri yükleyin ve yapılandırmayı onaylayın.",
                      color="warning", style=_ALERT_STYLE),
            no, no, no, False, "",
        )

    name = name.strip()
    conn_info = {
        "server": server or "", "database": database or "", "driver": driver or "",
        "tables": [t for t in [t1, t2, t3] if t and t.strip()],
        "join_keys": [jk1 or "", jk2 or "", jk3 or ""],
    }
    try:
        _save_mon_profile(name, key, config, connection_info=conn_info)
        return (
            dbc.Alert(f"✓ '{name}' kaydedildi.", color="success", style=_ALERT_STYLE),
            _list_mon_profiles(),
            name,
            False,
            True,
            f"'{name}' başarıyla kaydedildi.",
        )
    except Exception as e:
        return (
            dbc.Alert(f"Kaydetme hatası: {e}", color="danger", style=_ALERT_STYLE),
            no, no, no, False, "",
        )


# ══════════════════════════════════════════════════════════════════════════════
#   Profil Yükle
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("store-mon-key", "data", allow_duplicate=True),
    Output("store-mon-config", "data", allow_duplicate=True),
    Output("store-mon-loaded", "data", allow_duplicate=True),
    Output("mon-collapse-config", "is_open", allow_duplicate=True),
    Output("mon-dd-target-col", "options", allow_duplicate=True),
    Output("mon-dd-target-col", "value", allow_duplicate=True),
    Output("mon-dd-date-col", "options", allow_duplicate=True),
    Output("mon-dd-date-col", "value", allow_duplicate=True),
    Output("mon-dd-pd-col", "options", allow_duplicate=True),
    Output("mon-dd-pd-col", "value", allow_duplicate=True),
    Output("mon-dd-id-col", "options", allow_duplicate=True),
    Output("mon-dd-id-col", "value", allow_duplicate=True),
    Output("mon-input-maturity", "value", allow_duplicate=True),
    Output("mon-radio-period-freq", "value", allow_duplicate=True),
    Output("mon-profile-status", "children"),
    Output("mon-load-status", "children", allow_duplicate=True),
    Output("mon-collapse-profile-actions", "is_open", allow_duplicate=True),
    Output("mon-input-sql-server", "value", allow_duplicate=True),
    Output("mon-input-sql-database", "value", allow_duplicate=True),
    Output("mon-dd-sql-driver", "value", allow_duplicate=True),
    Output("mon-input-table-1", "value", allow_duplicate=True),
    Output("mon-input-table-2", "value", allow_duplicate=True),
    Output("mon-input-table-3", "value", allow_duplicate=True),
    Output("mon-input-sql-jk-1", "value", allow_duplicate=True),
    Output("mon-input-sql-jk-2", "value", allow_duplicate=True),
    Output("mon-input-sql-jk-3", "value", allow_duplicate=True),
    Output("mon-chk-woe", "value", allow_duplicate=True),
    Output("mon-opt-pickle-status", "children", allow_duplicate=True),
    Output("store-mon-summaries-signal", "data", allow_duplicate=True),
    Output("mon-modal-compute", "is_open", allow_duplicate=True),
    Output("store-mon-compute-state", "data", allow_duplicate=True),
    Output("interval-mon-compute", "disabled", allow_duplicate=True),
    Input("mon-btn-profile-load", "n_clicks"),
    State("mon-dd-profile", "value"),
    prevent_initial_call=True,
)
def mon_load_profile_cb(_, profile_name):
    _N_OUTPUTS = 32
    no = dash.no_update
    if not profile_name:
        return (no,) * _N_OUTPUTS

    _IDX_PROFILE_STATUS = 14

    try:
        new_key, config, opt, is_new_format = _load_mon_profile(profile_name)
    except Exception as e:
        out = [no] * _N_OUTPUTS
        out[_IDX_PROFILE_STATUS] = dbc.Alert(f"Yükleme hatası: {e}", color="danger", style=_ALERT_STYLE)
        return tuple(out)

    # Meta bilgileri oku
    meta_path = _MON_PROFILES_DIR / profile_name / "meta.json"
    meta = {}
    conn = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        conn = meta.get("connection", {})
    tables = conn.get("tables", [])
    jkeys = conn.get("join_keys", ["", "", ""])

    if is_new_format:
        # Yeni format — özetler yüklendi
        ref_n = meta.get("ref_n_total", 0)
        mon_n = meta.get("mon_n_total", 0)
        n_periods = meta.get("n_periods", 0)
        loaded = {"ref_rows": ref_n, "mon_rows": mon_n}

        # SQL'de yeni veri var mı kontrol et
        last_period = meta.get("last_period", "")
        saved_count = meta.get("mon_total_count", 0)
        saved_columns = meta.get("mon_columns")
        sql_status, new_df, sql_detail = _check_sql_for_new_data(
            conn, config, last_period, saved_count, saved_columns)

        if sql_status == "schema_changed":
            # Kolon yapısı değişmiş — hesaplama güvenli değil
            profile_msg = dbc.Alert(
                [html.Strong("⚠ Tablo yapısı değişmiş!"),
                 html.Br(),
                 f"Fark: {sql_detail}",
                 html.Br(), html.Br(),
                 "Mevcut profil bu yapıyla uyumlu değil. ",
                 "Lütfen yeni bir profil oluşturun veya tabloyu eski yapısına döndürün."],
                color="warning", style=_ALERT_STYLE,
            )
            load_msg = dbc.Alert(
                "Kolon uyumsuzluğu — hesaplama durduruldu",
                color="warning", style=_ALERT_STYLE,
            )
            summaries_signal = no
            compute_modal_open = False
            compute_state = no
            interval_disabled = no
        elif sql_status == "new_data" and new_df is not None:
            # Yeni veri var — incremental compute başlat
            _MON_STORE[new_key + "_new_mon"] = new_df
            prog_key = start_mon_incremental(new_key, config, new_df)
            new_rows = len(new_df)

            profile_msg = dbc.Alert(
                [f"✓ '{profile_name}' yüklendi  ·  ",
                 html.Strong(f"{new_rows:,} yeni satır bulundu"),
                 f"  ·  Yeni dönemler hesaplanıyor…"],
                color="info", style=_ALERT_STYLE,
            )
            load_msg = dbc.Alert(
                f"Ref: {ref_n:,}  ·  İzleme: {mon_n:,} + {new_rows:,} yeni",
                color="info", style=_ALERT_STYLE,
            )
            summaries_signal = no
            compute_modal_open = True
            compute_state = {"prog_key": prog_key, "key": new_key}
            interval_disabled = False
        else:
            # Değişiklik yok — direkt göster
            profile_msg = dbc.Alert(
                f"✓ '{profile_name}' yüklendi  ·  Ref: {ref_n:,}  ·  İzleme: {mon_n:,}  ·  {n_periods} dönem",
                color="success", style=_ALERT_STYLE,
            )
            load_msg = dbc.Alert(
                "Özetler profilden yüklendi — değişiklik yok",
                color="info", style=_ALERT_STYLE,
            )
            summaries_signal = {"key": new_key, "ts": 0}
            compute_modal_open = False
            compute_state = no
            interval_disabled = no
    else:
        # Eski format — ham veriler yüklendi, otomatik hesaplama başlat
        ref_df = _MON_STORE.get(new_key + "_ref")
        mon_df = _MON_STORE.get(new_key + "_mon")
        if ref_df is None and mon_df is None:
            out = [no] * _N_OUTPUTS
            out[_IDX_PROFILE_STATUS] = dbc.Alert(
                "Profilde veri dosyası bulunamadı.", color="danger", style=_ALERT_STYLE)
            return tuple(out)

        loaded = {}
        if ref_df is not None:
            loaded["ref_rows"] = len(ref_df)
        if mon_df is not None:
            loaded["mon_rows"] = len(mon_df)

        ref_info = f"Ref: {len(ref_df):,}" if ref_df is not None else "Ref: —"
        mon_info = f"İzleme: {len(mon_df):,}" if mon_df is not None else "İzleme: —"

        # Hesaplamayı otomatik başlat
        prog_key = start_mon_compute(new_key, config)

        profile_msg = dbc.Alert(
            [f"✓ '{profile_name}' yüklendi (eski format)  ·  {ref_info}  ·  {mon_info}  ·  ",
             html.Strong("Özetler hesaplanıyor…")],
            color="info", style=_ALERT_STYLE,
        )
        load_msg = dbc.Alert(
            f"{ref_info}  ·  {mon_info}  ·  Profilden yüklendi",
            color="info", style=_ALERT_STYLE,
        )
        summaries_signal = no
        compute_modal_open = True
        compute_state = {"prog_key": prog_key, "key": new_key}
        interval_disabled = False

    # Dropdown seçenekleri — config'den model_vars varsa kullan
    model_vars = config.get("model_vars", [])
    config_cols = [config.get("target_col"), config.get("date_col"),
                   config.get("pd_col"), config.get("id_col")] + model_vars
    config_cols = [c for c in config_cols if c]
    all_opts = [{"label": c, "value": c} for c in config_cols]
    pd_opts = all_opts  # profil yüklendikten sonra tam liste gerekmez

    return (
        new_key,                                        # store-mon-key
        config,                                         # store-mon-config
        loaded,                                         # store-mon-loaded
        True,                                           # mon-collapse-config is_open
        all_opts,                                       # mon-dd-target-col options
        config.get("target_col"),                       # mon-dd-target-col value
        all_opts,                                       # mon-dd-date-col options
        config.get("date_col"),                         # mon-dd-date-col value
        pd_opts,                                        # mon-dd-pd-col options
        config.get("pd_col"),                           # mon-dd-pd-col value
        all_opts,                                       # mon-dd-id-col options
        config.get("id_col"),                           # mon-dd-id-col value
        config.get("maturity_months", 12),              # mon-input-maturity
        config.get("period_freq", "M"),                 # mon-radio-period-freq
        profile_msg,                                    # mon-profile-status
        load_msg,                                       # mon-load-status
        True,                                           # mon-collapse-profile-actions
        conn.get("server", ""),                         # mon-input-sql-server
        conn.get("database", ""),                       # mon-input-sql-database
        conn.get("driver", ""),                         # mon-dd-sql-driver
        tables[0] if len(tables) > 0 else "",           # mon-input-table-1
        tables[1] if len(tables) > 1 else "",           # mon-input-table-2
        tables[2] if len(tables) > 2 else "",           # mon-input-table-3
        jkeys[0] if len(jkeys) > 0 else "",             # mon-input-sql-jk-1
        jkeys[1] if len(jkeys) > 1 else "",             # mon-input-sql-jk-2
        jkeys[2] if len(jkeys) > 2 else "",             # mon-input-sql-jk-3
        ["woe"] if opt is not None else [],              # mon-chk-woe value
        (html.Span("✓ opt pickle yüklendi (profilden)",  # mon-opt-pickle-status
                   style={"color": "#10b981"})
         if opt is not None else ""),
        summaries_signal,                               # store-mon-summaries-signal
        compute_modal_open,                             # mon-modal-compute is_open
        compute_state,                                  # store-mon-compute-state
        interval_disabled,                              # interval-mon-compute disabled
    )


# ══════════════════════════════════════════════════════════════════════════════
#   Profil Sil — modal aç
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-modal-profile-delete", "is_open", allow_duplicate=True),
    Output("mon-dd-profile-delete", "options"),
    Output("mon-dd-profile-delete", "value"),
    Output("mon-delete-confirm-area", "children", allow_duplicate=True),
    Output("mon-btn-profile-delete-confirm", "style", allow_duplicate=True),
    Input("mon-btn-profile-delete", "n_clicks"),
    prevent_initial_call=True,
)
def mon_open_delete_modal(_):
    return True, _list_mon_profiles(), None, "", {"display": "none"}


# ══════════════════════════════════════════════════════════════════════════════
#   Profil seçilince onay butonu göster
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-delete-confirm-area", "children"),
    Output("mon-btn-profile-delete-confirm", "style"),
    Input("mon-dd-profile-delete", "value"),
    prevent_initial_call=True,
)
def mon_show_delete_confirm(profile_name):
    if not profile_name:
        return "", {"display": "none"}
    return (
        dbc.Alert(
            f"'{profile_name}' silinecek. Bu işlem geri alınamaz.",
            color="warning", style=_ALERT_STYLE,
        ),
        {},
    )


# ══════════════════════════════════════════════════════════════════════════════
#   Silme onayı
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("mon-profile-status", "children", allow_duplicate=True),
    Output("mon-dd-profile", "options", allow_duplicate=True),
    Output("mon-dd-profile", "value", allow_duplicate=True),
    Output("mon-modal-profile-delete", "is_open"),
    Input("mon-btn-profile-delete-confirm", "n_clicks"),
    State("mon-dd-profile-delete", "value"),
    prevent_initial_call=True,
)
def mon_confirm_delete_profile_cb(_, profile_name):
    if not profile_name:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    try:
        _delete_mon_profile(profile_name)
        return (
            dbc.Alert(f"'{profile_name}' silindi.", color="info", style=_ALERT_STYLE),
            _list_mon_profiles(),
            None,
            False,
        )
    except Exception as e:
        return (
            dbc.Alert(f"Silme hatası: {e}", color="danger", style=_ALERT_STYLE),
            dash.no_update, dash.no_update, True,
        )
