"""Profil Kaydet / Yükle / Sil — kullanıcı oturumunu dosyaya persist eder."""

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
from server_state import _SERVER_STORE

_PROFILES_DIR = Path(__file__).parent.parent / "profiles"
_PROFILES_DIR.mkdir(exist_ok=True)

_ALERT_STYLE = {"padding": "0.4rem 0.75rem", "fontSize": "0.78rem"}


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _list_profiles() -> list[dict]:
    """profiles/ altındaki profilleri tara, dropdown option listesi döndür."""
    profiles = []
    if not _PROFILES_DIR.exists():
        return profiles
    for d in sorted(_PROFILES_DIR.iterdir()):
        meta_file = d / "meta.json"
        if d.is_dir() and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                label = f"{d.name}  ({meta.get('saved_at', '?')})"
                profiles.append({"label": label, "value": d.name})
            except Exception:
                profiles.append({"label": d.name, "value": d.name})
        elif d.is_dir() and not any(d.iterdir()):
            # Boş klasör — silme artığı, temizle
            try:
                d.rmdir()
            except Exception:
                pass
    return profiles


def _save_profile(name: str, key: str, config: dict, expert_exclude: list,
                   connection_info: dict | None = None):
    """Profili diske yaz: meta.json + data.parquet + cache.pkl"""
    profile_dir = _PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    # 1. Meta
    from datetime import datetime
    meta = {
        "config": config,
        "expert_exclude": expert_exclude or [],
        "connection": connection_info or {},
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    (profile_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2. DataFrame → pickle
    df = _SERVER_STORE.get(key)
    if df is not None:
        with open(profile_dir / "data.pkl", "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Cache → pickle (UUID prefix çıkar, sadece suffix sakla)
    cache = {}
    prefix = f"{key}_"
    for k, v in _SERVER_STORE.items():
        if k.startswith(prefix):
            suffix = k[len(prefix):]
            cache[suffix] = v
    with open(profile_dir / "cache.pkl", "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_profile(name: str) -> tuple[str, dict, list, pd.DataFrame | None]:
    """Profili diskten oku, yeni UUID ile _SERVER_STORE'a yükle.
    Returns: (new_key, config, expert_exclude, df)
    """
    profile_dir = _PROFILES_DIR / name
    meta = json.loads((profile_dir / "meta.json").read_text(encoding="utf-8"))
    config = meta["config"]
    expert_exclude = meta.get("expert_exclude", [])

    new_key = str(uuid.uuid4())

    # DataFrame — pickle (eski parquet dosyası varsa onu da dene)
    pkl_data_path = profile_dir / "data.pkl"
    parquet_path = profile_dir / "data.parquet"
    df = None
    if pkl_data_path.exists():
        with open(pkl_data_path, "rb") as f:
            df = pickle.load(f)
    if df is None and parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            pass
    if df is not None:
        _SERVER_STORE[new_key] = df

    # Cache — re-key with new UUID
    cache_path = profile_dir / "cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        for suffix, value in cache.items():
            _SERVER_STORE[f"{new_key}_{suffix}"] = value

    return new_key, config, expert_exclude, df


def _delete_profile(name: str):
    """Profil klasörünü sil (OneDrive readonly kilidini kaldırarak)."""
    import os, stat
    profile_dir = _PROFILES_DIR / name
    if profile_dir.exists():
        def _force_remove(func, path, _exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(profile_dir, onerror=_force_remove)


# ── Callback: Sayfa açılışında profil listesini doldur ───────────────────────
@app.callback(
    Output("dd-profile", "options"),
    Input("dd-profile", "id"),  # sayfa açılışında tetikle
)
def populate_profile_dropdown(_):
    return _list_profiles()


# ── Callback: Profil Kaydet modal aç ────────────────────────────────────────
@app.callback(
    Output("modal-profile-save", "is_open"),
    Input("btn-profile-save", "n_clicks"),
    Input("btn-profile-save-confirm", "n_clicks"),
    State("modal-profile-save", "is_open"),
    prevent_initial_call=True,
)
def toggle_save_modal(open_clicks, confirm_clicks, is_open):
    ctx = dash.callback_context.triggered_id
    if ctx == "btn-profile-save":
        return True
    return False


# ── Callback: Profil Kaydet (confirm) ───────────────────────────────────────
@app.callback(
    Output("profile-status", "children", allow_duplicate=True),
    Output("dd-profile", "options", allow_duplicate=True),
    Output("dd-profile", "value", allow_duplicate=True),
    Output("modal-profile-save", "is_open", allow_duplicate=True),
    Output("toast-profile-saved", "is_open", allow_duplicate=True),
    Output("toast-profile-saved", "children", allow_duplicate=True),
    Output("store-profile-loaded", "data", allow_duplicate=True),
    Input("btn-profile-save-confirm", "n_clicks"),
    State("input-profile-name", "value"),
    State("store-key", "data"),
    State("store-config", "data"),
    State("store-expert-exclude", "data"),
    State("input-sql-server", "value"),
    State("input-sql-database", "value"),
    State("dd-sql-driver", "value"),
    State("input-table-1", "value"),
    State("input-table-2", "value"),
    State("input-table-3", "value"),
    State("input-sql-jk-1", "value"),
    State("input-sql-jk-2", "value"),
    State("input-sql-jk-3", "value"),
    prevent_initial_call=True,
)
def save_profile_cb(_, name, key, config, expert_exclude,
                    server, database, driver, t1, t2, t3, jk1, jk2, jk3):
    no = dash.no_update
    if not name or not name.strip():
        return (
            dbc.Alert("Profil adı boş olamaz.", color="warning", style=_ALERT_STYLE),
            no, no, no, False, "", no,
        )
    if not key or not config:
        return (
            dbc.Alert("Önce veri yükleyin ve yapılandırmayı onaylayın.",
                      color="warning", style=_ALERT_STYLE),
            no, no, no, False, "", no,
        )

    name = name.strip()
    conn_info = {
        "server": server or "", "database": database or "", "driver": driver or "",
        "tables": [t for t in [t1, t2, t3] if t and t.strip()],
        "join_keys": [jk1 or "", jk2 or "", jk3 or ""],
    }
    try:
        _save_profile(name, key, config, expert_exclude or [], connection_info=conn_info)
        return (
            dbc.Alert(f"✓ '{name}' kaydedildi.", color="success", style=_ALERT_STYLE),
            _list_profiles(),
            name,
            False,  # modal kapat
            True,   # toast aç
            f"'{name}' başarıyla kaydedildi.",
            name,   # store-profile-loaded
        )
    except Exception as e:
        return (
            dbc.Alert(f"Kaydetme hatası: {e}", color="danger", style=_ALERT_STYLE),
            no, no, no, False, "", no,
        )


# ── Callback: Profil Yükle ──────────────────────────────────────────────────
@app.callback(
    Output("store-key", "data", allow_duplicate=True),
    Output("store-config", "data", allow_duplicate=True),
    Output("store-expert-exclude", "data", allow_duplicate=True),
    Output("collapse-config", "is_open", allow_duplicate=True),
    Output("dd-target-col", "options", allow_duplicate=True),
    Output("dd-target-col", "value", allow_duplicate=True),
    Output("dd-date-col", "options", allow_duplicate=True),
    Output("dd-date-col", "value", allow_duplicate=True),
    Output("dd-segment-col", "options", allow_duplicate=True),
    Output("dd-segment-col", "value", allow_duplicate=True),
    Output("dd-segment-val", "options", allow_duplicate=True),
    Output("dd-segment-val", "value", allow_duplicate=True),
    Output("collapse-segment", "is_open", allow_duplicate=True),
    Output("profile-status", "children"),
    Output("load-status", "children", allow_duplicate=True),
    Output("collapse-profile-actions", "is_open", allow_duplicate=True),
    # Bağlantı bilgileri
    Output("input-sql-server", "value", allow_duplicate=True),
    Output("input-sql-database", "value", allow_duplicate=True),
    Output("dd-sql-driver", "value", allow_duplicate=True),
    Output("input-table-1", "value", allow_duplicate=True),
    Output("input-table-2", "value", allow_duplicate=True),
    Output("input-table-3", "value", allow_duplicate=True),
    Output("input-sql-jk-1", "value", allow_duplicate=True),
    Output("input-sql-jk-2", "value", allow_duplicate=True),
    Output("input-sql-jk-3", "value", allow_duplicate=True),
    Output("store-profile-loaded", "data", allow_duplicate=True),
    # Profil değişince playground model çıktısını ve sonuç sinyalini temizle
    Output("pg-model-output", "children", allow_duplicate=True),
    Output("store-model-signal", "data", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Input("btn-profile-load", "n_clicks"),
    State("dd-profile", "value"),
    prevent_initial_call=True,
)
def load_profile_cb(_, profile_name):
    _N_OUTPUTS = 29
    no = dash.no_update
    if not profile_name:
        return (no,) * _N_OUTPUTS

    try:
        new_key, config, expert_exclude, df = _load_profile(profile_name)
    except Exception as e:
        out = [no] * _N_OUTPUTS
        out[13] = dbc.Alert(f"Yükleme hatası: {e}", color="danger", style=_ALERT_STYLE)
        return tuple(out)

    if df is None:
        out = [no] * _N_OUTPUTS
        out[13] = dbc.Alert("Profilde veri dosyası bulunamadı.", color="danger", style=_ALERT_STYLE)
        return tuple(out)

    # Dropdown seçeneklerini oluştur
    all_opts = [{"label": f"{c}  [{df[c].dtype}]", "value": c} for c in df.columns]

    n_rows = len(df)
    n_cols = df.shape[1]

    # Segment filtresi varsa filtrelenmiş satır sayısını da göster
    from utils.helpers import apply_segment_filter
    seg_col_cfg = config.get("segment_col")
    seg_val_cfg = config.get("segment_val")
    if seg_col_cfg and seg_val_cfg and seg_col_cfg in df.columns:
        df_filtered = apply_segment_filter(df, seg_col_cfg, seg_val_cfg)
        n_filtered = len(df_filtered)
        if n_filtered != n_rows:
            row_info = f"{n_filtered:,} satır (toplam {n_rows:,})"
        else:
            row_info = f"{n_rows:,} satır"
    else:
        row_info = f"{n_rows:,} satır"

    profile_msg = dbc.Alert(
        f"✓ '{profile_name}' yüklendi  ·  {row_info}  ·  {n_cols} kolon",
        color="success", style=_ALERT_STYLE,
    )
    load_msg = dbc.Alert(
        f"{row_info}  ·  {n_cols} kolon  ·  Profilden yüklendi",
        color="info", style=_ALERT_STYLE,
    )

    # Segment değeri
    seg_col = config.get("segment_col")
    seg_val = config.get("segment_val")
    if seg_col and seg_col in df.columns:
        unique_vals = sorted(df[seg_col].dropna().astype(str).unique().tolist())
        seg_opts = [{"label": "Tümü", "value": "Tümü"}] + [
            {"label": v, "value": v} for v in unique_vals
        ]
        seg_dropdown_val = seg_val if seg_val else ["Tümü"]
        seg_open = True
    else:
        seg_opts = []
        seg_dropdown_val = ["Tümü"]
        seg_open = False

    # Bağlantı bilgileri
    meta_path = _PROFILES_DIR / profile_name / "meta.json"
    conn = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        conn = meta.get("connection", {})
    tables = conn.get("tables", [])
    jkeys = conn.get("join_keys", ["", "", ""])

    return (
        new_key,                                        # store-key
        config,                                         # store-config
        expert_exclude,                                 # store-expert-exclude
        True,                                           # collapse-config is_open
        all_opts,                                       # dd-target-col options
        config.get("target_col"),                       # dd-target-col value
        all_opts,                                       # dd-date-col options
        config.get("date_col"),                         # dd-date-col value
        all_opts,                                       # dd-segment-col options
        seg_col,                                        # dd-segment-col value
        seg_opts,                                       # dd-segment-val options
        seg_dropdown_val,                               # dd-segment-val value
        seg_open,                                       # collapse-segment is_open
        profile_msg,                                    # profile-status
        load_msg,                                       # load-status
        True,                                           # collapse-profile-actions
        conn.get("server", ""),                         # input-sql-server
        conn.get("database", ""),                       # input-sql-database
        conn.get("driver", ""),                         # dd-sql-driver
        tables[0] if len(tables) > 0 else "",           # input-table-1
        tables[1] if len(tables) > 1 else "",           # input-table-2
        tables[2] if len(tables) > 2 else "",           # input-table-3
        jkeys[0] if len(jkeys) > 0 else "",             # input-sql-jk-1
        jkeys[1] if len(jkeys) > 1 else "",             # input-sql-jk-2
        jkeys[2] if len(jkeys) > 2 else "",             # input-sql-jk-3
        profile_name,                                   # store-profile-loaded
        "",                                             # pg-model-output (temizle)
        None,                                           # store-model-signal (temizle)
        None,                                           # store-loaded-model-index (temizle)
    )


# ── Callback: Profil Sil — modal aç + profil listesi doldur ─────────────────
@app.callback(
    Output("modal-profile-delete", "is_open", allow_duplicate=True),
    Output("dd-profile-delete", "options"),
    Output("dd-profile-delete", "value"),
    Output("delete-confirm-area", "children", allow_duplicate=True),
    Output("btn-profile-delete-confirm", "style", allow_duplicate=True),
    Input("btn-profile-delete", "n_clicks"),
    prevent_initial_call=True,
)
def open_delete_modal(_):
    return True, _list_profiles(), None, "", {"display": "none"}


# ── Callback: Profil seçilince onay butonu göster ───────────────────────────
@app.callback(
    Output("delete-confirm-area", "children"),
    Output("btn-profile-delete-confirm", "style"),
    Input("dd-profile-delete", "value"),
    prevent_initial_call=True,
)
def show_delete_confirm(profile_name):
    if not profile_name:
        return "", {"display": "none"}
    return (
        dbc.Alert(
            f"'{profile_name}' silinecek. Bu işlem geri alınamaz.",
            color="warning", style=_ALERT_STYLE,
        ),
        {},  # visible
    )


# ── Callback: Silme onayı ──────────────────────────────────────────────────
@app.callback(
    Output("profile-status", "children", allow_duplicate=True),
    Output("dd-profile", "options", allow_duplicate=True),
    Output("dd-profile", "value", allow_duplicate=True),
    Output("modal-profile-delete", "is_open"),
    Input("btn-profile-delete-confirm", "n_clicks"),
    State("dd-profile-delete", "value"),
    prevent_initial_call=True,
)
def confirm_delete_profile_cb(_, profile_name):
    if not profile_name:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    try:
        _delete_profile(profile_name)
        return (
            dbc.Alert(f"'{profile_name}' silindi.", color="info", style=_ALERT_STYLE),
            _list_profiles(),
            None,
            False,  # modal kapat
        )
    except Exception as e:
        return (
            dbc.Alert(f"Silme hatası: {e}", color="danger", style=_ALERT_STYLE),
            dash.no_update, dash.no_update, True,
        )


# ── Callback: Config onaylandığında Profil Kaydet/Sil butonlarını göster ────
@app.callback(
    Output("collapse-profile-actions", "is_open"),
    Input("store-config", "data"),
)
def show_profile_actions(config):
    return bool(config and config.get("target_col"))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL KAYDET / YÜKLE / SİL
# ══════════════════════════════════════════════════════════════════════════════

def _list_saved_models(profile_name: str) -> list[dict]:
    """Bir profildeki kayıtlı modellerin dropdown listesini döndür."""
    if not profile_name:
        return []
    meta_path = _PROFILES_DIR / profile_name / "meta.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        models = meta.get("saved_models", [])
        return [{"label": m["name"], "value": i} for i, m in enumerate(models)]
    except Exception:
        return []


def _get_saved_models(profile_name: str) -> list[dict]:
    """Profildeki model listesini ham olarak döndür."""
    if not profile_name:
        return []
    meta_path = _PROFILES_DIR / profile_name / "meta.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("saved_models", [])
    except Exception:
        return []


def _save_model_to_profile(profile_name: str, model_entry: dict,
                            overwrite_index: int | None = None):
    """Modeli profil meta.json'a ekle veya mevcut modelin üstüne yaz."""
    meta_path = _PROFILES_DIR / profile_name / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "saved_models" not in meta:
        meta["saved_models"] = []
    if overwrite_index is not None and 0 <= overwrite_index < len(meta["saved_models"]):
        meta["saved_models"][overwrite_index] = model_entry
    else:
        meta["saved_models"].append(model_entry)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _delete_model_from_profile(profile_name: str, model_index: int):
    """Profildeki belirli bir modeli sil."""
    meta_path = _PROFILES_DIR / profile_name / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = meta.get("saved_models", [])
    if 0 <= model_index < len(models):
        models.pop(model_index)
        meta["saved_models"] = models
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ── Callback: Profil yüklendiğinde model panelini göster + listeyi doldur ────
@app.callback(
    Output("collapse-model-actions", "is_open"),
    Output("dd-saved-models", "options"),
    Output("dd-saved-models", "value"),
    Input("store-profile-loaded", "data"),
    prevent_initial_call=True,
)
def show_model_panel(profile_name):
    if not profile_name:
        return False, [], None
    return True, _list_saved_models(profile_name), None


# ── Yardımcı: Model entry oluştur ────────────────────────────────────────────
def _build_model_entry(profile_name, model_vars, model_type, test_size,
                       c_val, thr_method, thr_val, pg_target,
                       split_method, split_date, key):
    """Kayıt için model_entry dict oluştur."""
    from datetime import datetime
    algo_labels = {"lr": "LR", "lgbm": "LGBM", "xgb": "XGB", "rf": "RF"}
    algo_short = algo_labels.get(model_type, model_type or "?")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    auto_name = f"{algo_short} — {len(model_vars)} değişken — {now}"

    # Model özeti — cache'den al
    model_note = ""
    if key:
        cache_key = f"{key}_model_results"
        if cache_key in _SERVER_STORE:
            model_note = _SERVER_STORE[cache_key].get("model_note", "")

    return {
        "name": auto_name,
        "saved_at": now,
        "model_note": model_note,
        "params": {
            "model_vars": model_vars,
            "model_type": model_type,
            "test_size": test_size,
            "c_val": c_val,
            "threshold_method": thr_method,
            "threshold_val": thr_val,
            "pg_target": pg_target,
            "split_method": split_method,
            "split_date": split_date,
        },
    }


# ── Callback: Model Kaydet (yeni veya üstüne yazma onayı) ───────────────────
@app.callback(
    Output("model-save-status", "children"),
    Output("dd-saved-models", "options", allow_duplicate=True),
    Output("dd-saved-models", "value", allow_duplicate=True),
    Output("modal-model-overwrite", "is_open", allow_duplicate=True),
    Output("modal-overwrite-body", "children", allow_duplicate=True),
    Input("btn-model-save", "n_clicks"),
    State("store-profile-loaded", "data"),
    State("store-pg-model-vars", "data"),
    State("pg-model-type", "value"),
    State("pg-test-size", "value"),
    State("pg-c-value", "value"),
    State("pg-threshold-method", "value"),
    State("pg-threshold-val", "value"),
    State("pg-target-col", "value"),
    State("pg-split-method", "value"),
    State("pg-split-date", "value"),
    State("store-key", "data"),
    State("store-loaded-model-index", "data"),
    prevent_initial_call=True,
)
def save_model_cb(_, profile_name, model_vars, model_type, test_size,
                  c_val, thr_method, thr_val, pg_target, split_method,
                  split_date, key, loaded_idx):
    no = dash.no_update
    if not profile_name:
        return (
            dbc.Alert("Önce profil kaydedin.", color="warning", style=_ALERT_STYLE),
            no, no, False, "",
        )
    if not model_vars:
        return (
            dbc.Alert("Model değişkenleri boş.", color="warning", style=_ALERT_STYLE),
            no, no, False, "",
        )

    # Yüklü model varsa → onay modalı aç
    if loaded_idx is not None:
        models = _get_saved_models(profile_name)
        if 0 <= loaded_idx < len(models):
            old_name = models[loaded_idx]["name"]
            return (
                no, no, no, True,
                html.Div([
                    html.P([
                        "Daha önce yüklemiş olduğunuz ",
                        html.Strong(old_name),
                        " modelinin üstüne yazılacak.",
                    ]),
                    html.P("Bu işlem geri alınamaz. Onaylıyor musunuz?",
                           style={"color": "#f59e0b"}),
                ]),
            )

    # Yeni model → direkt kaydet
    entry = _build_model_entry(profile_name, model_vars, model_type, test_size,
                               c_val, thr_method, thr_val, pg_target,
                               split_method, split_date, key)
    try:
        _save_model_to_profile(profile_name, entry)
        new_opts = _list_saved_models(profile_name)
        return (
            dbc.Alert(f"✓ Model kaydedildi: {entry['name']}", color="success",
                      style=_ALERT_STYLE),
            new_opts,
            len(new_opts) - 1,
            False, "",
        )
    except Exception as e:
        return (
            dbc.Alert(f"Kaydetme hatası: {e}", color="danger", style=_ALERT_STYLE),
            no, no, False, "",
        )


# ── Callback: Üstüne Kaydet — onay sonrası ──────────────────────────────────
@app.callback(
    Output("model-save-status", "children", allow_duplicate=True),
    Output("dd-saved-models", "options", allow_duplicate=True),
    Output("dd-saved-models", "value", allow_duplicate=True),
    Output("modal-model-overwrite", "is_open", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Input("btn-model-overwrite-confirm", "n_clicks"),
    State("store-profile-loaded", "data"),
    State("store-pg-model-vars", "data"),
    State("pg-model-type", "value"),
    State("pg-test-size", "value"),
    State("pg-c-value", "value"),
    State("pg-threshold-method", "value"),
    State("pg-threshold-val", "value"),
    State("pg-target-col", "value"),
    State("pg-split-method", "value"),
    State("pg-split-date", "value"),
    State("store-key", "data"),
    State("store-loaded-model-index", "data"),
    prevent_initial_call=True,
)
def overwrite_model_cb(_, profile_name, model_vars, model_type, test_size,
                       c_val, thr_method, thr_val, pg_target,
                       split_method, split_date, key, loaded_idx):
    no = dash.no_update
    if not profile_name or loaded_idx is None or not model_vars:
        return no, no, no, False, no

    entry = _build_model_entry(profile_name, model_vars, model_type, test_size,
                               c_val, thr_method, thr_val, pg_target,
                               split_method, split_date, key)
    try:
        _save_model_to_profile(profile_name, entry, overwrite_index=loaded_idx)
        new_opts = _list_saved_models(profile_name)
        return (
            dbc.Alert(f"✓ Model güncellendi: {entry['name']}", color="success",
                      style=_ALERT_STYLE),
            new_opts,
            loaded_idx,
            False,
            loaded_idx,  # index'i koru
        )
    except Exception as e:
        return (
            dbc.Alert(f"Kaydetme hatası: {e}", color="danger", style=_ALERT_STYLE),
            no, no, False, no,
        )


# ── Callback: Vazgeç — modalı kapat ─────────────────────────────────────────
@app.callback(
    Output("modal-model-overwrite", "is_open", allow_duplicate=True),
    Input("btn-model-overwrite-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_overwrite(_):
    return False


# ── Callback: Model Yükle → Playground parametrelerini set et ─────────────────
@app.callback(
    Output("pg-var-dropdown", "value", allow_duplicate=True),
    Output("store-pg-model-vars", "data", allow_duplicate=True),
    Output("pg-model-type", "value", allow_duplicate=True),
    Output("pg-c-value", "value", allow_duplicate=True),
    Output("pg-threshold-method", "value", allow_duplicate=True),
    Output("pg-threshold-val", "value", allow_duplicate=True),
    Output("pg-target-col", "value", allow_duplicate=True),
    Output("pg-split-method", "value", allow_duplicate=True),
    Output("pg-split-date", "value", allow_duplicate=True),
    Output("btn-pg-build", "n_clicks", allow_duplicate=True),
    Output("model-save-status", "children", allow_duplicate=True),
    Output("store-pending-note", "data", allow_duplicate=True),
    Output("store-loaded-model-index", "data", allow_duplicate=True),
    Input("btn-model-load", "n_clicks"),
    State("store-profile-loaded", "data"),
    State("dd-saved-models", "value"),
    prevent_initial_call=True,
)
def load_model_cb(_, profile_name, model_index):
    no = dash.no_update
    _N = 13
    if not profile_name or model_index is None:
        return (no,) * _N

    models = _get_saved_models(profile_name)
    if model_index < 0 or model_index >= len(models):
        out = [no] * _N
        out[10] = dbc.Alert("Model bulunamadı.", color="warning", style=_ALERT_STYLE)
        return tuple(out)

    p = models[model_index]["params"]
    model_name = models[model_index]["name"]
    model_note = models[model_index].get("model_note", "")

    mvars = p.get("model_vars", [])
    return (
        mvars,                             # pg-var-dropdown
        mvars,                             # store-pg-model-vars
        p.get("model_type", "lr"),         # pg-model-type
        p.get("c_val", 1.0),              # pg-c-value
        p.get("threshold_method", "fixed"),# pg-threshold-method
        p.get("threshold_val", 0.50),     # pg-threshold-val
        p.get("pg_target"),               # pg-target-col
        p.get("split_method", "random"),  # pg-split-method
        p.get("split_date", ""),          # pg-split-date
        1,                                 # btn-pg-build n_clicks → tetikle
        dbc.Alert(f"✓ '{model_name}' yüklendi, model kuruluyor…",
                  color="info", style=_ALERT_STYLE),
        model_note,                        # store-pending-note
        model_index,                       # store-loaded-model-index
    )


# ── Callback: Model Sil — onay sor ───────────────────────────────────────────
@app.callback(
    Output("model-save-status", "children", allow_duplicate=True),
    Output("btn-model-delete-confirm", "style"),
    Input("btn-model-delete", "n_clicks"),
    State("store-profile-loaded", "data"),
    State("dd-saved-models", "value"),
    prevent_initial_call=True,
)
def ask_delete_model_cb(_, profile_name, model_index):
    hide = {"display": "none", "fontSize": "0.75rem"}
    if not profile_name or model_index is None:
        return dbc.Alert("Silinecek model seçin.", color="warning", style=_ALERT_STYLE), hide

    models = _get_saved_models(profile_name)
    if model_index < 0 or model_index >= len(models):
        return dbc.Alert("Model bulunamadı.", color="warning", style=_ALERT_STYLE), hide

    model_name = models[model_index]["name"]
    return (
        dbc.Alert(f"'{model_name}' silinecek. Emin misiniz?",
                  color="warning", style=_ALERT_STYLE),
        {"fontSize": "0.75rem"},  # göster
    )


# ── Callback: Model Sil — onay sonrası ──────────────────────────────────────
@app.callback(
    Output("model-save-status", "children", allow_duplicate=True),
    Output("dd-saved-models", "options", allow_duplicate=True),
    Output("dd-saved-models", "value", allow_duplicate=True),
    Output("btn-model-delete-confirm", "style", allow_duplicate=True),
    Input("btn-model-delete-confirm", "n_clicks"),
    State("store-profile-loaded", "data"),
    State("dd-saved-models", "value"),
    prevent_initial_call=True,
)
def confirm_delete_model_cb(_, profile_name, model_index):
    no = dash.no_update
    hide = {"display": "none", "fontSize": "0.75rem"}
    if not profile_name or model_index is None:
        return no, no, no, hide

    models = _get_saved_models(profile_name)
    if model_index < 0 or model_index >= len(models):
        return (dbc.Alert("Model bulunamadı.", color="warning", style=_ALERT_STYLE),
                no, no, hide)

    model_name = models[model_index]["name"]
    try:
        _delete_model_from_profile(profile_name, model_index)
        return (
            dbc.Alert(f"'{model_name}' silindi.", color="info", style=_ALERT_STYLE),
            _list_saved_models(profile_name),
            None,
            hide,
        )
    except Exception as e:
        return (
            dbc.Alert(f"Silme hatası: {e}", color="danger", style=_ALERT_STYLE),
            no, no, hide,
        )
