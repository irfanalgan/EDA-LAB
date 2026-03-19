##############################################################################
# KAPSAMLI FONKSİYONELİTE TESTİ
##############################################################################
import sys, traceback, time, uuid, io, pickle
import pandas as pd
import numpy as np

results = []
def test(name, fn):
    t0 = time.perf_counter()
    try:
        r = fn()
        dt = round(time.perf_counter() - t0, 2)
        results.append(("OK", name, dt, str(r)[:120] if r else ""))
        print(f"  OK  {name} ({dt}s)")
    except Exception as e:
        dt = round(time.perf_counter() - t0, 2)
        tb = traceback.format_exc()
        results.append(("FAIL", name, dt, str(e)[:200]))
        print(f"  FAIL {name} ({dt}s): {str(e)[:150]}")
        for line in tb.strip().split("\n")[-4:]:
            print(f"       {line}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
print("=== 1. DATA LOADING ===")
from data.loader import get_data_from_sql
from utils.helpers import coerce_numeric_columns
from server_state import _SERVER_STORE

df = None
key = None

def load_data():
    global df, key
    df = get_data_from_sql("dbo.MODEL_DATA", server="IRPHAN", database="MASTER",
                           driver="ODBC Driver 17 for SQL Server")
    df, converted = coerce_numeric_columns(df)
    key = str(uuid.uuid4())
    _SERVER_STORE[key] = df
    _SERVER_STORE[f"{key}_quality"] = {"converted": converted}
    return f"{len(df):,} rows x {df.shape[1]} cols"
test("SQL veri cekme", load_data)

TARGET = "tgt__dpd30_in_next90"
DATE_COL = "as_of_date"
SEG_COL = "prof__age_band"
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
model_vars = num_cols[:5]
print(f"  model_vars = {model_vars}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. PROFILING
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. PROFILING ===")
from modules.profiling import compute_profile, profile_summary

def t_profiling():
    prof = compute_profile(df)
    return f"{len(prof)} rows"
test("compute_profile", t_profiling)

def t_profile_summary():
    prof = compute_profile(df)
    s = profile_summary(prof, len(df))
    return s
test("profile_summary", t_profile_summary)

# ═══════════════════════════════════════════════════════════════════════════
# 3. TARGET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. TARGET ANALYSIS ===")
from modules.target_analysis import compute_target_stats, compute_target_over_time

def t_target_stats():
    r = compute_target_stats(df, TARGET)
    return f"keys={list(r.keys()) if isinstance(r,dict) else type(r)}"
test("compute_target_stats", t_target_stats)

def t_target_time():
    r = compute_target_over_time(df, TARGET, DATE_COL)
    return f"type={type(r).__name__}, len={len(r) if r is not None else 0}"
test("compute_target_over_time", t_target_time)

# ═══════════════════════════════════════════════════════════════════════════
# 4. SCREENING
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. SCREENING ===")
from modules.screening import screen_columns

def t_screening():
    kept, details = screen_columns(df, TARGET, date_col=DATE_COL, segment_col=SEG_COL)
    return f"{len(kept)} vars kept, {len(details)} detail rows"
test("screen_variables", t_screening)

# ═══════════════════════════════════════════════════════════════════════════
# 5. CORRELATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. CORRELATION ===")
from modules.correlation import compute_correlation_matrix, find_high_corr_pairs, compute_vif

def t_corr():
    cols = num_cols[:10]
    corr_matrix = compute_correlation_matrix(df, cols)
    pairs = find_high_corr_pairs(corr_matrix, threshold=0.7)
    return f"corr_matrix={corr_matrix.shape}, high_pairs={len(pairs)}"
test("compute_correlation_groups", t_corr)

# ═══════════════════════════════════════════════════════════════════════════
# 6. DEEP DIVE (WoE)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 6. DEEP DIVE (WoE) ===")
from modules.deep_dive import get_woe_encoder, get_woe_detail

def t_woe_encoder():
    col = num_cols[0]
    woe_s, iv, ok, optb = get_woe_encoder(df, col, TARGET)
    return f"col={col}, iv={iv:.4f}, ok={ok}, optb_type={type(optb).__name__}"
test("get_woe_encoder", t_woe_encoder)

def t_woe_detail():
    col = num_cols[0]
    woe_df, iv_total, bin_edges, _optb = get_woe_detail(df, col, TARGET)
    return f"woe_rows={len(woe_df)}, iv={iv_total:.4f}, bin_edges_type={type(bin_edges).__name__}"
test("get_woe_detail", t_woe_detail)

# ═══════════════════════════════════════════════════════════════════════════
# 7. WoE DATASET BUILD
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 7. WoE DATASET BUILD ===")
from utils.chart_helpers import _build_woe_dataset

def t_woe_build():
    woe_df, failed, opt_dict = _build_woe_dataset(df, TARGET, model_vars)
    return f"shape={woe_df.shape}, failed={failed}, opt_keys={len(opt_dict)}"
test("_build_woe_dataset", t_woe_build)

# ═══════════════════════════════════════════════════════════════════════════
# 8. MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 8. MODEL FITTING ===")
from callbacks.playground import SmLogitWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from utils.helpers import apply_segment_filter

df_active = apply_segment_filter(df, SEG_COL, None)
y = df_active[TARGET]

# OOT split
dates = pd.to_datetime(df_active[DATE_COL], errors="coerce")
date_periods = dates.dt.to_period("M").astype(str)
unique_months = sorted(date_periods.dropna().unique())
oot_date = unique_months[int(len(unique_months)*0.8)] if len(unique_months) > 3 else None

oot_mask = date_periods >= oot_date if oot_date else pd.Series(False, index=df_active.index)
non_oot = ~oot_mask

from sklearn.model_selection import train_test_split
non_oot_idx = df_active[non_oot].index.tolist()
train_idx, test_idx = train_test_split(non_oot_idx, test_size=0.2, random_state=42)

train_mask = pd.Series(False, index=df_active.index)
test_mask = pd.Series(False, index=df_active.index)
train_mask.iloc[train_idx] = True
test_mask.iloc[test_idx] = True

X_raw = pd.get_dummies(df_active[model_vars].copy(), drop_first=True)
print(f"  Split: Train={train_mask.sum()}, Test={test_mask.sum()}, OOT={oot_mask.sum()}")

# LR HAM
def t_lr_raw():
    import statsmodels.api as sm
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_mask])
    y_train = y[train_mask]
    X_const = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_const).fit(disp=0, maxiter=100)
    y_prob = model.predict(X_const)
    auc = roc_auc_score(y_train, y_prob)
    return f"LR Raw Train Gini={2*auc-1:.4f}"
test("LR Ham model", t_lr_raw)

# WoE MODEL
def t_lr_woe():
    import statsmodels.api as sm
    woe_df, failed, opt_dict = _build_woe_dataset(df_active, TARGET, model_vars)
    woe_cols = [f"{v}_woe" for v in model_vars if f"{v}_woe" in woe_df.columns]
    if not woe_cols:
        return "No WoE cols"
    X_woe = woe_df[woe_cols]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_woe[train_mask])
    y_train = y[train_mask]
    X_const = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_const).fit(disp=0, maxiter=100)
    y_prob = model.predict(X_const)
    auc = roc_auc_score(y_train, y_prob)
    return f"WoE Train Gini={2*auc-1:.4f}, opt={len(opt_dict)}, woe_cols={len(woe_cols)}"
test("LR WoE model", t_lr_woe)

# LightGBM
def t_lgbm():
    try:
        import lightgbm as lgb
    except ImportError:
        return "lightgbm not installed"
    model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1, random_state=42)
    model.fit(X_raw[train_mask], y[train_mask])
    y_prob = model.predict_proba(X_raw[train_mask])[:, 1]
    auc = roc_auc_score(y[train_mask], y_prob)
    return f"LGBM Train Gini={2*auc-1:.4f}"
test("LightGBM model", t_lgbm)

# ═══════════════════════════════════════════════════════════════════════════
# 9. PICKLE EXPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 9. PICKLE EXPORT ===")

def t_pickle_model():
    import statsmodels.api as sm
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_mask])
    y_train = y[train_mask]
    X_const = sm.add_constant(X_train)
    sm_model = sm.Logit(y_train, X_const).fit(disp=0, maxiter=100)
    wrapper = SmLogitWrapper(sm_model)
    bundle = {"algo": "lr", "tab": "raw", "model": wrapper, "scaler": scaler,
              "model_vars": model_vars, "opt_thr": 0.5}
    buf = io.BytesIO()
    pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    loaded = pickle.loads(buf.getvalue())
    return f"Pickle OK, size={len(buf.getvalue())} bytes, type={type(loaded['model']).__name__}"
test("Model pickle serialize/deserialize", t_pickle_model)

def t_pickle_opt():
    _, _, opt_dict = _build_woe_dataset(df_active, TARGET, model_vars)
    buf = io.BytesIO()
    pickle.dump(opt_dict, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    loaded = pickle.loads(buf.getvalue())
    return f"OPT Pickle OK, {len(loaded)} binnings, size={len(buf.getvalue())} bytes"
test("OPT pickle serialize/deserialize", t_pickle_opt)

# ═══════════════════════════════════════════════════════════════════════════
# 10. SQL PUSH (build only, no write)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 10. SQL PUSH (build only) ===")

def t_sql_push_build():
    out_cols = model_vars + [TARGET, DATE_COL, SEG_COL]
    out_df = df_active[out_cols].copy()
    out_df["TRAIN_OOT_FLAG"] = "TRAIN"
    out_df.loc[test_mask, "TRAIN_OOT_FLAG"] = "TEST"
    out_df.loc[oot_mask, "TRAIN_OOT_FLAG"] = "OOT"
    return f"DF ready: {out_df.shape}, flags: {out_df['TRAIN_OOT_FLAG'].value_counts().to_dict()}"
test("SQL push DataFrame build", t_sql_push_build)

# ═══════════════════════════════════════════════════════════════════════════
# 11. EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 11. EXCEL EXPORT ===")

def t_excel():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Test"
    ws.cell(row=1, column=1, value="Test")
    buf = io.BytesIO()
    wb.save(buf)
    return f"Excel OK, size={len(buf.getvalue())} bytes"
test("Excel workbook create", t_excel)

# ═══════════════════════════════════════════════════════════════════════════
# 12. SHAP
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 12. SHAP ===")

def t_shap():
    try:
        import shap
        import lightgbm as lgb
    except ImportError:
        return "shap/lightgbm not installed"
    model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1)
    model.fit(X_raw[train_mask], y[train_mask])
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_raw[train_mask].head(100))
    return f"SHAP OK, shape={np.array(shap_vals).shape}"
test("SHAP TreeExplainer", t_shap)

# ═══════════════════════════════════════════════════════════════════════════
# 13. PROFILE SAVE/LOAD
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 13. PROFILE SAVE/LOAD ===")
from callbacks.profile import _save_profile, _load_profile, _list_profiles
import shutil

def t_profile_save():
    conn_info = {"server": "IRPHAN", "database": "MASTER",
                 "driver": "ODBC Driver 17 for SQL Server",
                 "tables": ["dbo.MODEL_DATA"], "join_keys": []}
    config_test = {"target_col": TARGET, "date_col": DATE_COL, "segment_col": SEG_COL}
    _save_profile("__test_profile__", key, config_test, [], connection_info=conn_info)
    return "Profile saved"
test("Profile save", t_profile_save)

def t_profile_load():
    new_key, cfg, excl, loaded_df = _load_profile("__test_profile__")
    return f"Loaded: key={new_key[:8]}..., shape={loaded_df.shape}, cfg_keys={list(cfg.keys())}"
test("Profile load", t_profile_load)

def t_profile_list():
    profiles = _list_profiles()
    return f"{len(profiles)} profiles found"
test("Profile list", t_profile_list)

# Cleanup
try:
    shutil.rmtree("profiles/__test_profile__", ignore_errors=True)
except:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# 14. PRECOMPUTE
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 14. PRECOMPUTE ===")
from callbacks.precompute import _run_precompute_background

def t_precompute():
    prog_key = f"__test_prog_{uuid.uuid4().hex[:8]}"
    _run_precompute_background(prog_key, key, TARGET, DATE_COL, SEG_COL, None)
    cached_keys = [k for k in _SERVER_STORE if k.startswith(key) and k != key]
    return f"Precomputed {len(cached_keys)} cache keys"
test("Precompute pipeline", t_precompute)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SONUC RAPORU")
print("="*70)
ok_count = sum(1 for r in results if r[0] == "OK")
fail_count = sum(1 for r in results if r[0] == "FAIL")
print(f"Toplam: {len(results)} test  |  OK: {ok_count}  |  FAIL: {fail_count}")
print()
for status, name, dt, detail in results:
    icon = "v" if status == "OK" else "X"
    line = f"  [{icon}] {name} ({dt}s)"
    if status == "FAIL":
        line += f"\n      --> {detail}"
    print(line)
