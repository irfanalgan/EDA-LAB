from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from modules.deep_dive import get_woe_encoder, calc_total_iv


# ── Tab açıklama kutusu yardımcısı ────────────────────────────────────────────
def _tab_info(title: str, subtitle: str, body: str, color: str = "#4F8EF7") -> html.Div:
    """Her tab'ın üstünde gösterilen açıklama kartı."""
    return html.Div([
        html.Div([
            html.Span(title, style={"color": "#E8EAF0", "fontWeight": "700",
                                    "fontSize": "0.82rem"}),
            html.Span(f"  ·  {subtitle}",
                      style={"color": "#7e8fa4", "fontSize": "0.78rem"}),
        ], style={"marginBottom": "0.3rem"}),
        html.Div(body, style={"color": "#7e8fa4", "fontSize": "0.76rem",
                               "lineHeight": "1.55"}),
    ], style={
        "backgroundColor": "#111827",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "4px",
        "padding": "0.6rem 1rem",
        "marginBottom": "1.1rem",
    })


# ── Ortak grafik teması ───────────────────────────────────────────────────────
_PLOT_LAYOUT = dict(
    paper_bgcolor="#161C27",
    plot_bgcolor="#0E1117",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)

_AXIS_STYLE = dict(gridcolor="#232d3f", linecolor="#232d3f", zerolinecolor="#232d3f")

# ── DataTable ortak stil sabiti ────────────────────────────────────────────────
_TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "#111827", "color": "#a8b2c2",
                  "fontWeight": "700", "fontSize": "0.72rem",
                  "border": "1px solid #2d3a4f", "textTransform": "uppercase"},
    style_data={"backgroundColor": "#161C27", "color": "#c8cdd8",
                "fontSize": "0.82rem", "border": "1px solid #232d3f"},
    style_cell={"padding": "0.4rem 0.65rem"},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1a2035"}],
)


# ── WoE Dataset Builder ───────────────────────────────────────────────────────
def _build_woe_dataset(df: pd.DataFrame, target: str, cols: list) -> tuple:
    """
    Her kolon için WoE encode eder; kolon adı '{col}_woe' olarak kaydedilir.
    Returns: (woe_df, failed_cols, opt_dict)
    opt_dict: {col_name: OptimalBinning object} — pickle için saklanır.
    """
    result = {}
    failed = []
    opt_dict = {}
    for col in cols:
        try:
            woe_series, _, ok, optb = get_woe_encoder(df, col, target)
            if ok:
                result[f"{col}_woe"] = woe_series.values
                if optb is not None:
                    opt_dict[col] = optb
            else:
                failed.append(col)
        except Exception:
            failed.append(col)
    woe_df = pd.DataFrame(result, index=df.index)
    return woe_df, failed, opt_dict


def build_woe_datasets(df_train: pd.DataFrame, df_test, df_oot,
                       target: str, cols: list, max_bins: int = 4) -> dict:
    """
    Train üzerinde OptimalBinning fit → tek döngüde IV, optb, bin_edges,
    WoE transform (train + test + oot) üretir.

    Returns dict:
        "train_woe":  pd.DataFrame  — train WoE değerleri
        "test_woe":   pd.DataFrame | None
        "oot_woe":    pd.DataFrame | None
        "iv_df":      pd.DataFrame  — Değişken, IV, Eksik %
        "optb_dict":  dict          — {col: OptimalBinning}
        "bins_dict":  dict          — {col: [bin_edges]}
        "iv_tables":  dict          — {col: binning_table DataFrame}
        "failed":     list          — encode edilemeyen kolonlar
    """
    from optbinning import OptimalBinning as _OB
    from modules.deep_dive import SPECIAL_VALUES, is_special_column

    train_woe_data = {}
    test_woe_data = {}
    oot_woe_data = {}
    iv_records = []
    optb_dict = {}
    bins_dict = {}
    iv_tables = {}
    failed = []

    # Eksik % → train + test
    if df_test is not None:
        _eksik_df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        _eksik_df = df_train

    y_train = pd.to_numeric(df_train[target], errors="coerce")
    _clean_mask = y_train.notna()

    for col in cols:
        if col == target:
            continue
        try:
            is_numeric = pd.api.types.is_numeric_dtype(df_train[col])

            # Fit on train
            _local = df_train.loc[_clean_mask, [col, target]].copy()
            _local[target] = y_train[_clean_mask].astype(int)

            _special_col = is_numeric and is_special_column(_local[col])
            _bins = 2 if _special_col else max_bins
            kwargs = dict(name=col, monotonic_trend="auto_asc_desc",
                          max_n_bins=_bins, solver="cp",
                          dtype="numerical" if is_numeric else "categorical")
            if is_numeric:
                kwargs["special_codes"] = list(SPECIAL_VALUES)

            optb = _OB(**kwargs)
            optb.fit(_local[col].values, _local[target].values)

            # IV table + IV value (special değerler ayrıştırılarak)
            bt = optb.binning_table.build(show_digits=8)
            iv = calc_total_iv(bt, _local[col].values, _local[target].values)
            iv_tables[col] = bt
            optb_dict[col] = optb

            # Bin edges
            if is_numeric and hasattr(optb, "splits") and optb.splits is not None:
                edges = list(optb.splits)
                edges.insert(0, -np.inf)
                edges.append(np.inf)
                bins_dict[col] = edges

            # WoE transform — train
            _tr_woe = optb.transform(df_train[col].values, metric="woe",
                                     metric_missing="empirical",
                                     metric_special="empirical")
            train_woe_data[col] = pd.Series(_tr_woe, index=df_train.index).fillna(0.0)

            # WoE transform — test
            if df_test is not None and len(df_test) > 0:
                _te_woe = optb.transform(df_test[col].values, metric="woe",
                                         metric_missing="empirical",
                                         metric_special="empirical")
                test_woe_data[col] = pd.Series(_te_woe, index=df_test.index).fillna(0.0)

            # WoE transform — oot
            if df_oot is not None and len(df_oot) > 0:
                _oot_woe = optb.transform(df_oot[col].values, metric="woe",
                                          metric_missing="empirical",
                                          metric_special="empirical")
                oot_woe_data[col] = pd.Series(_oot_woe, index=df_oot.index).fillna(0.0)

            # IV record
            eksik_pct = round(_eksik_df[col].isna().mean() * 100, 2)
            iv_records.append({"Değişken": col, "IV": round(iv, 4), "Eksik %": eksik_pct})

        except Exception:
            eksik_pct = round(_eksik_df[col].isna().mean() * 100, 2) if col in _eksik_df.columns else 0.0
            iv_records.append({"Değişken": col, "IV": 0.0, "Eksik %": eksik_pct})
            failed.append(col)

    # Build DataFrames
    df_train_woe = pd.DataFrame(train_woe_data, index=df_train.index)
    df_test_woe = pd.DataFrame(test_woe_data, index=df_test.index) if df_test is not None and test_woe_data else None
    df_oot_woe = pd.DataFrame(oot_woe_data, index=df_oot.index) if df_oot is not None and oot_woe_data else None

    iv_df = pd.DataFrame(iv_records).sort_values("IV", ascending=False).reset_index(drop=True)

    # Güç etiketi ekle
    def _iv_label(iv: float) -> str:
        if iv < 0.02:  return "Çok Zayıf"
        if iv < 0.10:  return "Zayıf"
        if iv < 0.30:  return "Orta"
        if iv < 0.50:  return "Güçlü"
        return "Şüpheli"
    iv_df["Güç"] = iv_df["IV"].apply(_iv_label)

    return {
        "train_woe": df_train_woe,
        "test_woe": df_test_woe,
        "oot_woe": df_oot_woe,
        "iv_df": iv_df,
        "optb_dict": optb_dict,
        "bins_dict": bins_dict,
        "iv_tables": iv_tables,
        "failed": failed,
    }


# ── PSI — Train WoE vs OOT WoE ───────────────────────────────────────────────
def calc_psi(base: np.ndarray, comp: np.ndarray,
             n_bins: int = 10, discrete: bool = True,
             detail: bool = False):
    """
    Population Stability Index — TEK KAYNAK.
    Tüm PSI hesaplamaları bu fonksiyondan geçer.

    discrete=True  → WoE değerleri (kesikli): her unique değere düşen oran karşılaştırılır.
    discrete=False → Ham değerler (sürekli): n_bins eşit aralıklı bin ile histogram karşılaştırması.
    detail=False   → float döndürür (toplam PSI).
    detail=True    → dict döndürür: {"psi": float, "rows": [{"Bin", "Baseline %", "Karşılaştırma %", "Δ (pp)", "PSI Katkı"}, ...]}
    """
    eps = 1e-4

    if discrete:
        # ── WoE: her unique değer bir bin ──────────────────────────────────
        all_vals = np.union1d(np.unique(base), np.unique(comp))
        if len(all_vals) < 2:
            return {"psi": 0.0, "rows": []} if detail else 0.0
        n_base, n_comp = len(base), len(comp)
        base_counts = {v: 0 for v in all_vals}
        comp_counts = {v: 0 for v in all_vals}
        for v in base:
            base_counts[v] = base_counts.get(v, 0) + 1
        for v in comp:
            comp_counts[v] = comp_counts.get(v, 0) + 1
        psi = 0.0
        rows = []
        for v in all_vals:
            b_pct = max(base_counts[v] / n_base, eps)
            c_pct = max(comp_counts[v] / n_comp, eps)
            contrib = (c_pct - b_pct) * np.log(c_pct / b_pct)
            psi += contrib
            if detail:
                rows.append({
                    "Bin": str(round(float(v), 4)),
                    "Baseline %": round(b_pct * 100, 2),
                    "Karşılaştırma %": round(c_pct * 100, 2),
                    "Δ (pp)": round((c_pct - b_pct) * 100, 2),
                    "PSI Katkı": round(float(contrib), 5),
                })
        if detail:
            return {"psi": float(psi), "rows": rows}
        return float(psi)
    else:
        # ── Ham değerler: linspace binning ─────────────────────────────────
        mn, mx = float(base.min()), float(base.max())
        if mn == mx:
            return {"psi": 0.0, "rows": []} if detail else 0.0
        bins = np.linspace(mn, mx, n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf
        b_hist = np.histogram(base, bins=bins)[0]
        c_hist = np.histogram(comp, bins=bins)[0]
        b_pct = b_hist / len(base)
        c_pct = c_hist / len(comp)
        b_pct = np.where(b_pct < eps, eps, b_pct)
        c_pct = np.where(c_pct < eps, eps, c_pct)
        contribs = (c_pct - b_pct) * np.log(c_pct / b_pct)
        psi_total = float(np.sum(contribs))
        if not detail:
            return psi_total
        rows = []
        for i in range(len(contribs)):
            lo = f"{bins[i]:.2f}" if not np.isinf(bins[i]) else "-∞"
            hi = f"{bins[i+1]:.2f}" if not np.isinf(bins[i+1]) else "∞"
            rows.append({
                "Bin": f"({lo}, {hi}]",
                "Baseline %": round(float(b_pct[i]) * 100, 2),
                "Karşılaştırma %": round(float(c_pct[i]) * 100, 2),
                "Δ (pp)": round(float(c_pct[i] - b_pct[i]) * 100, 2),
                "PSI Katkı": round(float(contribs[i]), 5),
            })
        return {"psi": psi_total, "rows": rows}


def psi_label(psi_val: float) -> str:
    if psi_val < 0.10:
        return "Stabil"
    if psi_val < 0.25:
        return "Hafif Kayma"
    return "Kritik Kayma"


def _iv_label(iv: float) -> str:
    if iv < 0.02:  return "Çok Zayıf"
    if iv < 0.10:  return "Zayıf"
    if iv < 0.30:  return "Orta"
    if iv < 0.50:  return "Güçlü"
    return "Şüpheli"


# ── Ortak WoE yardımcıları (precompute + refit paylaşır) ────────────────────
def format_bt(bt_raw, col_name=None, df_train=None, target=None):
    """Optbinning binning_table → uygulama formatına dönüştür."""
    data_rows = bt_raw[bt_raw["Bin"].astype(str).str.len() > 0]
    data_rows = data_rows[~data_rows["Bin"].isin(["Special", "Missing", "Totals"])]
    rows = []
    for _, r in data_rows.iterrows():
        total = int(r["Count"]); bad = int(r["Event"]); good = int(r["Non-event"])
        rows.append({
            "Bin": str(r["Bin"]), "Toplam": total, "Bad": bad, "Good": good,
            "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
            "WOE": round(float(r["WoE"]), 4),
            "IV Katkı": round(float(r["IV"]), 4),
        })
    # Special — her değer ayrı satır
    sr_sp = bt_raw[bt_raw["Bin"] == "Special"]
    if not sr_sp.empty and int(sr_sp.iloc[0]["Count"]) > 0:
        sp_woe = round(float(sr_sp.iloc[0]["WoE"]), 4)
        sp_iv  = float(sr_sp.iloc[0]["IV"])
        sp_n   = int(sr_sp.iloc[0]["Count"])
        sv_found = False
        if col_name and df_train is not None and col_name in df_train.columns:
            from modules.deep_dive import SPECIAL_VALUES
            y_tr = pd.to_numeric(df_train[target], errors="coerce")
            total_bad = int(y_tr.sum())
            total_good = len(y_tr) - total_bad
            for sv in sorted(SPECIAL_VALUES):
                sv_mask = df_train[col_name] == sv
                if not sv_mask.any():
                    continue
                sv_found = True
                n_sv = int(sv_mask.sum())
                bad_sv = int(y_tr[sv_mask].sum())
                good_sv = n_sv - bad_sv
                # Her special değer için ayrı WoE ve IV
                d_b = bad_sv / total_bad if total_bad > 0 else 0
                d_g = good_sv / total_good if total_good > 0 else 0
                sv_woe = round(float(np.log(d_b / d_g)), 4) if d_b > 0 and d_g > 0 else 0.0
                if d_b > 0 and d_g > 0:
                    _log_val = float(np.log(d_b / d_g))
                    sv_iv = float(f"{(d_b - d_g) * _log_val:.4f}")
                else:
                    sv_iv = 0.0
                rows.append({
                    "Bin": f"Special ({int(sv)})", "Toplam": n_sv,
                    "Bad": bad_sv, "Good": good_sv,
                    "Bad Rate %": round(bad_sv / n_sv * 100, 2) if n_sv > 0 else 0.0,
                    "WOE": sv_woe, "IV Katkı": sv_iv,
                })
        if not sv_found:
            r = sr_sp.iloc[0]
            rows.append({
                "Bin": "Special", "Toplam": int(r["Count"]),
                "Bad": int(r["Event"]), "Good": int(r["Non-event"]),
                "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
                "WOE": sp_woe, "IV Katkı": round(sp_iv, 4),
            })
    # Missing
    sr_ms = bt_raw[bt_raw["Bin"] == "Missing"]
    if not sr_ms.empty and int(sr_ms.iloc[0]["Count"]) > 0:
        r = sr_ms.iloc[0]
        rows.append({
            "Bin": "Eksik", "Toplam": int(r["Count"]),
            "Bad": int(r["Event"]), "Good": int(r["Non-event"]),
            "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
            "WOE": round(float(r["WoE"]), 4),
            "IV Katkı": round(float(r["IV"]), 4),
        })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    t_n = result["Toplam"].sum(); t_b = result["Bad"].sum()
    total_row = pd.DataFrame([{
        "Bin": "TOPLAM", "Toplam": int(t_n), "Bad": int(t_b),
        "Good": int(result["Good"].sum()),
        "Bad Rate %": round(t_b / t_n * 100, 2) if t_n > 0 else 0.0,
        "WOE": "", "IV Katkı": round(float(result["IV Katkı"].sum()), 4),
    }])
    return pd.concat([result, total_row], ignore_index=True)


def mono_check(bt):
    """Bad Rate % üzerinden monotonluk kontrol et (Eksik/Special/TOPLAM hariç)."""
    m = bt[~bt["Bin"].astype(str).str.match(r"^(TOPLAM|Eksik|Special)")]
    nums = [float(w) for w in m["Bad Rate %"].dropna().tolist()
            if isinstance(w, (int, float))]
    if len(nums) < 2:
        return "–"
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    if all(d >= 0 for d in diffs):
        return "Artan ↑"
    if all(d <= 0 for d in diffs):
        return "Azalan ↓"
    return "Monoton Değil ✗"


# ── Tek değişken için bin sayısını değiştirip tüm cache'i güncelle ──────────
def refit_single_variable(col, new_max_bins, df_train, df_test, df_oot,
                          target, key, seg_col, seg_val):
    """
    Tek değişken için OptBinning yeniden fit et,
    8 cache key'ini atomik olarak güncelle.
    Diğer değişkenler hiç etkilenmez.

    Returns: (new_iv, actual_n_bins, old_iv)
    """
    from optbinning import OptimalBinning as _OB
    from modules.deep_dive import SPECIAL_VALUES, is_special_column, get_woe_detail
    from server_state import _SERVER_STORE

    _pfx = f"{key}_ds_{seg_col}_{seg_val}"

    # ── 1. OptBinning fit (train) ────────────────────────────────────────────
    y_train = pd.to_numeric(df_train[target], errors="coerce")
    _clean = y_train.notna()
    _local = df_train.loc[_clean, [col, target]].copy()
    _local[target] = y_train[_clean].astype(int)

    is_numeric = pd.api.types.is_numeric_dtype(df_train[col])
    _special_col = is_numeric and is_special_column(_local[col])
    _bins = 2 if _special_col else new_max_bins

    kwargs = dict(name=col, monotonic_trend="auto_asc_desc",
                  max_n_bins=_bins, solver="cp",
                  dtype="numerical" if is_numeric else "categorical")
    if is_numeric:
        kwargs["special_codes"] = list(SPECIAL_VALUES)

    optb = _OB(**kwargs)
    optb.fit(_local[col].values, _local[target].values)

    # ── 2. Binning table + IV (special değerler ayrıştırılarak) ──────────────
    bt = optb.binning_table.build(show_digits=8)
    _local_refit = df_train.loc[df_train[target].notna(), [col, target]]
    new_iv = calc_total_iv(bt, _local_refit[col].values, _local_refit[target].astype(int).values)

    # ── 3. Bin edges ─────────────────────────────────────────────────────────
    new_edges = None
    if is_numeric and hasattr(optb, "splits") and optb.splits is not None:
        new_edges = [-np.inf] + list(optb.splits) + [np.inf]

    # Gerçek bin sayısı (OptBinning üst sınırdan az verebilir)
    n_data_bins = len(bt) - 3  # Totals, Special, Missing hariç
    # Special/Missing boşsa sayma
    if bt[bt["Bin"] == "Special"].empty or int(bt[bt["Bin"] == "Special"].iloc[0]["Count"]) == 0:
        n_data_bins = len(bt) - 2  # sadece Totals ve Missing
    actual_n_bins = max(n_data_bins, 1)

    # ── 4. WoE transform ────────────────────────────────────────────────────
    tr_woe = optb.transform(df_train[col].values, metric="woe",
                            metric_missing="empirical",
                            metric_special="empirical")

    te_woe = None
    if df_test is not None and len(df_test) > 0:
        te_woe = optb.transform(df_test[col].values, metric="woe",
                                metric_missing="empirical",
                                metric_special="empirical")

    oot_woe = None
    if df_oot is not None and len(df_oot) > 0:
        oot_woe = optb.transform(df_oot[col].values, metric="woe",
                                 metric_missing="empirical",
                                 metric_special="empirical")

    # ── 5. Cache güncelleme (per-column, atomik) ─────────────────────────────
    # 5a. optb_dict
    optb_dict = _SERVER_STORE.get(f"{_pfx}_optb", {})
    optb_dict[col] = optb
    _SERVER_STORE[f"{_pfx}_optb"] = optb_dict

    # 5b. bins_dict
    bins_dict = _SERVER_STORE.get(f"{_pfx}_bins", {})
    if new_edges:
        bins_dict[col] = new_edges
    _SERVER_STORE[f"{_pfx}_bins"] = bins_dict

    # 5b2. per-variable bin override
    pv_bins = _SERVER_STORE.get(f"{_pfx}_pv_bins", {})
    pv_bins[col] = new_max_bins
    _SERVER_STORE[f"{_pfx}_pv_bins"] = pv_bins

    # 5c. iv_tables
    iv_tables = _SERVER_STORE.get(f"{_pfx}_iv_tables", {})
    iv_tables[col] = bt
    _SERVER_STORE[f"{_pfx}_iv_tables"] = iv_tables

    # 5d. train_woe DataFrame — sadece o kolonu güncelle
    train_woe_df = _SERVER_STORE.get(f"{_pfx}_train_woe")
    old_iv = None
    if train_woe_df is not None:
        train_woe_df[col] = pd.Series(tr_woe, index=df_train.index).fillna(0.0)

    # 5e. test_woe DataFrame
    test_woe_df = _SERVER_STORE.get(f"{_pfx}_test_woe")
    if test_woe_df is not None and te_woe is not None:
        test_woe_df[col] = pd.Series(te_woe, index=df_test.index).fillna(0.0)

    # 5f. oot_woe DataFrame
    oot_woe_df = _SERVER_STORE.get(f"{_pfx}_oot_woe")
    if oot_woe_df is not None and oot_woe is not None:
        oot_woe_df[col] = pd.Series(oot_woe, index=df_oot.index).fillna(0.0)

    # 5g. iv_df — o satırın IV'ünü güncelle
    iv_key = f"{key}_iv_{seg_col}_{seg_val}"
    iv_df = _SERVER_STORE.get(iv_key)
    if iv_df is not None:
        mask = iv_df["Değişken"] == col
        if mask.any():
            old_iv = float(iv_df.loc[mask, "IV"].iloc[0])
        iv_df.loc[mask, "IV"] = round(new_iv, 4)
        iv_df.loc[mask, "Güç"] = _iv_label(new_iv)
        _SERVER_STORE[iv_key] = iv_df

    # 5h. woe_tables[col] — train + test + oot tabloları
    woe_tables = _SERVER_STORE.get(f"{_pfx}_woe_tables", {})
    bt_train = format_bt(bt, col_name=col, df_train=df_train, target=target)
    if not bt_train.empty:
        entry = {
            "train_table": bt_train.to_dict("records"),
            "iv_train": round(new_iv, 4),
            "monoton": mono_check(bt_train),
        }
        # Test tablosu
        if df_test is not None and len(df_test) > 0:
            try:
                bt_test, iv_test, _, _ = get_woe_detail(
                    df_test, col, target, fitted_optb=optb, use_edges=True)
                if not bt_test.empty:
                    entry["test_table"] = bt_test.to_dict("records")
                    entry["iv_test"] = round(iv_test, 4)
                    entry["monoton_test"] = mono_check(bt_test)
            except Exception:
                pass
        # OOT tablosu
        if df_oot is not None and len(df_oot) > 0:
            try:
                bt_oot, iv_oot, _, _ = get_woe_detail(
                    df_oot, col, target, fitted_optb=optb, use_edges=True)
                if not bt_oot.empty:
                    entry["oot_table"] = bt_oot.to_dict("records")
                    entry["iv_oot"] = round(iv_oot, 4)
                    entry["monoton_oot"] = mono_check(bt_oot)
            except Exception:
                pass
        woe_tables[col] = entry
        _SERVER_STORE[f"{_pfx}_woe_tables"] = woe_tables

    # ── 6. var_summary cache'ini invalidate et (next render'da taze hesaplansın)
    _SERVER_STORE.pop(f"{key}_varsummary_{seg_col}_{seg_val}", None)
    _SERVER_STORE.pop(f"{key}_summary_{seg_col}_{seg_val}", None)
    _SERVER_STORE.pop(f"{_pfx}_psi_map", None)

    return new_iv, actual_n_bins, old_iv


# ── Yardımcı: r Badge ────────────────────────────────────────────────────────
def _make_r_badge(r):
    if np.isnan(r):
        return html.Div()
    r_str   = f"{r:+.4f}"
    r_color = "#10b981" if abs(r) < 0.4 else "#f59e0b" if abs(r) < 0.75 else "#ef4444"
    label   = "Zayıf" if abs(r) < 0.4 else "Orta" if abs(r) < 0.75 else "Yüksek" if abs(r) < 0.9 else "Çok Yüksek"
    return html.Div([
        html.Div("Korelasyon (r)", className="metric-label"),
        html.Div(r_str, className="metric-value",
                 style={"color": r_color, "fontSize": "1.35rem"}),
        html.Div(label, style={"fontSize": "0.65rem", "color": r_color,
                                "fontWeight": "700", "letterSpacing": "0.06em",
                                "textTransform": "uppercase", "marginTop": "0.15rem"}),
    ], className="metric-card", style={"padding": "0.55rem 1rem", "minWidth": "120px"})


# ── Yardımcı: Çift Analiz Grafikleri ─────────────────────────────────────────
def _make_pair_scatter(df_active, var1, var2, target):
    local = df_active[[var1, var2, target]].dropna(subset=[var1, var2]).copy()
    n_total = len(local)

    _no_data = html.Div("Yeterli çakışan veri yok.",
                        style={"color": "#7e8fa4", "fontSize": "0.8rem",
                               "padding": "1rem 0"})
    if n_total < 5:
        return _no_data

    is_num1 = pd.api.types.is_numeric_dtype(local[var1])
    is_num2 = pd.api.types.is_numeric_dtype(local[var2])

    def _trunc(s, n=24):
        return s[:n] + "…" if len(s) > n else s

    overall_br = float(local[target].mean() * 100)
    colors_t   = {0: "#4F8EF7", 1: "#ef4444"}

    # target'ın gerçek unique değerleri (NaN hariç)
    t_vals_raw = sorted(local[target].dropna().unique())
    if not t_vals_raw:
        return _no_data

    # ════════════════════════════════════════════════════════════════════════
    # NUMERIC × NUMERIC  —  Kantil Tabanlı Bad Rate Isı Haritası
    # Her iki değişkeni 10 kantile böler, hücre rengi = o kombinasyonun
    # kötü oranı (bad rate). Yeşil=düşük risk, Kırmızı=yüksek risk.
    # Sadece agregasyon yapıldığı için 5M+ satırda anlık çalışır.
    # ════════════════════════════════════════════════════════════════════════
    if is_num1 and is_num2:
        try:
            r = float(local[[var1, var2]].corr().iloc[0, 1])
        except Exception:
            r = float("nan")
        r_str = f"{r:+.4f}" if not np.isnan(r) else "—"

        N_BINS = 10
        work = local[[var1, var2, target]].dropna().copy()

        # Kantil tabanlı binleme — tekrar eden değerler için duplicates='drop'
        try:
            work["_b1"] = pd.qcut(work[var1], q=N_BINS, duplicates="drop", labels=False)
            work["_b2"] = pd.qcut(work[var2], q=N_BINS, duplicates="drop", labels=False)
        except Exception:
            work["_b1"] = pd.cut(work[var1], bins=N_BINS, labels=False)
            work["_b2"] = pd.cut(work[var2], bins=N_BINS, labels=False)

        grid = (
            work.groupby(["_b1", "_b2"], observed=True)
            .agg(bad_rate=(target, "mean"), n=(target, "count"))
            .reset_index()
        )
        if grid.empty:
            return _no_data

        # Bin orta noktaları — eksen etiketleri için
        b1_mid = work.groupby("_b1", observed=True)[var1].median().sort_index()
        b2_mid = work.groupby("_b2", observed=True)[var2].median().sort_index()

        pivot_br = grid.pivot(index="_b2", columns="_b1", values="bad_rate")
        pivot_n  = grid.pivot(index="_b2", columns="_b1", values="n")

        x_labels = [f"{b1_mid.get(c, c):.3g}" for c in pivot_br.columns]
        y_labels = [f"{b2_mid.get(i, i):.3g}" for i in pivot_br.index]

        hover = [
            [
                "Veri yok" if pd.isna(pivot_br.values[r][c])
                else (f"Bad Rate: %{pivot_br.values[r][c]*100:.1f}<br>"
                      f"n: {int(pivot_n.values[r][c]) if not pd.isna(pivot_n.values[r][c]) else 0:,}")
                for c in range(pivot_br.shape[1])
            ]
            for r in range(pivot_br.shape[0])
        ]

        fig = go.Figure(go.Heatmap(
            z=pivot_br.values * 100,
            x=x_labels,
            y=y_labels,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.0,  "#10b981"],   # yeşil  — düşük risk
                [0.5,  "#f59e0b"],   # sarı   — orta risk
                [1.0,  "#ef4444"],   # kırmızı — yüksek risk
            ],
            colorbar=dict(
                title=dict(text="Bad Rate %", font=dict(color="#8892a4", size=10)),
                tickfont=dict(color="#8892a4", size=9),
                ticksuffix="%",
                bgcolor="#161C27",
                bordercolor="#232d3f",
            ),
            zmid=overall_br,
        ))

        fig.update_layout(
            paper_bgcolor="#161C27",
            plot_bgcolor="#0E1117",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
            title=dict(
                text=(f"Bad Rate Isı Haritası  ·  {_trunc(var1)} × {_trunc(var2)}"
                      f"  ·  r = {r_str}  ·  Genel Bad Rate %{overall_br:.1f}"),
                font=dict(color="#E8EAF0", size=12),
            ),
            xaxis=dict(**_AXIS_STYLE,
                       title=dict(text=f"{_trunc(var1)} (kantil medyanı)",
                                  font=dict(color="#8892a4", size=10)),
                       tickangle=-35),
            yaxis=dict(**_AXIS_STYLE,
                       title=dict(text=f"{_trunc(var2)} (kantil medyanı)",
                                  font=dict(color="#8892a4", size=10))),
            height=480,
            margin=dict(l=80, r=20, t=65, b=80),
        )

        n_note = (f"n = {n_total:,}  ·  {N_BINS}×{N_BINS} kantil hücre  "
                  f"·  Renk = hücre bad rate  ·  Genel ort. = %{overall_br:.1f}")

        return html.Div([
            html.P(n_note, style={"fontSize": "0.72rem", "color": "#7e8fa4",
                                   "marginBottom": "0.3rem"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORICAL × NUMERIC
    # Yan yana Box plot — kategori başına hedef kırılımı
    # ════════════════════════════════════════════════════════════════════════
    else:
        num_col = var1 if is_num1 else var2
        cat_col = var2 if is_num1 else var1

        local[cat_col] = local[cat_col].fillna("Eksik").astype(str)
        top_cats  = local[cat_col].value_counts().head(15).index.tolist()
        local_f   = local[local[cat_col].isin(top_cats)].copy()
        cat_order = (local_f.groupby(cat_col)[num_col]
                     .median().sort_values().index.tolist())

        fig = go.Figure()
        for tv in t_vals_raw:
            g    = local_f[local_f[target] == tv]
            clr  = colors_t.get(int(tv), "#8892a4")
            fill = f"rgba({'79,142,247' if int(tv)==0 else '239,68,68'}, 0.18)"
            fig.add_trace(go.Box(
                x=g[cat_col], y=g[num_col],
                name=f"Target = {int(tv)}",
                marker_color=clr,
                line=dict(color=clr, width=1.5),
                fillcolor=fill,
                boxmean=True,
                hovertemplate=(
                    f"Target = {int(tv)}<br>"
                    f"{_trunc(cat_col)}: %{{x}}<br>"
                    f"{_trunc(num_col)}: %{{y:.3f}}<extra></extra>"
                ),
            ))

        fig.update_layout(
            paper_bgcolor="#161C27", plot_bgcolor="#0E1117",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#8892a4", size=11),
            boxmode="group",
            title=dict(
                text=f"{_trunc(cat_col)} × {_trunc(num_col)}  ·  Genel Bad Rate %{overall_br:.1f}",
                font=dict(color="#E8EAF0", size=12),
            ),
            xaxis=dict(**_AXIS_STYLE,
                       title=dict(text=cat_col, font=dict(color="#8892a4", size=10)),
                       categoryorder="array", categoryarray=cat_order, tickangle=-30),
            yaxis=dict(**_AXIS_STYLE,
                       title=dict(text=num_col, font=dict(color="#8892a4", size=10))),
            legend=dict(bgcolor="#161C27", bordercolor="#232d3f", font=dict(size=10)),
            height=430,
            margin=dict(l=60, r=20, t=50, b=80),
        )

        n_note = (f"n = {n_total:,}  ·  İlk 15 kategori  "
                  f"·  {_trunc(num_col, 20)} medyanına göre sıralı  "
                  f"·  — ortalama  ·  Mavi = Target 0  ·  Kırmızı = Target 1")

        return html.Div([
            html.P(n_note, style={"fontSize": "0.72rem", "color": "#7e8fa4",
                                   "marginBottom": "0.3rem"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])


def _safe_pair_scatter(df_active, var1, var2, target):
    try:
        return _make_pair_scatter(df_active, var1, var2, target)
    except Exception as exc:
        return html.Div(
            f"Grafik oluşturulamadı: {exc}",
            style={"color": "#ef4444", "padding": "1rem", "fontSize": "0.8rem"},
        )
