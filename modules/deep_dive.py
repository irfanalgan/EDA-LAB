import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

SPECIAL_VALUES = {9999999999, 8888888888}


# ── sklearn >= 1.6 uyumluluk yaması (optbinning metrics.py) ──────────────────
# sklearn 1.6'da force_all_finite → ensure_all_finite olarak yeniden adlandırıldı.
# optbinning bu değişikliği henüz yansıtmadıysa runtime'da monkey-patch uygula.
try:
    import sklearn as _sklearn
    _sk_ver = tuple(int(v) for v in _sklearn.__version__.split(".")[:2])
    if _sk_ver >= (1, 6):
        import optbinning.binning.metrics as _ob_metrics
        from sklearn.utils import check_array as _check_array
        from sklearn.utils import check_consistent_length as _check_cl

        def _patched_check_x_y(x, y):
            x = _check_array(x, ensure_2d=False, ensure_all_finite=True)
            y = _check_array(y, ensure_2d=False, ensure_all_finite=True)
            _check_cl(x, y)
            return x, y

        _ob_metrics._check_x_y = _patched_check_x_y
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────


def get_woe_detail(df: pd.DataFrame, col: str, target: str,
                   max_n_bins: int = 4,
                   force_dtype: str = None) -> tuple[pd.DataFrame, float, object]:
    """
    OptimalBinning ile monoton WOE/IV tablosu döndürür.
    Returns: (woe_df, iv_total, bin_edges)
    bin_edges: [-inf, split1, ..., +inf] numpy array (numerik), None (kategorik).
    force_dtype: None | "numerical" | "categorical"  — otomatik algıyı ezer.
    """
    from optbinning import OptimalBinning as _OB

    local = df[[col, target]].copy()
    if local[target].dtype == object:
        local[target] = (local[target].astype(str)
                         .str.replace('%', '', regex=False).str.strip())
    local[target] = pd.to_numeric(local[target], errors='coerce')
    local = local.dropna(subset=[target])

    total_bad  = local[target].sum()
    total_good = len(local) - total_bad

    if total_bad == 0 or total_good == 0:
        return pd.DataFrame(), 0.0, None

    is_numeric = pd.api.types.is_numeric_dtype(local[col])
    if force_dtype == "categorical":
        is_numeric = False
    elif force_dtype == "numerical":
        is_numeric = True
    X = local[col].values
    y = local[target].values.astype(int)

    try:
        kwargs = dict(
            name=col,
            monotonic_trend="auto_asc_desc",
            max_n_bins=max_n_bins,
            solver="cp",
            dtype="numerical" if is_numeric else "categorical",
        )
        if is_numeric:
            kwargs["special_codes"] = list(SPECIAL_VALUES)
        optb = _OB(**kwargs)
        optb.fit(X, y)
        bt = optb.binning_table.build(show_digits=8)
        iv_total = float(bt.loc["Totals", "IV"])
    except Exception:
        return pd.DataFrame(), 0.0, None

    # Main bins: non-empty Bin string, not "Special" or "Missing"
    data_rows = bt[bt["Bin"].astype(str).str.len() > 0]
    data_rows = data_rows[~data_rows["Bin"].isin(["Special", "Missing"])]

    rows = []
    for _, row in data_rows.iterrows():
        total   = int(row["Count"])
        bad     = int(row["Event"])
        good    = int(row["Non-event"])
        bad_rate = round(float(row["Event rate"]) * 100, 2)
        woe     = round(float(row["WoE"]), 4)
        iv_part = round(float(row["IV"]), 4)
        rows.append({"Bin": str(row["Bin"]), "Toplam": total, "Bad": bad,
                     "Good": good, "Bad Rate %": bad_rate, "WOE": woe, "IV Katkı": iv_part})

    # Special and Missing rows with count > 0
    for special_lbl in ["Special", "Missing"]:
        sr = bt[bt["Bin"] == special_lbl]
        if not sr.empty and int(sr.iloc[0]["Count"]) > 0:
            r = sr.iloc[0]
            display_lbl = "Eksik" if special_lbl == "Missing" else "Special"
            rows.append({"Bin": display_lbl, "Toplam": int(r["Count"]),
                         "Bad": int(r["Event"]), "Good": int(r["Non-event"]),
                         "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
                         "WOE": round(float(r["WoE"]), 4),
                         "IV Katkı": round(float(r["IV"]), 4)})

    if not rows:
        return pd.DataFrame(), 0.0, None

    result = pd.DataFrame(rows)
    total_all_n = result["Toplam"].sum()
    total_bad_n = result["Bad"].sum()
    total_row = pd.DataFrame([{
        "Bin": "TOPLAM", "Toplam": int(total_all_n), "Bad": int(total_bad_n),
        "Good": int(result["Good"].sum()),
        "Bad Rate %": round(total_bad_n / total_all_n * 100, 2) if total_all_n > 0 else 0.0,
        "WOE": "", "IV Katkı": round(iv_total, 4),
    }])
    result = pd.concat([result, total_row], ignore_index=True)

    # bin_edges from optbinning splits (for PSI and compute_period_badrate)
    _bin_edges = None
    if is_numeric:
        try:
            splits = list(optb.splits)
            if splits:
                _bin_edges = np.array([-np.inf] + splits + [np.inf], dtype=float)
        except Exception:
            pass

    return result, iv_total, _bin_edges


def compute_period_badrate(df: pd.DataFrame, col: str, target: str,
                           woe_df: pd.DataFrame, bin_edges) -> pd.DataFrame:
    """
    Train WOE binlerini yeni bir df'ye (test/OOT) uygulayarak her bin için
    bad rate hesaplar. Bin eşleştirmesi pozisyon bazlıdır (label formatından bağımsız).

    Returns DataFrame with columns [Bin, Toplam, Bad, Bad Rate %] aligned with woe_df bins.
    """
    local = df[[col, target]].copy()
    local[target] = pd.to_numeric(local[target], errors="coerce")
    local = local.dropna(subset=[target])
    if local.empty:
        return pd.DataFrame()

    is_numeric = pd.api.types.is_numeric_dtype(local[col])

    # Main train bins (excluding TOPLAM, Eksik, Special) — in order
    main_bins_df = woe_df[
        ~woe_df["Bin"].isin(["TOPLAM", "Eksik"]) &
        ~woe_df["Bin"].astype(str).str.startswith("Special", na=False)
    ].reset_index(drop=True)

    if is_numeric and bin_edges is not None and len(bin_edges) >= 2:
        special_mask = local[col].isin(SPECIAL_VALUES)
        special_data = local[special_mask]
        main_data    = local[~special_mask]
        present      = main_data[main_data[col].notna()].copy()
        missing      = main_data[main_data[col].isna()]

        # Use integer labels (0-based) to match by position, not by label string
        _cut = pd.cut(present[col], bins=bin_edges, include_lowest=False, labels=False)
        present["bin_idx"] = _cut

        rows = []
        for i, bin_lbl in enumerate(main_bins_df["Bin"]):
            grp   = present[present["bin_idx"] == i]
            total = len(grp)
            bad   = int(grp[target].sum())
            rows.append({"Bin": bin_lbl, "Toplam": total, "Bad": bad,
                         "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0})

        # Special bins
        for sv in SPECIAL_VALUES:
            sv_grp = special_data[special_data[col] == sv]
            if len(sv_grp):
                lbl   = f"Special ({int(sv)})"
                total = len(sv_grp)
                bad   = int(sv_grp[target].sum())
                rows.append({"Bin": lbl, "Toplam": total, "Bad": bad,
                             "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0})
        # Eksik bin
        if len(missing):
            total = len(missing)
            bad   = int(missing[target].sum())
            rows.append({"Bin": "Eksik", "Toplam": total, "Bad": bad,
                         "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0})
    else:
        # Kategorik: eşleştirme değer adına göre
        local["bin_label"] = local[col].fillna("Eksik").astype(str)
        all_train_bins = woe_df[woe_df["Bin"] != "TOPLAM"]["Bin"].tolist()
        rows = []
        for lbl in all_train_bins:
            grp   = local[local["bin_label"] == lbl]
            total = len(grp)
            bad   = int(grp[target].sum())
            rows.append({"Bin": lbl, "Toplam": total, "Bad": bad,
                         "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    total_all = result["Toplam"].sum()
    total_bad_r = result["Bad"].sum()
    total_row = pd.DataFrame([{
        "Bin": "TOPLAM", "Toplam": int(total_all), "Bad": int(total_bad_r),
        "Bad Rate %": round(total_bad_r / total_all * 100, 2) if total_all > 0 else 0.0,
    }])
    return pd.concat([result, total_row], ignore_index=True)


def compute_iv_ranking_optimal(df: pd.DataFrame, target: str,
                                max_n_bins: int = 4) -> pd.DataFrame:
    """Tüm değişkenler için IV hesaplar (tam veri)."""
    records = []
    for col in df.columns:
        if col == target:
            continue
        try:
            _, iv, _ = get_woe_detail(df, col, target, max_n_bins)
            records.append({
                "Değişken": col,
                "IV":       round(iv, 4),
                "Eksik %":  round(df[col].isna().mean() * 100, 2),
                "Güç":      _iv_label(iv),
            })
        except Exception:
            records.append({"Değişken": col, "IV": 0.0,
                             "Eksik %": 0.0, "Güç": "Çok Zayıf"})

    return pd.DataFrame(records).sort_values("IV", ascending=False).reset_index(drop=True)


def _iv_label(iv: float) -> str:
    if iv < 0.02:  return "Çok Zayıf"
    if iv < 0.10:  return "Zayıf"
    if iv < 0.30:  return "Orta"
    if iv < 0.50:  return "Güçlü"
    return "Şüpheli"


# ── PSI — Tarih Bazlı Split ───────────────────────────────────────────────────

def compute_psi(df: pd.DataFrame, col: str, target: str,
                date_col: str = None, cutoff_date: str = None,
                max_n_bins: int = 4, bin_edges=None,
                force_dtype: str = None) -> dict:
    """
    PSI hesaplar.
    1. Baseline (tarih öncesi) üzerinde bin sınırları belirlenir.
    2. Aynı sınırlar comparison (tarih sonrası) üzerinde uygulanır.
    3. İki dönemin bin dağılımı karşılaştırılır → PSI.
    force_dtype: None | "numerical" | "categorical"  — otomatik algıyı ezer.
    """
    cols_needed = [col, target] + ([date_col] if date_col and date_col in df.columns else [])
    local = df[cols_needed].copy()
    if local[target].dtype == object:
        local[target] = (local[target].astype(str)
                         .str.replace('%', '', regex=False).str.strip())
    local[target] = pd.to_numeric(local[target], errors='coerce')
    is_numeric    = pd.api.types.is_numeric_dtype(local[col])
    if force_dtype == "categorical":
        is_numeric = False
    elif force_dtype == "numerical":
        is_numeric = True

    # ── Split ─────────────────────────────────────────────────────────────────
    if date_col and cutoff_date and date_col in local.columns:
        local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
        cutoff      = pd.to_datetime(cutoff_date)
        df_base     = local[local[date_col] <  cutoff]
        df_comp     = local[local[date_col] >= cutoff]
        split_label = f"< {cutoff_date}"
        comp_label  = f"≥ {cutoff_date}"
    else:
        mid         = len(local) // 2
        df_base     = local.iloc[:mid]
        df_comp     = local.iloc[mid:]
        split_label = "İlk Yarı"
        comp_label  = "İkinci Yarı"

    if len(df_base) == 0 or len(df_comp) == 0:
        return {"psi": None, "error": "Split sonucu boş segment."}

    # ── Bin sınırları: WOE'dan geldiyse kullan, yoksa baseline'dan hesapla ────
    if is_numeric:
        if bin_edges is not None:
            # WOE ile aynı sınırları kullan
            edges = np.array(bin_edges, dtype=float)
            edges[0]  = -np.inf
            edges[-1] =  np.inf
        else:
            base_clean = df_base[col].dropna()
            base_clean = base_clean[~base_clean.isin(SPECIAL_VALUES)]
            try:
                _, edges = pd.qcut(base_clean, q=max_n_bins,
                                   duplicates="drop", retbins=True)
                edges[0]  = -np.inf
                edges[-1] =  np.inf
            except Exception as e:
                return {"psi": None, "error": f"Bin hatası: {e}"}

        try:
            # Interval nesnelerini koru — str'e çevirme, sıralama için gerekli
            base_binned = pd.cut(df_base[col], bins=edges, include_lowest=True)
            comp_binned = pd.cut(df_comp[col], bins=edges, include_lowest=True)
        except Exception as e:
            return {"psi": None, "error": f"Cut hatası: {e}"}
    else:
        top_cats    = df_base[col].value_counts().head(max_n_bins).index
        base_binned = df_base[col].where(df_base[col].isin(top_cats), "Diğer").fillna("Eksik").astype(str)
        comp_binned = df_comp[col].where(df_comp[col].isin(top_cats), "Diğer").fillna("Eksik").astype(str)

    # ── PSI hesabı ────────────────────────────────────────────────────────────
    base_dist = base_binned.value_counts(normalize=True)
    comp_dist = comp_binned.value_counts(normalize=True)

    # Numerik: Interval nesneleri sol sınıra göre doğal sıralanır
    # Kategorik: string olarak alfabetik sıra
    try:
        all_bins = sorted(set(base_dist.index) | set(comp_dist.index))
    except TypeError:
        all_bins = sorted(set(base_dist.index) | set(comp_dist.index), key=str)
    eps       = 1e-9
    psi_total = 0.0
    rows      = []

    for b in all_bins:
        if b is None or (hasattr(b, '__class__') and b.__class__.__name__ == 'float' and np.isnan(b)):
            continue   # NaN bin'i atla
        p_base   = float(base_dist.get(b, eps))
        p_comp   = float(comp_dist.get(b, eps))
        psi_part = (p_comp - p_base) * np.log((p_comp + eps) / (p_base + eps))
        psi_total += psi_part
        rows.append({
            "Bin":              str(b),
            "Baseline %":       round(p_base * 100, 2),
            "Karşılaştırma %":  round(p_comp * 100, 2),
            "Δ (pp)":           round((p_comp - p_base) * 100, 2),
            "PSI Katkı":        round(psi_part, 5),
        })

    psi_label = ("Stabil" if psi_total < 0.1
                 else "Hafif Kayma" if psi_total < 0.25
                 else "Kritik Kayma")

    return {
        "psi":         round(psi_total, 5),
        "label":       psi_label,
        "n_baseline":  len(df_base),
        "n_compare":   len(df_comp),
        "split_label": split_label,
        "comp_label":  comp_label,
        "detail_df":   pd.DataFrame(rows),
    }


# ── Temel Değişken İstatistikleri ─────────────────────────────────────────────

def get_variable_stats(df: pd.DataFrame, col: str, target: str) -> dict:
    """Tek değişken özet istatistikler. Orijinal df'ye dokunmaz."""
    s           = df[col].copy()
    n           = len(s)
    missing     = int(s.isna().sum())
    missing_pct = round(missing / n * 100, 2)
    unique      = int(s.nunique(dropna=True))
    is_numeric  = pd.api.types.is_numeric_dtype(s)

    base = {"n": n, "missing": missing, "missing_pct": missing_pct,
            "unique": unique, "dtype": str(s.dtype), "is_numeric": is_numeric}

    if is_numeric:
        nn = s.dropna()
        if len(nn) > 2:
            q1, q3 = nn.quantile(0.25), nn.quantile(0.75)
            iqr    = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_n = int(((nn < lower) | (nn > upper)).sum())
            base.update({
                "mean": round(float(nn.mean()), 4), "std": round(float(nn.std()), 4),
                "min":  round(float(nn.min()),  4),
                "p1":   round(float(nn.quantile(.01)), 4),
                "p5":   round(float(nn.quantile(.05)), 4),
                "p25":  round(float(q1), 4),
                "median": round(float(nn.median()), 4),
                "p75":  round(float(q3), 4),
                "p95":  round(float(nn.quantile(.95)), 4),
                "p99":  round(float(nn.quantile(.99)), 4),
                "max":  round(float(nn.max()), 4),
                "skewness":      round(float(scipy_stats.skew(nn)), 4),
                "kurtosis":      round(float(scipy_stats.kurtosis(nn)), 4),
                "iqr_lower":     round(float(lower), 4),
                "iqr_upper":     round(float(upper), 4),
                "outlier_count": outlier_n,
                "outlier_pct":   round(outlier_n / len(nn) * 100, 2),
            })
    else:
        base["top5"] = s.value_counts().head(5).to_dict()

    lc = df[[col, target]].copy()
    # String-formatted numbers ('1.49%', '0.5', vb.) için güvenli dönüşüm
    if lc[target].dtype == object:
        lc[target] = (lc[target].astype(str)
                      .str.replace('%', '', regex=False)
                      .str.strip())
    lc[target] = pd.to_numeric(lc[target], errors='coerce')
    if missing > 0:
        base["missing_bad_rate"] = round(float(lc[lc[col].isna()][target].mean() * 100), 2)
        base["present_bad_rate"] = round(float(lc[lc[col].notna()][target].mean() * 100), 2)
    else:
        base["missing_bad_rate"] = None
        base["present_bad_rate"] = None

    return base


# ── WoE Encoding ──────────────────────────────────────────────────────────────

def get_woe_encoder(df: pd.DataFrame, col: str, target: str,
                    max_n_bins: int = 4) -> tuple:
    """
    OptimalBinning ile her satır için WoE değeri hesaplar.
    Returns: (woe_series, iv_total, success: bool)
    Başarısızsa woe_series = NaN dolu, success = False.
    """
    from optbinning import OptimalBinning as _OB

    local = df[[col, target]].copy()
    if local[target].dtype == object:
        local[target] = (local[target].astype(str)
                         .str.replace('%', '', regex=False).str.strip())
    local[target] = pd.to_numeric(local[target], errors='coerce')
    local = local.dropna(subset=[target])

    total_bad  = local[target].sum()
    total_good = len(local) - total_bad
    if total_bad == 0 or total_good == 0:
        return pd.Series(np.nan, index=df.index), 0.0, False

    is_numeric = pd.api.types.is_numeric_dtype(local[col])

    try:
        kwargs = dict(name=col, monotonic_trend="auto_asc_desc",
                      max_n_bins=max_n_bins, solver="cp",
                      dtype="numerical" if is_numeric else "categorical")
        if is_numeric:
            kwargs["special_codes"] = list(SPECIAL_VALUES)
        optb = _OB(**kwargs)
        optb.fit(local[col].values, local[target].values.astype(int))
        bt   = optb.binning_table.build(show_digits=8)
        iv   = float(bt.loc["Totals", "IV"])
        woe_vals = optb.transform(df[col].values, metric="woe",
                                   metric_missing="empirical",
                                   metric_special="empirical")
        woe_series = pd.Series(woe_vals, index=df.index).fillna(0.0)
    except Exception:
        return pd.Series(np.nan, index=df.index), 0.0, False

    return woe_series, iv, True
