import re
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

SPECIAL_VALUES = {9999999999, 8888888888}

_INTERVAL_RE = re.compile(r'^([\(\[])\s*(.*?)\s*,\s*(.*?)\s*([\)\]])$')


def _merge_label(a: str, b: str) -> str:
    """Merge two interval label strings into a clean combined range."""
    ma = _INTERVAL_RE.match(a)
    mb = _INTERVAL_RE.match(b)
    if ma and mb:
        return f"{ma.group(1)}{ma.group(2)}, {mb.group(3)}{mb.group(4)}"
    return f"{a} | {b}"


# ── WOE / IV — Monoton Optimal Binning ───────────────────────────────────────

def _merge_bins(bins_df: pd.DataFrame) -> pd.DataFrame:
    """
    WOE monotonluğunu sağlamak için komşu non-monoton bin'leri birleştirir.
    bins_df: [bin_label, total, bad, good, woe] kolonları olmalı.
    """
    df = bins_df.copy().reset_index(drop=True)
    changed = True
    while changed and len(df) > 2:
        changed = False
        woes = df["woe"].values
        # Yön: çoğunluk yönünde monotonluk ara
        diffs = np.diff(woes)
        if np.sum(diffs > 0) >= np.sum(diffs < 0):
            # Artan olmalı — azalan geçişleri bul
            bad_idx = np.where(diffs < 0)[0]
        else:
            # Azalan olmalı — artan geçişleri bul
            bad_idx = np.where(diffs > 0)[0]

        if len(bad_idx) == 0:
            break

        # En küçük IV katkılı komşu çifti birleştir
        merge_at = min(bad_idx, key=lambda i: df.loc[i, "iv_part"] + df.loc[i+1, "iv_part"])
        i, j = merge_at, merge_at + 1
        merged = {
            "bin_label": _merge_label(df.loc[i,'bin_label'], df.loc[j,'bin_label']),
            "total":     df.loc[i,"total"] + df.loc[j,"total"],
            "bad":       df.loc[i,"bad"]   + df.loc[j,"bad"],
            "good":      df.loc[i,"good"]  + df.loc[j,"good"],
        }
        df = df.drop([i, j]).reset_index(drop=True)
        df = pd.concat([df.iloc[:i],
                        pd.DataFrame([merged]),
                        df.iloc[i:]], ignore_index=True)
        changed = True

    return df


def _compute_woe_stats(df: pd.DataFrame, total_bad: float, total_good: float,
                       eps: float = 1e-9) -> pd.DataFrame:
    """WOE ve IV katkısını hesaplar."""
    df = df.copy()
    df["dist_bad"]  = df["bad"]  / (total_bad  + eps)
    df["dist_good"] = df["good"] / (total_good + eps)
    df["woe"]       = np.log((df["dist_bad"] + eps) / (df["dist_good"] + eps))
    df["iv_part"]   = (df["dist_bad"] - df["dist_good"]) * df["woe"]
    df["bad_rate"]  = df["bad"] / (df["total"] + eps) * 100
    return df


def get_woe_detail(df: pd.DataFrame, col: str, target: str,
                   max_n_bins: int = 4) -> tuple[pd.DataFrame, float, object]:
    """
    Monoton WOE/IV tablosu döndürür.
    Returns: (woe_df, iv_total, bin_edges)
    bin_edges: numerik kolonlar için WOE'de kullanılan bin sınırları (numpy array),
               kategorik kolonlar için None. PSI'ya geçirilerek aynı sınırlar kullanılır.
    """
    local = df[[col, target]].copy()
    local[target] = local[target].astype(float)
    local = local.dropna(subset=[target])

    total_bad  = local[target].sum()
    total_good = len(local) - total_bad

    if total_bad == 0 or total_good == 0:
        return pd.DataFrame(), 0.0, None

    is_numeric = pd.api.types.is_numeric_dtype(local[col])

    # Special değerleri ayır
    if is_numeric:
        special_mask = local[col].isin(SPECIAL_VALUES)
        special_data = local[special_mask]
        main_data    = local[~special_mask]
    else:
        special_data = pd.DataFrame()
        main_data    = local

    # ── Binning ───────────────────────────────────────────────────────────────
    bins_list = []

    _bin_edges = None   # PSI'ya aktarılacak WOE bin sınırları (merge sonrası)

    if is_numeric:
        # Eksik bin
        missing = main_data[main_data[col].isna()]
        present = main_data[main_data[col].notna()]

        # qcut ile quantile sınırları belirle, uçları ±inf yap, pd.cut ile uygula
        n_init = max(max_n_bins * 2, 6)
        present = present.copy()
        try:
            _, edges = pd.qcut(present[col], q=n_init, duplicates="drop", retbins=True)
        except Exception:
            try:
                _, edges = pd.qcut(present[col], q=max_n_bins, duplicates="drop", retbins=True)
            except Exception:
                edges = None

        if edges is not None:
            edges[0], edges[-1] = -np.inf, np.inf
            _cut = pd.cut(present[col], bins=edges, include_lowest=False)
        else:
            _cut = pd.Categorical(present[col].astype(str))

        # cat.codes → kesin sayısal sıra (string sort değil)
        present["_bin_code"] = _cut.cat.codes if hasattr(_cut, "cat") else 0
        present["bin_label"] = _cut.astype(str).replace("nan", "Eksik")

        for _, grp in present.groupby("_bin_code", sort=True):
            lbl = grp["bin_label"].iloc[0]
            if lbl in ("nan", "Eksik", ""):
                continue
            bins_list.append({
                "bin_label": str(lbl),
                "total": len(grp),
                "bad":   grp[target].sum(),
                "good":  len(grp) - grp[target].sum(),
            })

        # Special bins
        for sv in SPECIAL_VALUES:
            sv_grp = special_data[special_data[col] == sv]
            if len(sv_grp):
                bins_list.append({
                    "bin_label": f"Special ({int(sv)})",
                    "total": len(sv_grp),
                    "bad":   sv_grp[target].sum(),
                    "good":  len(sv_grp) - sv_grp[target].sum(),
                })

        # Missing bin
        if len(missing):
            bins_list.append({
                "bin_label": "Eksik",
                "total": len(missing),
                "bad":   missing[target].sum(),
                "good":  len(missing) - missing[target].sum(),
            })
    else:
        # Kategorik: top N kategoriler + Diğer + Eksik
        local_cat = main_data.copy()
        top_cats  = local_cat[col].value_counts().head(max_n_bins).index
        local_cat["bin_label"] = (local_cat[col]
                                  .where(local_cat[col].isin(top_cats), "Diğer")
                                  .fillna("Eksik")
                                  .astype(str))
        for lbl, grp in local_cat.groupby("bin_label"):
            bins_list.append({
                "bin_label": str(lbl),
                "total": len(grp),
                "bad":   grp[target].sum(),
                "good":  len(grp) - grp[target].sum(),
            })

    if not bins_list:
        return pd.DataFrame(), 0.0, None

    bins_df = pd.DataFrame(bins_list)
    bins_df = _compute_woe_stats(bins_df, total_bad, total_good)

    # ── Monotonluk zorla (sadece numeric main binler için) ────────────────────
    if is_numeric and len(bins_df) > 2:
        # Special ve Eksik binleri ayır, sadece main binlerde monotonluk
        special_rows = bins_df[bins_df["bin_label"].str.startswith(("Special", "Eksik"))]
        main_rows    = bins_df[~bins_df["bin_label"].str.startswith(("Special", "Eksik"))]

        if len(main_rows) > 2:
            main_rows = _merge_bins(main_rows)
            main_rows = _compute_woe_stats(main_rows, total_bad, total_good)

        bins_df = pd.concat([main_rows, special_rows], ignore_index=True)

    iv_total = float(bins_df["iv_part"].sum())

    result = bins_df[[
        "bin_label", "total", "bad", "good", "bad_rate", "woe", "iv_part"
    ]].rename(columns={
        "bin_label": "Bin",
        "total":     "Toplam",
        "bad":       "Bad",
        "good":      "Good",
        "bad_rate":  "Bad Rate %",
        "woe":       "WOE",
        "iv_part":   "IV Katkı",
    })
    result["Bad Rate %"] = result["Bad Rate %"].round(2)
    result["WOE"]        = result["WOE"].round(4)
    result["IV Katkı"]   = result["IV Katkı"].round(4)

    total_bad_n  = result["Bad"].sum()
    total_all_n  = result["Toplam"].sum()
    total_row = pd.DataFrame([{
        "Bin":       "TOPLAM",
        "Toplam":    int(total_all_n),
        "Bad":       int(total_bad_n),
        "Good":      int(result["Good"].sum()),
        "Bad Rate %": round(total_bad_n / total_all_n * 100, 2) if total_all_n > 0 else 0.0,
        "WOE":       "",
        "IV Katkı":  round(iv_total, 4),
    }])
    result = pd.concat([result, total_row], ignore_index=True)

    # ── Birleştirilmiş bin sınırlarını parse et (PSI için) ───────────────────
    if is_numeric:
        main_bins = result[~result["Bin"].isin(["TOPLAM", "Eksik"])
                          & ~result["Bin"].str.startswith("Special")]
        parsed_edges = []
        for label in main_bins["Bin"]:
            m = _INTERVAL_RE.match(str(label))
            if m:
                lo = m.group(2).strip()
                hi = m.group(3).strip()
                try:
                    lo_val = -np.inf if lo in ("-inf", "-Inf", "−inf") else float(lo)
                except ValueError:
                    lo_val = -np.inf
                try:
                    hi_val = np.inf if hi in ("inf", "Inf") else float(hi)
                except ValueError:
                    hi_val = np.inf
                if not parsed_edges:
                    parsed_edges.append(lo_val)
                parsed_edges.append(hi_val)
        if len(parsed_edges) >= 2:
            _bin_edges = np.array(parsed_edges, dtype=float)

    return result, iv_total, _bin_edges


def compute_iv_ranking_optimal(df: pd.DataFrame, target: str,
                                max_n_bins: int = 4) -> pd.DataFrame:
    """Tüm değişkenler için IV hesaplar (tam veri)."""
    local = df.copy()
    records = []
    for col in local.columns:
        if col == target:
            continue
        try:
            _, iv, _ = get_woe_detail(local, col, target, max_n_bins)
            records.append({
                "Değişken": col,
                "IV":       round(iv, 4),
                "Eksik %":  round(local[col].isna().mean() * 100, 2),
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
                max_n_bins: int = 4, bin_edges=None) -> dict:
    """
    PSI hesaplar.
    1. Baseline (tarih öncesi) üzerinde bin sınırları belirlenir.
    2. Aynı sınırlar comparison (tarih sonrası) üzerinde uygulanır.
    3. İki dönemin bin dağılımı karşılaştırılır → PSI.
    """
    cols_needed = [col, target] + ([date_col] if date_col and date_col in df.columns else [])
    local = df[cols_needed].copy()
    local[target] = local[target].astype(float)
    is_numeric    = pd.api.types.is_numeric_dtype(local[col])

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
    lc[target] = lc[target].astype(float)
    if missing > 0:
        base["missing_bad_rate"] = round(float(lc[lc[col].isna()][target].mean() * 100), 2)
        base["present_bad_rate"] = round(float(lc[lc[col].notna()][target].mean() * 100), 2)
    else:
        base["missing_bad_rate"] = None
        base["present_bad_rate"] = None

    return base


# ── WoE Encoding ──────────────────────────────────────────────────────────────

def _parse_interval_bounds(label: str):
    """'(-inf, 0.5]' → (lo, hi) float tuple. None if not an interval."""
    m = _INTERVAL_RE.match(str(label).strip())
    if not m:
        return None
    try:
        lo_s = m.group(2).strip()
        hi_s = m.group(3).strip()
        lo = float('-inf') if lo_s in ('-inf', '-Inf') else float(lo_s)
        hi = float('inf')  if hi_s in ('inf',  'Inf')  else float(hi_s)
        return lo, hi
    except ValueError:
        return None


def get_woe_encoder(df: pd.DataFrame, col: str, target: str,
                    max_n_bins: int = 4) -> tuple:
    """
    Her satır için WoE değeri hesaplar.
    Returns: (woe_series, iv_total, success: bool)
    Başarısızsa woe_series = NaN dolu, success = False.
    """
    woe_df, iv, _ = get_woe_detail(df, col, target, max_n_bins)
    if woe_df.empty:
        return pd.Series(np.nan, index=df.index), 0.0, False

    # TOPLAM satırını çıkar
    woe_map = {
        row["Bin"]: row["WOE"]
        for _, row in woe_df.iterrows()
        if row["Bin"] != "TOPLAM" and row["WOE"] != ""
    }

    is_numeric = pd.api.types.is_numeric_dtype(df[col])

    if is_numeric:
        no_special = df[col][~df[col].isin(SPECIAL_VALUES) & df[col].notna()]
        n_init = max(max_n_bins * 2, 6)
        try:
            _, edges = pd.qcut(no_special, q=n_init, duplicates="drop", retbins=True)
            edges[0], edges[-1] = -np.inf, np.inf
            cut_result = pd.cut(df[col], bins=edges, include_lowest=False)
            bin_labels = cut_result.astype(str)
        except Exception:
            bin_labels = df[col].astype(str)

        for sv in SPECIAL_VALUES:
            bin_labels = bin_labels.where(df[col] != sv, f"Special ({int(sv)})")
        bin_labels = bin_labels.where(df[col].notna(), "Eksik")
    else:
        top_cats = df[col].dropna().value_counts().head(max_n_bins).index
        bin_labels = (df[col]
                      .where(df[col].isin(top_cats), "Diğer")
                      .fillna("Eksik")
                      .astype(str))

    woe_series = bin_labels.map(woe_map)

    # Birleştirilmiş bin etiketleriyle eşleşemeyen numerik satırlar için
    # ham değer üzerinden aralık kontrolü yap
    if is_numeric:
        unmatched = woe_series.isna() & df[col].notna() & ~df[col].isin(SPECIAL_VALUES)
        if unmatched.any():
            vals = df.loc[unmatched, col]
            for merged_label, woe_val in woe_map.items():
                bounds = _parse_interval_bounds(str(merged_label))
                if bounds is None:
                    continue
                lo, hi = bounds
                in_range = (vals > lo) & (vals <= hi)
                woe_series.loc[in_range[in_range].index] = woe_val

    woe_series = woe_series.fillna(0.0)
    return woe_series, iv, True
