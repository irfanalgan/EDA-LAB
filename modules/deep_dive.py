import logging
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

SPECIAL_VALUES = {9999999999, 8888888888}

# Eşik: kolondaki special value oranı >= %2 ise "special kolon" say
_SPECIAL_RATIO_THRESHOLD = 0.02


def is_special_column(series: "pd.Series") -> bool:
    """Kolonda SPECIAL_VALUES var mı ve oranı >= %2 mi?"""
    mask = series.isin(SPECIAL_VALUES)
    if not mask.any():
        return False
    return mask.sum() / len(series) >= _SPECIAL_RATIO_THRESHOLD


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


# ── WoE / IV Hesaplama Fonksiyonları ─────────────────────────────────────────

SPECIAL_CODES = {'special_1': 9999999999, 'special_2': 8888888888}


def _iv_label(iv):
    """IV gücü etiketi."""
    if iv < 0.02: return "Çok Zayıf"
    if iv < 0.1:  return "Zayıf"
    if iv < 0.3:  return "Orta"
    if iv < 0.5:  return "Güçlü"
    return "Şüpheli"


def format_binning_table(bt: pd.DataFrame) -> pd.DataFrame:
    """OptBinning binning_table.build() çıktısını UI formatına dönüştürür.

    Dönüşümler:
      Count → Toplam, Event rate → Bad Rate % (×100),
      Totals → TOPLAM, Missing → Eksik,
      special_1 → Special (8888888888), special_2 → Special (9999999999)
    Sıralama: bin aralıkları → Eksik → Special → TOPLAM
    """
    df = bt.copy().reset_index(drop=True)
    df = df.rename(columns={
        "Count":      "Toplam",
        "Count (%)":  "Toplam (%)",
        "Event rate": "Bad Rate %",
    })

    # JS kolonunu kaldır (OptBinning'den geliyor, UI'da gerekmez)
    if "JS" in df.columns:
        df = df.drop(columns=["JS"])

    # Tüm sayısal kolonları float'a çevir (to_dict round-trip'inde tip bozulmasın)
    for c in ["Toplam", "Toplam (%)", "Non-event", "Event", "Bad Rate %", "WoE", "IV"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Event rate 0-1 aralığında → yüzdeye çevir
    if "Bad Rate %" in df.columns:
        df["Bad Rate %"] = (df["Bad Rate %"] * 100).round(2)

    # Satır etiketleri
    df["Bin"] = df["Bin"].astype(str)
    df["Bin"] = df["Bin"].replace({"Totals": "TOPLAM", "Missing": "Eksik"})

    # Totals satırı Bin="" olabilir (OptBinning versiyonuna göre)
    empty_mask = df["Bin"].str.strip().isin(["", "nan"])
    df.loc[empty_mask, "Bin"] = "TOPLAM"

    # Special etiketleri: "special_1" → "Special (8888888888)"
    def _rename_special(b):
        if b in SPECIAL_CODES:
            return f"Special ({SPECIAL_CODES[b]})"
        return b
    df["Bin"] = df["Bin"].apply(_rename_special)

    # Virgülden sonra 3 basamak, WoE 8 basamak
    for c in ["Toplam (%)", "Bad Rate %", "IV"]:
        if c in df.columns:
            df[c] = df[c].round(4)
    if "WoE" in df.columns:
        df["WoE"] = df["WoE"].round(8)

    # Sıralama: bin aralıkları → Eksik → Special → TOPLAM
    is_toplam  = df["Bin"] == "TOPLAM"
    is_eksik   = df["Bin"] == "Eksik"
    is_special = df["Bin"].str.startswith("Special", na=False)
    is_normal  = ~(is_toplam | is_eksik | is_special)

    df = pd.concat([
        df[is_normal],
        df[is_eksik],
        df[is_special],
        df[is_toplam],
    ], ignore_index=True)

    return df


def build_period_table(df_period, col, target, bin_edges, train_bt):
    """Test/OOT binning tablosu oluşturur.

    Train bin edge'leri ile veriyi bin'lere ayırır,
    train WoE değerlerini kullanarak IV hesaplar.
    """
    if df_period is None or len(df_period) == 0:
        return None

    series = df_period[col]
    y = df_period[target]

    # Special ve missing mask'leri
    sp1_mask = series.isin([SPECIAL_CODES['special_1']])
    sp2_mask = series.isin([SPECIAL_CODES['special_2']])
    miss_mask = pd.isna(series)
    exclude_mask = sp1_mask | sp2_mask | miss_mask

    # Normal bin'lere ayır
    rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            bin_mask = (series < hi) & ~exclude_mask
        else:
            bin_mask = (series >= lo) & (series < hi) & ~exclude_mask
        cnt = int(bin_mask.sum())
        evt = int(y[bin_mask].sum()) if cnt > 0 else 0
        rows.append({"count": cnt, "Event": evt})

    # Train tablosundan Bin ve WoE etiketlerini al (TOPLAM hariç)
    train_data = train_bt[train_bt["Bin"] != "TOPLAM"].copy().reset_index(drop=True)

    # Eksik satırı — sadece train tablosunda Eksik varsa ekle
    if "Eksik" in train_data["Bin"].values:
        miss_cnt = int(miss_mask.sum())
        miss_evt = int(y[miss_mask].sum()) if miss_cnt > 0 else 0
        rows.append({"count": miss_cnt, "Event": miss_evt})

    # Special satırları — sadece train tablosunda varsa ekle
    for sp_key, sp_val in SPECIAL_CODES.items():
        sp_label = f"Special ({sp_val})"
        if sp_label in train_data["Bin"].values:
            sp_mask = series.isin([sp_val])
            cnt = int(sp_mask.sum())
            evt = int(y[sp_mask].sum()) if cnt > 0 else 0
            rows.append({"count": cnt, "Event": evt})

    if len(rows) != len(train_data):
        logger.warning("build_period_table satır uyuşmazlığı: rows=%d, train_data=%d, col=%s",
                        len(rows), len(train_data), col)
        return None

    result = pd.DataFrame(rows)
    result["Bin"] = train_data["Bin"].values
    result["WoE"] = pd.to_numeric(train_data["WoE"], errors="coerce").fillna(0.0).values
    result["Non-event"] = result["count"] - result["Event"]

    total_count = result["count"].sum()
    total_event = result["Event"].sum()
    total_nonevent = result["Non-event"].sum()

    result["Toplam (%)"] = (
        (result["count"] / total_count * 100).round(4) if total_count > 0
        else 0.0
    )
    result["Bad Rate %"] = np.where(
        result["count"] > 0,
        (result["Event"] / result["count"] * 100).round(4),
        0.0,
    )

    # IV = (good_prob - bad_prob) * WoE
    if total_event > 0 and total_nonevent > 0:
        bad_prob  = result["Event"] / total_event
        good_prob = result["Non-event"] / total_nonevent
        result["IV"] = ((good_prob - bad_prob) * result["WoE"]).round(4)
    else:
        result["IV"] = 0.0

    # Totals satırı
    totals = {
        "Bin": "TOPLAM",
        "count": int(total_count),
        "Toplam (%)": 100.0,
        "Non-event": int(total_nonevent),
        "Event": int(total_event),
        "Bad Rate %": round(total_event / total_count * 100, 3) if total_count > 0 else 0.0,
        "WoE": np.nan,
        "IV": float(result["IV"].sum()),
    }
    result = pd.concat([result, pd.DataFrame([totals])], ignore_index=True)
    result = result.rename(columns={"count": "Toplam"})

    # Sayısal kolonları float'a zorla (to_dict round-trip güvenliği)
    for c in ["Toplam", "Toplam (%)", "Non-event", "Event", "Bad Rate %", "WoE", "IV"]:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors="coerce")

    # Virgülden sonra 3 basamak, WoE 8 basamak — format_binning_table ile tutarlı
    for c in ["Toplam (%)", "Bad Rate %", "IV"]:
        if c in result.columns:
            result[c] = result[c].round(4)
    if "WoE" in result.columns:
        result["WoE"] = result["WoE"].round(8)

    result = result[["Bin", "Toplam", "Toplam (%)", "Non-event", "Event",
                      "Bad Rate %", "WoE", "IV"]]
    return result


def _check_monotonicity(bt):
    """Binning tablosundan monotonluk kontrolü. Eksik/Special/TOPLAM hariç."""
    if bt is None or bt.empty:
        return "—"
    mask = (~bt["Bin"].isin(["Eksik", "TOPLAM"])
            & ~bt["Bin"].astype(str).str.startswith("Special"))
    br = pd.to_numeric(bt.loc[mask, "Bad Rate %"], errors="coerce").values
    diffs = np.diff(br)
    if len(diffs) == 0:
        return "—"
    if all(d >= 0 for d in diffs):
        return "Artan"
    if all(d <= 0 for d in diffs):
        return "Azalan"
    return "Monoton Değil"


