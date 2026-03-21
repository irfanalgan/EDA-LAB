import logging
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

SPECIAL_VALUES = {9999999999, 8888888888}

# Eşik: kolondaki special value oranı >= %2 ise "special kolon" say
_SPECIAL_RATIO_THRESHOLD = 0.02


def calc_total_iv(bt, X, y) -> float:
    """OptimalBinning bt'den toplam IV hesapla — special değerleri ayrıştırarak.
    Tüm IV hesaplamaları bu fonksiyonu kullanmalı.
    """
    total_bad = int(np.asarray(y).sum())
    total_good = len(y) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0

    iv = 0.0
    # Normal bin'ler + Missing → OptimalBinning'in IV'leri doğru
    for _, row in bt.iterrows():
        bin_name = str(row["Bin"])
        if bin_name in ("Special", "Totals", ""):
            continue
        iv += float(row["IV"])

    # Special → her değer için ayrı IV hesapla
    x_series = pd.Series(X)
    for sv in SPECIAL_VALUES:
        sv_mask = x_series == sv
        if not sv_mask.any():
            continue
        bad_sv = int(np.asarray(y)[sv_mask.values].sum())
        good_sv = int(sv_mask.sum()) - bad_sv
        d_b = bad_sv / total_bad if total_bad > 0 else 0
        d_g = good_sv / total_good if total_good > 0 else 0
        if d_b > 0 and d_g > 0:
            iv += (d_b - d_g) * np.log(d_b / d_g)

    return round(iv, 4)


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


def _build_binning_table_from_edges(optb, X, y, col, is_numeric):
    """
    Train'in fitted optb'sindeki bin sınırları + WoE değerleriyle
    yeni verinin (test/oot) dağılım tablosunu hesaplar.
    WoE → train'den (sabit), Count/Event Rate → yeni veriden,
    IV → yeni verinin dağılımı × train WoE.
    """
    # Train binning tablosundan WoE değerlerini al
    try:
        train_bt = optb.binning_table.build(show_digits=8)
        splits = list(optb.splits) if hasattr(optb, "splits") and optb.splits is not None else []
    except Exception as e:
        logger.debug("binning_table build failed for %s: %s", col, e)
        return pd.DataFrame(), 0.0

    if not splits and is_numeric:
        return pd.DataFrame(), 0.0

    edges = [-np.inf] + splits + [np.inf]
    eps = 1e-9

    total_n = len(y)
    total_bad = int(y.sum())
    total_good = total_n - total_bad
    if total_bad == 0 or total_good == 0:
        return pd.DataFrame(), 0.0

    # Train WoE per bin (Totals/Special/Missing hariç, sıralı)
    train_main = train_bt[~train_bt["Bin"].isin(["Special", "Missing", "Totals"])]
    train_main = train_main[train_main["Bin"].astype(str).str.len() > 0]
    train_woe_list = train_main["WoE"].tolist()  # bin sırasına göre

    # Special / Missing WoE
    _sp_row = train_bt[train_bt["Bin"] == "Special"]
    train_woe_special = float(_sp_row.iloc[0]["WoE"]) if not _sp_row.empty else 0.0
    _ms_row = train_bt[train_bt["Bin"] == "Missing"]
    train_woe_missing = float(_ms_row.iloc[0]["WoE"]) if not _ms_row.empty else 0.0

    # Special + Missing ayır
    x_series = pd.Series(X)
    special_mask = x_series.isin(SPECIAL_VALUES)
    missing_mask = x_series.isna()
    normal_mask = ~special_mask & ~missing_mask

    x_normal = x_series[normal_mask].values
    y_normal = y[normal_mask]

    # Bin'e ata
    bin_idx = np.digitize(x_normal, edges[1:-1], right=True)

    rows = []
    iv_total = 0.0
    for i in range(len(edges) - 1):
        mask_i = (bin_idx == i)
        n_i = int(mask_i.sum())
        bad_i = int(y_normal[mask_i].sum()) if n_i > 0 else 0
        good_i = n_i - bad_i
        bad_rate = bad_i / n_i if n_i > 0 else 0.0

        # Train WoE (sabit — tabloda gösterilecek)
        woe = float(train_woe_list[i]) if i < len(train_woe_list) else 0.0

        # IV: test dağılımı × train WoE → (d_b - d_g) * WoE_train
        dist_bad = bad_i / total_bad if total_bad > 0 else 0
        dist_good = good_i / total_good if total_good > 0 else 0
        iv_part = (dist_bad - dist_good) * woe
        iv_total += iv_part

        # Bin label — show_digits=8 ile tutarlı format
        lo, hi = edges[i], edges[i + 1]
        if lo == -np.inf:
            lbl = f"(-inf, {hi:.8f})"
        elif hi == np.inf:
            lbl = f"[{lo:.8f}, inf)"
        else:
            lbl = f"[{lo:.8f}, {hi:.8f})"

        rows.append({
            "Bin": lbl, "Toplam": n_i, "Bad": bad_i, "Good": good_i,
            "Bad Rate %": round(bad_rate * 100, 2),
            "WOE": round(woe, 4),
            "IV Katkı": round(float(iv_part), 4),
        })

    # Special — her değer ayrı satır, WoE ve IV test dağılımı × kendi WoE
    x_series_sp = pd.Series(X)
    for sv in sorted(SPECIAL_VALUES):
        sv_mask = x_series_sp == sv
        if not sv_mask.any():
            continue
        y_sv = y[sv_mask.values]
        n_sv = int(sv_mask.sum())
        bad_sv = int(y_sv.sum())
        good_sv = n_sv - bad_sv
        d_b = bad_sv / total_bad if total_bad > 0 else 0
        d_g = good_sv / total_good if total_good > 0 else 0
        sv_woe = round(float(np.log(d_b / d_g)), 4) if d_b > 0 and d_g > 0 else 0.0
        iv_sv = (d_b - d_g) * sv_woe
        iv_total += iv_sv
        rows.append({
            "Bin": f"Special ({int(sv)})", "Toplam": n_sv, "Bad": bad_sv,
            "Good": good_sv,
            "Bad Rate %": round(bad_sv / n_sv * 100, 2) if n_sv > 0 else 0.0,
            "WOE": sv_woe,
            "IV Katkı": round(float(iv_sv), 4),
        })

    # Missing — IV: test dağılımı × train WoE
    if missing_mask.any():
        y_ms = y[missing_mask]
        n_ms = int(missing_mask.sum())
        bad_ms = int(y_ms.sum())
        good_ms = n_ms - bad_ms
        d_b = bad_ms / total_bad if total_bad > 0 else 0
        d_g = good_ms / total_good if total_good > 0 else 0
        iv_ms = (d_b - d_g) * train_woe_missing
        iv_total += iv_ms
        rows.append({
            "Bin": "Eksik", "Toplam": n_ms, "Bad": bad_ms, "Good": good_ms,
            "Bad Rate %": round(bad_ms / n_ms * 100, 2) if n_ms > 0 else 0.0,
            "WOE": round(train_woe_missing, 4),
            "IV Katkı": round(float(iv_ms), 4),
        })

    if not rows:
        return pd.DataFrame(), 0.0

    result = pd.DataFrame(rows)
    total_all_n = result["Toplam"].sum()
    total_bad_n = result["Bad"].sum()
    total_row = pd.DataFrame([{
        "Bin": "TOPLAM", "Toplam": int(total_all_n), "Bad": int(total_bad_n),
        "Good": int(result["Good"].sum()),
        "Bad Rate %": round(total_bad_n / total_all_n * 100, 2) if total_all_n > 0 else 0.0,
        "WOE": "", "IV Katkı": round(float(iv_total), 4),
    }])
    result = pd.concat([result, total_row], ignore_index=True)
    return result, round(float(iv_total), 4)


def get_woe_detail(df: pd.DataFrame, col: str, target: str,
                   max_n_bins: int = 4,
                   force_dtype: str = None,
                   fitted_optb=None,
                   use_edges: bool = True) -> tuple[pd.DataFrame, float, object, object]:
    """
    OptimalBinning ile monoton WOE/IV tablosu döndürür.
    Returns: (woe_df, iv_total, bin_edges)
    bin_edges: [-inf, split1, ..., +inf] numpy array (numerik), None (kategorik).
    force_dtype: None | "numerical" | "categorical"  — otomatik algıyı ezer.
    fitted_optb: Önceden fit edilmiş OptimalBinning objesi verilirse yeniden fit etmez,
    use_edges: True ise test/oot için _build_binning_table_from_edges kullanır.
               False ise optb.binning_table.build() kullanır (train için).
                 bu objenin bin sınırlarıyla yeni veri üzerinde tablo oluşturur.
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
        return pd.DataFrame(), 0.0, None, None

    is_numeric = pd.api.types.is_numeric_dtype(local[col])
    if force_dtype == "categorical":
        is_numeric = False
    elif force_dtype == "numerical":
        is_numeric = True
    X = local[col].values
    y = local[target].values.astype(int)

    try:
        if fitted_optb is not None:
            optb = fitted_optb
        else:
            _special_col = is_numeric and is_special_column(local[col])
            _bins = 2 if _special_col else max_n_bins
            kwargs = dict(
                name=col,
                monotonic_trend="auto_asc_desc",
                max_n_bins=_bins,
                solver="cp",
                dtype="numerical" if is_numeric else "categorical",
            )
            if is_numeric:
                kwargs["special_codes"] = list(SPECIAL_VALUES)
            optb = _OB(**kwargs)
            optb.fit(X, y)

        # fitted_optb + use_edges ise: train'in bin sınırlarıyla yeni verinin tablosunu hesapla
        if fitted_optb is not None and is_numeric and use_edges:
            result, iv_total = _build_binning_table_from_edges(
                optb, X, y, col, is_numeric)
            if result.empty:
                return pd.DataFrame(), 0.0, None, None
            _bin_edges = None
            try:
                splits = list(optb.splits)
                if splits:
                    _bin_edges = np.array([-np.inf] + splits + [np.inf], dtype=float)
            except Exception:
                pass
            return result, iv_total, _bin_edges, optb

        bt = optb.binning_table.build(show_digits=8)
        iv_total = calc_total_iv(bt, X, y)
    except Exception as e:
        logger.debug("OptBinning fit/build failed for %s: %s", col, e)
        return pd.DataFrame(), 0.0, None, None

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

    # Special — her değer ayrı satır, kendi WoE ve IV'ü hesaplanır
    sr_special = bt[bt["Bin"] == "Special"]
    if not sr_special.empty and int(sr_special.iloc[0]["Count"]) > 0:
        x_series = pd.Series(X)
        sv_count = 0
        for sv in sorted(SPECIAL_VALUES):
            sv_mask = x_series == sv
            if not sv_mask.any():
                continue
            sv_count += 1
            n_sv = int(sv_mask.sum())
            bad_sv = int(y[sv_mask.values].sum())
            good_sv = n_sv - bad_sv
            # Her special değerin kendi WoE ve IV katkısını hesapla
            d_b = bad_sv / total_bad if total_bad > 0 else 0
            d_g = good_sv / total_good if total_good > 0 else 0
            if d_b > 0 and d_g > 0:
                sv_woe = round(np.log(d_b / d_g), 4)
                sv_iv = round((d_b - d_g) * np.log(d_b / d_g), 4)
            else:
                sv_woe = 0.0
                sv_iv = 0.0
            rows.append({"Bin": f"Special ({int(sv)})", "Toplam": n_sv,
                         "Bad": bad_sv, "Good": good_sv,
                         "Bad Rate %": round(bad_sv / n_sv * 100, 2) if n_sv > 0 else 0.0,
                         "WOE": sv_woe, "IV Katkı": sv_iv})
        # Hiçbir special value ayrı bulunamadıysa orijinal satırı koy
        if sv_count == 0:
            r = sr_special.iloc[0]
            sp_woe = round(float(r["WoE"]), 4)
            sp_iv = round(float(r["IV"]), 4)
            rows.append({"Bin": "Special", "Toplam": int(r["Count"]),
                         "Bad": int(r["Event"]), "Good": int(r["Non-event"]),
                         "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
                         "WOE": sp_woe, "IV Katkı": sp_iv})
    # Missing
    sr_missing = bt[bt["Bin"] == "Missing"]
    if not sr_missing.empty and int(sr_missing.iloc[0]["Count"]) > 0:
        r = sr_missing.iloc[0]
        rows.append({"Bin": "Eksik", "Toplam": int(r["Count"]),
                     "Bad": int(r["Event"]), "Good": int(r["Non-event"]),
                     "Bad Rate %": round(float(r["Event rate"]) * 100, 2),
                     "WOE": round(float(r["WoE"]), 4),
                     "IV Katkı": round(float(r["IV"]), 4)})

    if not rows:
        return pd.DataFrame(), 0.0, None, None

    result = pd.DataFrame(rows)
    total_all_n = result["Toplam"].sum()
    total_bad_n = result["Bad"].sum()
    iv_total = float(result["IV Katkı"].sum())
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

    return result, iv_total, _bin_edges, optb


def compute_period_badrate(df: pd.DataFrame, col: str, target: str,
                           woe_df: pd.DataFrame, bin_edges) -> tuple:
    """
    Train WOE binlerini yeni bir df'ye (test/OOT) uygulayarak her bin için
    bad rate, WOE (train'den) ve IV Katkı hesaplar.

    Returns (DataFrame, iv_total).
    DataFrame columns: [Bin, Toplam, Bad, Bad Rate %, WOE, IV Katkı]
    """
    local = df[[col, target]].copy()
    local[target] = pd.to_numeric(local[target], errors="coerce")
    local = local.dropna(subset=[target])
    if local.empty:
        return pd.DataFrame(), 0.0

    is_numeric = pd.api.types.is_numeric_dtype(local[col])
    eps = 1e-9
    total_n = len(local)
    total_bad = int(local[target].sum())
    total_good = total_n - total_bad

    # Train WOE referansı (TOPLAM hariç)
    woe_ref_df = woe_df[woe_df["Bin"] != "TOPLAM"].copy()
    woe_ref_map = dict(zip(woe_ref_df["Bin"], woe_ref_df["WOE"]))

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

        _cut = pd.cut(present[col], bins=bin_edges, include_lowest=False, labels=False)
        present["bin_idx"] = _cut

        rows = []
        for i, bin_lbl in enumerate(main_bins_df["Bin"]):
            grp   = present[present["bin_idx"] == i]
            total = len(grp)
            bad   = int(grp[target].sum())
            good  = total - bad
            woe   = woe_ref_map.get(bin_lbl, 0.0)
            woe   = float(woe) if woe != "" else 0.0
            # IV: test dağılımı × train WoE
            d_b = bad / total_bad if total_bad > 0 else 0
            d_g = good / total_good if total_good > 0 else 0
            iv_part = (d_b - d_g) * woe
            rows.append({"Bin": bin_lbl, "Toplam": total, "Bad": bad,
                         "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0,
                         "WOE": round(woe, 4), "IV Katkı": round(float(iv_part), 4)})

        # Special bins
        for sv in SPECIAL_VALUES:
            sv_grp = special_data[special_data[col] == sv]
            if len(sv_grp):
                lbl   = f"Special ({int(sv)})"
                total = len(sv_grp)
                bad   = int(sv_grp[target].sum())
                good  = total - bad
                woe   = woe_ref_map.get(lbl, 0.0)
                woe   = float(woe) if woe != "" else 0.0
                d_b = bad / total_bad if total_bad > 0 else 0
                d_g = good / total_good if total_good > 0 else 0
                iv_part = (d_b - d_g) * woe
                rows.append({"Bin": lbl, "Toplam": total, "Bad": bad,
                             "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0,
                             "WOE": round(woe, 4), "IV Katkı": round(float(iv_part), 4)})
        # Eksik bin
        if len(missing):
            total = len(missing)
            bad   = int(missing[target].sum())
            good  = total - bad
            woe   = woe_ref_map.get("Eksik", 0.0)
            woe   = float(woe) if woe != "" else 0.0
            d_b = bad / total_bad if total_bad > 0 else 0
            d_g = good / total_good if total_good > 0 else 0
            iv_part = (d_b - d_g) * woe
            rows.append({"Bin": "Eksik", "Toplam": total, "Bad": bad,
                         "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0,
                         "WOE": round(woe, 4), "IV Katkı": round(float(iv_part), 4)})
    else:
        # Kategorik: eşleştirme değer adına göre
        local["bin_label"] = local[col].fillna("Eksik").astype(str)
        all_train_bins = woe_df[woe_df["Bin"] != "TOPLAM"]["Bin"].tolist()
        rows = []
        for lbl in all_train_bins:
            grp   = local[local["bin_label"] == lbl]
            total = len(grp)
            bad   = int(grp[target].sum())
            good  = total - bad
            woe   = woe_ref_map.get(lbl, 0.0)
            woe   = float(woe) if woe != "" else 0.0
            d_b = bad / total_bad if total_bad > 0 else 0
            d_g = good / total_good if total_good > 0 else 0
            iv_part = (d_b - d_g) * woe
            rows.append({"Bin": lbl, "Toplam": total, "Bad": bad,
                         "Bad Rate %": round(bad / total * 100, 2) if total > 0 else 0.0,
                         "WOE": round(woe, 4), "IV Katkı": round(float(iv_part), 4)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result, 0.0
    iv_total = float(result["IV Katkı"].sum())
    total_all = result["Toplam"].sum()
    total_bad_r = result["Bad"].sum()
    total_row = pd.DataFrame([{
        "Bin": "TOPLAM", "Toplam": int(total_all), "Bad": int(total_bad_r),
        "Bad Rate %": round(total_bad_r / total_all * 100, 2) if total_all > 0 else 0.0,
        "WOE": "", "IV Katkı": round(iv_total, 4),
    }])
    return pd.concat([result, total_row], ignore_index=True), round(iv_total, 4)


def compute_iv_ranking_optimal(df: pd.DataFrame, target: str,
                                max_n_bins: int = 4) -> pd.DataFrame:
    """Tüm değişkenler için IV hesaplar (tam veri)."""
    records = []
    for col in df.columns:
        if col == target:
            continue
        try:
            _, iv, _, _ = get_woe_detail(df, col, target, max_n_bins)
            records.append({
                "Değişken": col,
                "IV":       round(iv, 4),
                "Eksik %":  round(df[col].isna().mean() * 100, 2),
                "Güç":      _iv_label(iv),
            })
        except Exception as e:
            logger.debug("IV ranking failed for %s: %s", col, e)
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
    Returns: (woe_series, iv_total, success: bool, optb_obj | None)
    Başarısızsa woe_series = NaN dolu, success = False, optb_obj = None.
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
        return pd.Series(np.nan, index=df.index), 0.0, False, None

    is_numeric = pd.api.types.is_numeric_dtype(local[col])

    try:
        _special_col = is_numeric and is_special_column(local[col])
        _bins = 2 if _special_col else max_n_bins
        kwargs = dict(name=col, monotonic_trend="auto_asc_desc",
                      max_n_bins=_bins, solver="cp",
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
        return pd.Series(np.nan, index=df.index), 0.0, False, None

    return woe_series, iv, True, optb
