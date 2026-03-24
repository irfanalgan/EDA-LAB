"""İzleme — hesaplama motoru.

Referans ve izleme verilerinden dönemsel özetler çıkarır.
Tüm metrikler bu özetlerden hesaplanır — ham veri saklanmaz.

Background thread ile çalışır, iptal edilebilir (threading.Event).
"""

import threading
import math
import logging
import numpy as np
import pandas as pd
from scipy.stats import binom

logger = logging.getLogger(__name__)

from server_state import _MON_STORE, _PRECOMPUTE_PROGRESS

# ══════════════════════════════════════════════════════════════════════════════
#   Sabit Değerler — Rating tablosu
# ══════════════════════════════════════════════════════════════════════════════

# PD → Rating ataması için eşik değerleri (25 aralık)
RATING_THRESHOLDS = [
    0.00032, 0.00044, 0.0006, 0.00083, 0.00114, 0.00156, 0.00215, 0.00297,
    0.00409, 0.00563, 0.00775, 0.01067, 0.0147, 0.02024, 0.02788, 0.03839,
    0.05287, 0.0728, 0.10026, 0.13807, 0.19014, 0.26185, 0.36059, 0.49659, 1.0,
]

# Her rating'in ortalama PD'si (MIDPD) — backtesting için
MIDPD = [
    0.0002, 0.00037, 0.00051, 0.0007, 0.00097, 0.00133, 0.00184, 0.00253,
    0.00348, 0.0048, 0.0066, 0.0091, 0.01253, 0.01725, 0.02375, 0.03271,
    0.04505, 0.06204, 0.08543, 0.11765, 0.16203, 0.22313, 0.30728, 0.42316,
    0.58275,
]

N_RATINGS = 25


# ══════════════════════════════════════════════════════════════════════════════
#   Thread yönetimi
# ══════════════════════════════════════════════════════════════════════════════

_mon_active_thread: threading.Thread | None = None
_mon_cancel_event = threading.Event()


def cancel_mon_compute():
    """Çalışan hesaplama thread'ini iptal et."""
    global _mon_active_thread
    _mon_cancel_event.set()
    if _mon_active_thread and _mon_active_thread.is_alive():
        _mon_active_thread.join(timeout=3)
    _mon_active_thread = None
    _mon_cancel_event.clear()


# ══════════════════════════════════════════════════════════════════════════════
#   Yardımcı fonksiyonlar
# ══════════════════════════════════════════════════════════════════════════════

def assign_rating(pd_value):
    """PD değerine göre 1-25 rating ata."""
    for i, threshold in enumerate(RATING_THRESHOLDS):
        if pd_value < threshold:
            return i + 1
    if pd_value >= 1.0:
        return 26
    return 0


def assign_ratings_vectorized(pd_series):
    """Vektörize rating ataması — büyük veri setleri için."""
    ratings = np.searchsorted(RATING_THRESHOLDS, pd_series.values, side="left") + 1
    # PD == 1.0 → rating 26
    ratings[pd_series.values >= 1.0] = 26
    # PD < 0 veya NaN → rating 0
    ratings[pd_series.isna().values] = 0
    return ratings


def detect_pd_or_rating(series):
    """PD kolonu mu (0-1 float) yoksa Rating kolonu mu (1-25 int) algıla.

    Returns: "pd" veya "rating"
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return "pd"
    # Tüm değerler 1-25 arası tam sayı ise → rating
    if clean.min() >= 1 and clean.max() <= 25:
        int_vals = clean.astype(int)
        if (clean == int_vals).all():
            return "rating"
    # 1-26 dahil (26 = default rating)
    if clean.min() >= 1 and clean.max() <= 26:
        int_vals = clean.astype(int)
        if (clean == int_vals).all():
            return "rating"
    return "pd"


def get_ratings(series):
    """PD veya Rating serisinden 1-25 rating dizisi döndür.

    Otomatik algılama yapar:
      - PD (0-1): assign_ratings_vectorized ile dönüştürür
      - Rating (1-25): doğrudan kullanır, 25'i aşanları N_RATINGS'e kırpar
    """
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    col_type = detect_pd_or_rating(series)
    if col_type == "rating":
        ratings = numeric.astype(int).values.copy()
        ratings[ratings < 1] = 0
        ratings[ratings > N_RATINGS] = N_RATINGS
        return ratings
    return assign_ratings_vectorized(numeric)


def _safe_psi(ref_pct, mon_pct):
    """Tek bir bin için PSI katkısı. Sıfır koruma ile."""
    eps = 1e-8
    r = max(ref_pct, eps)
    m = max(mon_pct, eps)
    return (m - r) * math.log(m / r)


def _compute_var_edges_from_ref(values, n_bins=10):
    """Referans verisinden quantile bin edge'leri hesapla."""
    clean = values.dropna()
    if len(clean) == 0:
        return [-np.inf, np.inf]
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = list(np.unique(np.quantile(clean, quantiles)))
    # -inf ve +inf ekle
    if edges[0] != -np.inf:
        edges = [-np.inf] + edges
    if edges[-1] != np.inf:
        edges = edges + [np.inf]
    return edges


def _bin_counts(values, edges):
    """Değerleri verilen edge'lere göre bin'le, count dizisi döndür."""
    counts = np.histogram(values.dropna(), bins=edges)[0]
    return counts.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#   Referans Özeti Hesaplama
# ══════════════════════════════════════════════════════════════════════════════

def compute_ref_summary(ref_df, config, opt_dict=None):
    """Referans verisinden özet çıkar.

    Returns: ref_summary dict
    """
    target_col = config["target_col"]
    pd_col = config["pd_col"]
    model_vars = config.get("model_vars", [])
    woe_enabled = config.get("woe_enabled", False)

    target = pd.to_numeric(ref_df[target_col], errors="coerce").fillna(0)

    # Rating ataması (PD veya Rating kolonu otomatik algılanır)
    ratings = get_ratings(ref_df[pd_col])
    rating_counts = [0] * N_RATINGS
    rating_defaults = [0] * N_RATINGS
    for r in range(1, N_RATINGS + 1):
        mask = ratings == r
        rating_counts[r - 1] = int(mask.sum())
        rating_defaults[r - 1] = int(target[mask].sum())

    n_total = int(target.count())
    n_bad = int(target.sum())

    # Değişken PSI referans bin'leri
    var_psi = {}
    for var in model_vars:
        if var not in ref_df.columns:
            continue
        col_data = ref_df[var]

        if woe_enabled and opt_dict and var in opt_dict:
            # WoE dönüşümü uygula
            try:
                woe_values = opt_dict[var].transform(
                    col_data.values,
                    metric="woe",
                    metric_missing="empirical",
                    metric_special="empirical",
                )
                # WoE unique değerleri edge olarak kullan (discrete)
                unique_woe = sorted(set(woe_values[~np.isnan(woe_values)]))
                # Discrete bin'ler: her unique WoE değeri bir bin
                edges = [-np.inf] + unique_woe + [np.inf]
                ref_counts = _bin_counts(pd.Series(woe_values), edges)
                # Bad counts per bin
                woe_series = pd.Series(woe_values)
                ref_bad_counts = []
                for j in range(len(edges) - 1):
                    lo, hi = edges[j], edges[j + 1]
                    if j == len(edges) - 2:
                        mask = (woe_series >= lo) & (woe_series <= hi)
                    else:
                        mask = (woe_series >= lo) & (woe_series < hi)
                    ref_bad_counts.append(int(target[mask].sum()))

                var_psi[var] = {
                    "edges": edges,
                    "woe_values": unique_woe,
                    "ref_counts": ref_counts,
                    "ref_bad_counts": ref_bad_counts,
                }
                continue  # WoE başarılı, sonraki değişkene geç
            except Exception as e:
                logger.warning("WoE transform failed for %s: %s — raw fallback", var, e)

        # Ham değerler — quantile bin'leme (WoE kapalıysa VEYA WoE fail olduysa)
        num_data = pd.to_numeric(col_data, errors="coerce")
        edges = _compute_var_edges_from_ref(num_data, n_bins=10)
        ref_counts = _bin_counts(num_data, edges)
        ref_bad_counts = []
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            if j == len(edges) - 2:
                mask = (num_data >= lo) & (num_data <= hi)
            else:
                mask = (num_data >= lo) & (num_data < hi)
            ref_bad_counts.append(int(target[mask.fillna(False)].sum()))

        var_psi[var] = {
            "edges": edges,
            "woe_values": None,
            "ref_counts": ref_counts,
            "ref_bad_counts": ref_bad_counts,
        }

    return {
        "period_label": "Referans",
        "n_total": n_total,
        "n_bad": n_bad,
        "bad_rate": n_bad / n_total if n_total > 0 else 0,
        "rating_counts": rating_counts,
        "rating_defaults": rating_defaults,
        "var_psi": var_psi,
    }


# ══════════════════════════════════════════════════════════════════════════════
#   Dönem Özeti Hesaplama
# ══════════════════════════════════════════════════════════════════════════════

def compute_period_summary(period_df, period_label, config, ref_summary,
                           opt_dict=None, ref_df=None, id_col=None):
    """Tek bir dönemin özetini hesapla.

    Args:
        period_df: O döneme ait DataFrame
        period_label: "2024-01" gibi etiket
        config: Yapılandırma dict'i
        ref_summary: Referans özeti (edge'ler için)
        opt_dict: WoE opt dict (opsiyonel)
        ref_df: Referans DataFrame (göç matrisi için, opsiyonel)
        id_col: ID kolonu adı (göç matrisi için, opsiyonel)

    Returns: period_summary dict
    """
    target_col = config["target_col"]
    pd_col = config["pd_col"]
    model_vars = config.get("model_vars", [])
    woe_enabled = config.get("woe_enabled", False)

    target = pd.to_numeric(period_df[target_col], errors="coerce").fillna(0)

    # Rating ataması (PD veya Rating kolonu otomatik algılanır)
    ratings = get_ratings(period_df[pd_col])
    rating_counts = [0] * N_RATINGS
    rating_defaults = [0] * N_RATINGS
    for r in range(1, N_RATINGS + 1):
        mask = ratings == r
        rating_counts[r - 1] = int(mask.sum())
        rating_defaults[r - 1] = int(target[mask].sum())

    n_total = int(target.count())
    n_bad = int(target.sum())

    # Değişken PSI — referans edge'lerini kullan
    var_psi = {}
    for var in model_vars:
        if var not in period_df.columns or var not in ref_summary.get("var_psi", {}):
            continue
        ref_var = ref_summary["var_psi"][var]
        edges = ref_var["edges"]

        if woe_enabled and opt_dict and var in opt_dict:
            try:
                woe_values = opt_dict[var].transform(
                    period_df[var].values,
                    metric="woe",
                    metric_missing="empirical",
                    metric_special="empirical",
                )
                mon_counts = _bin_counts(pd.Series(woe_values), edges)
                # Bad counts per bin
                woe_series = pd.Series(woe_values)
                mon_bad_counts = []
                for j in range(len(edges) - 1):
                    lo, hi = edges[j], edges[j + 1]
                    if j == len(edges) - 2:
                        m = (woe_series >= lo) & (woe_series <= hi)
                    else:
                        m = (woe_series >= lo) & (woe_series < hi)
                    mon_bad_counts.append(int(target[m].sum()))

                var_psi[var] = {
                    "edges": edges,
                    "woe_values": ref_var.get("woe_values"),
                    "mon_counts": mon_counts,
                    "mon_bad_counts": mon_bad_counts,
                }
                continue  # WoE başarılı
            except Exception as e:
                logger.warning("WoE transform failed (period) for %s: %s — raw fallback", var, e)

        # Ham değerler — quantile bin'leme (WoE kapalıysa VEYA fail olduysa)
        num_data = pd.to_numeric(period_df[var], errors="coerce")
        ref_edges = ref_var["edges"]
        mon_counts = _bin_counts(num_data, ref_edges)
        mon_bad_counts = []
        for j in range(len(ref_edges) - 1):
            lo, hi = ref_edges[j], ref_edges[j + 1]
            if j == len(ref_edges) - 2:
                m = (num_data >= lo) & (num_data <= hi)
            else:
                m = (num_data >= lo) & (num_data < hi)
            mon_bad_counts.append(int(target[m.fillna(False)].sum()))

        var_psi[var] = {
            "edges": ref_edges,
            "woe_values": None,
            "mon_counts": mon_counts,
            "mon_bad_counts": mon_bad_counts,
        }

    summary = {
        "period_label": period_label,
        "period_start": str(period_label),
        "n_total": n_total,
        "n_bad": n_bad,
        "bad_rate": n_bad / n_total if n_total > 0 else 0,
        "rating_counts": rating_counts,
        "rating_defaults": rating_defaults,
        "var_psi": var_psi,
        "migration_matrix": None,
        "migration_matched_count": 0,
    }

    # Göç Matrisi — ID eşleştirme
    if id_col and ref_df is not None and id_col in period_df.columns and id_col in ref_df.columns:
        try:
            ref_ratings = get_ratings(ref_df[pd_col])
            ref_id_rating = pd.DataFrame({
                "id": ref_df[id_col],
                "ref_rating": ref_ratings,
            })
            mon_id_rating = pd.DataFrame({
                "id": period_df[id_col],
                "mon_rating": ratings,
            })
            merged = pd.merge(ref_id_rating, mon_id_rating, on="id", how="inner")

            if len(merged) > 0:
                matrix = [[0] * N_RATINGS for _ in range(N_RATINGS)]
                for _, row in merged.iterrows():
                    rr = int(row["ref_rating"])
                    mr = int(row["mon_rating"])
                    if 1 <= rr <= N_RATINGS and 1 <= mr <= N_RATINGS:
                        matrix[rr - 1][mr - 1] += 1
                summary["migration_matrix"] = matrix
                summary["migration_matched_count"] = len(merged)
        except Exception:
            pass

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#   Kümülatif Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_summaries(summaries, mature_only=False):
    """Dönemsel özetleri birleştirerek kümülatif özet üret."""
    if not summaries:
        return None

    filtered = summaries
    if mature_only:
        filtered = [s for s in summaries if s.get("is_mature", False)]
    if not filtered:
        return None

    n_total = sum(s["n_total"] for s in filtered)
    n_bad = sum(s["n_bad"] for s in filtered)

    # Rating counts/defaults element-wise toplam
    rating_counts = [0] * N_RATINGS
    rating_defaults = [0] * N_RATINGS
    for s in filtered:
        for i in range(N_RATINGS):
            rating_counts[i] += s["rating_counts"][i]
            rating_defaults[i] += s["rating_defaults"][i]

    # Var PSI — mon_counts element-wise toplam
    var_psi = {}
    for s in filtered:
        for var, data in s.get("var_psi", {}).items():
            if var not in var_psi:
                var_psi[var] = {
                    "edges": data["edges"],
                    "woe_values": data.get("woe_values"),
                    "mon_counts": list(data["mon_counts"]),
                    "mon_bad_counts": list(data["mon_bad_counts"]),
                }
            else:
                existing = var_psi[var]
                for j in range(len(existing["mon_counts"])):
                    existing["mon_counts"][j] += data["mon_counts"][j]
                    existing["mon_bad_counts"][j] += data["mon_bad_counts"][j]

    # Migration matrix — element-wise toplam
    migration_matrix = None
    migration_matched = 0
    for s in filtered:
        m = s.get("migration_matrix")
        if m is not None:
            if migration_matrix is None:
                migration_matrix = [row[:] for row in m]
            else:
                for i in range(N_RATINGS):
                    for j in range(N_RATINGS):
                        migration_matrix[i][j] += m[i][j]
            migration_matched += s.get("migration_matched_count", 0)

    return {
        "n_total": n_total,
        "n_bad": n_bad,
        "bad_rate": n_bad / n_total if n_total > 0 else 0,
        "rating_counts": rating_counts,
        "rating_defaults": rating_defaults,
        "var_psi": var_psi,
        "migration_matrix": migration_matrix,
        "migration_matched_count": migration_matched,
    }


# ══════════════════════════════════════════════════════════════════════════════
#   Metrik Hesaplama Fonksiyonları (özetlerden)
# ══════════════════════════════════════════════════════════════════════════════

def calc_ks_from_summary(rating_counts, rating_defaults):
    """Rating bazlı KS hesapla. Returns (ks_value, ks_table_rows)."""
    total_good = sum(c - d for c, d in zip(rating_counts, rating_defaults))
    total_bad = sum(rating_defaults)
    if total_good == 0 or total_bad == 0:
        return 0.0, []

    rows = []
    cum_good = 0
    cum_bad = 0
    for i in range(N_RATINGS):
        count = rating_counts[i]
        default = rating_defaults[i]
        good = count - default
        cum_good += good
        cum_bad += default
        pct_cum_good = cum_good / total_good * 100
        pct_cum_bad = cum_bad / total_bad * 100
        diff = pct_cum_bad - pct_cum_good
        rows.append({
            "rating": i + 1,
            "good": good,
            "bad": default,
            "total": count,
            "bad_rate": default / count if count > 0 else 0,
            "cum_good": cum_good,
            "cum_bad": cum_bad,
            "pct_cum_good": pct_cum_good,
            "pct_cum_bad": pct_cum_bad,
            "diff": diff,
        })

    ks = max(abs(r["diff"]) for r in rows) if rows else 0.0
    return ks, rows


def calc_gini_from_summary(rating_counts, rating_defaults):
    """Rating bazlı Gini/AR hesapla (CAP eğrisi).
    Kötüden iyiye (25→1) sıralama ile.
    Returns (gini, ar, cap_area, details_dict).
    details_dict["rows"] = satır bazlı detaylar (Gini tablosu için).
    """
    total = sum(rating_counts)
    total_bad = sum(rating_defaults)
    total_good = total - total_bad
    if total == 0 or total_bad == 0:
        return 0.0, 0.0, 0.0, {}

    # Kötüden iyiye sırala
    indices = list(range(N_RATINGS - 1, -1, -1))

    cum_good = 0
    cum_bad = 0
    cum_total = 0
    prev_pct_cum_bad = 0.0
    prev_pct_cum_total = 0.0
    gini_area_sum = 0.0
    gini_rows = []

    for idx in indices:
        count = rating_counts[idx]
        default = rating_defaults[idx]
        good = count - default
        cum_good += good
        cum_bad += default
        cum_total += count

        pct_cum_good = cum_good / total_good * 100 if total_good > 0 else 0
        pct_cum_bad = cum_bad / total_bad * 100 if total_bad > 0 else 0
        pct_cum_total = cum_total / total * 100 if total > 0 else 0

        # Trapez alanı (0-1 ölçeğinde hesapla)
        pct_bad_01 = cum_bad / total_bad if total_bad > 0 else 0
        pct_total_01 = cum_total / total if total > 0 else 0
        gini_area = (pct_bad_01 + prev_pct_cum_bad) / 2 * (pct_total_01 - prev_pct_cum_total)
        gini_area_sum += gini_area

        gini_rows.append({
            "rating": idx + 1,
            "good": good,
            "bad": default,
            "total": count,
            "bad_rate": default / count if count > 0 else 0,
            "cum_good": cum_good,
            "cum_bad": cum_bad,
            "cum_total": cum_total,
            "pct_cum_good": pct_cum_good,
            "pct_cum_bad": pct_cum_bad,
            "pct_cum_total": pct_cum_total,
            "gini_area": gini_area,
            "random": pct_cum_total,       # Random = kümülatif % total
            "perfect_curve": pct_cum_bad,  # Perfect Curve = kümülatif % bad
        })

        prev_pct_cum_bad = pct_bad_01
        prev_pct_cum_total = pct_total_01

    bad_rate = total_bad / total
    random_area = 0.5
    perfect_area = 1 - bad_rate / 2
    cap_minus_random = gini_area_sum - random_area
    perfect_minus_random = perfect_area - random_area
    ar = cap_minus_random / perfect_minus_random if perfect_minus_random > 0 else 0
    gini = 2 * gini_area_sum - 1

    return gini, ar, gini_area_sum, {
        "total_area_cap": gini_area_sum,
        "cap_minus_random": cap_minus_random,
        "perfect_minus_random": perfect_minus_random,
        "accuracy_ratio": ar,
        "gini": gini,
        "rows": gini_rows,
    }


def calc_backtesting_table(rating_counts, rating_defaults, confidence=0.95):
    """Binomial test tablosu üret.
    Returns: list of row dicts.
    """
    total_count = sum(rating_counts)
    total_default = sum(rating_defaults)
    rows = []

    for i in range(N_RATINGS):
        count = rating_counts[i]
        default = rating_defaults[i]
        midpd = MIDPD[i]
        dr = default / count if count > 0 else 0
        concentration = count / total_count if total_count > 0 else 0

        if count > 0:
            upper = binom.ppf(confidence, count, midpd) / count
            lower = binom.ppf(1 - confidence, count, midpd) / count
        else:
            upper = 0
            lower = 0

        upper = round(upper, 4) if not np.isnan(upper) else 0
        lower = round(lower, 4) if not np.isnan(lower) else 0

        upper_flag = "Exceeding the Range" if dr > upper else "Within the Range"
        if dr < lower:
            lower_flag = "Below the Range"
        elif dr > upper:
            lower_flag = "Exceeding the Range"
        else:
            lower_flag = "Within the Range"

        conservatism = midpd >= dr

        rows.append({
            "rating": i + 1,
            "count": count,
            "concentration": concentration,
            "default": default,
            "dr": dr,
            "midpd": midpd,
            "conservatism": conservatism,
            "upper_limit": upper,
            "lower_limit": lower,
            "upper_flag": upper_flag,
            "lower_flag": lower_flag,
        })

    # Monotonicity
    for j in range(len(rows)):
        if j == 0:
            rows[j]["monotonicity"] = True
        else:
            rows[j]["monotonicity"] = rows[j]["dr"] >= rows[j - 1]["dr"]

    # Grand Total
    grand_dr = total_default / total_count if total_count > 0 else 0
    grand_midpd = (sum(c * m for c, m in zip(rating_counts, MIDPD)) / total_count
                   if total_count > 0 else 0)

    if total_count > 0:
        grand_upper = binom.ppf(confidence, total_count, grand_midpd) / total_count
        grand_lower = binom.ppf(1 - confidence, total_count, grand_midpd) / total_count
    else:
        grand_upper = 0
        grand_lower = 0

    grand_upper = round(grand_upper, 4) if not np.isnan(grand_upper) else 0
    grand_lower = round(grand_lower, 4) if not np.isnan(grand_lower) else 0

    rows.append({
        "rating": "Grand_Total",
        "count": total_count,
        "concentration": 1.0,
        "default": total_default,
        "dr": grand_dr,
        "midpd": grand_midpd,
        "conservatism": grand_midpd >= grand_dr,
        "monotonicity": True,
        "upper_limit": grand_upper,
        "lower_limit": grand_lower,
        "upper_flag": "Exceeding the Range" if grand_dr > grand_upper else "Within the Range",
        "lower_flag": ("Below the Range" if grand_dr < grand_lower
                       else ("Exceeding the Range" if grand_dr > grand_upper
                             else "Within the Range")),
    })

    return rows


def calc_hhi_from_summary(rating_counts):
    """Rating bazlı HHI hesapla. Returns (hhi_value, table_rows)."""
    total = sum(rating_counts)
    if total == 0:
        return 0.0, []

    rows = []
    hhi = 0.0
    for i in range(N_RATINGS):
        count = rating_counts[i]
        share = count / total
        hhi_contrib = share ** 2
        hhi += hhi_contrib
        rows.append({
            "rating": i + 1,
            "count": count,
            "share": share,
            "hhi_contrib": hhi_contrib,
        })

    return hhi, rows


def calc_var_psi(ref_var_psi, mon_var_psi):
    """Değişken PSI hesapla (tek değişken için).
    Returns (psi_total, bin_rows).
    """
    ref_counts = ref_var_psi["ref_counts"]
    mon_counts = mon_var_psi["mon_counts"]
    edges = ref_var_psi["edges"]
    woe_values = ref_var_psi.get("woe_values")

    ref_total = sum(ref_counts)
    mon_total = sum(mon_counts)
    if ref_total == 0 or mon_total == 0:
        return 0.0, []

    ref_bad = ref_var_psi.get("ref_bad_counts", [0] * len(ref_counts))
    mon_bad = mon_var_psi.get("mon_bad_counts", [0] * len(mon_counts))

    rows = []
    psi_total = 0.0
    for j in range(len(ref_counts)):
        ref_pct = ref_counts[j] / ref_total
        mon_pct = mon_counts[j] / mon_total
        psi_bin = _safe_psi(ref_pct, mon_pct)
        psi_total += psi_bin

        # IV katkı (ref ve mon ayrı)
        ref_good_j = ref_counts[j] - ref_bad[j]
        mon_good_j = mon_counts[j] - mon_bad[j]
        ref_total_good = ref_total - sum(ref_bad)
        ref_total_bad = sum(ref_bad)
        mon_total_good = mon_total - sum(mon_bad)
        mon_total_bad = sum(mon_bad)

        row = {
            "bin_idx": j,
            "edge_lo": edges[j] if j < len(edges) else None,
            "edge_hi": edges[j + 1] if j + 1 < len(edges) else None,
            "ref_count": ref_counts[j],
            "ref_pct": ref_pct,
            "mon_count": mon_counts[j],
            "mon_pct": mon_pct,
            "psi_contrib": psi_bin,
            "ref_bad": ref_bad[j],
            "mon_bad": mon_bad[j],
        }

        if woe_values and j < len(woe_values):
            row["woe"] = woe_values[j]

            # Ref IV katkı
            if ref_total_good > 0 and ref_total_bad > 0:
                g_pct = ref_good_j / ref_total_good if ref_total_good > 0 else 0
                b_pct = ref_bad[j] / ref_total_bad if ref_total_bad > 0 else 0
                eps = 1e-8
                row["ref_iv_contrib"] = (g_pct - b_pct) * math.log(max(g_pct, eps) / max(b_pct, eps))
            else:
                row["ref_iv_contrib"] = 0

            # Mon IV katkı
            if mon_total_good > 0 and mon_total_bad > 0:
                g_pct = mon_good_j / mon_total_good if mon_total_good > 0 else 0
                b_pct = mon_bad[j] / mon_total_bad if mon_total_bad > 0 else 0
                eps = 1e-8
                row["mon_iv_contrib"] = (g_pct - b_pct) * math.log(max(g_pct, eps) / max(b_pct, eps))
            else:
                row["mon_iv_contrib"] = 0

        rows.append(row)

    return psi_total, rows


def calc_rating_psi(ref_rating_counts, mon_rating_counts):
    """Rating bazlı PSI hesapla. Returns (psi_total, bin_rows)."""
    ref_total = sum(ref_rating_counts)
    mon_total = sum(mon_rating_counts)
    if ref_total == 0 or mon_total == 0:
        return 0.0, []

    rows = []
    psi_total = 0.0
    for i in range(N_RATINGS):
        ref_pct = ref_rating_counts[i] / ref_total
        mon_pct = mon_rating_counts[i] / mon_total
        psi_bin = _safe_psi(ref_pct, mon_pct)
        psi_total += psi_bin
        rows.append({
            "rating": i + 1,
            "ref_count": ref_rating_counts[i],
            "ref_pct": ref_pct,
            "mon_count": mon_rating_counts[i],
            "mon_pct": mon_pct,
            "psi_contrib": psi_bin,
        })

    return psi_total, rows


# ══════════════════════════════════════════════════════════════════════════════
#   Olgunlaşma (Maturity) Kontrolü
# ══════════════════════════════════════════════════════════════════════════════

def apply_maturity(summaries, maturity_months):
    """Her dönemin is_mature flag'ını ayarla."""
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    cutoff = datetime.now() - relativedelta(months=maturity_months)

    for s in summaries:
        try:
            period = pd.Period(s["period_label"])
            s["is_mature"] = period.end_time <= cutoff
        except Exception:
            s["is_mature"] = False

    return summaries


# ══════════════════════════════════════════════════════════════════════════════
#   Background Hesaplama
# ══════════════════════════════════════════════════════════════════════════════

def run_mon_compute(key, config, cancel_event):
    """Background thread'de tüm hesaplamaları yap.

    Sonuçlar _MON_STORE'a yazılır:
      key + "_ref_summary"      → referans özeti
      key + "_period_summaries" → dönemsel özetler listesi
    """
    prog_key = f"mon_compute_{key}"
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 0, "total_steps": 4, "done": False, "error": None}

    try:
        ref_df = _MON_STORE.get(key + "_ref")
        mon_df = _MON_STORE.get(key + "_mon")
        opt_dict = _MON_STORE.get(key + "_opt")

        if ref_df is None or mon_df is None:
            _PRECOMPUTE_PROGRESS[prog_key]["error"] = "Referans veya İzleme verisi bulunamadı."
            _PRECOMPUTE_PROGRESS[prog_key]["done"] = True
            return

        # Step 1: Referans özeti
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 1
        ref_summary = compute_ref_summary(ref_df, config, opt_dict)
        _MON_STORE[key + "_ref_summary"] = ref_summary

        # Step 2: Dönemlere ayır
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 2
        date_col = config["date_col"]
        period_freq = config.get("period_freq", "M")
        id_col = config.get("id_col")

        dates = pd.to_datetime(mon_df[date_col], errors="coerce")
        mon_df = mon_df.copy()
        mon_df["_period"] = dates.dt.to_period(period_freq)
        periods = sorted(mon_df["_period"].dropna().unique())

        # Step 3: Her dönem için özet hesapla
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 3
        summaries = []
        for period in periods:
            if cancel_event.is_set():
                return
            period_data = mon_df[mon_df["_period"] == period].copy()
            if len(period_data) == 0:
                continue
            summary = compute_period_summary(
                period_data, str(period), config, ref_summary,
                opt_dict=opt_dict, ref_df=ref_df, id_col=id_col,
            )
            summaries.append(summary)

        # Step 4: Olgunlaşma flag'ı
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 4
        maturity_months = config.get("maturity_months", 12)
        summaries = apply_maturity(summaries, maturity_months)

        _MON_STORE[key + "_period_summaries"] = summaries
        _PRECOMPUTE_PROGRESS[prog_key]["done"] = True

    except Exception as e:
        _PRECOMPUTE_PROGRESS[prog_key]["error"] = str(e)
        _PRECOMPUTE_PROGRESS[prog_key]["done"] = True


def run_mon_incremental(key, config, cancel_event, new_mon_df):
    """Artımlı hesaplama — sadece yeni dönemleri hesapla, mevcut özetlere ekle.

    new_mon_df: Sadece yeni satırları içeren DataFrame (WHERE tarih > last_date)
    """
    prog_key = f"mon_compute_{key}"
    _PRECOMPUTE_PROGRESS[prog_key] = {"step": 0, "total_steps": 3, "done": False, "error": None}

    try:
        ref_summary = _MON_STORE.get(key + "_ref_summary")
        existing_summaries = _MON_STORE.get(key + "_period_summaries", [])
        opt_dict = _MON_STORE.get(key + "_opt")
        ref_df = _MON_STORE.get(key + "_ref")  # göç matrisi için

        if ref_summary is None:
            _PRECOMPUTE_PROGRESS[prog_key]["error"] = "Referans özeti bulunamadı."
            _PRECOMPUTE_PROGRESS[prog_key]["done"] = True
            return

        if new_mon_df is None or len(new_mon_df) == 0:
            _PRECOMPUTE_PROGRESS[prog_key]["done"] = True
            return

        # Step 1: Yeni veriyi dönemlere ayır
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 1
        date_col = config["date_col"]
        period_freq = config.get("period_freq", "M")
        id_col = config.get("id_col")

        new_mon_df = new_mon_df.copy()
        dates = pd.to_datetime(new_mon_df[date_col], errors="coerce")
        new_mon_df["_period"] = dates.dt.to_period(period_freq)
        new_periods = sorted(new_mon_df["_period"].dropna().unique())

        # Mevcut dönemleri atla (sadece gerçekten yeni olanları hesapla)
        existing_labels = {s["period_label"] for s in existing_summaries}
        new_periods = [p for p in new_periods if str(p) not in existing_labels]

        if not new_periods:
            _PRECOMPUTE_PROGRESS[prog_key]["done"] = True
            return

        # Step 2: Yeni dönem özetleri hesapla
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 2
        new_summaries = []
        for period in new_periods:
            if cancel_event.is_set():
                return
            period_data = new_mon_df[new_mon_df["_period"] == period].copy()
            if len(period_data) == 0:
                continue
            summary = compute_period_summary(
                period_data, str(period), config, ref_summary,
                opt_dict=opt_dict, ref_df=ref_df, id_col=id_col,
            )
            new_summaries.append(summary)

        # Step 3: Mevcut özetlere ekle + olgunlaşma güncelle
        if cancel_event.is_set():
            return
        _PRECOMPUTE_PROGRESS[prog_key]["step"] = 3
        all_summaries = existing_summaries + new_summaries
        maturity_months = config.get("maturity_months", 12)
        all_summaries = apply_maturity(all_summaries, maturity_months)

        _MON_STORE[key + "_period_summaries"] = all_summaries
        _PRECOMPUTE_PROGRESS[prog_key]["done"] = True

    except Exception as e:
        _PRECOMPUTE_PROGRESS[prog_key]["error"] = str(e)
        _PRECOMPUTE_PROGRESS[prog_key]["done"] = True


def start_mon_compute(key, config):
    """Tam hesaplama thread'ini başlat."""
    global _mon_active_thread, _mon_cancel_event

    cancel_mon_compute()

    _mon_cancel_event = threading.Event()
    _mon_active_thread = threading.Thread(
        target=run_mon_compute,
        args=(key, config, _mon_cancel_event),
        daemon=True,
    )
    _mon_active_thread.start()
    return f"mon_compute_{key}"


def start_mon_incremental(key, config, new_mon_df):
    """Artımlı hesaplama thread'ini başlat."""
    global _mon_active_thread, _mon_cancel_event

    cancel_mon_compute()

    _mon_cancel_event = threading.Event()
    _mon_active_thread = threading.Thread(
        target=run_mon_incremental,
        args=(key, config, _mon_cancel_event, new_mon_df),
        daemon=True,
    )
    _mon_active_thread.start()
    return f"mon_compute_{key}"
