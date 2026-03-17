import pandas as pd


# ── Target Dağılımı ───────────────────────────────────────────────────────────

def compute_target_stats(df: pd.DataFrame, target: str) -> dict:
    """Target değişkeninin temel istatistiklerini hesaplar."""
    local = df[[target]].copy()
    if local[target].dtype == object:
        local[target] = (local[target].astype(str)
                         .str.replace('%', '', regex=False).str.strip())
    s = pd.to_numeric(local[target], errors='coerce')

    total    = len(s)
    missing  = int(s.isna().sum())
    valid    = total - missing
    bad      = int(s.sum())
    good     = valid - bad
    bad_rate = bad / valid * 100 if valid > 0 else 0.0
    ratio    = good / bad if bad > 0 else float("inf")

    return {
        "total":    total,
        "missing":  missing,
        "valid":    valid,
        "bad":      bad,
        "good":     good,
        "bad_rate": round(bad_rate, 4),
        "ratio":    round(ratio, 2),
    }


def compute_target_over_time(df: pd.DataFrame, target: str, date_col: str,
                              freq: str = "ME") -> pd.DataFrame:
    """
    Zaman içinde bad rate değişimini hesaplar.
    freq: pandas resample frekansı — 'ME' (ay sonu), 'QE' (çeyrek sonu)
    """
    local = df[[date_col, target]].copy()
    local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
    if local[target].dtype == object:
        local[target] = (local[target].astype(str)
                         .str.replace('%', '', regex=False).str.strip())
    local[target] = pd.to_numeric(local[target], errors='coerce')
    local = local.dropna(subset=[date_col])
    local = local.set_index(date_col).sort_index()

    resampled = local.resample(freq)[target].agg(
        bad_count="sum",
        total_count="count"
    ).reset_index()
    resampled["bad_rate"] = (resampled["bad_count"] / resampled["total_count"] * 100).round(4)
    resampled = resampled.rename(columns={date_col: "Tarih"})

    return resampled
