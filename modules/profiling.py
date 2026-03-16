import pandas as pd
import numpy as np


def compute_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kolon bazında profiling tablosu döndürür.
    df'nin kopyası üzerinde çalışır — orijinale dokunmaz.
    """
    local = df.copy()
    n = len(local)
    rows = []

    for col in local.columns:
        s = local[col]
        missing     = int(s.isna().sum())
        missing_pct = round(missing / n * 100, 2) if n > 0 else 0.0
        unique      = int(s.nunique(dropna=True))

        mode_series = s.mode(dropna=True)
        if len(mode_series) > 0:
            mode_val  = mode_series.iloc[0]
            mode_freq = round((s == mode_val).sum() / n * 100, 2)
        else:
            mode_val  = None
            mode_freq = None

        is_numeric = pd.api.types.is_numeric_dtype(s)

        row = {
            "Kolon":        col,
            "Tip":          str(s.dtype),
            "Dolu Sayı":    n - missing,
            "Eksik Sayı":   missing,
            "Eksik %":      missing_pct,
            "Tekil Değer":  unique,
            "En Sık Değer": str(mode_val) if mode_val is not None else "—",
            "En Sık %":     mode_freq if mode_freq is not None else 0.0,
            "Ortalama":     "—",
            "Std":          "—",
            "Min":          "—",
            "Medyan":       "—",
            "Max":          "—",
        }

        if is_numeric:
            non_null = s.dropna()
            if len(non_null):
                pcts = non_null.quantile([.01,.05,.10,.25,.50,.75,.90,.95,.99])
                row["Ortalama"] = round(float(non_null.mean()),   4)
                row["Std"]      = round(float(non_null.std()),    4)
                row["Min"]      = round(float(non_null.min()),    4)
                row["P1"]       = round(float(pcts[.01]),         4)
                row["P5"]       = round(float(pcts[.05]),         4)
                row["P10"]      = round(float(pcts[.10]),         4)
                row["P25"]      = round(float(pcts[.25]),         4)
                row["Medyan"]   = round(float(pcts[.50]),         4)
                row["P75"]      = round(float(pcts[.75]),         4)
                row["P90"]      = round(float(pcts[.90]),         4)
                row["P95"]      = round(float(pcts[.95]),         4)
                row["P99"]      = round(float(pcts[.99]),         4)
                row["Max"]      = round(float(non_null.max()),    4)

        rows.append(row)

    return pd.DataFrame(rows)


def profile_summary(profile_df: pd.DataFrame, total_rows: int) -> dict:
    """Profiling tablosundan özet istatistikler çıkarır."""
    return {
        "total_cols":      len(profile_df),
        "numeric_cols":    int(profile_df["Tip"].str.contains("int|float").sum()),
        "categorical_cols": int((profile_df["Tip"] == "object").sum()),
        "full_cols":       int((profile_df["Eksik %"] == 0).sum()),
        "high_missing":    int((profile_df["Eksik %"] > 50).sum()),
        "mid_missing":     int(((profile_df["Eksik %"] > 5) & (profile_df["Eksik %"] <= 50)).sum()),
        "total_rows":      total_rows,
    }
