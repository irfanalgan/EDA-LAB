import numpy as np
import pandas as pd


# ── Kolon seçimi ──────────────────────────────────────────────────────────────

def get_numeric_cols(df: pd.DataFrame,
                     exclude: list[str] = None,
                     max_cols: int = 40) -> list[str]:
    """
    Numerik kolonları döndürür (exclude listesi + max_cols ile sınırlı).
    Fazla kolon varsa varyansa göre en bilgilendirici olanları alır.
    """
    excl = set(exclude or [])
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in excl]
    if len(cols) <= max_cols:
        return cols
    variances = df[cols].var().sort_values(ascending=False)
    return variances.head(max_cols).index.tolist()


# ── Korelasyon ────────────────────────────────────────────────────────────────

def compute_correlation_matrix(df: pd.DataFrame,
                                cols: list[str]) -> pd.DataFrame:
    """Pearson korelasyon matrisi. NaN'lar pairwise hariç tutulur."""
    return df[cols].corr(method="pearson")


def find_high_corr_pairs(corr_df: pd.DataFrame,
                          threshold: float = 0.75) -> pd.DataFrame:
    """Mutlak korelasyonu eşik üzerinde olan tüm çiftleri döndürür."""
    cols = corr_df.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr_df.iloc[i, j])
            if not np.isnan(val) and abs(val) >= threshold:
                pairs.append({
                    "Değişken 1":    cols[i],
                    "Değişken 2":    cols[j],
                    "Korelasyon":    round(val, 4),
                    "|Korelasyon|":  round(abs(val), 4),
                })
    if not pairs:
        return pd.DataFrame(columns=["Değişken 1", "Değişken 2",
                                     "Korelasyon", "|Korelasyon|"])
    return (pd.DataFrame(pairs)
            .sort_values("|Korelasyon|", ascending=False)
            .reset_index(drop=True))


# ── VIF ───────────────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame,
                cols: list[str]) -> pd.DataFrame:
    """
    Her kolon için VIF hesaplar (statsmodels variance_inflation_factor).
    Eşikler:  < 5 Normal  |  5–10 Orta  |  > 10 Yüksek
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    local = df[cols].dropna()
    if len(local) < len(cols) + 2:
        return pd.DataFrame()

    if len(cols) < 2:
        return pd.DataFrame({
            "Değişken": cols,
            "VIF":      [1.0] * len(cols),
            "Uyarı":    ["✓ Normal"] * len(cols),
        })

    X = add_constant(local[cols])
    records = []
    for i, col in enumerate(cols):
        try:
            col_idx = list(X.columns).index(col)
            vif = round(variance_inflation_factor(X.values, col_idx), 2)
        except Exception:
            vif = 999.0
        uyari = "✓ Normal" if vif < 5 else "⚠ Orta" if vif < 10 else "✗ Yüksek"
        records.append({"Değişken": col, "VIF": vif, "Uyarı": uyari})

    return (pd.DataFrame(records)
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True))
