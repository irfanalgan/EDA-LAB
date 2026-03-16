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
                cols: list[str],
                sample_size: int = 50_000) -> pd.DataFrame:
    """
    Her kolon için VIF hesaplar (numpy lstsq, bağımlılık yok).
    VIF_i = 1 / (1 - R²_i),  R²_i: col_i'nin diğer kolonlara regresyonu.
    Eşikler:  < 5 Normal  |  5–10 Orta  |  > 10 Yüksek
    """
    local = df[cols].dropna()
    if len(local) < len(cols) + 2:
        return pd.DataFrame()

    if len(local) > sample_size:
        local = local.sample(n=sample_size, random_state=42)

    X = local.values.astype(float)
    n, k = X.shape

    if k < 2:
        return pd.DataFrame({
            "Değişken": cols,
            "VIF":      [1.0] * k,
            "Uyarı":    ["✓ Normal"] * k,
        })

    records = []
    for idx, col in enumerate(cols):
        y      = X[:, idx]
        X_rest = np.delete(X, idx, axis=1)
        A      = np.c_[np.ones(n), X_rest]
        try:
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            y_hat  = A @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2     = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            vif    = round(1.0 / (1.0 - r2), 2) if r2 < 0.9999 else 999.0
        except Exception:
            vif = 999.0

        uyari = "✓ Normal" if vif < 5 else "⚠ Orta" if vif < 10 else "✗ Yüksek"
        records.append({"Değişken": col, "VIF": vif, "Uyarı": uyari})

    return (pd.DataFrame(records)
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True))
