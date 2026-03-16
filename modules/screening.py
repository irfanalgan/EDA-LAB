import pandas as pd
import numpy as np


def screen_columns(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = None,
    segment_col: str = None,
    missing_threshold: float = 0.80,
) -> tuple[list[str], pd.DataFrame]:
    """
    Kalite kontrolü — düşük kaliteli kolonları eleyin.

    Kurallar (sırasıyla uygulanır):
      1. Yüksek Eksik  : dolu oran < (1 - missing_threshold)  → varsayılan %80 üstü boş
      2. Sabit          : dolu değerler arasında tekil değer sayısı ≤ 1

    Returns
    -------
    passed : list[str]   — geçen kolon adları (hedef/tarih/segment hariç)
    report : pd.DataFrame — [Kolon, Kural, Detay] — elenen kolonlar
    """
    exclude = {c for c in [target_col, date_col, segment_col] if c}
    n = len(df)
    records = []
    passed = []

    for col in df.columns:
        if col in exclude:
            continue

        missing_n   = int(df[col].isna().sum())
        missing_pct = missing_n / n if n > 0 else 0.0
        nunique     = int(df[col].nunique(dropna=True))

        if missing_pct > missing_threshold:
            records.append({
                "Kolon":  col,
                "Kural":  "Yüksek Eksik",
                "Detay":  f"%{missing_pct * 100:.1f} boş  ({missing_n:,} / {n:,})",
            })
        elif nunique <= 1:
            records.append({
                "Kolon":  col,
                "Kural":  "Sabit Değişken",
                "Detay":  f"Yalnızca {nunique} tekil değer (dolu satırlarda)",
            })
        else:
            passed.append(col)

    report = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["Kolon", "Kural", "Detay"])
    )
    return passed, report
