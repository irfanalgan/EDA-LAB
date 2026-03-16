import pandas as pd


def apply_segment_filter(
    df: pd.DataFrame,
    segment_col: str | None,
    segment_val: str | None,
) -> pd.DataFrame:
    """
    df_original'e dokunmadan segment maskesi uygular ve kopya döndürür.
    Seçim yoksa df'nin kendisini döndürür.
    """
    if segment_col and segment_val and segment_val != "Tümü":
        return df[df[segment_col].astype(str) == segment_val].copy()
    return df


def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Segmentasyon için uygun kolonları döndürür (object veya az unique değerli)."""
    return [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= max_unique
    ]
