import pandas as pd


# Temizlenebilecek formatlama karakterleri (sıralı olarak denenir)
_STRIP_PATTERNS = [
    ('%',  ''),   # yüzde: '1.49%' → '1.49'
    (',',  ''),   # binlik ayracı: '1,234' → '1234'
    ('$',  ''),   # dolar
    ('€',  ''),   # euro
    ('£',  ''),   # sterlin
    (' ',  ''),   # boşluk
]


def coerce_numeric_columns(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """
    Tüm object kolonları tara; formatlama karakterleri temizlenince ≥threshold
    oranında numerik dönüşüm başarılıysa kolonu numeric'e çevir.

    Returns:
        (df_cleaned, converted_cols)  — converted_cols değiştirilenlerin listesi
    """
    converted = []
    changes: dict[str, pd.Series] = {}

    for col in df.select_dtypes(include="object").columns:
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue

        # Önce ham haliyle dene
        candidate = pd.to_numeric(non_null, errors="coerce")
        success = candidate.notna().sum() / len(non_null)

        if success < threshold:
            # Formatlama karakterlerini sırayla soy, en iyi adayı bul
            cleaned = non_null.astype(str).str.strip()
            for char, repl in _STRIP_PATTERNS:
                cleaned = cleaned.str.replace(char, repl, regex=False)
            candidate = pd.to_numeric(cleaned, errors="coerce")
            success = candidate.notna().sum() / len(non_null)

        if success >= threshold:
            # Tüm seri (null'lar dahil) için dönüştür
            full_cleaned = s.astype(str).str.strip()
            for char, repl in _STRIP_PATTERNS:
                full_cleaned = full_cleaned.str.replace(char, repl, regex=False)
            full_cleaned = full_cleaned.where(s.notna(), other=None)
            changes[col] = pd.to_numeric(full_cleaned, errors="coerce")
            converted.append(col)

    if changes:
        df = df.assign(**changes)

    return df, converted


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


def detect_target_type(s: pd.Series) -> str:
    """
    Target değişkeninin ölçüm tipini tespit eder.
    Dönüş: 'binary' | 'continuous' | 'multiclass' | 'categorical'
    """
    if pd.api.types.is_object_dtype(s):
        return "categorical"
    vals = s.dropna()
    if len(vals) == 0:
        return "binary"
    unique_vals = set(vals.unique())
    if unique_vals <= {0, 1, 0.0, 1.0}:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        n_unique = vals.nunique()
        is_integer_like = (vals % 1 == 0).all()
        if is_integer_like and n_unique <= 20:
            return "multiclass"
        return "continuous"
    return "categorical"


def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Segmentasyon için uygun kolonları döndürür (object veya az unique değerli)."""
    return [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= max_unique
    ]
