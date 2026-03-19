import re
import numpy as np
import pandas as pd


# Türkçe sayı formatı kalıpları
# "10.000,50" veya "1.234.567" — nokta binlik, virgül ondalık
_TR_FULL = re.compile(r'^-?\d{1,3}(\.\d{3})+(,\d*)?$')
# "1234,56" veya "100,5" — virgül ondalık, en fazla 2 basamak
_TR_DEC  = re.compile(r'^-?\d+,\d{1,2}$')

# Diğer formatlama karakterleri (virgül & nokta hariç)
_SYMBOL_STRIP = ['%', '$', '€', '£', ' ', '\xa0']


def _turkish_fraction(series_clean: pd.Series) -> float:
    """Kolondaki değerlerin Türkçe sayı formatına uyma oranını döndürür."""
    sample = series_clean.head(300)
    if len(sample) == 0:
        return 0.0
    hits = sum(1 for v in sample
               if _TR_FULL.match(v) or _TR_DEC.match(v))
    return hits / len(sample)


def _apply_turkish(s_str: pd.Series) -> pd.Series:
    """Türkçe formatı temizler: nokta binlik sil, virgül → nokta."""
    c = s_str.str.replace('.', '', regex=False)
    c = c.str.replace(',', '.', regex=False)
    return pd.to_numeric(c, errors="coerce")


def coerce_numeric_columns(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Tüm object kolonları tara; çeşitli temizleme stratejileri uygulayarak
    ≥threshold oranında numerik dönüşüm başarılıysa kolonu numeric'e çevir.

    Dönüşüm sırası:
      1. Ham parse (zaten numeric-like)
      2. Türkçe format  — "10.000,50" veya "1234,56"
      3. Sembol temizleme — %, $, €, boşluk vb.
      4. İngiliz binlik virgülü — "1,234" → 1234

    Returns:
        (df_cleaned, converted)
        converted: [{"col", "fix", "sample_before", "sample_after", "n_converted"}, ...]
    """
    converted: list[dict] = []
    changes:   dict[str, pd.Series] = {}

    for col in df.select_dtypes(include="object").columns:
        s        = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue

        sample_before = str(non_null.iloc[0])

        # ── 1. Ham parse ──────────────────────────────────────────────────────
        cand = pd.to_numeric(non_null, errors="coerce")
        if cand.notna().sum() / len(non_null) >= threshold:
            # Sessizce dönüştür; kullanıcıya raporlama gerektirmez
            changes[col] = pd.to_numeric(s, errors="coerce")
            continue

        cleaned = non_null.astype(str).str.strip()

        # ── 2. Türkçe format ─────────────────────────────────────────────────
        if _turkish_fraction(cleaned) >= 0.70:
            full_str = s.astype(str).str.strip().where(s.notna(), other=None)
            result   = _apply_turkish(full_str.fillna(''))
            result   = result.where(s.notna(), other=pd.NA)
            n_ok     = result.notna().sum()
            if n_ok / len(non_null) >= threshold:
                changes[col] = result
                sample_after = str(_apply_turkish(
                    pd.Series([sample_before])).iloc[0])
                converted.append({
                    "col": col, "fix": "turkish_decimal",
                    "sample_before": sample_before,
                    "sample_after":  sample_after,
                    "n_converted":   int(n_ok),
                })
                continue

        # ── 3. Sembol temizleme (%  $  €  £  boşluk) ─────────────────────────
        stripped = cleaned.copy()
        for ch in _SYMBOL_STRIP:
            stripped = stripped.str.replace(ch, '', regex=False)
        cand = pd.to_numeric(stripped, errors="coerce")
        n_ok = cand.notna().sum()
        if n_ok / len(non_null) >= threshold:
            full_str = s.astype(str).str.strip()
            for ch in _SYMBOL_STRIP:
                full_str = full_str.str.replace(ch, '', regex=False)
            full_str = full_str.where(s.notna(), other=None)
            changes[col] = pd.to_numeric(full_str, errors="coerce")
            sample_after = str(cand.dropna().iloc[0]) if len(cand.dropna()) else ""
            converted.append({
                "col": col, "fix": "symbol_strip",
                "sample_before": sample_before,
                "sample_after":  sample_after,
                "n_converted":   int(n_ok),
            })
            continue

        # ── 4. İngiliz binlik virgülü "1,234" → 1234 ─────────────────────────
        stripped_comma = stripped.str.replace(',', '', regex=False)
        cand = pd.to_numeric(stripped_comma, errors="coerce")
        n_ok = cand.notna().sum()
        if n_ok / len(non_null) >= threshold:
            full_str = s.astype(str).str.strip()
            for ch in _SYMBOL_STRIP + [',']:
                full_str = full_str.str.replace(ch, '', regex=False)
            full_str = full_str.where(s.notna(), other=None)
            changes[col] = pd.to_numeric(full_str, errors="coerce")
            sample_after = str(cand.dropna().iloc[0]) if len(cand.dropna()) else ""
            converted.append({
                "col": col, "fix": "thousands_comma",
                "sample_before": sample_before,
                "sample_after":  sample_after,
                "n_converted":   int(n_ok),
            })

    if changes:
        df = df.assign(**changes)

    return df, converted


def apply_segment_filter(
    df: pd.DataFrame,
    segment_col: str | None,
    segment_val: str | list | None,
) -> pd.DataFrame:
    """
    df_original'e dokunmadan segment maskesi uygular ve kopya döndürür.
    segment_val tek değer veya liste olabilir.
    Seçim yoksa veya "Tümü" içeriyorsa df'nin kendisini döndürür.
    """
    if not segment_col or not segment_val:
        return df
    # Tekil değeri listeye çevir
    vals = segment_val if isinstance(segment_val, list) else [segment_val]
    if "Tümü" in vals:
        return df
    return df[df[segment_col].astype(str).isin(vals)].copy()


def get_splits(df: pd.DataFrame, config: dict) -> tuple:
    """
    df'yi config'e göre (df_train, df_test, df_oot) üçlüsüne böler.

    Kurallar:
    - OOT: oot_date + date_col varsa tarihe göre bölünür, yoksa df_oot = None
    - Test: has_test_split True ise config["test_size"] oranında rastgele bölünür
      False ise df_test = None — hiçbir yerde kafasına göre split yapılmaz
    - Train: Geri kalan veri
    """
    from sklearn.model_selection import train_test_split as _tts

    date_col  = config.get("date_col")
    oot_date  = config.get("oot_date")
    has_test  = bool(config.get("has_test_split"))
    _ts = config.get("test_size")
    test_pct  = float(_ts) / 100 if _ts is not None else 0.20
    target    = config.get("target_col")

    def _random_split(df_pool):
        indices = np.arange(len(df_pool))
        y = df_pool[target] if target and target in df_pool.columns else None
        try:
            stratify = y.values if y is not None and y.nunique() <= 10 else None
            tr_idx, te_idx = _tts(indices, test_size=test_pct, random_state=42,
                                  stratify=stratify)
        except Exception:
            tr_idx, te_idx = _tts(indices, test_size=test_pct, random_state=42)
        return df_pool.iloc[tr_idx].copy(), df_pool.iloc[te_idx].copy()

    # Tarih kolonuna göre sırala — OOT/train/test tutarlılığı için
    # NOT: reset_index YAPILMAZ — orijinal index korunmalı ki mask eşleşmesi doğru olsun
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col, na_position="last")

    if oot_date and date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        pool_mask = dates < pd.to_datetime(oot_date)
        df_pool = df[pool_mask].copy()
        df_oot  = df[~pool_mask].copy()
        if has_test and len(df_pool) >= 20:
            df_train, df_test = _random_split(df_pool)
        else:
            df_train, df_test = df_pool, None
    else:
        df_oot = None
        if has_test and len(df) >= 20:
            df_train, df_test = _random_split(df)
        else:
            df_train, df_test = df.copy(), None

    return df_train, df_test, df_oot


def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Segmentasyon için uygun kolonları döndürür (object veya az unique değerli)."""
    return [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= max_unique
    ]
