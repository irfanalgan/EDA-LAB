import logging
import sys

import pandas as pd

log = logging.getLogger(__name__)

# Sunucu tarafı veri deposu — Geliştirme
_SERVER_STORE: dict[str, pd.DataFrame] = {}

# Precompute ilerleme deposu
_PRECOMPUTE_PROGRESS: dict = {}

# Sunucu tarafı veri deposu — İzleme (tamamen bağımsız)
_MON_STORE: dict[str, pd.DataFrame] = {}

# Uyarı eşiği (MB) — bu değeri aşınca log.warning
_WARN_MB = 4_096


def _store_size_mb(store: dict) -> float:
    """Store'daki toplam bellek kullanımını MB olarak tahmin eder."""
    total = 0
    for v in store.values():
        if isinstance(v, pd.DataFrame):
            try:
                total += int(v.memory_usage(deep=True).sum())
            except Exception:
                total += sys.getsizeof(v)
        else:
            total += sys.getsizeof(v)
    return total / (1024 * 1024)


def get_df(key) -> "pd.DataFrame | None":
    if not key:
        return None
    return _SERVER_STORE.get(key)


def clear_store(keep_key: str | None = None):
    """Store'u temizle. keep_key verilirse sadece ham veri ({key} ve {key}_quality) kalır."""
    before = _store_size_mb(_SERVER_STORE)
    if keep_key is None:
        _SERVER_STORE.clear()
        log.info("SERVER_STORE temizlendi (önceki: %.0f MB)", before)
        return
    raw = _SERVER_STORE.get(keep_key)
    quality = _SERVER_STORE.get(f"{keep_key}_quality")
    _SERVER_STORE.clear()
    if raw is not None:
        _SERVER_STORE[keep_key] = raw
    if quality is not None:
        _SERVER_STORE[f"{keep_key}_quality"] = quality
    after = _store_size_mb(_SERVER_STORE)
    log.info("SERVER_STORE temizlendi: %.0f MB → %.0f MB", before, after)


def log_store_usage():
    """Mevcut store boyutlarını loglar. Eşik aşılırsa uyarı verir."""
    dev_mb = _store_size_mb(_SERVER_STORE)
    mon_mb = _store_size_mb(_MON_STORE)
    msg = "Store kullanımı — Geliştirme: %.0f MB (%d anahtar), İzleme: %.0f MB (%d anahtar)"
    args = (dev_mb, len(_SERVER_STORE), mon_mb, len(_MON_STORE))
    if dev_mb + mon_mb > _WARN_MB:
        log.warning(msg, *args)
    else:
        log.info(msg, *args)


# ── İzleme store fonksiyonları ───────────────────────────────────────────────

def get_mon_df(key) -> "pd.DataFrame | None":
    if not key:
        return None
    return _MON_STORE.get(key)


def clear_mon_store(keep_key: str | None = None):
    """İzleme store'unu temizle. Geliştirme store'una dokunmaz."""
    before = _store_size_mb(_MON_STORE)
    if keep_key is None:
        _MON_STORE.clear()
        log.info("MON_STORE temizlendi (önceki: %.0f MB)", before)
        return
    raw = _MON_STORE.get(keep_key)
    quality = _MON_STORE.get(f"{keep_key}_quality")
    _MON_STORE.clear()
    if raw is not None:
        _MON_STORE[keep_key] = raw
    if quality is not None:
        _MON_STORE[f"{keep_key}_quality"] = quality
    after = _store_size_mb(_MON_STORE)
    log.info("MON_STORE temizlendi: %.0f MB → %.0f MB", before, after)
