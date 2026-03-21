import pandas as pd

# Sunucu tarafı veri deposu
_SERVER_STORE: dict[str, pd.DataFrame] = {}

# Precompute ilerleme deposu
_PRECOMPUTE_PROGRESS: dict = {}


def get_df(key) -> "pd.DataFrame | None":
    if not key:
        return None
    return _SERVER_STORE.get(key)


def clear_store(keep_key: str | None = None):
    """Store'u temizle. keep_key verilirse sadece ham veri ({key} ve {key}_quality) kalır."""
    if keep_key is None:
        _SERVER_STORE.clear()
        return
    raw = _SERVER_STORE.get(keep_key)
    quality = _SERVER_STORE.get(f"{keep_key}_quality")
    _SERVER_STORE.clear()
    if raw is not None:
        _SERVER_STORE[keep_key] = raw
    if quality is not None:
        _SERVER_STORE[f"{keep_key}_quality"] = quality
