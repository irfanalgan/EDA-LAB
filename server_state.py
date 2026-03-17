import pandas as pd

# Sunucu tarafı veri deposu
_SERVER_STORE: dict[str, pd.DataFrame] = {}

# Precompute ilerleme deposu
_PRECOMPUTE_PROGRESS: dict = {}


def get_df(key) -> "pd.DataFrame | None":
    if not key:
        return None
    return _SERVER_STORE.get(key)
