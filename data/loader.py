import tomllib
import pandas as pd
import pyodbc
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

_DRIVER_OPTIONS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
]


def _get_config() -> dict:
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _build_conn_str(server: str, database: str, driver: str) -> str:
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )


def _quote_table(name: str) -> str:
    """Tablo adını SQL Server için güvenli hâle getirir.
    'dbo.MODEL_DATA' → '[dbo].[MODEL_DATA]'
    'MODEL_DATA'     → '[MODEL_DATA]'
    """
    parts = name.strip().split(".")
    return ".".join(f"[{p}]" for p in parts)


def get_config_defaults() -> dict:
    """config.toml'dan varsayılan bağlantı bilgilerini döndürür."""
    cfg = _get_config().get("database", {})
    return {
        "server":   cfg.get("server",   ""),
        "database": cfg.get("database", ""),
        "driver":   cfg.get("driver",   _DRIVER_OPTIONS[0]),
    }


def get_data_from_sql(
    table_name: str,
    server:   str | None = None,
    database: str | None = None,
    driver:   str | None = None,
) -> pd.DataFrame:
    """
    Belirtilen tabloyu MS SQL Server'dan çeker.
    Parametreler verilmezse config.toml değerleri kullanılır.
    """
    cfg = _get_config().get("database", {})
    server   = server   or cfg.get("server",   "")
    database = database or cfg.get("database", "")
    driver   = driver   or cfg.get("driver",   _DRIVER_OPTIONS[0])

    conn_str = _build_conn_str(server, database, driver)
    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(f"SELECT * FROM {_quote_table(table_name)}", conn)
    return df


def get_data_from_sql_multi(
    tables:   list[str],
    join_keys_per_table: list[list[str]],
    server:   str | None = None,
    database: str | None = None,
    driver:   str | None = None,
) -> pd.DataFrame:
    """
    Birden fazla tabloyu LEFT JOIN mantığıyla birleştirir (pd.merge).
    tables[0] master kabul edilir; diğerlerinden tekrar eden kolonlar çıkarılır.

    join_keys_per_table: Her tablonun kendi key listesi.
      join_keys_per_table[0] → master (left_on)
      join_keys_per_table[i] → i. tablo (right_on)
    Aynı isimse aynı değeri yaz, farklıysa her tabloya kendi key'ini yaz.
    """
    cfg = _get_config().get("database", {})
    server   = server   or cfg.get("server",   "")
    database = database or cfg.get("database", "")
    driver   = driver   or cfg.get("driver",   _DRIVER_OPTIONS[0])

    conn_str = _build_conn_str(server, database, driver)
    left_keys = join_keys_per_table[0] if join_keys_per_table else []

    with pyodbc.connect(conn_str) as conn:
        result = pd.read_sql(f"SELECT * FROM {_quote_table(tables[0])}", conn)

        if not left_keys or len(tables) < 2:
            return result

        for i, tbl in enumerate(tables[1:], start=1):
            df = pd.read_sql(f"SELECT * FROM {_quote_table(tbl)}", conn)
            right_keys = join_keys_per_table[i] if i < len(join_keys_per_table) and join_keys_per_table[i] else left_keys
            # Tekrar eden kolonları düşür (join key hariç)
            all_keys = set(left_keys) | set(right_keys)
            drop_cols = [c for c in df.columns if c in result.columns and c not in all_keys]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            if left_keys == right_keys:
                result = pd.merge(result, df, on=left_keys, how="left")
            else:
                result = pd.merge(result, df, left_on=left_keys, right_on=right_keys, how="left")
                # Sağ taraftaki key kolonunu düşür (duplicate olmaması için)
                extra = [rk for lk, rk in zip(left_keys, right_keys) if lk != rk and rk in result.columns]
                if extra:
                    result = result.drop(columns=extra)

    return result
