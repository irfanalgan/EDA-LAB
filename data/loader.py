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
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df


def get_data_from_sql_multi(
    tables:   list[str],
    join_key: list[str],
    server:   str | None = None,
    database: str | None = None,
    driver:   str | None = None,
) -> pd.DataFrame:
    """
    Birden fazla tabloyu çekip UNIQUE_ID / join_key üzerinden axis=1 birleştirir.
    tables[0] master kabul edilir; diğerlerinden join_key + tekrar eden kolonlar çıkarılır.
    join_key boşsa sadece tables[0] döndürülür.
    """
    cfg = _get_config().get("database", {})
    server   = server   or cfg.get("server",   "")
    database = database or cfg.get("database", "")
    driver   = driver   or cfg.get("driver",   _DRIVER_OPTIONS[0])

    conn_str = _build_conn_str(server, database, driver)
    jk = [k.strip() for k in join_key if k.strip()]

    with pyodbc.connect(conn_str) as conn:
        result = pd.read_sql(f"SELECT * FROM {tables[0]}", conn)
        if jk:
            result = result.set_index(jk)

        for tbl in tables[1:]:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            if jk:
                df = df.set_index(jk)
            # Tekrar eden kolonları düşür (master'da zaten var)
            drop_cols = [c for c in df.columns if c in result.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            result = pd.concat([result, df], axis=1)

    if jk:
        result = result.reset_index()
    return result
