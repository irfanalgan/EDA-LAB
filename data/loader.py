import tomllib
import pandas as pd
import pyodbc
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def _get_config() -> dict:
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def get_data_from_sql(table_name: str) -> pd.DataFrame:
    """
    Belirtilen tabloyu MS SQL Server'dan çeker ve DataFrame döndürür.
    Windows Authentication kullanır — username/password gerekmez.
    """
    cfg = _get_config()["database"]

    conn_str = (
        f"DRIVER={{{cfg['driver']}}};"
        f"SERVER={cfg['server']};"
        f"DATABASE={cfg['database']};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    return df
