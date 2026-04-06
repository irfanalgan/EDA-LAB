import logging
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def get_database_config() -> dict:
    """config.toml'daki [database] bölümünü oku. Boşsa boş dict döndürür."""
    try:
        with open(_CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("database", {})
    except Exception as e:
        log.warning("config.toml [database] okunamadı: %s", e)
    return {}


def get_profiles_root() -> Path:
    """config.toml'daki shared_dir doluysa onu, yoksa yerel profiles/ döndürür."""
    try:
        with open(_CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        raw = cfg.get("profiles", {}).get("shared_dir", "").strip()
        if raw:
            p = Path(raw)
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception as e:
        log.warning("config.toml okunamadı, yerel profiles/ kullanılacak: %s", e)
    return Path(__file__).parent.parent / "profiles"
