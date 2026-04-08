# ── Merkezi dağıtım ─────────────────────────────────────────────────
# shared_dir doluysa: lokali temizle → merkezdeki dosyaları kopyala
import sys, pathlib, os, shutil

_HERE = pathlib.Path(__file__).resolve().parent
_SKIP = {"logs", "__pycache__", ".git", "app.py", "profiles"}


def _auto_sync():
    """config.toml'daki shared_dir'den merkez app kaynağını bul, birebir kopyala."""
    try:
        import tomllib
        with open(_HERE / "config.toml", "rb") as f:
            shared = tomllib.load(f).get("profiles", {}).get("shared_dir", "").strip()
        if not shared:
            return
        source = pathlib.Path(shared).parent / "app"
        if not source.exists():
            return
        if source.resolve() == _HERE:
            return
    except Exception:
        return

    # 1. Lokaldeki eski dosya/klasörleri sil (SKIP hariç)
    for item in _HERE.iterdir():
        if item.name in _SKIP:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # 2. Merkezdeki her şeyi kopyala (SKIP hariç)
    count = 0
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(source)
        if any(part in _SKIP for part in rel.parts):
            continue
        if rel.name in _SKIP:
            continue
        dest = _HERE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)
        count += 1
    if count:
        print(f"[sync] {count} dosya güncellendi.")


_auto_sync()
# ─────────────────────────────────────────────────────────────────────

# ── Bağımlılık kontrolü — diğer importlardan önce çalışmalı ──────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
    from setup_deps import ensure_deps
    ensure_deps(verbose=True)
except Exception as _dep_err:
    print(f"[setup_deps] Atlandı: {_dep_err}")
# ─────────────────────────────────────────────────────────────────────────────

# ── Merkezi loglama ──────────────────────────────────────────────────────────
import logging, os
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/calibris.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.getLogger(__name__).info("Calibris başlatılıyor…")
# ─────────────────────────────────────────────────────────────────────────────

import threading
import webbrowser

from app_instance import app
from layout import build_layout
import callbacks  # noqa: F401 — tüm @app.callback dekoratörlerini kaydeder

app.layout = build_layout()

if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:8060")).start()
    app.run(debug=False, port=8060)
