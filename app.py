# ── Bağımlılık kontrolü — diğer importlardan önce çalışmalı ──────────────────
import sys, pathlib
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
        logging.FileHandler("logs/eda_lab.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.getLogger(__name__).info("EDA Laboratuvarı başlatılıyor…")
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
