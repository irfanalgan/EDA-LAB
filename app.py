# ── Bağımlılık kontrolü — diğer importlardan önce çalışmalı ──────────────────
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
    from setup_deps import ensure_deps
    ensure_deps(verbose=True)
except Exception as _dep_err:
    print(f"[setup_deps] Atlandı: {_dep_err}")
# ─────────────────────────────────────────────────────────────────────────────

import threading
import webbrowser

from app_instance import app
from layout import build_layout
import callbacks  # noqa: F401 — tüm @app.callback dekoratörlerini kaydeder

app.layout = build_layout()

if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:8050")).start()
    app.run(debug=False, port=8050)
