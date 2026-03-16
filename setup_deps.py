"""
Bağımlılık yükleyici — app.py başlamadan önce çalışır.

Kullanım:
  1. pip_prefix.txt dosyasını aç.
  2. İş yerinin gerektirdiği ön komutu yapıştır.
     Örnek:  pip install --index-url https://... --trusted-host ...
  3. Dosyayı kaydet. Bir daha dokunmana gerek yok.

Mantık:
  - Her paket için önce import dener → başarılıysa atlar (hızlı).
  - Başarısızsa pip_prefix.txt'teki komuta paket adını ekleyerek yükler.
"""

import importlib
import subprocess
import sys
from pathlib import Path

# ── Paket listesi: (import_adı, pip_adı) ─────────────────────────────────────
PACKAGES = [
    ("dash",                    "dash>=2.14"),
    ("dash_bootstrap_components","dash-bootstrap-components>=1.5"),
    ("pandas",                  "pandas>=2.0"),
    ("numpy",                   "numpy"),
    ("plotly",                  "plotly>=5.18"),
    ("scipy",                   "scipy>=1.11"),
    ("sklearn",                 "scikit-learn>=1.3"),
    ("lightgbm",                "lightgbm>=4.0"),
    ("xgboost",                 "xgboost>=2.0"),
    ("pyodbc",                  "pyodbc"),
    ("ydata_profiling",         "ydata-profiling"),
]

# ── Prefix dosyasını oku ──────────────────────────────────────────────────────
PREFIX_FILE = Path(__file__).parent / "pip_prefix.txt"

def _get_prefix() -> list[str]:
    if not PREFIX_FILE.exists():
        return [sys.executable, "-m", "pip", "install"]
    raw = PREFIX_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return [sys.executable, "-m", "pip", "install"]
    # "pip install ..." → python -m pip install ...
    # Kullanıcı sadece "pip install" yazmışsa ya da tam komut yazmışsa ikisini de destekle
    parts = raw.split()
    if parts[0].lower() == "pip":
        return [sys.executable, "-m"] + parts
    return parts  # tam yol verilmişse direkt kullan


def ensure_deps(verbose: bool = True) -> bool:
    """
    Eksik paketleri yükler.
    Returns: True → herşey tamam / yüklendi, False → en az bir yükleme başarısız.
    """
    missing = []
    for import_name, pip_name in PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        if verbose:
            print("[setup_deps] Tüm bağımlılıklar yüklü — atlanıyor.")
        return True

    prefix = _get_prefix()
    print(f"[setup_deps] {len(missing)} eksik paket bulundu: {', '.join(missing)}")
    print(f"[setup_deps] Prefix: {' '.join(prefix)}")

    all_ok = True
    for pkg in missing:
        cmd = prefix + [pkg]
        print(f"[setup_deps] Yükleniyor: {pkg} ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("OK")
        else:
            print("HATA")
            print(result.stderr[-400:] if result.stderr else "(çıktı yok)")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    success = ensure_deps(verbose=True)
    sys.exit(0 if success else 1)
