"""
Bağımlılık yükleyici — app.py başlamadan önce çalışır.

Kullanım:
  1. pip_prefix.txt dosyasını aç.
  2. İş yerinin gerektirdiği ön komutu yapıştır.
     Örnek:  pip install --index-url https://... --trusted-host ...
  3. Dosyayı kaydet. Bir daha dokunmana gerek yok.

Mantık:
  - Her paket için önce import dener → başarılıysa atlar (hızlı).
  - Minimum versiyon belirtilmişse kurulu versiyonu kontrol eder.
  - Başarısızsa pip_prefix.txt'teki komuta paket adını ekleyerek yükler.
"""

import importlib
import subprocess
import sys
from pathlib import Path

# ── Paket listesi: (import_adı, pip_adı, minimum_versiyon | None) ────────────
PACKAGES = [
    ("dash",                    "dash>=4.0",                        "4.0"),
    ("dash_bootstrap_components","dash-bootstrap-components>=1.5",  None),
    ("pandas",                  "pandas>=2.0",                      None),
    ("numpy",                   "numpy",                            None),
    ("plotly",                  "plotly>=5.18",                     None),
    ("scipy",                   "scipy>=1.11",                      None),
    ("sklearn",                 "scikit-learn>=1.3",                None),
    ("lightgbm",                "lightgbm>=4.0",                    None),
    ("xgboost",                 "xgboost>=2.0",                     None),
    ("statsmodels",             "statsmodels",                      None),
    ("optbinning",              "optbinning",                       None),
    ("matplotlib",              "matplotlib",                       None),
    ("pyodbc",                  "pyodbc",                           None),
    ("sqlalchemy",              "sqlalchemy",                       None),
    ("shap",                    "shap",                             None),
    ("psutil",                  "psutil",                           None),
    ("openpyxl",                "openpyxl",                         None),
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


def _ver_tuple(v: str) -> tuple:
    """'4.0.1' → (4, 0, 1) — basit versiyon karşılaştırma için."""
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def _check_min_version(import_name: str, min_ver: str) -> bool:
    """Kurulu paketin versiyonu >= min_ver mi kontrol eder."""
    try:
        from importlib.metadata import version as _pkg_ver
        # import_name ve pip_name farklı olabilir (sklearn → scikit-learn)
        # Dash için import_name = "dash", metadata name = "dash" → sorunsuz
        installed = _pkg_ver(import_name)
        return _ver_tuple(installed) >= _ver_tuple(min_ver)
    except Exception:
        return False


def ensure_deps(verbose: bool = True) -> bool:
    """
    Eksik veya sürümü yetersiz paketleri yükler.
    Returns: True → herşey tamam / yüklendi, False → en az bir yükleme başarısız.
    """
    missing = []
    for import_name, pip_name, min_ver in PACKAGES:
        try:
            importlib.import_module(import_name)
            # Paket var ama versiyon yetersiz mi?
            if min_ver and not _check_min_version(import_name, min_ver):
                if verbose:
                    print(f"[setup_deps] {import_name} kurulu ama >= {min_ver} gerekli — güncelleniyor.")
                missing.append(pip_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        if verbose:
            print("[setup_deps] Tüm bağımlılıklar yüklü — atlanıyor.")
        return True

    prefix = _get_prefix()
    print(f"[setup_deps] {len(missing)} eksik/eski paket bulundu: {', '.join(missing)}")
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
