"""
Gelişmiş Değişken Eleme — saf hesaplama fonksiyonları.
Dash bağımsız. Tüm fonksiyonlar DataFrame alır, sonuç dict döner.
"""

import threading
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import logging
log = logging.getLogger(__name__)


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _make_estimator(model_type="lgbm", n_estimators=100, max_depth=None, seed=42):
    """Hafif LightGBM veya XGBoost estimator üret."""
    if model_type == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 6,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
    # default: lgbm
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth or -1,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def _quick_auc(model, X_tr, y_tr, X_te, y_te):
    """Fit + predict → AUC. NaN-safe."""
    try:
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        return roc_auc_score(y_te, proba)
    except Exception:
        return 0.5


def _cancelled(cancel_event):
    return cancel_event is not None and cancel_event.is_set()


# ── Stratified Sampling ──────────────────────────────────────────────────────

def stratified_sample(X, y, max_rows=300_000):
    """Target oranını koruyarak örneklem al.
    max_rows=None veya 0 → tümü. Zaten küçükse dokunma."""
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y
    frac = max_rows / len(X)
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=42)
    return X_s, y_s


# ── 1. Tekli Gini Kontrolü ───────────────────────────────────────────────────

def single_gini_check(X, y, threshold=0.85, model_type="lgbm",
                       cancel_event=None, progress_cb=None):
    """Her değişken için tek başına Gini hesapla.
    Eşik üstü → leakage şüphesi → eleme önerisi.

    Returns: {"eliminated": [...], "details": {var: gini_value}}
    """
    details = {}
    eliminated = []
    n = len(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    for i, col in enumerate(X.columns):
        if _cancelled(cancel_event):
            break

        mdl = _make_estimator(model_type, n_estimators=50, max_depth=3)
        x_tr = X_tr[[col]].copy()
        x_te = X_te[[col]].copy()

        auc = _quick_auc(mdl, x_tr, y_tr, x_te, y_te)
        gini = 2 * auc - 1
        details[col] = round(gini, 4)

        if gini >= threshold:
            eliminated.append(col)

        if progress_cb:
            progress_cb(int((i + 1) / n * 100))

    return {"eliminated": eliminated, "details": details}


# ── 2. Sıfır Kazanç Eleme ────────────────────────────────────────────────────

def zero_gain_elimination(X, y, model_type="lgbm", cancel_event=None):
    """Tüm değişkenlerle model kur, importance=0 olanları ele.

    Returns: {"eliminated": [...], "importances": {var: importance}}
    """
    mdl = _make_estimator(model_type, n_estimators=100)
    try:
        mdl.fit(X, y)
    except Exception as e:
        log.warning("zero_gain_elimination fit hatası: %s", e)
        return {"eliminated": [], "importances": {}}

    imp = mdl.feature_importances_
    importances = dict(zip(X.columns, [round(float(v), 4) for v in imp]))
    eliminated = [col for col, v in importances.items() if v == 0]

    return {"eliminated": eliminated, "importances": importances}


# ── 3. RFE ────────────────────────────────────────────────────────────────────

def rfe_elimination(X, y, min_features=10, step=5, model_type="lgbm",
                    cancel_event=None, progress_cb=None):
    """Recursive Feature Elimination — sklearn RFE wrapper.

    Returns: {"eliminated": [...], "selected": [...], "ranking": {var: rank}}
    """
    from sklearn.feature_selection import RFE

    mdl = _make_estimator(model_type, n_estimators=100)
    n_features = max(1, min_features)
    step_size = max(1, step)

    rfe = RFE(estimator=mdl, n_features_to_select=n_features, step=step_size)

    try:
        rfe.fit(X, y)
    except Exception as e:
        log.warning("rfe_elimination fit hatası: %s", e)
        return {"eliminated": [], "selected": list(X.columns), "ranking": {}}

    ranking = dict(zip(X.columns, [int(r) for r in rfe.ranking_]))
    selected = [col for col, r in ranking.items() if r == 1]
    eliminated = [col for col, r in ranking.items() if r > 1]

    if progress_cb:
        progress_cb(100)

    return {"eliminated": eliminated, "selected": selected, "ranking": ranking}


# ── 4. Performans Eğrisi ─────────────────────────────────────────────────────

def performance_curve(X, y, step=5, metric="gini", model_type="lgbm",
                      cancel_event=None, progress_cb=None):
    """Importance sırasına göre artan değişken sayısıyla metrik eğrisi.

    1. Tüm değişkenlerle model → importance sıralaması
    2. İlk step, 2*step, ... ile mini modeller → metrik
    3. Grafik verisi döner (eleme yapmaz — slider callback'te yapılır)

    Returns: {
        "ranked_vars": [var1, var2, ...],
        "curve": [(n_vars, metric_value), ...],
        "importances": {var: importance}
    }
    """
    # Tüm değişkenlerle model → importance sıralaması
    full_mdl = _make_estimator(model_type, n_estimators=100)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    try:
        full_mdl.fit(X_tr, y_tr)
    except Exception as e:
        log.warning("performance_curve fit hatası: %s", e)
        return {"ranked_vars": [], "curve": [], "importances": {}}

    imp = full_mdl.feature_importances_
    importances = dict(zip(X.columns, [round(float(v), 4) for v in imp]))

    # Importance'a göre sırala (büyükten küçüğe)
    ranked_vars = sorted(X.columns, key=lambda c: importances.get(c, 0), reverse=True)

    # Artan değişken sayısıyla metrik hesapla
    step_size = max(1, step)
    n_points = list(range(step_size, len(ranked_vars) + 1, step_size))
    if n_points and n_points[-1] != len(ranked_vars):
        n_points.append(len(ranked_vars))
    if not n_points:
        n_points = [len(ranked_vars)]

    curve = []
    total_points = len(n_points)
    for idx, n in enumerate(n_points):
        if _cancelled(cancel_event):
            break

        top_vars = ranked_vars[:n]
        mdl = _make_estimator(model_type, n_estimators=100)
        auc = _quick_auc(mdl, X_tr[top_vars], y_tr, X_te[top_vars], y_te)

        if metric == "gini":
            val = round(2 * auc - 1, 4)
        else:  # auc
            val = round(auc, 4)

        curve.append((n, val))

        if progress_cb:
            progress_cb(int((idx + 1) / total_points * 100))

    return {
        "ranked_vars": ranked_vars,
        "curve": curve,
        "importances": importances,
    }


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(X, y, steps, max_rows=300_000,
                 cancel_event=None, progress_dict=None):
    """Gelişmiş eleme pipeline'ı. Her adım öncekinin çıktısı üzerinde çalışır.

    steps: [
        {"method": "gini",       "params": {"threshold": 0.85}},
        {"method": "zero_gain",  "params": {}},
        {"method": "rfe",        "params": {"min_features": 10, "step": 5}},
        {"method": "perf_curve", "params": {"step": 5, "metric": "gini"}},
    ]

    progress_dict: paylaşılan dict, callback tarafından okunur:
        {
            "current_step": 0,
            "current_method": "gini",
            "step_progress": 0,
            "results": {method: result_dict},
            "step_counts": {"gini": (before, after), ...},
            "done": False,
            "error": None,
        }
    """
    if progress_dict is None:
        progress_dict = {}

    progress_dict.update({
        "current_step": 0,
        "current_method": "",
        "step_progress": 0,
        "results": {},
        "step_counts": {},
        "done": False,
        "error": None,
    })

    try:
        # Stratified sample (bir kez)
        X_s, y_s = stratified_sample(X, y, max_rows)
        model_type = steps[0].get("model_type", "lgbm") if steps else "lgbm"

        current_X = X_s.copy()
        all_eliminated = []

        for i, step in enumerate(steps):
            if _cancelled(cancel_event):
                progress_dict["error"] = "İptal edildi"
                break

            method = step["method"]
            params = step.get("params", {})
            params["model_type"] = model_type

            progress_dict["current_step"] = i + 1
            progress_dict["current_method"] = method
            progress_dict["step_progress"] = 0

            before_count = len(current_X.columns)

            def _prog(pct, _m=method):
                progress_dict["step_progress"] = pct

            if method == "gini":
                result = single_gini_check(
                    current_X, y_s,
                    threshold=params.get("threshold", 0.85),
                    model_type=model_type,
                    cancel_event=cancel_event,
                    progress_cb=_prog,
                )
            elif method == "zero_gain":
                result = zero_gain_elimination(
                    current_X, y_s,
                    model_type=model_type,
                    cancel_event=cancel_event,
                )
                progress_dict["step_progress"] = 100
            elif method == "rfe":
                result = rfe_elimination(
                    current_X, y_s,
                    min_features=params.get("min_features", 10),
                    step=params.get("step", 5),
                    model_type=model_type,
                    cancel_event=cancel_event,
                    progress_cb=_prog,
                )
            elif method == "perf_curve":
                result = performance_curve(
                    current_X, y_s,
                    step=params.get("step", 5),
                    metric=params.get("metric", "gini"),
                    model_type=model_type,
                    cancel_event=cancel_event,
                    progress_cb=_prog,
                )
            else:
                result = {"eliminated": []}

            progress_dict["results"][method] = result

            # Perf curve eleme yapmaz (slider callback'te yapılır)
            elim = result.get("eliminated", [])
            if method != "perf_curve" and elim:
                current_X = current_X.drop(columns=elim, errors="ignore")
                all_eliminated.extend(elim)

            after_count = len(current_X.columns)
            progress_dict["step_counts"][method] = (before_count, after_count)

        progress_dict["all_eliminated"] = all_eliminated
        progress_dict["done"] = True

    except Exception as e:
        log.exception("Advanced elim pipeline hatası")
        progress_dict["error"] = str(e)
        progress_dict["done"] = True

    return progress_dict
