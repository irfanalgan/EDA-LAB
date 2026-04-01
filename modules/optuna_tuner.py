"""Optuna hiperparametre optimizasyonu — saf Python, Dash bağımsız."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Arama uzayları ──────────────────────────────────────────────────────────

def _suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 40, 300, step=20),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "max_depth":         trial.suggest_int("max_depth", 4, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, step=20),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0, 10),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1, 15, step=1),
        # sabitler
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def _suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 40, 300, step=20),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 15),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0, 10),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1, 15, step=1),
        # sabitler
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
        "verbosity": 0,
    }


# ── Optuna çalıştırıcı ─────────────────────────────────────────────────────

def run_optuna(
    X_train,
    y_train,
    X_test,
    y_test,
    model_type: str = "lgbm",
    n_trials: int = 50,
    cancel_event: Optional[threading.Event] = None,
    progress_dict: Optional[dict] = None,
) -> dict:
    """Optuna ile hiperparametre optimizasyonu çalıştır.

    Objektif: test_gini - max(0, train_gini - test_gini)
    (test performansını maximize et, overfitting'e ceza ver)

    progress_dict güncellenir:
        {"trial": int, "total": int, "best_score": float,
         "best_train_gini": float, "best_test_gini": float, "done": bool}

    Returns dict:
        {"best_params", "best_score", "best_train_gini",
         "best_test_gini", "best_gap", "n_trials_completed"}
    """
    if progress_dict is not None:
        progress_dict.update(
            trial=0, total=n_trials, best_score=None,
            best_train_gini=None, best_test_gini=None, done=False,
        )

    # lazy import — büyük kütüphaneler sadece gerektiğinde yüklensin
    if model_type == "xgb":
        from xgboost import XGBClassifier as _Clf
        _suggest = _suggest_xgb_params
    else:
        from lightgbm import LGBMClassifier as _Clf
        _suggest = _suggest_lgbm_params

    best_info: dict = {"train_gini": None, "test_gini": None}

    def _objective(trial: optuna.Trial) -> float:
        if cancel_event and cancel_event.is_set():
            raise optuna.exceptions.OptunaError("Cancelled")

        params = _suggest(trial)
        mdl = _Clf(**params)
        mdl.fit(X_train, y_train)

        tr_auc = roc_auc_score(y_train, mdl.predict_proba(X_train)[:, 1])
        te_auc = roc_auc_score(y_test,  mdl.predict_proba(X_test)[:, 1])

        tr_gini = 2 * tr_auc - 1
        te_gini = 2 * te_auc - 1
        overfit = max(0.0, tr_gini - te_gini)
        score   = te_gini - overfit

        # En iyi sonucu sakla (Optuna study.best_trial'dan da alınabilir
        # ama train_gini/test_gini ayrı lazım)
        if best_info["test_gini"] is None or score > (
            best_info["test_gini"] - max(0, best_info["train_gini"] - best_info["test_gini"])
        ):
            best_info["train_gini"] = tr_gini
            best_info["test_gini"]  = te_gini

        if progress_dict is not None:
            progress_dict["trial"] = trial.number + 1
            progress_dict["best_score"] = score
            progress_dict["best_train_gini"] = best_info["train_gini"]
            progress_dict["best_test_gini"]  = best_info["test_gini"]

        return score

    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(
            _objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[
                lambda study, trial: (
                    study.stop()
                    if cancel_event and cancel_event.is_set()
                    else None
                ),
            ],
        )
    except optuna.exceptions.OptunaError:
        pass  # iptal edildi — kısmi sonuç döndür

    # Sonuçları derle
    n_completed = len(study.trials)
    if n_completed == 0:
        result = {
            "best_params": {},
            "best_score": None,
            "best_train_gini": None,
            "best_test_gini": None,
            "best_gap": None,
            "n_trials_completed": 0,
        }
    else:
        bp = study.best_params
        result = {
            "best_params": bp,
            "best_score": study.best_value,
            "best_train_gini": best_info["train_gini"],
            "best_test_gini": best_info["test_gini"],
            "best_gap": round(
                (best_info["train_gini"] or 0) - (best_info["test_gini"] or 0), 4
            ),
            "n_trials_completed": n_completed,
        }

    if progress_dict is not None:
        progress_dict["done"] = True
        progress_dict["result"] = result

    return result
