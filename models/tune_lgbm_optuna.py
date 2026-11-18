"""Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE sampler).

- Search: Tree-structured Parzen Estimator (Bayesian) -- no grid or random search.
- Objective: mean GroupKFold (by trip) observation-level Top-1 accuracy on the training trips.
- Pruning: MedianPruner on per-fold intermediate scores.
- Record: results/optuna/lgbm.db (sqlite study), trials.csv, plots, tuning_comparison.csv.

    python -m models.tune_lgbm_optuna --trials 30 --timeout 900
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from evaluation.metrics import top1_accuracy
from models.common import SEED, load_split, run_experiment
from pathsnap.features import FEATURE_COLUMNS

OUT = os.path.join("results", "optuna")
CV_FOLDS = 3   # group-aware folds inside the search loop (5-fold is used for the final report numbers)


def make_model(p):
    return LGBMClassifier(**p, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)


def search_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=900, help="seconds")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    train = load_split("train")
    X, y, g = train[FEATURE_COLUMNS].values, train["label"].values, train["trip_id"].values
    splits = list(GroupKFold(n_splits=CV_FOLDS).split(X, y, g))

    def objective(trial):
        p = search_space(trial)
        scores = []
        for k, (tr, va) in enumerate(splits):
            m = make_model(p).fit(X[tr], y[tr])
            pf = train.iloc[va][["trip_id", "obs_idx", "cand_arc_id", "label"]].copy()
            pf["prob"] = m.predict_proba(X[va])[:, 1]
            scores.append(top1_accuracy(pf))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=f"lgbm_top1_{time.strftime('%Y%m%d_%H%M%S')}", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=f"sqlite:///{OUT}/lgbm.db", load_if_exists=False)
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout, show_progress_bar=False)
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(f"{OUT}/trials.csv", index=False)
    best = study.best_params
    with open(f"{OUT}/best_params.json", "w") as f:
        json.dump(dict(best_params=best, best_cv_top1=study.best_value, n_trials=len(study.trials),
                       n_complete=int((trials.state == "COMPLETE").sum()), search_seconds=round(search_s, 1),
                       sampler="TPE", objective=f"mean GroupKFold({CV_FOLDS}) top-1 accuracy"), f, indent=2)

    # plots of the search record
    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(done.number, done.value, s=14, label="trial")
    ax.plot(done.number, done.value.cummax(), color="C1", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel(f"CV top-1 ({CV_FOLDS}-fold)"); ax.legend(); ax.set_title("Optuna TPE optimisation history")
    fig.tight_layout(); fig.savefig(f"{OUT}/optimization_history.png", dpi=120); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_title("Hyperparameter importance (fANOVA)")
        fig.tight_layout(); fig.savefig(f"{OUT}/param_importance.png", dpi=120); plt.close(fig)
    except Exception as e:  # needs enough completed trials
        print("importance plot skipped:", e)

    # refit with best params and evaluate with the same harness (5-fold CV + student_test)
    tuned_cfg = dict(best, subsample_freq=1)
    row_t, est = run_experiment("lightgbm_tuned", "Boosting (Bayesian-tuned)", make_model(best), tuned_cfg)
    os.makedirs("artifacts", exist_ok=True)
    import joblib
    joblib.dump(est, "artifacts/final_model.joblib", compress=3)
    with open("artifacts/model_meta.json", "w") as f:
        json.dump(dict(model_name="lightgbm_tuned", feature_columns=FEATURE_COLUMNS, params=tuned_cfg,
                       trained_on="train partition (data-quality-filtered)", seed=SEED,
                       date=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=2)

    comp = pd.read_csv("results/model_comparison.csv")
    cols = ["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1", "test_logloss",
            "test_route_continuity_clf", "train_time_s", "infer_ms_per_trip"]
    tc = comp[comp.model.isin(["lightgbm", "lightgbm_tuned"])][cols]
    tc["search_seconds"] = [np.nan if m == "lightgbm" else round(search_s, 1) for m in tc.model]
    tc.to_csv("results/tuning_comparison.csv", index=False)
    print(tc.to_string(index=False))


if __name__ == "__main__":
    main()
