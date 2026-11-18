"""Shared training harness: GroupKFold CV on the 80 training trips -> refit -> student-test evaluation.

    from models.common import run_experiment
    run_experiment(name, family, estimator, config, train_sample_frac=None)
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURE_COLUMNS
from evaluation.metrics import all_metrics, confusion

SEED = 42
N_FOLDS = 5
PROC = os.path.join("data", "processed")
MAX_MEDIAN_ALIGN_M = 50.0   # data-quality filter: route does not match GPS track


def excluded_trips():
    """Training trips whose reference route is far from the GPS track (median > 50 m)."""
    s = pd.read_csv(os.path.join("results", "dataset_summary_train.csv"), dtype={"trip_id": str})
    return sorted(s.loc[s.median_align_dist_m > MAX_MEDIAN_ALIGN_M, "trip_id"].tolist())


def load_split(split):
    df = pd.read_parquet(os.path.join(PROC, f"{split}.parquet"))
    df["trip_id"] = df["trip_id"].astype(str)
    if split == "train":
        df = df[~df.trip_id.isin(excluded_trips())].reset_index(drop=True)
    return df


def sample_observations(df, frac, seed=SEED):
    """Sample whole observations (all candidates of an observation kept) so Top-1 stays computable."""
    keys = df[["trip_id", "obs_idx"]].drop_duplicates()
    keys = keys.sample(frac=frac, random_state=seed)
    return df.merge(keys, on=["trip_id", "obs_idx"]).reset_index(drop=True)


def _pred_frame(df, prob):
    out = df[["trip_id", "obs_idx", "cand_arc_id", "label"]].copy()
    out["prob"] = prob
    return out


def cross_validate(estimator, df, n_folds=N_FOLDS):
    X, y, groups = df[FEATURE_COLUMNS].values, df["label"].values, df["trip_id"].values
    oof = np.zeros(len(df))
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        est = clone(estimator)
        t0 = time.time()
        est.fit(X[tr], y[tr])
        fit_s = time.time() - t0
        oof[va] = est.predict_proba(X[va])[:, 1]
        m = all_metrics(_pred_frame(df.iloc[va], oof[va]))
        m.update(fold=k, fit_s=round(fit_s, 2))
        folds.append(m)
    return oof, folds


def rebuild_comparison():
    """results/model_comparison.csv is rebuilt from all per-model JSON logs (safe for parallel runs)."""
    rows = []
    for fn in sorted(os.listdir("results/model_runs")):
        if fn.endswith(".json"):
            with open(os.path.join("results/model_runs", fn)) as f:
                r = json.load(f)
            rows.append({k: v for k, v in r.items() if k not in ("folds", "features", "excluded_train_trips")})
    pd.DataFrame(rows).sort_values("cv_top1_mean", ascending=False).to_csv("results/model_comparison.csv", index=False)


def run_experiment(name, family, estimator, config, train_sample_frac=None, seed=SEED, save_model=True):
    os.makedirs("results/model_runs", exist_ok=True); os.makedirs("results/preds", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True); os.makedirs("artifacts", exist_ok=True)
    train, test = load_split("train"), load_split("student_test")
    if train_sample_frac:
        train = sample_observations(train, train_sample_frac, seed)
    print(f"[{name}] train rows={len(train):,} trips={train.trip_id.nunique()} | test rows={len(test):,}")

    oof, folds = cross_validate(estimator, train)
    _pred_frame(train, oof).to_parquet(f"results/preds/{name}_train_oof.parquet", index=False)
    fdf = pd.DataFrame(folds)

    est = clone(estimator)
    t0 = time.time()
    est.fit(train[FEATURE_COLUMNS].values, train["label"].values)
    train_s = time.time() - t0
    t0 = time.time()
    prob = est.predict_proba(test[FEATURE_COLUMNS].values)[:, 1]
    infer_ms_per_trip = (time.time() - t0) * 1000 / test.trip_id.nunique()
    tp = _pred_frame(test, prob)
    tp.to_parquet(f"results/preds/{name}_student_test.parquet", index=False)
    tm = all_metrics(tp, split="student_test")
    train_fit_m = all_metrics(_pred_frame(train, est.predict_proba(train[FEATURE_COLUMNS].values)[:, 1]))

    row = dict(
        model=name, family=family, config_json=json.dumps(config, default=str),
        cv_macro_f1_mean=fdf.macro_f1.mean(), cv_macro_f1_std=fdf.macro_f1.std(),
        cv_top1_mean=fdf.top1_accuracy.mean(), cv_top1_std=fdf.top1_accuracy.std(),
        cv_logloss=fdf.logloss.mean(),
        train_top1=train_fit_m["top1_accuracy"], train_macro_f1=train_fit_m["macro_f1"],
        test_top1=tm["top1_accuracy"], test_macro_f1=tm["macro_f1"], test_precision=tm["precision"],
        test_recall=tm["recall"], test_logloss=tm["logloss"], test_cand_miss_rate=tm["candidate_miss_rate"],
        test_route_continuity_clf=tm["route_continuity"],
        train_time_s=round(train_s, 2), infer_ms_per_trip=round(infer_ms_per_trip, 2),
        train_rows=len(train), seed=seed,
    )
    with open(f"results/model_runs/{name}.json", "w") as f:
        json.dump(dict(row, folds=folds, features=FEATURE_COLUMNS,
                       excluded_train_trips=excluded_trips(),
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=2, default=float)
    rebuild_comparison()

    cm = confusion(tp)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name} (student_test)")
    fig.tight_layout(); fig.savefig(f"results/plots/confusion_{name}.png", dpi=120); plt.close(fig)
    if save_model:
        joblib.dump(est, f"artifacts/{name}.joblib", compress=3)
    print(f"[{name}] CV macroF1={row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} "
          f"CV top1={row['cv_top1_mean']:.4f} | test top1={row['test_top1']:.4f} "
          f"macroF1={row['test_macro_f1']:.4f} logloss={row['test_logloss']:.4f} "
          f"cont={row['test_route_continuity_clf']:.4f} | fit {train_s:.1f}s, infer {infer_ms_per_trip:.0f} ms/trip")
    return row, est
