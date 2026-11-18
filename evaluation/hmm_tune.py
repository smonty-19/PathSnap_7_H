"""Bayesian (Optuna TPE) tuning of the HMM decoder parameters.   Owner: Monty (M2).

    python -m evaluation.hmm_tune --trials 40 --timeout 900

Objective : mean observation-level Top-1 of the HMM on TRAIN out-of-fold classifier probabilities
            (every second training trip, to keep a trial at a few seconds). student_test and the evaluator
            trips are never used for tuning.
Writes    : configs/hmm.yaml (best params), results/hmm_tuning.csv (all trials).
"""
import argparse
import os
import time

import numpy as np
import optuna
import pandas as pd
import yaml

from evaluation.hmm_eval import load_joined, trip_context
from pathsnap.hmm import DEFAULT_CFG, decode

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm_tuned")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--every", type=int, default=2, help="use every n-th training trip")
    a = ap.parse_args()

    df = load_joined(a.model, "train")
    trips = sorted(df.trip_id.unique())[::a.every]
    data = []
    for tid in trips:
        g = df[df.trip_id == tid]
        net, pts, t = trip_context("train", tid)
        data.append((net, pts, t, g[["obs_idx", "cand_arc_id", "proj_frac", "dist_m", "prob"]],
                     g.groupby("obs_idx")["true_arc_id"].first()))
    print(f"tuning on {len(data)} train trips, {sum(len(d[4]) for d in data):,} observations")

    def top1(cfg):
        hit = n = 0
        for net, pts, t, g, truth in data:
            dec, _ = decode(net, pts, t, g, cfg)
            hit += int((dec.hmm_arc.values == truth.reindex(dec.obs_idx).values).sum()); n += len(dec)
        return hit / n

    base = top1(DEFAULT_CFG)
    print(f"default config Top-1: {base:.4f}")

    def objective(trial):
        return top1(dict(
            sigma_m=trial.suggest_float("sigma_m", 5.0, 40.0, log=True),
            beta_m=trial.suggest_float("beta_m", 1.0, 50.0, log=True),
            w=trial.suggest_float("w", 0.0, 1.0),
            eps=trial.suggest_float("eps", 1e-8, 1e-3, log=True),
            gap_s=trial.suggest_float("gap_s", 3.0, 30.0),
            max_m=300.0, back_penalty_m=trial.suggest_float("back_penalty_m", 0.0, 100.0)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
    study.enqueue_trial({k: DEFAULT_CFG[k] for k in ("sigma_m", "beta_m", "w", "eps", "gap_s", "back_penalty_m")})
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)
    tr = study.trials_dataframe()
    tr.to_csv(os.path.join("results", "hmm_tuning.csv"), index=False)
    best = {**DEFAULT_CFG, **study.best_params}
    with open(os.path.join("configs", "hmm.yaml"), "w") as f:
        f.write(f"# Tuned with Optuna TPE on train OOF ({a.model}), {len(study.trials)} trials, best train-subset Top-1 "
                f"{study.best_value:.4f} (default {base:.4f}).\n")
        yaml.safe_dump({k: float(v) for k, v in best.items()}, f)
    print(f"best Top-1 {study.best_value:.4f} vs default {base:.4f} after {len(study.trials)} trials, {time.time() - t0:.0f}s")
    print(best)


if __name__ == "__main__":
    main()
