# Sanchita: Metrics, Training Harness, LightGBM, Bayesian Optimisation, FastAPI (Laptop 1)

**Tools:** Claude Code. Your harness is what Patil and Sheetal plug their models into, so **keep it simple to call and ship it early**.
Read [`00_TEAM_PLAN.md`](00_TEAM_PLAN.md) first (contracts in §4.1–4.3).

## Where you push
You push **only** to:
- `api/` (whole folder)
- `models/common.py`, `models/train_lgbm.py`, `models/tune_lgbm_optuna.py`
- `evaluation/metrics.py`
- `artifacts/` (final model + `model_meta.json`)
- `results/model_comparison.csv`, `results/model_runs/`, `results/preds/` for LightGBM runs, `results/optuna/`, `results/tuning_comparison.csv`, `results/plots/lgbm_*`
- `report/sections/4_bayesian_opt.md`, `report/sections/3_models_lightgbm.md`, API part of `report/sections/7_api_ui.md`

Always run `git pull --rebase origin main` before `git push origin main`, and `git add` only your own files. Need a change outside your folders? Ask the owner. Full rules: [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## ⏰ Before 11:30 (phase-1 evaluation)
**Already done and pushed:** data pipeline, features, all 5 models trained with the shared harness, model comparison (`results/model_comparison.csv`), and Bayesian tuning of LightGBM (Optuna TPE).
Your job before 11:30 is to **own and be able to explain your part**. The evaluators will ask *you*.

Setup (10 min): `git pull`, `pip install -r requirements.txt`, then copy the dataset into `data/raw/train` and `data/raw/student_test`.

1. Read and own `models/common.py` (harness: GroupKFold(5) by trip → refit → student_test → metrics/logs) and `evaluation/metrics.py` (Top-1, macro F1, log loss, candidate miss rate, route continuity).
2. Own **LightGBM** (`models/train_lgbm.py`): untuned CV Top-1 0.874, test Top-1 0.883, log loss 0.115, 6 s fit.
3. Own **Bayesian tuning** (`models/tune_lgbm_optuna.py`):
   - Optuna **TPE** sampler + MedianPruner.
   - Objective: mean GroupKFold(3) Top-1.
   - Search space and its justification, the number of trials, the best params (`results/optuna/best_params.json`), and tuned vs untuned (`results/tuning_comparison.csv`).
   - Open `results/optuna/optimization_history.png` and `param_importance.png`, and be able to explain them.
4. Explain **why LightGBM was selected**: best CV and test Top-1 (the primary metric), best route continuity, fast fit (so many Bayesian trials are affordable), a 1.3 MB model file and fast inference.
5. Remember: **no grid or random search** anywhere, only Bayesian optimisation.

Read the team viva guide shared by Monty (not in the repo) before 11:30.

## Summary of deliverables
| ID | Deliverable | Files | Phase | Unblocks |
|---|---|---|---|---|
| S0 | Metrics module + training harness | `evaluation/metrics.py`, `models/common.py` | 1 (first ~1.5 h) | Patil, Sheetal, Monty HMM |
| S1 | LightGBM baseline (untuned) | `models/train_lgbm.py` | 1 | model comparison |
| S2 | Bayesian optimisation + final model export | `models/tune_lgbm_optuna.py`, `artifacts/final_model.joblib`, `artifacts/model_meta.json` | 2 | API, HMM |
| S3 | FastAPI backend + deployment notes | `api/main.py`, `api/config.py`, `api/README.md` | skeleton in 1, real in 2 | Sheetal UI, final demo |
| R | Report: §4 (Bayesian opt), LightGBM subsection in §3.2, API part of §7 | `report/sections/` | 2 | — |

---

## S0: `evaluation/metrics.py`
Pure functions over a predictions DataFrame `df[trip_id, obs_idx, cand_arc_id, label, prob]`:
- `top1_accuracy(df)`: argmax of `prob` per (trip_id, obs_idx); correct if that row has label 1. **Observations whose true arc is not among the candidates count as wrong.**
- `macro_f1(df, thr=0.5)`, `precision_recall(df)`, `logloss(df)` (clip probs), `confusion(df)`
- `candidate_miss_rate(df)`: fraction of observations with no `label==1` row
- `route_continuity(pred_arcs_seq, successors)`: fraction of consecutive pairs that are the same arc or `b in successors[a]`. Use Monty's `pathsnap.graph`. Until that exists, write it against a plain `dict[int, set[int]]`.

## S0: `models/common.py` (harness)
Signature (already promised to the team, don't change it):
```python
run_experiment(name, family, estimator, config, train_sample_frac=None, seed=42)
```
Steps:
1. Load `data/processed/train.parquet` and `student_test.parquet`. `X = df[FEATURE_COLUMNS]`, `y = label`, `groups = trip_id`. Import `FEATURE_COLUMNS` from `pathsnap.features`.
2. Optional subsampling for slow models (`train_sample_frac`): sample whole observations (keep all candidates of a sampled observation) so Top-1 stays computable.
3. `GroupKFold(5)` CV: per fold record macro F1, Top-1, log loss and fit time. Keep **out-of-fold probabilities** and save them to `results/preds/<name>_train_oof.parquet` (Monty needs these for decoder tuning).
4. Refit on all train, then predict student_test. Record inference time per trip (ms).
5. Write/update one row in `results/model_comparison.csv` with columns:
   `model, family, config_json, cv_macro_f1_mean, cv_macro_f1_std, cv_top1_mean, cv_top1_std, cv_logloss, test_top1, test_macro_f1, test_precision, test_recall, test_logloss, test_cand_miss_rate, test_route_continuity_clf, train_time_s, infer_ms_per_trip`
6. Save `results/model_runs/<name>.json` (all folds, config, seed, timestamp), `results/plots/confusion_<name>.png`, and `artifacts/<name>.joblib`.
7. Return a dict of results. If the estimator exposes training history (e.g. MLP `loss_curve_`), let the caller save it. Sheetal handles the MLP curves in a separate script.

**Unblock trick:** before the real parquet exists, build a tiny synthetic DataFrame with the same schema to test the harness end to end. Then post "harness ready" with a 10-line usage example.

## S1: LightGBM baseline (`models/train_lgbm.py`)
`LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, class_weight="balanced" or scale_pos_weight, random_state=42)` → `run_experiment("lightgbm", "Boosting", ...)`. Also save feature importances (gain) to `results/plots/lgbm_importance.png` for the report.

## S2: Bayesian optimisation (`models/tune_lgbm_optuna.py`)
- **Organiser rule: Bayesian optimisation only. No grid or random search anywhere.**
- Tune the model that wins the comparison table. Expected: LightGBM. If RF or MLP clearly wins, tune that one instead and adapt the search space.
- **Sampler:** Optuna `TPESampler(seed=42)` (Bayesian, tree-structured Parzen estimator), with `MedianPruner`.
- **Objective:** mean GroupKFold(5) **CV observation-level Top-1 accuracy** on the 80 train trips. Also log macro F1 and log loss per trial. Justify Top-1 as the primary metric: it is what the route depends on.
- **Search space** (justify it in the report):
  - `num_leaves` 15–255 (log)
  - `learning_rate` 0.01–0.3 (log)
  - `n_estimators` 100–1500 (or early stopping)
  - `min_child_samples` 10–300 (log)
  - `feature_fraction` 0.5–1.0
  - `bagging_fraction` 0.5–1.0 with `bagging_freq` 1
  - `lambda_l1`, `lambda_l2` 1e-3–10 (log)
  - `scale_pos_weight` 1–15
- **Trials:** 40–60 (watch the clock; time each trial). Use storage `sqlite:///results/optuna/lgbm.db` so the record persists, and export `results/optuna/trials.csv`.
- Plots: optimisation history, parameter importances, and a slice plot → `results/optuna/*.png`.
- Refit with the best params, then evaluate on student_test with the same harness under the name `lightgbm_tuned`. Write `results/tuning_comparison.csv` with **untuned vs tuned**: CV macro F1, test Top-1, test macro F1, log loss, inference time, plus total tuning compute time.
- Export `artifacts/final_model.joblib` and `artifacts/model_meta.json` (`feature_columns`, `model_name`, `params`, `trained_on: "train(80)"`, `seed`, `date`). **Never touch evaluator trips.**

## S3: FastAPI (`api/`)
- Implements the contract in team plan §4.3 exactly (`/health`, `/trips`, `/predict`). Sheetal builds against it.
- **Phase 1 skeleton:** `/predict` returns a mock using real GPS from a student_test trip and fake metrics, so Sheetal can integrate early.
- **Phase 2:** load the model, meta and `configs/hmm.yaml` once at startup. `/predict` calls Monty's `pipeline.predict_trip()` and then `pipeline.evaluate_trip()`.
- `PATHSNAP_EVAL_DIR` env var points to the folder of evaluator trips (use `data/raw/student_test` for rehearsal). `/trips` lists the subfolders.
- **Logging (must show in CMD during evaluation):** one clear line per request with timestamp, client IP, trip_id, n_points, timings, and HMM/classifier top-1. Use Python `logging` with uvicorn's console.
- Errors: unknown trip → 404, pipeline failure → 500 with a message (never crash the server).
- Response contains **no reference route geometry** (ground truth stays server-side).
- `api/README.md`: `uvicorn api.main:app --host 0.0.0.0 --port 8000`, how to find the laptop IP (`ipconfig`), Windows Firewall rule for port 8000, and both laptops on the same Wi-Fi/hotspot.
- Rehearse with Sheetal: run all 10 student_test trips through UI → API on two laptops and record the timings.

## Dependencies
- **You depend on:** Monty's parquet schema (sample parquet first) and `pathsnap.features.FEATURE_COLUMNS`. For the real API you need `pipeline.predict_trip`/`evaluate_trip` (agree the signature with Monty early).
- **Depend on you:** Patil and Sheetal (harness), Monty (OOF and test preds), Sheetal (API).

## Viva prep
Why GroupKFold by trip, why Top-1 is primary, how log loss relates to HMM emissions, LightGBM leaf-wise growth vs RF bagging, what TPE does differently from grid or random search, the search-space choices, tuned vs untuned results, and how the API keeps ground truth server-side.
