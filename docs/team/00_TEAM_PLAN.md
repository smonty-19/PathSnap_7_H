# PathSnap — Team Plan (Team 7_H)

Read this file first, then your own file:

| Member | File | Tools | Owns |
|---|---|---|---|
| **Monty** | [`monty.md`](monty.md) | Claude Code | Data pipeline (load → candidates → labels → features), HMM/Viterbi, inference pipeline, repo/README |
| **Sanchita** | [`sanchita.md`](sanchita.md) | Claude Code | Metrics + training harness, LightGBM, Bayesian optimisation (Optuna), final model export, FastAPI backend (Laptop 1) |
| **Patil** | [`patil.md`](patil.md) | Free AI tools | EDA, Logistic Regression + KNN, Folium maps module |
| **Sheetal** | [`sheetal.md`](sheetal.md) | Free AI tools | Streamlit UI (Laptop 2), Random Forest + MLP, model-comparison plots, report assembly |

Patil's and Sheetal's tasks need no deep knowledge of the rest of the codebase. Each one depends only on a small **contract** (a file format or a function signature) that is written out in full in their file. Each file also has starter code, so the free AI tiers are enough.

Everyone owns at least one model and one report section, so **everyone can answer viva questions about their own part**. Read the other three files once too: the viva can ask anyone about anything.

---

## 1. The problem in one paragraph

Each trip has a noisy GPS track (`.track`: lon, lat, t at about 1 Hz), its own road network (`.nodes`: lon/lat by row index; `.arcs`: from_node/to_node by row index = `arc_id`), and a reference route (`.route`: ordered `arc_id`s). For every GPS point we:
1. Generate **candidate arcs** near it.
2. Use a **classifier** to score each (point, candidate) pair with P(this is the true arc).
3. Run **HMM + Viterbi** over the candidates, using road connectivity, to get a connected route.

A FastAPI server (Laptop 1) runs this on any of the 10 evaluator trips. A UI (Laptop 2) lets the evaluator pick a trip and shows the route and the accuracy.

### Data facts we verified (train + student_test only)
- 80 train trips and 10 student_test trips. Points per trip: min 262, median about 1,500, **max 19,313**. Arcs per trip: median about 13–19k.
- **Trips are all over the world** (longitude −100° to +137°, latitude −35° to +62°). Every trip needs its own metre-based projection (UTM zone picked from the trip centroid). Never use raw degrees as distance.
- About 85% of arcs also exist in reverse (A→B and B→A). **Direction matters**, so heading features are essential.
- Consecutive arcs in `.route` connect (`to_node == next.from_node`) 99.9% of the time, which is good for the HMM.
- `.route` is **per arc, not per GPS point**. Monty builds the point → arc labels with a monotone alignment (see `monty.md`).

---

## 2. Problem framing (everyone must be able to explain this)

- **One row = one (GPS observation, candidate arc) pair.** Binary label: `1` if the candidate is the reference arc for that observation, else `0`.
- **Groups = `trip_id`.** All CV is `GroupKFold(n_splits=5)` by trip, so no trip appears in both train and validation.
- **Metrics** (all computed by `evaluation/metrics.py`, owned by Sanchita):
  - **Observation-level Top-1 Accuracy (PRIMARY):** for each observation, take the candidate with the highest probability and check whether it is the reference arc. This is what the route depends on.
  - **Macro F1** over classes {0,1} on candidate rows (threshold 0.5), plus precision and recall.
  - **Log loss:** probability quality matters because the probabilities feed the HMM emissions.
  - **Candidate miss rate:** fraction of observations whose reference arc is not among the candidates (an upper bound on accuracy).
  - **Route continuity:** fraction of consecutive predicted arcs that are the same arc or directly connected.
  - **Training time and inference time** (ms per trip).

## 3. The 5 models (and why these 5)

They come from five different algorithm families in the guide, and each tests a different hypothesis.

| # | Model | Family | Owner | Why it's in the set |
|---|---|---|---|---|
| 1 | **Logistic Regression** (balanced class weight, scaled) | Linear / probabilistic | Patil | Calibrated, interpretable baseline. Tests whether a linear decision boundary is enough. |
| 2 | **KNN, distance-weighted** (scaled) | Instance-based | Patil | Tests similarity in feature space. Shows the cost of scaling and inference time on about 1M rows. |
| 3 | **Random Forest** | Ensemble (bagging) | Sheetal | Strong non-linear model, robust, little tuning, gives feature importances. |
| 4 | **MLP** (2 hidden layers, early stopping) | Neural network | Sheetal | Non-linear representation. Has **real epoch-wise train/val loss curves**, which the report needs. |
| 5 | **LightGBM** | Boosting | Sanchita | Expected best: fast, handles class imbalance, good probabilities, small model file. **Candidate for Bayesian tuning.** |

Optional 6th model if time allows (cheap): Gaussian Naive Bayes (Patil).
Not chosen:
- **SVM:** O(n²) on about 1M rows, and probabilities need extra Platt calibration.
- **Clustering:** produces no supervised probabilities.

We pick the final model **from the measured results**, not this prediction. If RF or MLP wins on CV Top-1 and log loss, we tune that one instead.

---

## 4. Shared contracts (DO NOT change without telling the team)

### 4.1 Features file: `data/processed/{train,student_test}.parquet` (producer: Monty)
| column | type | meaning |
|---|---|---|
| `trip_id` | str | e.g. `"00000001"` (the CV group) |
| `obs_idx` | int | row index of the GPS point in `.track` |
| `t` | float | timestamp in seconds |
| `cand_arc_id` | int | candidate arc (row index in `.arcs`) |
| `label` | int 0/1 | 1 if `cand_arc_id` is the reference arc for this observation |
| `true_arc_id` | int | aligned reference arc for this observation (−1 if unknown). **Never use as a feature.** |
| feature columns | float | listed in `pathsnap/features.py` as `FEATURE_COLUMNS` |

**Rule:** models use **only** `FEATURE_COLUMNS`. `trip_id`, `obs_idx`, `t`, `cand_arc_id`, `label` and `true_arc_id` are never features.

### 4.2 Training harness: `models/common.py` (producer: Sanchita)
```python
from models.common import run_experiment
run_experiment(
    name="logreg",                 # file-safe name
    family="Linear/probabilistic",
    estimator=my_sklearn_estimator_or_pipeline,
    config={...},                  # hyperparameters, saved to the log
    train_sample_frac=None,        # e.g. 0.3 for slow models (KNN); sampled by trip-aware stratification
)
```
It does: 5-fold GroupKFold CV → fit on all train → predict student_test → compute all metrics → save:
- `results/model_comparison.csv` (one row per model, appended or updated)
- `results/model_runs/<name>.json` (config, per-fold scores, timings, seed)
- `results/preds/<name>_student_test.parquet` and `<name>_train_oof.parquet` (columns `trip_id, obs_idx, cand_arc_id, prob`), which Monty's HMM uses
- `results/plots/confusion_<name>.png`
- `artifacts/<name>.joblib`

### 4.3 API contract (producer: Sanchita, consumer: Sheetal)
Base URL: `http://<laptop1-ip>:8000`
- `GET /health` → `{"status": "ok", "model": "lightgbm_tuned", "version": "..."}`
- `GET /trips` → `{"trips": ["00000003", "00000017", ...]}` (the 10 evaluator trip IDs found on the server)
- `POST /predict` with body `{"trip_id": "00000003"}` → 
```json
{
  "trip_id": "00000003",
  "n_points": 1520,
  "gps": [[lat, lon], ...],                       // raw GPS, LAT-LON order for folium
  "predicted_route": [[[lat, lon], [lat, lon]], ...], // one polyline per predicted arc, in travel order
  "matched_points": [[lat, lon], ...],            // each GPS point snapped to its predicted arc
  "metrics": {
    "hmm":        {"top1_accuracy": 0.93, "route_continuity": 0.99},
    "classifier": {"top1_accuracy": 0.88, "route_continuity": 0.81},
    "candidate_miss_rate": 0.02
  },
  "timing_ms": {"features": 210, "model": 35, "viterbi": 120, "total": 400}
}
```
**Ground truth NEVER leaves the server.** The response has no reference route geometry, only the metrics computed against it on the server.

---

## 5. Repository layout & folder ownership

Each person commits **only in their own folders**. This avoids merge conflicts. Always `git pull --rebase` before `git push`.

```
PathSnap_7_H/
├── README.md, requirements.txt, .gitignore        Monty
├── configs/            hmm.yaml, candidates.yaml   Monty
├── pathsnap/           io, geo, graph, candidates,
│                       labels, features, hmm,
│                       pipeline                     Monty
├── scripts/            build_dataset.py            Monty
├── evaluation/         metrics.py                  Sanchita
│                       hmm_eval.py                 Monty
│                       compare_models.py           Sheetal
├── models/             common.py, train_lgbm.py,
│                       tune_lgbm_optuna.py          Sanchita
│                       train_logreg.py, train_knn.py Patil
│                       train_rf.py, train_mlp.py    Sheetal
├── eda/                notebooks + DATA_NOTES.md   Patil
├── viz/                folium_maps.py              Patil
├── api/                FastAPI app                 Sanchita
├── ui/                 Streamlit app               Sheetal
├── artifacts/          final model + meta          (generated)
├── results/            csv, json, plots, maps      (generated)
├── report/             report sections + final PDF Sheetal assembles
└── data/               raw/ and processed/ → GIT-IGNORED (not committed)
```

**Data setup (everyone):** copy `Student folder/train` and `Student folder/student_test` to `PathSnap_7_H/data/raw/train` and `PathSnap_7_H/data/raw/student_test`. Monty shares `data/processed/*.parquet` over Drive or USB, because it is git-ignored.

---

## 6. Dependency graph & timeline

```
           ┌─────────── Patil: EDA (no dependency, start immediately)
           │
Monty M0 (repo skeleton, 45 min) ──► Monty M1 (features parquet) ─────────┐
           │                              │ (sample parquet early)         │
           │                              ▼                                ▼
Sanchita S0 (metrics + harness) ──► Patil LR/KNN, Sheetal RF/MLP, Sanchita LightGBM
                                                     │
                                                     ▼
                        Sheetal compare plots ◄── results/model_comparison.csv
                                                     │
                       Sanchita Optuna tuning ◄──────┘
                                │
Monty M2 HMM (uses preds/*.parquet) ──► Monty M3 pipeline.predict_trip ──► Sanchita API /predict
                                                                            │
Sheetal UI (mock mode first, NO dependency) ─────────────────────────────────► integration test (2 laptops)
                                                                            │
Patil Folium maps of predictions ◄── Monty HMM outputs                    ▼
                         Everyone's report sections ──► Sheetal assembles PDF
```

### Phase 1 (before the lunch-break evaluation)
| Who | Must be done |
|---|---|
| Monty | Repo skeleton pushed. `data/processed/train.parquet` and `student_test.parquet` built. Candidate miss-rate numbers. |
| Sanchita | `metrics.py` and `common.py` working. LightGBM baseline trained. FastAPI skeleton returning mock data. |
| Patil | EDA notebook with inventory, gaps, connectivity, at least 1 Folium map. LogReg trained. |
| Sheetal | UI working in mock mode. RF trained. MLP training. |

### Phase 2 (after lunch)
| Who | Must be done |
|---|---|
| Monty | HMM/Viterbi + classifier-vs-HMM comparison. `pipeline.predict_trip`. README. |
| Sanchita | Optuna study, tuned vs untuned comparison, final model export, real `/predict`, 2-laptop test. |
| Patil | KNN (+ optional GNB). `folium_maps.py` with predicted vs reference maps. Report sections. |
| Sheetal | MLP loss curves, `compare_models.py` plots, UI wired to the real API, screenshots, report assembly. |

### Hand-off signals (post in the team chat)
1. **Monty → all:** "sample parquet ready" (about 5 trips, so harness and model work can start early).
2. **Sanchita → Patil, Sheetal:** "harness ready".
3. **Monty → all:** "full parquet ready, re-run your models".
4. **Sanchita → Sheetal:** "API on `http://<ip>:8000`".
5. **Monty → Patil:** "HMM preds ready" (for the prediction maps).

---

## 7. Rules (from the official docs, violations = deductions or disqualification)
- **Never** train, tune, select features or pick thresholds using evaluator trips. Decoder parameters are tuned on train and student_test only.
- **No hardcoded predictions or answers.** The API must run the full pipeline for each request.
- Fit scalers and other preprocessing **inside the CV folds** (use sklearn `Pipeline`).
- Every number and graph in the report must come from files in `results/`. No invented numbers and no fake training curves for models that have no epochs.
- Don't commit secrets, `data/raw`, or model files over 100 MB.
- AI use must be disclosed in the report. **Write report text in your own words** about your own results.
- Fix random seeds (`SEED = 42`) and log them.
