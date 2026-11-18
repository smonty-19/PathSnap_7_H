# PathSnap: Distributed Map-Matching & Route Reconstruction (Team 7_H)

Map-matches noisy GPS trajectories to a road network: **candidate generation → ML classifier → HMM/Viterbi decoding**. It is served by a FastAPI backend (Laptop 1) and a Streamlit UI (Laptop 2).

> Work in progress. Installation, execution and deployment instructions will be completed before submission.

## Team & plans
- Team plan, contracts and timeline: [`docs/team/00_TEAM_PLAN.md`](docs/team/00_TEAM_PLAN.md)
- Individual plans: [Monty](docs/team/monty.md) · [Sanchita](docs/team/sanchita.md) · [Sheetal](docs/team/sheetal.md) · [Patil](docs/team/patil.md)
- **Who pushes where:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Repository layout
```
pathsnap/     shared library (io, geo, graph, candidates, labels, features, hmm, pipeline)   Monty
scripts/      dataset build scripts                                                         Monty
configs/      candidate + HMM configs                                                       Monty
models/       training harness + one training script per model                              shared (see models/README.md)
evaluation/   metrics, HMM evaluation, model comparison                                     shared (see evaluation/README.md)
eda/          exploratory data analysis                                                     Patil
viz/          Folium map generation                                                         Patil
api/          FastAPI backend (Laptop 1)                                                    Sanchita
ui/           Streamlit UI (Laptop 2)                                                       Sheetal
artifacts/    saved models + metadata
results/      CSVs, logs, plots, maps (every report number comes from here)
report/       technical report                                                              Sheetal assembles
data/         raw + processed data (git-ignored)
docs/team/    team plans
```

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Copy the dataset to `data/raw/train` and `data/raw/student_test` (see `data/README.md`; `data/` is git-ignored).

## Reproduce everything (run from the repo root)
```bash
# 1. features (about 10 s each)
python -m scripts.build_dataset --split student_test
python -m scripts.build_dataset --split train --stride 2

# 2. the five models (each writes results/model_comparison.csv, results/model_runs/, results/preds/, artifacts/)
python -m models.train_logreg
python -m models.train_knn
python -m models.train_rf
python -m models.train_mlp
python -m models.train_lgbm

# 3. Bayesian tuning of the selected model (LightGBM, Optuna TPE) -> artifacts/final_model.joblib + model_meta.json
python -m models.tune_lgbm_optuna --trials 40 --timeout 720

# 4. HMM: tune the decoder on train out-of-fold probabilities, then compare classifier-only vs classifier + HMM
python -m evaluation.hmm_tune --trials 40 --timeout 900     # writes configs/hmm.yaml, results/hmm_tuning.csv
python -m evaluation.hmm_eval                               # writes results/hmm_comparison.csv, results/preds/hmm_<trip>.parquet

# 5. model comparison plots
python -m evaluation.compare_models

# 6. Folium prediction maps (GPS, network, reference, classifier-only, HMM) -> results/maps/prediction_<trip>.html
python -m viz.folium_maps
```
Notebooks (outputs saved): `eda/01_eda.ipynb` (EDA), then `notebooks/02` to `10`: preprocessing & features, one notebook per model,
comparison & selection, Bayesian tuning, prediction maps. Index with owners: [`notebooks/README.md`](notebooks/README.md).

## Tests
```bash
python -m pytest -q api/test_api.py                    # API contract: 404s, response fields, no ground truth in responses
uvicorn api.main:app --port 8000 & python -m pytest -q ui/test_ui.py    # UI against a running API
```

## Run the demo (two laptops, same Wi-Fi or hotspot)
**Laptop 1 (API):**
```bash
export PATHSNAP_EVAL_DIR=/path/to/evaluator_trips     # default: data/raw/student_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open port 8000 in the firewall (Windows: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`)
and find the laptop IP with `ipconfig`. Details: [`api/README.md`](api/README.md).

**Laptop 2 (UI):**
```bash
streamlit run ui/app.py --server.port 8501     # set the API URL in the sidebar to http://<laptop1-ip>:8000
```

## Pipeline in one paragraph
`pathsnap.io` loads a trip (the reference route only when asked), `geo` projects it to a local metric frame, `candidates` finds the 10 nearest arcs
within 50 m of every GPS point, `features` builds 21 features per (point, candidate) pair, the tuned LightGBM scores each pair, and `hmm` decodes the most
likely connected arc sequence with Viterbi. `pathsnap.pipeline.predict_trip` runs all of this for one trip; `evaluate_trip` scores it server side.

## Headline results (student_test, 10 trips)
| | Top-1 | Route continuity |
|---|---|---|
| Tuned LightGBM, classifier only | 0.8847 | 0.9543 |
| Tuned LightGBM + HMM/Viterbi | 0.9301 | 0.9790 |

Numbers come from `results/model_comparison.csv` and `results/hmm_comparison.csv`.

