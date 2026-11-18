# Sheetal: Streamlit UI (Laptop 2), Random Forest + MLP, Model-Comparison Plots, Report Assembly

**Tools:** free AI tools (no Pro). Your UI needs **only the API contract** below, so you can start immediately in mock mode.
Read [`00_TEAM_PLAN.md`](00_TEAM_PLAN.md) §1–3 first (5 minutes).

## Where you push
You push **only** to:
- `ui/` (whole folder)
- `models/train_rf.py`, `models/train_mlp.py`
- `evaluation/compare_models.py`
- `results/plots/` (comparison plots, `rf_importance.png`, `mlp_loss.png`), plus the outputs of your own RF/MLP runs in `results/`
- `report/REPORT.md`, `report/PathSnap_7_H_Report.pdf`, `report/sections/3_1_3_3_comparison.md`, `report/sections/7_api_ui.md`, `report/sections/3_models_rf.md`, `report/sections/3_models_mlp.md`

Always run `git pull --rebase origin main` before `git push origin main`, and `git add` only your own files. Need a change outside your folders? Ask the owner. Full rules: [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

### How to use a free AI assistant efficiently
- Paste **only the section you're working on** plus your error message, and ask for one function at a time.
- Test with small data first (2–3 trips, or `train_sample_frac=0.1`).
- Stuck for more than 20 minutes? Ping Monty or Sanchita (Claude Code).
- Understand every line you commit. The viva will ask *you* about the UI, RF, MLP and the comparison plots.

## Summary of deliverables
| ID | Deliverable | Files | Phase | Depends on |
|---|---|---|---|---|
| P1 | Evaluator UI (Streamlit) | `ui/app.py`, `ui/mock_response.json`, `ui/README.md` | **1, start now (mock)** → 2 (real API) | API contract only; Sanchita's API for the final hookup |
| P2 | Random Forest + MLP | `models/train_rf.py`, `models/train_mlp.py`, `results/plots/mlp_loss.png` | 1–2 | Sanchita's harness + Monty's parquet |
| P3 | Model-comparison plots | `evaluation/compare_models.py`, `results/plots/*.png` | 2 | `results/model_comparison.csv` (all 5 models) |
| R | Report: §3.1 table, §3.3 plots, §7 (UI + API screenshots), RF & MLP subsections in §3.2 | `report/sections/` | 2 | — |
| P4 | **Report assembly** (one consolidated PDF) | `report/` | 2 (start the skeleton in phase 1) | everyone's sections |

---

## P1: UI (`ui/app.py`)
Setup: `pip install streamlit streamlit-folium folium requests`

**API contract** (Sanchita implements it; don't change it on your own):
- `GET {API}/trips` → `{"trips": ["00000003", ...]}`
- `POST {API}/predict` with `{"trip_id": "..."}` → `gps` ([lat, lon] list), `predicted_route` (list of polylines, each a list of [lat, lon]), `matched_points`, `metrics` {`hmm`: {top1_accuracy, route_continuity}, `classifier`: {...}, `candidate_miss_rate`}, `timing_ms` {...}
- The UI **never** receives or shows the ground-truth route. It only shows metrics the server computed.

Requirements:
1. Sidebar: API URL text box (default `http://localhost:8000`), a "Test connection" button that calls `/health`, and a **Mock mode** checkbox.
2. Main area: dropdown/radio of the **10 trips** from `/trips` → "Run map-matching" button → spinner.
3. Show a **Folium map** (via `streamlit_folium.st_folium`): raw GPS points (blue dots), predicted route (red line), optional matched points; layer control and legend; zoom fitted to the GPS.
4. Show a **metrics table**: HMM vs classifier-only Top-1 accuracy and route continuity, candidate miss rate, and the timing breakdown.
5. Clear error messages (API unreachable, 404, timeout of about 60 s). Don't crash.

Starter skeleton:
```python
import json, requests, streamlit as st, folium
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
api = st.sidebar.text_input("API URL", "http://localhost:8000")
mock = st.sidebar.checkbox("Mock mode", value=False)

def get_trips():
    if mock: return ["MOCK_TRIP"]
    return requests.get(f"{api}/trips", timeout=10).json()["trips"]

def predict(trip_id):
    if mock: return json.load(open("ui/mock_response.json"))
    r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=60)
    r.raise_for_status(); return r.json()

trip = st.selectbox("Evaluation trip", get_trips())
if st.button("Run map-matching"):
    with st.spinner("Matching..."):
        st.session_state["res"] = predict(trip)
res = st.session_state.get("res")
if res:
    m = folium.Map(tiles="OpenStreetMap")
    gps = folium.FeatureGroup("Raw GPS")
    for lat, lon in res["gps"][::2]:
        folium.CircleMarker([lat, lon], radius=2, color="blue").add_to(gps)
    pred = folium.FeatureGroup("Predicted route")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="red", weight=4).add_to(pred)
    gps.add_to(m); pred.add_to(m); folium.LayerControl().add_to(m)
    m.fit_bounds(res["gps"])
    st_folium(m, width=None, height=600)
    st.json(res["metrics"]); st.json(res["timing_ms"])
```
`ui/mock_response.json`: make one by hand from a few GPS points of any **student_test** trip (convert lon,lat → [lat, lon]) with fake metrics. It's only for building the UI and must never be shown at evaluation.
Run: `streamlit run ui/app.py --server.port 8501`. `ui/README.md`: install, run, and how to set the API URL to Laptop 1's IP (both laptops on the same Wi-Fi/hotspot).
Phase 2: when Sanchita posts the API URL, test all 10 student_test trips end to end and take **screenshots of the UI** and of the **API CMD logs on Laptop 1** (needed for report §7).

## P2: Models (after "harness ready" from Sanchita)
`models/train_rf.py`:
```python
from sklearn.ensemble import RandomForestClassifier
from models.common import run_experiment
cfg = dict(n_estimators=300, max_depth=20, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1)
run_experiment("random_forest", "Ensemble (bagging)", RandomForestClassifier(**cfg, random_state=42), cfg)
```
Also save feature importances to `results/plots/rf_importance.png`. Check that the `.joblib` is under 100 MB; if not, reduce `max_depth` or `n_estimators`.

`models/train_mlp.py`: `make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, early_stopping=True, max_iter=200, random_state=42))` → `run_experiment("mlp", "Neural network", est, cfg, train_sample_frac=0.5)`.
**Loss curves (report requirement, must be real):** in a separate small script/function, split train **by trip** (e.g. `GroupShuffleSplit`, 80/20). Train with `partial_fit` one epoch at a time, record train log-loss and val log-loss each epoch, and plot them → `results/plots/mlp_loss.png`. Only MLP gets loss curves (RF has no epochs, so don't draw curves for it).

For each model write down: why it suits the problem, strengths and limitations **as seen in your numbers**, hyperparameters, the loss/objective (RF: Gini impurity per split; MLP: cross-entropy with Adam), student-test Top-1/F1/log loss, and over- or underfitting evidence (RF: train vs CV gap; MLP: where the val-loss curve flattens or rises).

## P3: `evaluation/compare_models.py`
Reads `results/model_comparison.csv` (and `results/model_runs/*.json` for per-fold scores) and saves to `results/plots/`:
1. `cv_macro_f1.png`: bar per model with error bars = std across the 5 folds (shows stability).
2. `test_top1_f1.png`: grouped bars of student-test Top-1 accuracy and macro F1.
3. `logloss.png`: bar chart (lower is better).
4. `inference_time.png`: ms per trip (log scale if KNN dominates).
5. `confusion_grid.png`: 2×3 grid of the confusion matrices.
6. `tuned_vs_untuned.png` from `results/tuning_comparison.csv` (once Sanchita has it).
Also print a Markdown version of the §3.1 comparison table to paste into the report.

## P4: Report assembly
- Early in phase 1, create `report/REPORT.md` (or a shared Google Doc) with the required headings: **§2.2 EDA, §3.1 comparison table, §3.2 per-model subsections, §3.3 visualisations, §4 Bayesian optimisation, §5 HMM/Viterbi, §6 Folium, §7 API/UI/deployment, AI-use disclosure**. Put each section's owner next to its heading.
- Owners:
  - §2.2: Patil (EDA) + Monty (preprocessing/candidates/features)
  - §3.1/3.3/7: you
  - §3.2: each model's owner
  - §4: Sanchita
  - §5: Monty
  - §6: Patil
- Chase missing sections. Make sure the wording is consistent and in **our own words**, not pasted AI text.
- Check that every number matches a file in `results/` and every graph comes from a script. Then export **one PDF** to `report/PathSnap_7_H_Report.pdf`.

## Dependencies
- **You depend on:** the API contract (have it now), Sanchita's harness (P2), and everyone's model results (P3).
- **Depend on you:** the final demo (UI on Laptop 2) and the report's results section.

## Viva prep
How the UI talks to the API (and why ground truth never reaches the UI), RF bagging and feature randomness vs boosting, MLP architecture/early stopping and how to read your loss curves, and how the comparison plots justify the final model choice.
