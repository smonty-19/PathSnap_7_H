# Monty: Data Pipeline, HMM/Viterbi, Inference Pipeline, Repo Lead

**Tools:** Claude Code. **You are on the critical path.** Everyone's models wait for your features parquet, so ship a *sample* parquet early.
Read [`00_TEAM_PLAN.md`](00_TEAM_PLAN.md) first (contracts in §4).

## Where you push
You push **only** to:
- `pathsnap/` (whole folder)
- `scripts/`
- `configs/`
- `evaluation/hmm_eval.py`
- `models/train_knn.py`, plus the outputs of your own KNN runs in `results/`, `report/sections/3_models_knn.md`
- `README.md`, `requirements.txt`, `.gitignore`, `CONTRIBUTING.md`, `docs/`
- `results/candidates_sweep.csv`, `results/hmm_comparison.csv`, `results/hmm_tuning.csv`, `results/preds/hmm_*.parquet`
- `report/sections/2_2_eda.md` (preprocessing/features part, shared with Patil), `report/sections/5_hmm.md`

Always run `git pull --rebase origin main` before `git push origin main`, and `git add` only your own files. Need a change outside your folders? Ask the owner. Full rules: [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Summary of deliverables
| ID | Deliverable | Files | Phase | Unblocks |
|---|---|---|---|---|
| M0 | Repo skeleton | `README.md`, `requirements.txt`, `.gitignore`, folders, `pathsnap/io.py`, `geo.py`, `graph.py` | 1 (first 45 min) | everyone |
| M1 | Candidates, labels, features dataset | `pathsnap/candidates.py`, `labels.py`, `features.py`, `scripts/build_dataset.py`, `configs/candidates.yaml` | 1 | all models |
| M2 | HMM + Viterbi + evaluation | `pathsnap/hmm.py`, `evaluation/hmm_eval.py`, `configs/hmm.yaml` | 2 | final route, report §5 |
| M3 | Single-trip inference pipeline | `pathsnap/pipeline.py` | 2 | Sanchita's API |
| M4 | README, integration, repo audit | `README.md` | 2 | submission |
| M5 | KNN model | `models/train_knn.py` | 1–2 (after the parquet + harness) | model comparison |
| R | Report: preprocessing/candidates/features part of §2.2, all of §5 (HMM), KNN subsection in §3.2 | `report/sections/` | 2 | — |

---

## M0: Repo skeleton (do this first)
- Create the folder layout from team plan §5 with empty `__init__.py` files.
- `requirements.txt`: numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, optuna, networkx, pyproj, shapely>=2, folium, matplotlib, seaborn, joblib, pyyaml, fastapi, uvicorn, requests, streamlit, streamlit-folium.
- `.gitignore`: `data/`, `__pycache__/`, `.venv/`, `*.sqlite` (optional), `artifacts/*.joblib` over 100 MB.
- `pathsnap/io.py`: `load_trip(trip_dir) -> dict(track=(N,3) lon/lat/t, nodes=(M,2) lon/lat, arcs=(A,2) int, route=(R,) int | None)`. Load `.route` **only** when asked (`with_route=True`) so ground truth stays separate from prediction.
- `pathsnap/geo.py`: per-trip UTM projection (EPSG chosen from the trip centroid via pyproj), `to_xy(lon, lat)`, `to_lonlat(x, y)`, bearings.
- `pathsnap/graph.py`: arc geometry arrays (start xy, end xy, length, bearing), `successors[arc] = arcs whose from_node == arcs[arc].to_node`, a networkx DiGraph with length weights, and a bounded Dijkstra helper `network_dist(arc_a, pos_a, arc_b, pos_b, max_m)`.
- Push to `main` (as smonty-19, **no Claude co-author line**). Post "skeleton ready".

## M1: Candidates → labels → features
1. **Candidates** (`candidates.py`): shapely `STRtree` over arc LineStrings (in metres). For each GPS point take the arcs within radius `R` (start at 50 m), keep the top-`K` nearest (start at K=10). If there are none, widen to 2R and flag it. Arcs are directed and about 85% have a reverse twin, so **keep both directions** and let heading features separate them.
   - Sweep R ∈ {25, 50, 75, 100} and K ∈ {5, 8, 10, 15} on **train** and save `results/candidates_sweep.csv` (miss rate vs average candidates). Pick the setting from this. It's report material.
2. **Labels** (`labels.py`): `.route` is a sequence of arcs, not per point. Assign each GPS point to a route index with a **monotone DP alignment**: route index never decreases over time, and cost = point-to-arc distance (plus a small penalty per skipped arc). `true_arc_id = route[aligned_idx]`. Write down the limitations (U-turns, route vs GPS mismatches) in `eda/DATA_NOTES.md` in a "labels" section, or tell Patil.
   - Sanity check: median distance from points to their aligned arcs should be a few metres. Plot a histogram for the report.
3. **Features** (`features.py`, export `FEATURE_COLUMNS`). Suggested set (your own design, adjust as needed):
   - Geometric: `dist_m`, `dist_rank`, `dist_minus_min`, `dist_ratio_min`, `n_cands`, `proj_frac` (0–1 along the arc), `at_endpoint`, `arc_length_m`.
   - Direction: `heading_diff` (0–180°) between the GPS bearing (smoothed over ±2 pts) and the arc bearing, `heading_valid` (0 when speed < 1 m/s), `heading_diff_x_speed`.
   - Kinematics: `speed_mps`, `accel`, `turn_rate`, `dt_prev`, `dt_next`, `gap_flag` (dt > 5 s).
   - Context/topology: `mean_dist_window` (mean distance of the same arc to points ±3), `prev_best_connected` (candidate equals or succeeds the nearest arc of the previous point), `arc_out_degree`, `arc_in_degree`, `cand_has_reverse`.
   - Everything is computed from the track and network only, **never from the route**.
4. `scripts/build_dataset.py --split train|student_test [--trips N] [--stride s]` → `data/processed/<split>.parquet`. Use multiprocessing per trip. The 19k-point trip is big, so `--stride` lets you subsample observations for training if the row count gets too large (keep stride=1 for student_test).
5. **Ship order:** `--trips 5` sample → post "sample parquet ready" → full build → post "full parquet ready". Share files via Drive or USB (`data/` is git-ignored).

## M2: HMM + Viterbi (`pathsnap/hmm.py`)
- **Hidden states:** the candidate arcs of each observation (with their projected position on the arc).
- **Emission:** `log P(obs | arc) = w · log(p_clf) + (1−w) · log N(dist; 0, σ)`. `p_clf` comes from the selected model, so **the decoder uses the classifier outputs consistently**. Compare w=0 (pure geometry) against the tuned w.
- **Transition:** Newson & Krumm style `exp(−|d_network − d_great_circle| / β)`. `d_network` is the shortest path from the projected position on arc i to arc j (bounded Dijkstra; same arc = along-arc distance; going backwards along the same arc gets a penalty).
- **Invalid or disconnected transitions:** if there is no path within `max_m`, use a small floor probability ε instead of 0, so decoding never dies. Record the count as "breaks".
- **Gaps and missing candidates:** if dt > gap threshold or an observation has no candidates, either scale β with dt, or **split the trip into segments** and decode each one separately. Observations without candidates get no prediction and count as wrong in accuracy (be honest about this).
- **Speed:** precompute transitions only between consecutive candidate sets, and cache Dijkstra per source arc. K≈10 means about 100 pairs per step.
- `evaluation/hmm_eval.py`: on student_test (and train OOF preds from `results/preds/*_train_oof.parquet`), compare **classifier-only argmax** vs **classifier+HMM**:
  - Top-1 accuracy
  - Route continuity
  - Sequence-level metric: route-arc precision/recall of the predicted *set* of arcs vs `.route`, or edit distance
  - Number of breaks
  
  Save `results/hmm_comparison.csv` plus per-trip rows, and note trips where the HMM *hurt*.
- **Decoder tuning:** small grid or Optuna over σ, β, w, ε, gap threshold, using **train OOF + student_test only**. Save to `configs/hmm.yaml` and `results/hmm_tuning.csv`.
- Post "HMM preds ready" and save `results/preds/hmm_<trip>.parquet` for Patil's maps.

## M3: `pathsnap/pipeline.py` (Sanchita's API depends on this; agree the signature early)
```python
def predict_trip(trip_dir, model, model_meta, hmm_cfg) -> dict:
    """Uses ONLY .track/.nodes/.arcs. Returns per-observation predicted arc (hmm & classifier),
    matched lat/lon, predicted route polylines (lat,lon), timings."""
def evaluate_trip(trip_dir, prediction) -> dict:
    """Loads .route separately (server side), aligns labels, returns metrics. Never called before predict."""
```
Test it on all 10 student_test trips (time per trip must be OK for a live demo, aim under 10 s, including the 19k-point case → stride/segment if needed).

## M4: README & audit
- README covers: setup (venv + `pip install -r requirements.txt`), data placement, how to rebuild features, train each model, tune, run HMM eval, start the API (Laptop 1), start the UI (Laptop 2), firewall note, and folder map.
- Audit before submission: grep for hardcoded trip IDs and arc lists, make sure no evaluator data is in git, make sure every report number exists in `results/`.

## M5: KNN (`models/train_knn.py`)
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from models.common import run_experiment

cfg = dict(n_neighbors=25, weights="distance")
est = make_pipeline(StandardScaler(), KNeighborsClassifier(**cfg, n_jobs=-1))
run_experiment("knn", "Instance-based", est, cfg, train_sample_frac=0.25)
```
(KNN is slow at prediction time, which is why it trains on a 20% sample. Mention this in the report.) One fixed configuration. **No grid or random search** (organiser rule).
- For the report: why similarity in feature space suits the problem, why scaling matters, the objective (none; a distance-weighted majority vote), the inference-time cost, student-test Top-1/F1/log loss, and over- or underfitting evidence (small k overfits, large k underfits; compare CV vs student-test).

## Dependencies
- **You depend on:** nothing for M0/M1. For M2 you need `results/preds/*` from the harness (Sanchita) and a chosen model. Until then you can test with LightGBM baseline preds.
- **Depend on you:** everyone (parquet), Sanchita (`pipeline.py`), Patil (HMM preds for maps).

## Viva prep (you must be able to explain)
Why UTM per trip, candidate radius and K choice (miss-rate sweep), the label alignment and its limitations, each feature's rationale, emission and transition formulas, how gaps and disconnections are handled, and where the HMM helped or hurt.
