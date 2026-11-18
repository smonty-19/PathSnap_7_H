# Patil: EDA, Logistic Regression + KNN, Folium Maps

**Tools:** free AI tools (no Pro). Every task below comes with ready-to-run code. Fill in the blanks, run it, look at the output, and write down what you notice.
Read [`00_TEAM_PLAN.md`](00_TEAM_PLAN.md) §1–3 first (5 minutes).

## Where you push
You push **only** to:
- `eda/` (whole folder)
- `viz/` (whole folder)
- `models/train_logreg.py`, `models/train_knn.py`, `models/train_gnb.py`
- `results/eda/`, `results/maps/`, plus the outputs of your own LogReg/KNN runs in `results/`
- `report/sections/2_2_eda.md` (EDA part, shared with Monty), `report/sections/6_folium.md`, `report/sections/3_models_logreg.md`, `report/sections/3_models_knn.md`

Always run `git pull --rebase origin main` before `git push origin main`, and `git add` only your own files. Need a change outside your folders? Ask the owner. Full rules: [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

### How to use a free AI assistant efficiently
- Paste **only the section you're working on** plus your error message, and ask for **one function at a time**.
- Run on 2–3 trips first, then on all trips.
- Stuck for more than 20 minutes? Ping Monty or Sanchita (Claude Code); they can fix it fast.
- Understand every line you commit. The viva will ask *you* about EDA, LogReg, KNN and the maps.

## Summary of deliverables
| ID | Deliverable | Files | Phase | Depends on |
|---|---|---|---|---|
| E1 | Exploratory data analysis | `eda/01_eda.ipynb`, `eda/DATA_NOTES.md`, `results/eda/*` | **1, start now** | nothing |
| E2 | Logistic Regression + KNN (+ optional GaussianNB) | `models/train_logreg.py`, `models/train_knn.py`, (`models/train_gnb.py`) | 1–2 | Sanchita's harness + Monty's parquet |
| E3 | Folium map module + prediction maps | `viz/folium_maps.py`, `results/maps/*.html` | 2 | Monty's HMM preds (for the predicted layer) |
| R | Your report sections: §2.2 EDA plots/tables, §6 Folium, LogReg & KNN subsections in §3.2. Send them to **Sheetal**, who assembles the PDF. | `report/sections/` | 2 | — |

---

## E1: EDA (use **train** and **student_test** only; label each output with its partition)
Setup: `pip install numpy pandas matplotlib seaborn networkx folium jupyter`

**Cell 1: loader and helpers (copy as-is)**
```python
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
ROOT = "data/raw"            # contains train/ and student_test/
OUT = "results/eda"; os.makedirs(OUT, exist_ok=True)

def load_trip(split, trip):
    d = os.path.join(ROOT, split, trip)
    track = pd.read_csv(f"{d}/{trip}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(f"{d}/{trip}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs  = pd.read_csv(f"{d}/{trip}.arcs",  sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(f"{d}/{trip}.route", header=None, names=["arc_id"])
    return track, nodes, arcs, route

def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))

trips = [(s, t) for s in ["train", "student_test"] for t in sorted(os.listdir(f"{ROOT}/{s}"))]
```

**Cell 2: one row of stats per trip (this one table covers most of the checklist)**
```python
rows = []
for split, trip in trips:
    track, nodes, arcs, route = load_trip(split, trip)
    dt = track.t.diff().dropna()
    step = haversine_m(track.lon.values[:-1], track.lat.values[:-1], track.lon.values[1:], track.lat.values[1:])
    speed = step / np.maximum(dt.values, 1e-9)
    pairs = set(zip(arcs.u, arcs.v))
    G = nx.DiGraph(); G.add_nodes_from(range(len(nodes))); G.add_edges_from(pairs)
    r = route.arc_id.values
    route_breaks = int(np.sum(arcs.v.values[r[:-1]] != arcs.u.values[r[1:]]))
    rows.append(dict(
        split=split, trip=trip, n_points=len(track), n_nodes=len(nodes), n_arcs=len(arcs), n_route_arcs=len(r),
        n_nan=int(track.isna().sum().sum()), n_dup=int(track.duplicated().sum()),
        t_sorted=bool(track.t.is_monotonic_increasing), pct_dt_1s=float((dt == 1).mean()),
        gaps_over_5s=int((dt > 5).sum()), gaps_over_30s=int((dt > 30).sum()), max_gap_s=float(dt.max()),
        median_speed=float(np.median(speed)), jumps_over_70mps=int((speed > 70).sum()),
        lon_c=track.lon.mean(), lat_c=track.lat.mean(),
        pct_arcs_with_reverse=float(np.mean([(v, u) in pairs for u, v in pairs])),
        n_weak_components=nx.number_weakly_connected_components(G),
        n_isolated_nodes=nx.number_of_isolates(G),
        route_breaks=route_breaks,
    ))
stats = pd.DataFrame(rows); stats.to_csv(f"{OUT}/trip_stats.csv", index=False)
stats.describe()
```

**Then make these outputs** (each is 1–3 lines of pandas or matplotlib; ask the AI if unsure):
1. `inventory.csv`: number of trips per split (`stats.groupby("split").size()`), plus file sizes (`os.path.getsize`).
2. Histogram of `n_points` (note the one huge trip of about 19k points).
3. Missing/invalid table: `n_nan`, `n_dup`, `t_sorted`, and points with lat outside ±90 or lon outside ±180.
4. Histogram of `dt` for 5 trips, plus a bar chart of `gaps_over_5s` per trip.
5. Histogram of speed in m/s, with jumps marked.
6. World scatter of `lon_c`, `lat_c` (shows trips are on several continents, which is why distances are computed in metres per trip).
7. Bar charts of `n_nodes`/`n_arcs`, and a histogram of `pct_arcs_with_reverse`.
8. Table of connectivity: `n_weak_components`, `n_isolated_nodes`.
9. Table of `route_breaks` (should be almost all 0).
10. `anomalies.csv`: `stats[(stats.gaps_over_30s > 0) | (stats.jumps_over_70mps > 0) | (stats.route_breaks > 0)]`.
11. **Folium map (mandatory).** Use `make_trip_map` from E3 below on 2–3 trips: GPS + reference route + nearby roads.
12. `eda/DATA_NOTES.md`: 5–10 bullets on what you found (e.g. "route is per-arc, not per-point", "x trips have gaps > 30 s", "trips are worldwide → per-trip projection").

Post a summary in the team chat when done. Monty will use your gap and anomaly findings.

## E2: Models (after "harness ready" from Sanchita)
The harness does CV, metrics, saving and plots. You only choose the model and its settings.

`models/train_logreg.py`:
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from models.common import run_experiment

cfg = dict(C=1.0, class_weight="balanced", max_iter=1000)
est = make_pipeline(StandardScaler(), LogisticRegression(**cfg, random_state=42))
run_experiment("logreg", "Linear/probabilistic", est, cfg)
```
`models/train_knn.py`:
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from models.common import run_experiment

for k in [5, 25, 75]:
    cfg = dict(n_neighbors=k, weights="distance")
    est = make_pipeline(StandardScaler(), KNeighborsClassifier(**cfg, n_jobs=-1))
    run_experiment(f"knn_k{k}", "Instance-based", est, cfg, train_sample_frac=0.2)
```
(KNN is slow at prediction time, which is why it trains on a 20% sample. Mention this in the report.) Pick the best k for the main table.
Optional: `models/train_gnb.py`, the same shape with `GaussianNB()` (adds a 6th model in 2 minutes).

For the report, write down for each model:
- Why it suits the problem.
- Strengths and limitations **as seen in your numbers**.
- The hyperparameters.
- The loss or objective: LogReg = log-loss with L2; KNN = no training objective, majority vote weighted by distance.
- Student-test Top-1, F1 and log loss (from `results/model_comparison.csv`).
- Over- or underfitting evidence: compare the CV score with the student-test score.

## E3: `viz/folium_maps.py`
Starter (copy and extend). All inputs are in **[lat, lon]** order:
```python
import folium

def make_trip_map(gps_latlon, network_segments_latlon=None, reference_route_latlon=None,
                  pred_classifier_latlon=None, pred_hmm_latlon=None, out_html="map.html"):
    m = folium.Map(tiles="OpenStreetMap")
    def layer(name): g = folium.FeatureGroup(name); g.add_to(m); return g
    if network_segments_latlon:
        g = layer("Road network")
        for seg in network_segments_latlon: folium.PolyLine(seg, color="#999", weight=1).add_to(g)
    if reference_route_latlon:
        folium.PolyLine(reference_route_latlon, color="green", weight=5, tooltip="Reference").add_to(layer("Reference route"))
    if pred_classifier_latlon:
        folium.PolyLine(pred_classifier_latlon, color="orange", weight=3, tooltip="Classifier only").add_to(layer("Classifier only"))
    if pred_hmm_latlon:
        folium.PolyLine(pred_hmm_latlon, color="red", weight=3, tooltip="HMM").add_to(layer("HMM prediction"))
    g = layer("Raw GPS")
    for lat, lon in gps_latlon[::2]: folium.CircleMarker([lat, lon], radius=2, color="blue").add_to(g)
    folium.LayerControl().add_to(m)
    m.fit_bounds(gps_latlon)
    m.save(out_html)
    return m
```
Converting from the files: `gps_latlon = track[["lat", "lon"]].values.tolist()`. The reference route is, for each `arc_id`, `[nodes.loc[u, ["lat", "lon"]], nodes.loc[v, ["lat", "lon"]]]`. Road network: only the arcs whose nodes fall inside the GPS bounding box (otherwise the map is too heavy).
To do:
- Add a small HTML legend (ask the AI: "add a fixed-position legend to a folium map").
- After Monty posts "HMM preds ready", make maps for 3 student_test trips → `results/maps/*.html`, and take screenshots (PNG) for the PDF.

## Viva prep
How the files relate (track → nodes/arcs → route), the data-quality issues you found, why degrees aren't distances (haversine), LogReg vs KNN (linear boundary vs similarity, why scaling matters, why KNN is slow), and how to read the Folium maps.
