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
.venv\Scripts\activate          # Windows (on macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
