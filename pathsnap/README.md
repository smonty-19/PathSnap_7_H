# `pathsnap/`

**Owner (only this person pushes here):** **Monty**

Shared library: loading, projection, road graph, candidates, labels, features, HMM, inference pipeline. Others import from here; **don't edit it**, ask Monty instead.

| File | What it is |
|---|---|
| `io.py` | `load_trip(trip_dir, with_route=False)` |
| `geo.py` | per-trip UTM projection, bearings |
| `graph.py` | arc geometry, successors, network distance |
| `candidates.py` | candidate arcs per GPS point |
| `labels.py` | monotone alignment of GPS points to route arcs |
| `features.py` | feature engineering + `FEATURE_COLUMNS` |
| `hmm.py` | HMM emissions/transitions + Viterbi |
| `pipeline.py` | `predict_trip()` / `evaluate_trip()` used by the API |
