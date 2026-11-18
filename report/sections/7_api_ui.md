<!-- DRAFT: API part by Sanchita (AI-assisted). Sheetal adds the UI part and the screenshots of the UI and of the API console on Laptop 1. -->
# 7. API, UI and deployment

## 7.1 FastAPI backend (Laptop 1)
The backend (`api/main.py`) exposes `GET /health`, `GET /trips` and `POST /predict {"trip_id"}`.
The model, its metadata and the decoder configuration are loaded once at start-up. Each `/predict` call runs the whole pipeline on the trip's
`.track`, `.nodes` and `.arcs` files: candidate generation (50 m radius, 10 nearest arcs), 21 features, the Bayesian-tuned LightGBM classifier,
then HMM/Viterbi decoding. Nothing is precomputed or cached per trip, and no predictions are hardcoded.

The response contains the raw GPS points, the predicted route as one polyline per arc in travel order, each GPS point snapped to its predicted arc,
the metrics and the timings. The reference route is read only on the server to compute the metrics (HMM and classifier Top-1, route continuity,
candidate miss rate); its geometry is never returned.

Robustness: only exact folder names found in `PATHSNAP_EVAL_DIR` are accepted (unknown trip: 404, which also blocks path traversal);
a pipeline failure returns 500 with a message and the server keeps running. Every request prints one console line with the timestamp, client IP,
trip id, number of points, timings and the HMM and classifier Top-1.

## 7.2 Measured performance (one laptop, no network)
student_test trips: total time per trip 60 ms to 720 ms, including the largest of them (3,958 points).
Largest training trips, run through the same pipeline for timing only: 15,783 points in 1.4 s, 16,031 points in 2.1 s and the 19,313-point
trip in 2.3 s (plus 0.7 s for the server-side metric computation), with a peak memory of about 350 MB. This is below the 10 s target for a live demo.
Network transfer of the response (GPS, matched points and polylines) is not included.

## 7.3 Deployment
Start with `uvicorn api.main:app --host 0.0.0.0 --port 8000`, set `PATHSNAP_EVAL_DIR` to the evaluator trips, put both laptops on the same Wi-Fi
or hotspot, open port 8000 in the firewall, and point the UI at `http://<laptop1-ip>:8000`. Details in `api/README.md`.
