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

## 7.2 Measured performance
**Rehearsal on the 10 student_test trips** (`python -m api.rehearse`, results in `results/api_rehearsal.csv`; client and server on the same laptop, so no
wireless network time): every trip completes in 86 ms to 1,118 ms round trip (server total 77 ms to 1,080 ms, which includes the server-side metric
computation). The largest trip (3,958 points) takes 1,118 ms and a 293 KB response; the median trip takes about 0.45 s. Time splits roughly into features
10 to 20%, classifier 25 to 40% and Viterbi 30 to 40%. Over Wi-Fi the transfer of up to about 300 KB adds a further few tens of milliseconds.
The two-laptop rehearsal with the UI is still to be recorded by re-running the same script from Laptop 2 with the URL of Laptop 1.

**Large trips (timing only).** The largest training trips were run through the same pipeline: 15,783 points in 1.4 s, 16,031 points in 2.1 s and the
19,313-point trip in 2.3 s (plus 0.7 s for the metrics), with a peak memory of about 350 MB. This is below the 10 s target for a live demo.

**Tests.** `api/test_api.py` (12 tests) checks the contract fields, the 404 and 422 error paths including path-traversal input, the metric ranges, valid
JSON without NaN, and that no ground-truth field appears in a response.

## 7.3 Deployment
Start with `uvicorn api.main:app --host 0.0.0.0 --port 8000`, set `PATHSNAP_EVAL_DIR` to the evaluator trips, put both laptops on the same Wi-Fi
or hotspot, open port 8000 in the firewall, and point the UI at `http://<laptop1-ip>:8000`. Details in `api/README.md`.
