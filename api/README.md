# `api/`

**Owner (only this person pushes here):** **Sanchita**

FastAPI backend that runs on **Laptop 1**. Contract: `docs/team/00_TEAM_PLAN.md` section 4.3.

| File | What it is |
|---|---|
| `main.py` | `/health`, `/trips`, `/predict` + one log line per request |
| `config.py` | paths and the `PATHSNAP_EVAL_DIR` setting |

## What `/predict` does
Every request runs the full pipeline on the trip's `.track`, `.nodes` and `.arcs` files
(`pathsnap.pipeline.predict_trip`: candidates, 21 features, tuned LightGBM, HMM/Viterbi). Nothing is cached or precomputed per trip.
The reference route is read **server side only** (`evaluate_trip`) to compute the metrics. The response contains no reference geometry.
Only exact names of trip folders found in `PATHSNAP_EVAL_DIR` are accepted; anything else returns **404**.
A pipeline failure returns **500** with a message and the server keeps running.

## Run (Laptop 1)
```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# rehearsal data is the default (data/raw/student_test). On evaluation day point it at the evaluator trips:
export PATHSNAP_EVAL_DIR=/path/to/evaluator_trips      # Windows CMD: set PATHSNAP_EVAL_DIR=D:\evaluator_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Startup loads `artifacts/final_model.joblib`, `artifacts/model_meta.json`, `configs/hmm.yaml` and `configs/candidates.yaml` once.
The console prints one line per request, for example:
```
2026-11-18 12:42:19 | INFO | client=192.168.1.7 trip=00000007 n_points=1354 timing_ms={'features': 42, 'model': 52, 'viterbi': 79, 'total': 215} hmm_top1=0.9941 clf_top1=0.9705 unmatched=0 breaks=0
```

## Connect Laptop 2
1. Both laptops on the **same Wi-Fi or phone hotspot**.
2. Find Laptop 1's IP: Windows `ipconfig` (IPv4 address), macOS `ipconfig getifaddr en0`.
3. Allow port 8000 through the firewall. Windows (admin CMD):
   `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
4. In the UI set the API URL to `http://<laptop1-ip>:8000`. Quick check from Laptop 2: `curl http://<laptop1-ip>:8000/health`.

## Try it
```bash
curl localhost:8000/health
curl localhost:8000/trips
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"trip_id": "00000007"}'
```
Interactive docs: `http://localhost:8000/docs`.

## Known notes
- `artifacts/final_model.joblib` was pickled with scikit-learn 1.7.2. Newer versions print an `InconsistentVersionWarning`
  while loading; the model still works. Pin `scikit-learn==1.7.2` in `requirements.txt` to silence it (Monty owns that file).
- Measured on the 10 student_test trips (one laptop, localhost): 86 ms to 1,118 ms round trip; server `timing_ms.total` 77 ms to 1,080 ms, which includes
  the server-side metric computation. The largest training trip (19,313 points) takes about 2.3 s for the prediction plus 0.7 s for the metrics.

## Tests and rehearsal
```bash
pip install pytest httpx
python -m pytest api/test_api.py -q                                   # 12 contract tests
python -m api.rehearse --url http://<laptop1-ip>:8000                 # run every trip through a live API, writes results/api_rehearsal.csv
```
Run the rehearsal from Laptop 2 with the real URL to record the timings including the network (needed for report section 7).
