# `ui/`

**Owner (only this person pushes here):** **Sheetal**

Streamlit evaluator UI that runs on **Laptop 2**. It talks to the API only, never reads data files, and never receives ground truth (accuracy is computed on the API server).

| File | What it is |
|---|---|
| `app.py` | Streamlit app: connection check, trip list from `GET /trips`, `POST /predict`, map (raw GPS + predicted route + matched points), server-side metrics (HMM vs classifier-only), timings, session history |
| `test_ui.py` | Headless test of the app against a running API (`streamlit.testing`) |

## Run (Laptop 2)
```bash
pip install -r requirements.txt
streamlit run ui/app.py --server.port 8501
```
Then enter the API URL of Laptop 1 in the sidebar, e.g. `http://192.168.1.7:8000`. Alternatively, set it before starting:
- Windows CMD: `set PATHSNAP_API=http://192.168.1.7:8000`
- macOS/Linux: `export PATHSNAP_API=http://192.168.1.7:8000`

Both laptops must be on the same Wi-Fi/hotspot, and port 8000 must be allowed through the firewall on Laptop 1 (see `api/README.md`). The sidebar shows a green "Connected" box when `/health` answers.

## Demo flow
1. The sidebar shows **Connected · model: lightgbm_tuned**.
2. Pick one of the evaluation trips from the dropdown (the list comes from the API) and click **Run map-matching**.
3. The map shows raw GPS (blue) and the predicted route (red). Matched points (orange) can be toggled.
4. The metric cards show HMM vs classifier-only Top-1, route continuity, candidate miss rate and response time.
5. Meanwhile, show the evaluator the request log lines in the CMD window on Laptop 1.

## Test
```bash
uvicorn api.main:app --port 8000          # terminal 1
python -m pytest -q ui/test_ui.py         # terminal 2
```
