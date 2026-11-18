# `api/`

**Owner (only this person pushes here):** **Sanchita**

FastAPI backend that runs on **Laptop 1**. Contract: `docs/team/00_TEAM_PLAN.md` §4.3.

| File | What it is |
|---|---|
| `main.py` | `/health`, `/trips`, `/predict` + request logging |
| `config.py` | paths, `PATHSNAP_EVAL_DIR` |

Run: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
