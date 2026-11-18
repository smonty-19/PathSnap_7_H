"""PathSnap FastAPI backend (Laptop 1).   Owner: Sanchita.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Contract (docs/team/00_TEAM_PLAN.md section 4.3):
    GET  /health   -> {"status", "model", "version"}
    GET  /trips    -> {"trips": [...]}
    POST /predict  {"trip_id"} -> gps, predicted_route, matched_points, metrics, timing_ms

Every /predict runs the full pipeline (candidates -> features -> classifier -> HMM) on the trip files.
The reference route is only read server-side to compute metrics; it is never part of a response.
"""
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from pathsnap.pipeline import evaluate_trip, load_cand_cfg, load_hmm_cfg, predict_trip

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    # load the model, its metadata and the decoder config once, not per request
    STATE["model"] = joblib.load(config.MODEL_PATH)
    with open(config.META_PATH) as f:
        STATE["meta"] = json.load(f)
    STATE["hmm_cfg"], STATE["cand_cfg"] = load_hmm_cfg(), load_cand_cfg()
    log.info("loaded model=%s from %s | eval dir=%s | %d trips",
             STATE["meta"]["model_name"], config.MODEL_PATH, config.EVAL_DIR, len(list_trips()))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)


class PredictRequest(BaseModel):
    trip_id: str


def list_trips():
    """Trip folders that contain a GPS track. Re-read on every call so trips can be dropped in live."""
    if not os.path.isdir(config.EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(config.EVAL_DIR)
                  if os.path.isfile(os.path.join(config.EVAL_DIR, d, f"{d}.track")))


def _clean(x):
    """JSON cannot carry NaN."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["meta"]["model_name"], "version": STATE["meta"].get("date", "1.0")}


@app.get("/trips")
def trips():
    return {"trips": list_trips()}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    # only exact names of trip folders found on disk are accepted (no path traversal)
    if req.trip_id not in list_trips():
        log.warning("client=%s trip=%s -> 404 unknown trip", client, req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{req.trip_id}'")
    trip_dir = os.path.join(config.EVAL_DIR, req.trip_id)
    t0 = time.perf_counter()
    try:
        pred = predict_trip(trip_dir, STATE["model"], STATE["meta"], STATE["hmm_cfg"], STATE["cand_cfg"])
        metrics = evaluate_trip(trip_dir, pred)
    except Exception as e:  # never crash the server
        log.exception("client=%s trip=%s -> 500", client, req.trip_id)
        raise HTTPException(status_code=500, detail=f"pipeline failed: {type(e).__name__}: {e}")
    pred.pop("_internal")
    pred["metrics"] = _clean(metrics) if metrics else None
    pred["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1e3)
    n_unmatched, breaks = pred.pop("n_unmatched"), pred.pop("breaks")
    m = metrics or {}
    log.info("client=%s trip=%s n_points=%d timing_ms=%s hmm_top1=%s clf_top1=%s unmatched=%d breaks=%d",
             client, req.trip_id, pred["n_points"], pred["timing_ms"],
             f"{m['hmm']['top1_accuracy']:.4f}" if m else "n/a",
             f"{m['classifier']['top1_accuracy']:.4f}" if m else "n/a", n_unmatched, breaks)
    return pred
