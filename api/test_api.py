"""API contract tests.   Run from the repo root:  python -m pytest api/test_api.py -q

Needs data/raw/student_test (rehearsal trips) and artifacts/final_model.joblib.
"""
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app

TRIP = "00000052"   # smallest student_test trip (262 points) keeps the test fast


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs the lifespan: model is loaded once
        yield c


def test_health(client):
    d = client.get("/health").json()
    assert d["status"] == "ok" and d["model"] == "lightgbm_tuned" and d["version"]


def test_trips_lists_folders(client):
    trips = client.get("/trips").json()["trips"]
    assert TRIP in trips and trips == sorted(trips)


@pytest.mark.parametrize("bad", ["does_not_exist", "../../etc", "", "00000052/../00000007"])
def test_unknown_trip_is_404(client, bad):
    assert client.post("/predict", json={"trip_id": bad}).status_code == 404


def test_missing_body_is_422(client):
    assert client.post("/predict", json={}).status_code == 422


@pytest.fixture(scope="module")
def pred(client):
    r = client.post("/predict", json={"trip_id": TRIP})
    assert r.status_code == 200
    return r.json()


def test_contract_fields(pred):
    assert set(pred) == {"trip_id", "n_points", "gps", "predicted_route", "matched_points", "metrics", "timing_ms"}
    n = pred["n_points"]
    assert len(pred["gps"]) == n == len(pred["matched_points"])
    assert all(len(p) == 2 for p in pred["gps"])
    assert pred["predicted_route"] and all(len(seg) == 2 and len(seg[0]) == 2 for seg in pred["predicted_route"])
    assert set(pred["timing_ms"]) == {"features", "model", "viterbi", "total"}


def test_gps_is_lat_lon_order(pred):
    lat, lon = pred["gps"][0]
    assert -90 <= lat <= 90 and -180 <= lon <= 180


def test_metrics_shape_and_range(pred):
    m = pred["metrics"]
    assert set(m) == {"hmm", "classifier", "candidate_miss_rate"}
    for k in ("hmm", "classifier"):
        assert set(m[k]) == {"top1_accuracy", "route_continuity"}
        assert all(v is None or 0 <= v <= 1 for v in m[k].values())
    assert 0 <= m["candidate_miss_rate"] <= 1
    assert m["hmm"]["top1_accuracy"] >= m["classifier"]["top1_accuracy"] - 0.05   # HMM must not be much worse


def test_no_ground_truth_leaves_the_server(pred):
    text = str(pred).lower()
    for forbidden in ("reference", "true_arc", "ground", "label", "_internal"):
        assert forbidden not in text
    # the only route-like geometry is the prediction itself
    assert set(pred) & {"route", "reference_route", "true_route"} == set()


def test_response_is_valid_json_no_nan(pred):
    def walk(x):
        if isinstance(x, float):
            assert not math.isnan(x)
        elif isinstance(x, dict):
            [walk(v) for v in x.values()]
        elif isinstance(x, list):
            [walk(v) for v in x[:50]]
    walk(pred)
