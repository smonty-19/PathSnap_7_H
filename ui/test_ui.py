"""Headless UI test against a running API (default http://localhost:8000).

    uvicorn api.main:app --port 8000
    python -m pytest -q ui/test_ui.py
"""
import os

import pytest
import requests
from streamlit.testing.v1 import AppTest

API = os.environ.get("PATHSNAP_API", "http://localhost:8000")
APP = os.path.join(os.path.dirname(__file__), "app.py")


def _api_up():
    try:
        return requests.get(f"{API}/health", timeout=3).ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason=f"API not running at {API}")


def test_connects_and_lists_trips():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Connected" in s.value for s in at.success)
    trips = requests.get(f"{API}/trips").json()["trips"]
    assert list(at.selectbox[0].options) == trips


def test_run_trip_shows_metrics():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "HMM Top-1 accuracy" in labels and "Classifier-only Top-1" in labels
    hmm = next(m for m in at.metric if m.label == "HMM Top-1 accuracy")
    assert hmm.value.endswith("%")


def test_unreachable_api_message():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = "http://127.0.0.1:9"   # nothing listens here
    at.run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
