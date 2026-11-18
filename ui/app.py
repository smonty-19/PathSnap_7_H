"""PathSnap evaluator UI (Laptop 2).   Owner: Sheetal.

    streamlit run ui/app.py --server.port 8501
    (set the API URL in the sidebar, or PATHSNAP_API=http://<laptop1-ip>:8000 before starting)

The UI talks ONLY to the FastAPI backend (contract: docs/team/00_TEAM_PLAN.md section 4.3):
    GET /health, GET /trips, POST /predict {"trip_id"}
It never reads data files and never receives the reference route: accuracy is computed on the server.
"""
import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")
COL = dict(gps="#1565c0", route="#d32f2f", matched="#ff9800")

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.title("🛰️ PathSnap: GPS map-matching")
st.caption("Pick an evaluation trip. The API on Laptop 1 runs the full pipeline (candidates → features → "
           "tuned LightGBM → HMM/Viterbi) and returns the matched route with server-side accuracy.")

# ------------------------------------------------------------------ sidebar: connection
with st.sidebar:
    st.header("API connection")
    api = st.text_input("API URL (Laptop 1)", st.session_state.get("api", DEFAULT_API)).strip().rstrip("/")
    st.session_state["api"] = api
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.success(f"Connected · model: {h.get('model')}")
        st.caption(f"version: {h.get('version')}")
        connected = True
    except Exception as e:
        st.error(f"API not reachable at {api}")
        st.caption(str(e)[:200])
        connected = False
    st.divider()
    show_gps = st.checkbox("Show raw GPS", True)
    show_route = st.checkbox("Show predicted route", True)
    show_matched = st.checkbox("Show matched points", False)

if not connected:
    st.info("Start the API on Laptop 1 (`uvicorn api.main:app --host 0.0.0.0 --port 8000`), make sure both laptops "
            "are on the same network, then enter `http://<laptop1-ip>:8000` in the sidebar.")
    st.stop()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_trips(api_url):
    r = requests.get(f"{api_url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


try:
    trips = fetch_trips(api)
except Exception as e:
    st.error(f"Could not load the trip list: {e}")
    st.stop()
if not trips:
    st.warning("The API reports no evaluation trips (check PATHSNAP_EVAL_DIR on Laptop 1).")
    st.stop()

# ------------------------------------------------------------------ trip selection
c1, c2 = st.columns([4, 1])
trip = c1.selectbox(f"Evaluation trip ({len(trips)} available)", trips)
run = c2.button("▶ Run map-matching", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Matching trip {trip} on the API…"):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip}, timeout=180)
            r.raise_for_status()
            res = r.json()
            res["_round_trip_ms"] = round((time.perf_counter() - t0) * 1000)
            st.session_state["res"] = res
            hist = st.session_state.setdefault("history", [])
            m = res.get("metrics") or {}
            hist.append({
                "trip": res["trip_id"], "points": res["n_points"],
                "HMM Top-1": (m.get("hmm") or {}).get("top1_accuracy"),
                "classifier Top-1": (m.get("classifier") or {}).get("top1_accuracy"),
                "HMM continuity": (m.get("hmm") or {}).get("route_continuity"),
                "round trip (ms)": res["_round_trip_ms"],
            })
        except requests.HTTPError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
        except requests.Timeout:
            st.error("The API did not answer within 180 s.")
        except Exception as e:
            st.error(f"Request failed: {e}")

res = st.session_state.get("res")
if not res:
    st.stop()


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


# ------------------------------------------------------------------ results
m = res.get("metrics") or {}
hmm, clf = m.get("hmm") or {}, m.get("classifier") or {}
st.subheader(f"Trip {res['trip_id']} · {res['n_points']:,} GPS points")
k1, k2, k3, k4, k5 = st.columns(5)
gain = None if hmm.get("top1_accuracy") is None or clf.get("top1_accuracy") is None else hmm["top1_accuracy"] - clf["top1_accuracy"]
k1.metric("HMM Top-1 accuracy", pct(hmm.get("top1_accuracy")), None if gain is None else f"{100 * gain:+.1f} pts vs classifier")
k2.metric("Classifier-only Top-1", pct(clf.get("top1_accuracy")))
k3.metric("HMM route continuity", pct(hmm.get("route_continuity")))
k4.metric("Candidate miss rate", pct(m.get("candidate_miss_rate")))
k5.metric("Response time", f"{res['_round_trip_ms']} ms")
if not m:
    st.info("This trip has no reference route on the server, so no accuracy can be computed.")

left, right = st.columns([3, 1])
with left:
    gps = res["gps"]
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if show_route and res.get("predicted_route"):
        g = folium.FeatureGroup("Predicted route (HMM)")
        for line in res["predicted_route"]:
            folium.PolyLine(line, color=COL["route"], weight=5, opacity=0.85).add_to(g)
        g.add_to(fmap)
    if show_gps:
        g = folium.FeatureGroup("Raw GPS")
        step = max(1, len(gps) // 2000)
        for lat, lon in gps[::step]:
            folium.CircleMarker([lat, lon], radius=2, color=COL["gps"], fill=True, fill_opacity=0.8, weight=0).add_to(g)
        g.add_to(fmap)
    if show_matched and res.get("matched_points"):
        g = folium.FeatureGroup("Matched points")
        step = max(1, len(res["matched_points"]) // 2000)
        for lat, lon in res["matched_points"][::step]:
            folium.CircleMarker([lat, lon], radius=2, color=COL["matched"], fill=True, fill_opacity=0.9, weight=0).add_to(g)
        g.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[], key=f"map_{res['trip_id']}")
    st.caption("🔵 raw GPS · 🔴 predicted route (HMM/Viterbi) · 🟠 GPS points snapped onto the predicted road")

with right:
    st.markdown("**Server timing (ms)**")
    t = res.get("timing_ms", {})
    st.dataframe(pd.DataFrame({"stage": list(t.keys()), "ms": list(t.values())}), hide_index=True, use_container_width=True)
    st.markdown("**Comparison**")
    st.dataframe(pd.DataFrame([
        {"method": "HMM / Viterbi", "Top-1": pct(hmm.get("top1_accuracy")), "continuity": pct(hmm.get("route_continuity"))},
        {"method": "Classifier only", "Top-1": pct(clf.get("top1_accuracy")), "continuity": pct(clf.get("route_continuity"))},
    ]), hide_index=True, use_container_width=True)
    st.caption(f"Predicted route: {len(res.get('predicted_route', []))} road segments")

hist = st.session_state.get("history", [])
if len(hist) > 1:
    st.subheader("Trips run in this session")
    st.dataframe(pd.DataFrame(hist), hide_index=True, use_container_width=True)
