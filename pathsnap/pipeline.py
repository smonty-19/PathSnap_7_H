"""Single-trip inference pipeline used by the API.

    predict_trip(trip_dir, model, model_meta, hmm_cfg)  -> dict   (uses ONLY .track / .nodes / .arcs)
    evaluate_trip(trip_dir, prediction)                 -> dict   (loads .route separately, server side)

`predict_trip` never opens the reference route. `evaluate_trip` is only called after prediction and only
returns metrics, never route geometry.
"""
import os
import time

import numpy as np
import yaml

from .features import build_features, prepare_trip
from .hmm import DEFAULT_CFG, decode
from .io import load_route, load_trip
from .labels import align_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAND = dict(radius_m=50.0, k_max=10)


def load_yaml(name, default):
    path = os.path.join(ROOT, "configs", name)
    if os.path.exists(path):
        with open(path) as f:
            return {**default, **(yaml.safe_load(f) or {})}
    return dict(default)


def load_hmm_cfg():
    return load_yaml("hmm.yaml", DEFAULT_CFG)


def load_cand_cfg():
    return load_yaml("candidates.yaml", DEFAULT_CAND)


def _latlon(proj, xy):
    ll = proj.to_lonlat(np.asarray(xy, float).reshape(-1, 2))
    return ll[:, ::-1]          # (lat, lon) order for folium


def predict_trip(trip_dir, model, model_meta, hmm_cfg=None, cand_cfg=None):
    """Candidates -> features -> classifier -> HMM/Viterbi. Returns the prediction dict (see module docstring)."""
    t_all = time.perf_counter()
    cand_cfg = {**DEFAULT_CAND, **(cand_cfg or load_cand_cfg())}
    hmm_cfg = {**DEFAULT_CFG, **(hmm_cfg or {})}
    trip = load_trip(trip_dir, with_route=False)

    t0 = time.perf_counter()
    prep = prepare_trip(trip, cand_cfg["radius_m"], cand_cfg["k_max"])
    df = build_features(prep)
    t_feat = time.perf_counter() - t0

    t0 = time.perf_counter()
    X = df[model_meta["feature_columns"]].to_numpy()
    df["prob"] = model.predict_proba(X)[:, 1]
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec, breaks = decode(prep["net"], prep["pts"], prep["t"], df[["obs_idx", "cand_arc_id", "proj_frac", "dist_m", "prob"]], hmm_cfg)
    t_vit = time.perf_counter() - t0

    net, proj, pts = prep["net"], prep["proj"], prep["pts"]
    N = len(pts)
    hmm_arc = np.full(N, -1); clf_arc = np.full(N, -1)
    hmm_arc[dec.obs_idx.values] = dec.hmm_arc.values
    clf_arc[dec.obs_idx.values] = dec.clf_arc.values

    # matched point = GPS point projected onto its predicted arc (raw GPS where no candidate exists)
    matched_xy = pts.copy()
    m = hmm_arc >= 0
    frac = np.zeros(N); frac[dec.obs_idx.values] = dec.hmm_frac.values
    matched_xy[m] = net.a_xy[hmm_arc[m]] + (net.b_xy[hmm_arc[m]] - net.a_xy[hmm_arc[m]]) * frac[m][:, None]

    # route polyline per arc in travel order (consecutive repeats collapsed)
    seq = hmm_arc[m]
    keep = np.r_[True, seq[1:] != seq[:-1]] if len(seq) else np.zeros(0, bool)
    route_arcs = seq[keep]
    route = [_latlon(proj, [net.a_xy[a], net.b_xy[a]]).tolist() for a in route_arcs]

    return {
        "trip_id": trip["trip_id"],
        "n_points": int(N),
        "gps": _latlon(proj, pts).tolist(),
        "predicted_route": route,
        "matched_points": _latlon(proj, matched_xy).tolist(),
        "n_unmatched": int((~m).sum()),
        "breaks": int(breaks),
        "timing_ms": {"features": round(t_feat * 1e3), "model": round(t_model * 1e3),
                      "viterbi": round(t_vit * 1e3), "total": round((time.perf_counter() - t_all) * 1e3)},
        "_internal": {"hmm_arc": hmm_arc, "clf_arc": clf_arc, "net": net, "pts": pts,
                      "cands": df[["obs_idx", "cand_arc_id"]]},
    }


def _continuity(net, arcs):
    a = arcs[arcs >= 0]
    if len(a) < 2:
        return float("nan")
    return float(np.mean(net.connected(a[:-1], a[1:])))


def evaluate_trip(trip_dir, prediction):
    """Server-side metrics against the reference route. Returns {} if the trip has no .route."""
    route = load_route(trip_dir)
    if route is None:
        return {}
    it = prediction["_internal"]
    net, pts = it["net"], it["pts"]
    _, true_arc, _ = align_route(net, pts, route)
    c = it["cands"]
    hit = np.zeros(len(pts), bool)
    hit[c.obs_idx.values[c.cand_arc_id.values == true_arc[c.obs_idx.values]]] = True
    return {
        "hmm": {"top1_accuracy": float(np.mean(it["hmm_arc"] == true_arc)),
                "route_continuity": _continuity(net, it["hmm_arc"])},
        "classifier": {"top1_accuracy": float(np.mean(it["clf_arc"] == true_arc)),
                       "route_continuity": _continuity(net, it["clf_arc"])},
        "candidate_miss_rate": float(1.0 - hit.mean()),
    }
