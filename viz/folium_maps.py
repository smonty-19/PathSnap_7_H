"""Folium maps of map-matching results (report section 6).   Owner: Patil.

Layers (each toggleable with the layer control):
    Road network (grey, clipped to the trip area) · Reference route (green) · Classifier-only prediction (orange)
    · HMM/Viterbi prediction (red) · Raw GPS (blue) · GPS points where the HMM arc is wrong (black, off by default)

Inputs: raw trip files + results/preds/hmm_<trip>.parquet (written by `python -m evaluation.hmm_eval`).
The reference route is used here only for offline visualisation of student_test results, never by the predictor.

    python -m viz.folium_maps                      # every student_test trip that has HMM predictions
    python -m viz.folium_maps --trips <id> <id>    # selected trips
"""
import argparse
import os

import folium
import numpy as np
import pandas as pd

from pathsnap.io import load_route, load_trip

RAW = os.path.join("data", "raw", "student_test")
PRED = os.path.join("results", "preds")
OUT = os.path.join("results", "maps")

COLORS = dict(network="#9e9e9e", reference="#2e7d32", classifier="#ff9800", hmm="#d32f2f", gps="#1565c0", wrong="#000000")


def _arc_line(nodes, arcs, a):
    u, v = arcs[a]
    return [[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]]      # (lat, lon) for folium


def _ordered_unique(seq):
    seq = [int(s) for s in seq if s >= 0]
    return list(dict.fromkeys(seq))


def _legend(title, stats):
    rows = "".join(
        f'<div><span style="display:inline-block;width:22px;height:4px;background:{c};vertical-align:middle;margin-right:6px"></span>{t}</div>'
        for t, c in [("Reference route", COLORS["reference"]), ("Classifier-only", COLORS["classifier"]),
                     ("HMM / Viterbi", COLORS["hmm"]), ("Road network", COLORS["network"])])
    rows += f'<div><span style="color:{COLORS["gps"]};margin-right:6px">&#9679;</span>Raw GPS</div>'
    rows += f'<div><span style="color:{COLORS["wrong"]};margin-right:6px">&#9679;</span>HMM wrong (layer off by default)</div>'
    return f"""<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;
        border:1px solid #888;border-radius:6px;font:13px Arial;box-shadow:0 1px 4px rgba(0,0,0,.3)">
        <b>{title}</b><br>{stats}<hr style="margin:6px 0">{rows}</div>"""


def make_prediction_map(trip_id, raw_dir=RAW, pred_dir=PRED, out_dir=OUT, max_network_arcs=4000):
    trip_dir = os.path.join(raw_dir, trip_id)
    trip = load_trip(trip_dir)
    route = load_route(trip_dir)
    nodes, arcs, track = trip["nodes"], trip["arcs"], trip["track"]
    pred = pd.read_parquet(os.path.join(pred_dir, f"hmm_{trip_id}.parquet")).sort_values("obs_idx")

    lat, lon = track[:, 1], track[:, 0]
    pad = 0.002
    lo, hi = (lat.min() - pad, lon.min() - pad), (lat.max() + pad, lon.max() + pad)

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([lo, hi])

    # road network clipped to the GPS bounding box (keeps the HTML small)
    inb = (nodes[:, 1] >= lo[0]) & (nodes[:, 1] <= hi[0]) & (nodes[:, 0] >= lo[1]) & (nodes[:, 0] <= hi[1])
    keep = np.flatnonzero(inb[arcs[:, 0]] & inb[arcs[:, 1]])[:max_network_arcs]
    g = folium.FeatureGroup("Road network (clipped)")
    for a in keep:
        folium.PolyLine(_arc_line(nodes, arcs, a), color=COLORS["network"], weight=1, opacity=0.6).add_to(g)
    g.add_to(m)

    for name, seq, color, weight, show in [
        ("Reference route", route, COLORS["reference"], 8, True),
        ("Classifier-only prediction", pred.clf_arc.values, COLORS["classifier"], 4, True),
        ("HMM / Viterbi prediction", pred.hmm_arc.values, COLORS["hmm"], 3, True),
    ]:
        g = folium.FeatureGroup(name, show=show)
        for a in _ordered_unique(seq):
            folium.PolyLine(_arc_line(nodes, arcs, a), color=color, weight=weight, opacity=0.85, tooltip=f"{name}: arc {a}").add_to(g)
        g.add_to(m)

    g = folium.FeatureGroup("Raw GPS")
    for la, lo_ in zip(lat[::2], lon[::2]):
        folium.CircleMarker([la, lo_], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8, weight=0).add_to(g)
    g.add_to(m)

    wrong = pred[(pred.hmm_arc != pred.true_arc)]
    g = folium.FeatureGroup("GPS points where HMM is wrong", show=False)
    for i in wrong.obs_idx.values:
        folium.CircleMarker([lat[i], lon[i]], radius=3, color=COLORS["wrong"], fill=True, fill_opacity=1, weight=0,
                            tooltip=f"t={track[i, 2]:.0f}s").add_to(g)
    g.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    clf_acc = float((pred.clf_arc == pred.true_arc).mean())
    hmm_acc = float((pred.hmm_arc == pred.true_arc).mean())
    stats = f"student_test trip {trip_id} · {len(track)} GPS points<br>Top-1: classifier {clf_acc:.3f} → HMM {hmm_acc:.3f}"
    m.get_root().html.add_child(folium.Element(_legend("PathSnap prediction map", stats)))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"prediction_{trip_id}.html")
    m.save(path)
    return m, path, dict(trip_id=trip_id, n_points=len(track), clf_top1=clf_acc, hmm_top1=hmm_acc, hmm_wrong=len(wrong))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+", default=None, help="trip ids (default: all with results/preds/hmm_<id>.parquet)")
    a = ap.parse_args()
    trips = a.trips or sorted(f[4:-8] for f in os.listdir(PRED) if f.startswith("hmm_") and f.endswith(".parquet"))
    for t in trips:
        _, path, s = make_prediction_map(t)
        print(f"{path}: classifier {s['clf_top1']:.3f} -> HMM {s['hmm_top1']:.3f} ({s['hmm_wrong']} wrong points)")


if __name__ == "__main__":
    main()
