"""Raw trip loading.

Each trip folder <trip_id>/ contains:
  <trip_id>.track  lon lat t          (GPS observations, ~1 Hz)
  <trip_id>.nodes  lon lat            (row index = node_id)
  <trip_id>.arcs   from_node to_node  (row index = arc_id, directed)
  <trip_id>.route  arc_id             (reference route, ordered)  -> ground truth

The route is only loaded when explicitly requested so that the prediction
path never touches ground truth.
"""
import os
import numpy as np


def _load(path, dtype=float, ndmin=2):
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def list_trips(split_dir):
    return sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))


def load_trip(trip_dir, with_route=False):
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    p = lambda ext: os.path.join(trip_dir, f"{trip_id}.{ext}")
    track = _load(p("track"))
    # basic cleaning: drop NaN rows, sort by time, drop duplicated timestamps
    track = track[~np.isnan(track).any(axis=1)]
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.ones(len(track), bool)
    keep[1:] = np.diff(track[:, 2]) > 0
    track = track[keep]
    out = {
        "trip_id": trip_id,
        "track": track,                       # (N,3) lon, lat, t
        "nodes": _load(p("nodes")),            # (M,2) lon, lat
        "arcs": _load(p("arcs"), dtype=np.int64),  # (A,2) u, v
    }
    if with_route:
        out["route"] = load_route(trip_dir)
    return out


def load_route(trip_dir):
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    path = os.path.join(trip_dir, f"{trip_id}.route")
    if not os.path.exists(path):
        return None
    return _load(path, dtype=np.int64, ndmin=1)
