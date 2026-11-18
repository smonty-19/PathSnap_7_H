"""Feature engineering for (GPS observation, candidate arc) pairs.

All features are computed from the GPS track and road network only. The
reference route is never used here (labels are attached separately).
"""
import numpy as np
import pandas as pd
from .geo import LocalProjection, point_segment
from .graph import RoadNetwork
from .candidates import generate_candidates

FEATURE_COLUMNS = [
    # geometric
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cands",
    "proj_frac", "at_endpoint", "arc_length_m",
    # direction
    "heading_diff", "heading_valid", "heading_cos_x_speed",
    # kinematics
    "speed_mps", "dt_prev", "gap_flag",
    # temporal context (same arc, neighbouring observations)
    "mean_dist_window", "max_dist_window",
    # topology
    "prev_best_connected", "next_best_connected", "arc_out_degree", "arc_in_degree", "has_reverse",
]

WIN = 2          # +-2 observations for heading and window features
GAP_S = 5.0


def prepare_trip(trip, radius=50.0, k_max=10):
    """Projection, network and per-observation kinematics for one trip."""
    proj = LocalProjection.from_lonlat(trip["track"][:, :2])
    pts = proj.to_xy(trip["track"][:, :2])
    net = RoadNetwork(proj.to_xy(trip["nodes"]), trip["arcs"])
    t = trip["track"][:, 2]
    N = len(pts)
    # heading over a +-WIN window (robust to 1 Hz jitter)
    lo = np.clip(np.arange(N) - WIN, 0, N - 1)
    hi = np.clip(np.arange(N) + WIN, 0, N - 1)
    vec = pts[hi] - pts[lo]
    move = np.hypot(vec[:, 0], vec[:, 1])
    heading = np.arctan2(vec[:, 1], vec[:, 0])
    dt_win = np.maximum(t[hi] - t[lo], 1e-6)
    speed = move / dt_win
    dt_prev = np.r_[0.0, np.diff(t)]
    cands = generate_candidates(net, pts, radius, k_max)
    return dict(proj=proj, pts=pts, net=net, t=t, heading=heading, move=move,
                speed=speed, dt_prev=dt_prev, cands=cands)


def build_features(prep):
    c, net, pts = prep["cands"], prep["net"], prep["pts"]
    obs, arc, d = c["obs_idx"], c["cand_arc_id"], c["dist_m"]
    N = len(pts)
    df = pd.DataFrame({"obs_idx": obs, "cand_arc_id": arc, "dist_m": d, "proj_frac": c["proj_frac"]})
    g = df.groupby("obs_idx")["dist_m"]
    df["dist_rank"] = g.cumcount()              # rows are already sorted by (obs, dist)
    dmin = g.transform("min")
    df["dist_minus_min"] = df.dist_m - dmin
    df["dist_ratio_min"] = df.dist_m / (dmin + 1.0)
    df["n_cands"] = g.transform("size")
    df["at_endpoint"] = ((df.proj_frac <= 1e-6) | (df.proj_frac >= 1 - 1e-6)).astype(np.int8)
    df["arc_length_m"] = net.length[arc]

    # direction: angle between GPS travel direction and arc direction u->v (0..180 deg)
    valid = prep["move"][obs] > 3.0
    diff = np.abs((prep["heading"][obs] - net.bearing[arc] + np.pi) % (2 * np.pi) - np.pi)
    df["heading_diff"] = np.where(valid, np.degrees(diff), 90.0)
    df["heading_valid"] = valid.astype(np.int8)
    df["heading_cos_x_speed"] = np.where(valid, np.cos(diff), 0.0) * np.minimum(prep["speed"][obs], 40)

    df["speed_mps"] = prep["speed"][obs]
    df["dt_prev"] = prep["dt_prev"][obs]
    df["gap_flag"] = (prep["dt_prev"][obs] > GAP_S).astype(np.int8)

    # same candidate arc evaluated at neighbouring observations (temporal smoothing)
    win = [d]
    for off in (-WIN, -1, 1, WIN):
        j = np.clip(obs + off, 0, N - 1)
        dj, _ = point_segment(pts[j], net.a_xy[arc], net.b_xy[arc])
        win.append(dj)
    win = np.vstack(win)
    df["mean_dist_window"] = win.mean(0)
    df["max_dist_window"] = win.max(0)

    # topology vs. nearest arc of previous / next observation
    best = np.full(N, -1)
    first = df.dist_rank.values == 0
    best[obs[first]] = arc[first]
    prev_best = best[np.clip(obs - 1, 0, N - 1)]
    next_best = best[np.clip(obs + 1, 0, N - 1)]
    pb_ok = prev_best >= 0
    nb_ok = next_best >= 0
    pbc = np.zeros(len(df), np.int8)
    nbc = np.zeros(len(df), np.int8)
    pbc[pb_ok] = net.connected(prev_best[pb_ok], arc[pb_ok])
    nbc[nb_ok] = net.connected(arc[nb_ok], next_best[nb_ok])
    df["prev_best_connected"] = pbc
    df["next_best_connected"] = nbc
    df["arc_out_degree"] = net.out_deg_node[net.v[arc]]
    df["arc_in_degree"] = net.in_deg_node[net.u[arc]]
    df["has_reverse"] = net.has_reverse[arc].astype(np.int8)
    df["t"] = prep["t"][obs]
    return df
