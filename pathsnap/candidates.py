"""Candidate generation: for each GPS point, the K nearest arcs within radius R.

Arcs are densified into sample points (every STEP m) and indexed with a KD-tree;
exact point-to-segment distances are then computed for the arcs that are hit.
Points without any candidate within R are retried with 2R.
"""
import numpy as np
from scipy.spatial import cKDTree
from .geo import point_segment

STEP = 15.0


def _densify(net):
    n = np.maximum(np.ceil(net.length / STEP).astype(int), 1) + 1
    arc_of = np.repeat(np.arange(len(n)), n)
    # position index within its arc
    start = np.cumsum(n) - n
    k = np.arange(n.sum()) - np.repeat(start, n)
    t = k / np.repeat(n - 1, n)
    pts = net.a_xy[arc_of] + (net.b_xy[arc_of] - net.a_xy[arc_of]) * t[:, None]
    return pts, arc_of


def _query(tree, arc_of, net, pts, idx, radius, k_max):
    hits = tree.query_ball_point(pts[idx], r=radius + STEP / 2)
    lens = np.fromiter((len(h) for h in hits), int, len(hits))
    if lens.sum() == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0), np.empty(0)
    obs = np.repeat(idx, lens)
    arc = arc_of[np.concatenate([np.asarray(h, int) for h in hits if len(h)])]
    key = np.unique(obs.astype(np.int64) * len(net.length) + arc)
    obs, arc = key // len(net.length), key % len(net.length)
    d, t = point_segment(pts[obs], net.a_xy[arc], net.b_xy[arc])
    m = d <= radius
    obs, arc, d, t = obs[m], arc[m], d[m], t[m]
    order = np.lexsort((d, obs))
    obs, arc, d, t = obs[order], arc[order], d[order], t[order]
    first = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    rank = np.arange(len(obs)) - np.repeat(first, np.diff(np.r_[first, len(obs)]))
    m = rank < k_max
    return obs[m], arc[m], d[m], t[m]


def generate_candidates(net, pts_xy, radius=50.0, k_max=10):
    """Returns dict of arrays: obs_idx, cand_arc_id, dist_m, proj_frac (sorted by obs, dist)."""
    samples, arc_of = _densify(net)
    tree = cKDTree(samples)
    all_idx = np.arange(len(pts_xy))
    obs, arc, d, t = _query(tree, arc_of, net, pts_xy, all_idx, radius, k_max)
    missing = np.setdiff1d(all_idx, obs)
    if len(missing):
        o2, a2, d2, t2 = _query(tree, arc_of, net, pts_xy, missing, 2 * radius, k_max)
        obs, arc, d, t = (np.concatenate(x) for x in ((obs, o2), (arc, a2), (d, d2), (t, t2)))
        order = np.lexsort((d, obs))
        obs, arc, d, t = obs[order], arc[order], d[order], t[order]
    return {"obs_idx": obs, "cand_arc_id": arc, "dist_m": d, "proj_frac": t}
