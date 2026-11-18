"""Per-observation labels from the per-arc reference route.

The .route file is an ordered list of arcs, not one arc per GPS point, so we
align GPS points to route positions with a monotone dynamic programme:

    D[i, j] = d(i, j) + min( D[i-1, j],                                   # stay on route arc j
                             min_{k<j} D[i-1, k] + LAM * (j - k - 1) )   # advance (skipping costs LAM per arc)

where d(i, j) is the point-to-segment distance (capped at CAP m). The route index
never decreases in time, which resolves parallel-road / reverse-twin ambiguity
that a plain nearest-arc assignment would get wrong.
"""
import numpy as np
from .geo import point_segment

LAM = 5.0     # metres-equivalent penalty per skipped route arc
CAP = 200.0   # cap on per-point distance so outliers don't dominate


def align_route(net, pts_xy, route):
    """Returns (route_pos per obs, true_arc_id per obs, align_dist per obs)."""
    N, R = len(pts_xy), len(route)
    a, b = net.a_xy[route], net.b_xy[route]
    jr = np.arange(R, dtype=np.float32)
    D = np.empty((N, R), np.float32)

    def dist_row(i):
        d, _ = point_segment(np.repeat(pts_xy[i:i + 1], R, 0), a, b)
        return np.minimum(d, CAP).astype(np.float32)

    D[0] = dist_row(0) + LAM * jr
    for i in range(1, N):
        prev = D[i - 1]
        pm = np.minimum.accumulate(prev - LAM * jr)
        adv = np.full(R, np.inf, np.float32)
        adv[1:] = LAM * (jr[1:] - 1) + pm[:-1]
        D[i] = dist_row(i) + np.minimum(prev, adv)

    # backtrack
    pos = np.empty(N, int)
    j = int(np.argmin(D[-1] + LAM * (R - 1 - jr)))
    pos[-1] = j
    for i in range(N - 1, 0, -1):
        prev = D[i - 1]
        if j > 0:
            k = int(np.argmin(prev[:j] - LAM * jr[:j]))
            adv_cost = LAM * (j - 1) + prev[k] - LAM * k
            if adv_cost < prev[j]:
                j = k
        pos[i - 1] = j
    true_arc = route[pos]
    d, _ = point_segment(pts_xy, net.a_xy[true_arc], net.b_xy[true_arc])
    return pos, true_arc, d
