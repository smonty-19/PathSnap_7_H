"""Road-network helpers (arcs are straight directed segments u -> v)."""
import numpy as np


class RoadNetwork:
    def __init__(self, nodes_xy, arcs):
        self.nodes_xy = nodes_xy
        self.arcs = arcs
        self.u, self.v = arcs[:, 0], arcs[:, 1]
        self.a_xy = nodes_xy[self.u]
        self.b_xy = nodes_xy[self.v]
        d = self.b_xy - self.a_xy
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 1], d[:, 0])          # radians, math convention
        n = len(nodes_xy)
        self.out_deg_node = np.bincount(self.u, minlength=n)
        self.in_deg_node = np.bincount(self.v, minlength=n)
        # reverse twin: arc v->u exists
        key = self.u.astype(np.int64) * (n + 1) + self.v
        rkey = self.v.astype(np.int64) * (n + 1) + self.u
        self.has_reverse = np.isin(rkey, key)

    def connected(self, a, b):
        """Vectorised: True if arc b equals arc a or directly follows it (a.v == b.u)."""
        a = np.asarray(a); b = np.asarray(b)
        return (a == b) | (self.v[a] == self.u[b])
