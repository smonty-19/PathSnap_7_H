"""HMM / Viterbi map-matching decoder on top of per-candidate classifier probabilities.

Hidden states   : the candidate arcs of each GPS observation (with the projected position on the arc).
Emission        : log e = w * log(p_clf) + (1 - w) * (-0.5 * (dist / sigma)^2)
Transition      : log t = -|d_network - d_gps| / beta      (Newson & Krumm style)
                  d_network = along-arc / connected / shortest-path distance between the projected positions.
                  No path within `max_m`  ->  floor probability `eps` (decoding never dies; counted as a "break").
Segmentation    : the trip is split into independent segments at time gaps (dt > gap_s) and at
                  observations without any candidate. Observations without candidates get arc -1.

Only the track, the network and the classifier probabilities are used. The reference route is never touched.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_CFG = dict(sigma_m=15.0, beta_m=10.0, w=0.7, eps=1e-6, gap_s=10.0, max_m=300.0, back_penalty_m=20.0)


class NetworkDistance:
    """Bounded Dijkstra between road nodes with a per-source cache (sparse rows)."""

    def __init__(self, net, max_m):
        n = len(net.nodes_xy)
        self.graph = csr_matrix((net.length, (net.u, net.v)), shape=(n, n))
        self.max_m = max_m
        self.cache = {}

    def _fill(self, sources):
        todo = [s for s in dict.fromkeys(sources) if s not in self.cache]
        if not todo:
            return
        rows = dijkstra(self.graph, directed=True, indices=np.asarray(todo), limit=self.max_m)
        for s, row in zip(todo, rows):
            idx = np.flatnonzero(np.isfinite(row))
            self.cache[s] = (idx, row[idx])

    def many(self, src_nodes, dst_nodes):
        """(len(src), len(dst)) matrix of shortest distances (inf if farther than max_m)."""
        self._fill(src_nodes.tolist())
        out = np.full((len(src_nodes), len(dst_nodes)), np.inf)
        for r, s in enumerate(src_nodes.tolist()):
            idx, dist = self.cache[s]
            pos = np.searchsorted(idx, dst_nodes)
            pos[pos >= len(idx)] = len(idx) - 1
            ok = idx[pos] == dst_nodes
            out[r, ok] = dist[pos[ok]]
        return out


def _transition_logp(net, nd, cfg, arc_i, frac_i, arc_j, frac_j, d_gps):
    """(Ki, Kj) log transition matrix and a boolean matrix marking pairs with no path."""
    L = net.length
    d_net = np.full((len(arc_i), len(arc_j)), np.inf)
    same = arc_i[:, None] == arc_j[None, :]
    along = (frac_j[None, :] - frac_i[:, None]) * L[arc_j][None, :]
    d_same = np.where(along >= 0, along, -along + cfg["back_penalty_m"])
    rem_i = (1.0 - frac_i) * L[arc_i]
    off_j = frac_j * L[arc_j]
    adj = (net.v[arc_i][:, None] == net.u[arc_j][None, :]) & ~same
    d_adj = rem_i[:, None] + off_j[None, :]
    d_net = np.where(same, d_same, np.where(adj, d_adj, d_net))
    need = ~same & ~adj
    if need.any():
        rows = np.flatnonzero(need.any(1))
        sp = nd.many(net.v[arc_i][rows], net.u[arc_j])
        d_sp = sp + rem_i[rows][:, None] + off_j[None, :]
        sub = d_net[rows]
        d_net[rows] = np.where(need[rows], d_sp, sub)
    nopath = ~np.isfinite(d_net)
    logp = np.where(nopath, np.log(cfg["eps"]), -np.abs(d_net - d_gps) / cfg["beta_m"])
    return logp, nopath


def decode(net, pts, t, cand, cfg=None):
    """Viterbi decode of one trip.

    net   : RoadNetwork,  pts : (N,2) projected GPS (metres),  t : (N,) timestamps
    cand  : DataFrame with columns obs_idx, cand_arc_id, proj_frac, dist_m, prob  (any row order)
    Returns DataFrame [obs_idx, clf_arc, hmm_arc, hmm_frac] (arc = -1 where there is no candidate)
    and the number of breaks (transitions that had to use the eps floor).
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    cand = cand.sort_values(["obs_idx", "cand_arc_id"], kind="stable").reset_index(drop=True)
    obs = cand["obs_idx"].to_numpy()
    arc = cand["cand_arc_id"].to_numpy()
    frac = cand["proj_frac"].to_numpy(float)
    prob = np.clip(cand["prob"].to_numpy(float), 1e-6, 1 - 1e-6)
    em = cfg["w"] * np.log(prob) - (1.0 - cfg["w"]) * 0.5 * (cand["dist_m"].to_numpy(float) / cfg["sigma_m"]) ** 2

    uobs, start = np.unique(obs, return_index=True)
    end = np.r_[start[1:], len(obs)]
    n_steps = len(uobs)

    # classifier-only argmax per observation
    clf_best = np.array([arc[s + np.argmax(prob[s:e])] for s, e in zip(start, end)])

    # segment boundaries: time gaps. Observations with no candidate are absent from uobs; if only one or two are
    # missing the transition simply spans them (d_gps is measured between the observations actually used).
    tt = t[uobs]
    new_seg = np.r_[True, np.diff(tt) > cfg["gap_s"]]
    seg_ids = np.cumsum(new_seg) - 1

    nd = NetworkDistance(net, cfg["max_m"])
    hmm_arc = np.full(n_steps, -1)
    hmm_frac = np.zeros(n_steps)
    breaks = 0
    for sg in range(seg_ids[-1] + 1 if n_steps else 0):
        steps = np.flatnonzero(seg_ids == sg)
        delta = em[start[steps[0]]:end[steps[0]]].copy()
        back, nop = [], []
        for a, b in zip(steps[:-1], steps[1:]):
            si, ei, sj, ej = start[a], end[a], start[b], end[b]
            d_gps = float(np.hypot(*(pts[uobs[b]] - pts[uobs[a]])))
            logp, nopath = _transition_logp(net, nd, cfg, arc[si:ei], frac[si:ei], arc[sj:ej], frac[sj:ej], d_gps)
            tot = delta[:, None] + logp
            bi = np.argmax(tot, axis=0)
            cols = np.arange(tot.shape[1])
            delta = tot[bi, cols] + em[sj:ej]
            back.append(bi)
            nop.append(nopath[bi, cols])
        j = int(np.argmax(delta))
        for k in range(len(steps) - 1, -1, -1):
            s_ = steps[k]
            hmm_arc[s_] = arc[start[s_] + j]
            hmm_frac[s_] = frac[start[s_] + j]
            if k > 0:
                breaks += int(nop[k - 1][j])
                j = int(back[k - 1][j])
    out = pd.DataFrame({"obs_idx": uobs, "clf_arc": clf_best, "hmm_arc": hmm_arc, "hmm_frac": hmm_frac})
    return out, breaks
