"""Geometry helpers.

Trips are spread worldwide (lon -100..137, lat -35..62), so we use a *per-trip*
local equirectangular projection centred on the trip's GPS centroid:
    x = R * cos(lat0) * (lon - lon0),  y = R * (lat - lat0)   [metres]
For the few-km extent of a single trip the distortion is far below GPS noise
(relative error ~0.05% median, <=0.2% across a ~50 km trip; centimetres at the <100 m
scales used for matching), and it is much faster than a full UTM transform.
"""
import numpy as np

EARTH_R = 6_371_000.0


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = EARTH_R * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    @classmethod
    def from_lonlat(cls, lonlat):
        return cls(np.mean(lonlat[:, 0]), np.mean(lonlat[:, 1]))

    def to_xy(self, lonlat):
        lonlat = np.asarray(lonlat, float)
        return np.column_stack([(lonlat[:, 0] - self.lon0) * self.kx, (lonlat[:, 1] - self.lat0) * self.ky])

    def to_lonlat(self, xy):
        xy = np.asarray(xy, float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def point_segment(p, a, b):
    """Vectorised point-to-segment distance.
    p, a, b: (n,2). Returns (dist, t) with t in [0,1] the projection fraction along a->b."""
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", p - a, ab) / np.maximum(L2, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.hypot(*(p - proj).T), t
