"""Build the (observation, candidate) feature table for a data split.

Usage:
    python -m scripts.build_dataset --split train --stride 2
    python -m scripts.build_dataset --split student_test
Output: data/processed/<split>.parquet and results/dataset_summary_<split>.csv
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pathsnap.io import list_trips, load_trip, load_route
from pathsnap.features import prepare_trip, build_features
from pathsnap.labels import align_route

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "processed")


def process(args):
    split, trip_id, stride, radius, k_max = args
    t0 = time.time()
    trip_dir = os.path.join(RAW, split, trip_id)
    trip = load_trip(trip_dir)
    prep = prepare_trip(trip, radius, k_max)
    df = build_features(prep)
    # --- labels (ground truth is attached only here, after features are built) ---
    route = load_route(trip_dir)
    pos, true_arc, align_d = align_route(prep["net"], prep["pts"], route)
    df["true_arc_id"] = true_arc[df.obs_idx.values]
    df["align_dist_m"] = align_d[df.obs_idx.values]
    df["label"] = (df.cand_arc_id.values == df.true_arc_id.values).astype(np.int8)
    df.insert(0, "trip_id", trip_id)
    N = len(prep["pts"])
    has_cand = np.zeros(N, bool); has_cand[df.obs_idx.values] = True
    hit = np.zeros(N, bool); hit[df.obs_idx.values[df.label.values == 1]] = True
    summary = dict(trip_id=trip_id, split=split, n_obs=N, n_rows=len(df),
                   obs_no_candidates=int((~has_cand).sum()),
                   candidate_miss_rate=float(1 - hit.mean()),
                   median_align_dist_m=float(np.median(align_d)),
                   p95_align_dist_m=float(np.percentile(align_d, 95)),
                   seconds=round(time.time() - t0, 2))
    if stride > 1:
        df = df[df.obs_idx % stride == 0]
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "student_test"])
    ap.add_argument("--stride", type=int, default=1, help="keep every s-th observation (train only)")
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--trips", type=int, default=0, help="limit number of trips (0 = all)")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    a = ap.parse_args()
    trips = list_trips(os.path.join(RAW, a.split))
    if a.trips:
        trips = trips[:a.trips]
    jobs = [(a.split, t, a.stride, a.radius, a.k) for t in trips]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(process, jobs, chunksize=1)
    df = pd.concat([r[0] for r in res], ignore_index=True)
    summ = pd.DataFrame([r[1] for r in res])
    os.makedirs(OUT, exist_ok=True); os.makedirs("results", exist_ok=True)
    suffix = "" if not a.trips else f"_sample{a.trips}"
    df.to_parquet(os.path.join(OUT, f"{a.split}{suffix}.parquet"), index=False)
    summ.to_csv(os.path.join("results", f"dataset_summary_{a.split}{suffix}.csv"), index=False)
    print(f"{a.split}: {len(trips)} trips, {len(df):,} rows, pos rate {df.label.mean():.3f}, "
          f"candidate miss rate {(summ.candidate_miss_rate * summ.n_obs).sum() / summ.n_obs.sum():.4f}, "
          f"median align {summ.median_align_dist_m.median():.1f} m, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
