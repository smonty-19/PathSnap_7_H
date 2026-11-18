"""Rehearsal: call a running API for every trip and record timings and metrics (what the evaluator UI will do).

    python -m api.rehearse --url http://localhost:8000 --out results/api_rehearsal.csv

Run it from the UI laptop with the URL of Laptop 1 to include the real network time.
`round_trip_ms` is measured by the client (request sent until the JSON is parsed); `server_total_ms` comes from the server.
"""
import argparse
import csv
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="results/api_rehearsal.csv")
    a = ap.parse_args()

    health = requests.get(f"{a.url}/health", timeout=10).json()
    trips = requests.get(f"{a.url}/trips", timeout=10).json()["trips"]
    print(f"API {a.url} model={health['model']} trips={len(trips)}")
    rows = []
    for tid in trips:
        t0 = time.perf_counter()
        r = requests.post(f"{a.url}/predict", json={"trip_id": tid}, timeout=120)
        rt = (time.perf_counter() - t0) * 1e3
        r.raise_for_status()
        d = r.json(); m = d["metrics"] or {}
        rows.append(dict(trip_id=tid, n_points=d["n_points"], payload_kb=round(len(r.content) / 1024),
                         round_trip_ms=round(rt), server_total_ms=d["timing_ms"]["total"],
                         features_ms=d["timing_ms"]["features"], model_ms=d["timing_ms"]["model"],
                         viterbi_ms=d["timing_ms"]["viterbi"],
                         hmm_top1=round(m["hmm"]["top1_accuracy"], 4) if m else "",
                         clf_top1=round(m["classifier"]["top1_accuracy"], 4) if m else ""))
        print(rows[-1])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"max round trip {max(r['round_trip_ms'] for r in rows)} ms -> {a.out}")


if __name__ == "__main__":
    main()
