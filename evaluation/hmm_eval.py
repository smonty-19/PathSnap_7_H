"""Classifier-only argmax vs classifier + HMM/Viterbi.   Owner: Monty (M2).

    python -m evaluation.hmm_eval                      # student_test, model = lightgbm_tuned
    python -m evaluation.hmm_eval --model lightgbm

Per-observation metrics follow the harness definition (observations that have at least one candidate),
so the classifier-only Top-1 here equals `test_top1` in results/model_comparison.csv.
Also reports route continuity, number of HMM breaks and the arc-set precision / recall of the predicted route
against the reference route. Writes:
    results/hmm_comparison.csv           overall + per-trip rows for every variant
    results/preds/hmm_<trip>.parquet     per-observation predictions (for the Folium maps)
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from pathsnap.features import FEATURE_COLUMNS, prepare_trip
from pathsnap.hmm import decode
from pathsnap.io import load_route, load_trip
from pathsnap.pipeline import load_hmm_cfg

PRED = os.path.join("results", "preds")
PROC = os.path.join("data", "processed")
KEYS = ["trip_id", "obs_idx", "cand_arc_id"]
GEOM_ONLY = dict(w=0.0)       # pure geometry (no classifier): ablation


def ensure_preds(model_name, split):
    """Probabilities of `model_name` on `split`. lightgbm_tuned is produced from artifacts/final_model.joblib
    (student_test) or by 5-fold GroupKFold OOF with the tuned params (train)."""
    path = os.path.join(PRED, f"{model_name}_{'train_oof' if split == 'train' else 'student_test'}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from models.common import cross_validate, load_split
    df = load_split(split)
    if split == "student_test":
        model = joblib.load(os.path.join("artifacts", "final_model.joblib"))
        prob = model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]
    else:
        from lightgbm import LGBMClassifier
        meta = json.load(open(os.path.join("artifacts", "model_meta.json")))
        est = LGBMClassifier(**meta["params"], random_state=42, n_jobs=-1, verbose=-1)
        prob, _ = cross_validate(est, df)
    out = df[KEYS].copy(); out["prob"] = prob
    os.makedirs(PRED, exist_ok=True); out.to_parquet(path, index=False)
    return out


def load_joined(model_name, split):
    from models.common import load_split
    feats = load_split(split)
    feats["trip_id"] = feats["trip_id"].astype(str)
    p = ensure_preds(model_name, split); p["trip_id"] = p["trip_id"].astype(str)
    cols = KEYS + ["proj_frac", "dist_m", "true_arc_id", "label", "t"]
    return feats[cols].merge(p[KEYS + ["prob"]], on=KEYS)


def trip_context(split, trip_id):
    prep = prepare_trip(load_trip(os.path.join("data", "raw", split, trip_id)))
    return prep["net"], prep["pts"], prep["t"]


def arc_set_pr(pred_arcs, route):
    p, r = set(pred_arcs.tolist()), set(route.tolist())
    inter = len(p & r)
    prec = inter / max(len(p), 1); rec = inter / max(len(r), 1)
    return prec, rec


def run(model_name="lightgbm_tuned", split="student_test", cfg=None):
    cfg = cfg or load_hmm_cfg()
    df = load_joined(model_name, split)
    variants = {"classifier_only": None, "hmm_geometry_only(w=0)": {**cfg, **GEOM_ONLY}, "hmm_tuned": cfg}
    rows, per_trip_pred = [], {}
    for tid, g in df.groupby("trip_id"):
        net, pts, t = trip_context(split, tid)
        truth = g.groupby("obs_idx")["true_arc_id"].first()
        route = load_route(os.path.join("data", "raw", split, tid))
        for name, vcfg in variants.items():
            dec, br = decode(net, pts, t, g[["obs_idx", "cand_arc_id", "proj_frac", "dist_m", "prob"]], vcfg or cfg)
            pred = dec["clf_arc"].values if name == "classifier_only" else dec["hmm_arc"].values
            tru = truth.reindex(dec.obs_idx).values
            valid = pred >= 0
            cont = float(np.mean(net.connected(pred[valid][:-1], pred[valid][1:]))) if valid.sum() > 1 else np.nan
            seq = pred[valid]; seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
            prec, rec = arc_set_pr(seq, route)
            rows.append(dict(trip_id=tid, variant=name, n_obs=len(dec), top1=float(np.mean(pred == tru)),
                             route_continuity=cont, breaks=int(br) if name != "classifier_only" else 0,
                             route_arc_precision=prec, route_arc_recall=rec))
            if name == "hmm_tuned":
                per_trip_pred[tid] = dec.assign(true_arc=tru)
    res = pd.DataFrame(rows)
    # overall rows: observation-weighted Top-1, mean of the rest
    allr = []
    for name, g in res.groupby("variant", sort=False):
        allr.append(dict(trip_id="ALL", variant=name, n_obs=int(g.n_obs.sum()),
                         top1=float(np.average(g.top1, weights=g.n_obs)),
                         route_continuity=float(g.route_continuity.mean()), breaks=int(g.breaks.sum()),
                         route_arc_precision=float(g.route_arc_precision.mean()),
                         route_arc_recall=float(g.route_arc_recall.mean())))
    out = pd.concat([pd.DataFrame(allr), res], ignore_index=True)
    return out, per_trip_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm_tuned")
    ap.add_argument("--split", default="student_test")
    a = ap.parse_args()
    out, preds = run(a.model, a.split)
    out.insert(0, "model", a.model); out.insert(1, "split", a.split)
    out.to_csv(os.path.join("results", "hmm_comparison.csv"), index=False)
    os.makedirs(PRED, exist_ok=True)
    for tid, d in preds.items():
        d.to_parquet(os.path.join(PRED, f"hmm_{tid}.parquet"), index=False)
    pd.set_option("display.width", 200)
    print(out[out.trip_id == "ALL"].round(4).to_string(index=False))
    piv = out[out.trip_id != "ALL"].pivot(index="trip_id", columns="variant", values="top1")
    hurt = piv[piv["hmm_tuned"] < piv["classifier_only"]]
    print("\nper-trip Top-1 (classifier -> HMM):")
    print(piv[["classifier_only", "hmm_tuned"]].round(4).to_string())
    print("\ntrips where the HMM hurt:", hurt.index.tolist() or "none")


if __name__ == "__main__":
    main()
