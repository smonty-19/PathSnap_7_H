"""Evaluation metrics on candidate-level predictions.

`df` columns: trip_id, obs_idx, cand_arc_id, label, prob
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix

RAW = os.path.join("data", "raw")


def top1(df):
    """Per observation, the candidate with the highest probability."""
    idx = df.groupby(["trip_id", "obs_idx"], sort=True)["prob"].idxmax()
    return df.loc[idx]


def top1_accuracy(df):
    return float(top1(df)["label"].mean())


def candidate_miss_rate(df):
    hit = df.groupby(["trip_id", "obs_idx"])["label"].max()
    return float(1 - hit.mean())


def classification_metrics(df, thr=0.5):
    y, p = df["label"].values, np.clip(df["prob"].values, 1e-7, 1 - 1e-7)
    yhat = (p >= thr).astype(int)
    return dict(
        macro_f1=float(f1_score(y, yhat, average="macro")),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        logloss=float(log_loss(y, p, labels=[0, 1])),
    )


def confusion(df, thr=0.5):
    return confusion_matrix(df["label"].values, (df["prob"].values >= thr).astype(int), labels=[0, 1])


def _arcs(split, trip_id):
    return np.loadtxt(os.path.join(RAW, split, trip_id, f"{trip_id}.arcs"), dtype=np.int64, ndmin=2)


def route_continuity(pred_arcs, arcs):
    """Fraction of consecutive predictions that are the same arc or directly connected (a.v == b.u)."""
    a, b = pred_arcs[:-1], pred_arcs[1:]
    if len(a) == 0:
        return 1.0
    return float(np.mean((a == b) | (arcs[a, 1] == arcs[b, 0])))


def mean_route_continuity(df, split):
    best = top1(df).sort_values(["trip_id", "obs_idx"])
    vals = [route_continuity(g["cand_arc_id"].values, _arcs(split, t)) for t, g in best.groupby("trip_id")]
    return float(np.mean(vals))


def all_metrics(df, split=None):
    m = classification_metrics(df)
    m["top1_accuracy"] = top1_accuracy(df)
    m["candidate_miss_rate"] = candidate_miss_rate(df)
    if split is not None:
        m["route_continuity"] = mean_route_continuity(df, split)
    return m
