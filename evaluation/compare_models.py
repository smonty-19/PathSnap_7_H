"""Model-comparison plots from results/model_comparison.csv and results/model_runs/*.json.

    python -m evaluation.compare_models
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

BASE_MODELS = ["logreg", "knn", "random_forest", "mlp", "lightgbm"]
OUT = os.path.join("results", "plots")


def load_comparison(models=BASE_MODELS):
    comp = pd.read_csv(os.path.join("results", "model_comparison.csv")).set_index("model")
    return comp.loc[[m for m in models if m in comp.index]]


def _fold_scores(model, key):
    with open(os.path.join("results", "model_runs", f"{model}.json")) as f:
        return [fold[key] for fold in json.load(f)["folds"]]


def _bar(ax, s, title, err=None, fmt="{:.3f}", ylim=None, highlight="max"):
    best = s.idxmax() if highlight == "max" else s.idxmin()
    colors = ["C1" if m == best else "C0" for m in s.index]
    ax.bar(s.index, s.values, yerr=err, capsize=4, color=colors)
    for i, v in enumerate(s.values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)


def make_all_plots(comp=None):
    comp = load_comparison() if comp is None else comp
    os.makedirs(OUT, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    _bar(ax[0], comp.cv_macro_f1_mean, "CV macro F1 (mean ± std, 5 folds)", err=comp.cv_macro_f1_std, ylim=(0.8, 0.96))
    _bar(ax[1], comp.cv_top1_mean, "CV Top-1 accuracy (mean ± std, 5 folds)", err=comp.cv_top1_std, ylim=(0.8, 0.92))
    fig.tight_layout(); p = os.path.join(OUT, "cv_comparison.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(comp)); w = 0.38
    ax.bar(x - w / 2, comp.test_top1, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, comp.test_macro_f1, w, label="macro F1")
    for i, (a, b) in enumerate(zip(comp.test_top1, comp.test_macro_f1)):
        ax.text(i - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, comp.index, rotation=25); ax.set_ylim(0.8, 0.96); ax.legend(); ax.set_title("Student-test Top-1 accuracy and macro F1")
    fig.tight_layout(); p = os.path.join(OUT, "test_top1_f1.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    _bar(ax[0], comp.test_logloss, "Student-test log loss (lower = better)", highlight="min")
    _bar(ax[1], comp.test_route_continuity_clf, "Route continuity, classifier-only", ylim=(0.9, 0.97))
    _bar(ax[2], comp.infer_ms_per_trip, "Inference time per trip (ms, log)", fmt="{:.0f}", highlight="min")
    ax[2].set_yscale("log")
    fig.tight_layout(); p = os.path.join(OUT, "logloss_continuity_speed.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = [_fold_scores(m, "top1_accuracy") for m in comp.index]
    ax.boxplot(data, tick_labels=list(comp.index)); ax.set_title("CV Top-1 per fold (stability across trip folds)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); p = os.path.join(OUT, "cv_fold_stability.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(comp)); w = 0.27
    for k, (col, lab) in enumerate([("train_top1", "train (fit data)"), ("cv_top1_mean", "CV"), ("test_top1", "student_test")]):
        ax.bar(x + (k - 1) * w, comp[col], w, label=lab)
    ax.set_xticks(x, comp.index, rotation=25); ax.set_ylim(0.8, 1.0); ax.legend(); ax.set_title("Over/underfitting: Top-1 on train vs CV vs test")
    fig.tight_layout(); p = os.path.join(OUT, "overfitting_check.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    imgs = [os.path.join(OUT, f"confusion_{m}.png") for m in comp.index]
    imgs = [i for i in imgs if os.path.exists(i)]
    if imgs:
        fig, axs = plt.subplots(1, len(imgs), figsize=(3.4 * len(imgs), 3.2))
        for a, i in zip(np.atleast_1d(axs), imgs):
            a.imshow(mpimg.imread(i)); a.axis("off")
        fig.tight_layout(); p = os.path.join(OUT, "confusion_grid.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    return paths


if __name__ == "__main__":
    comp = load_comparison()
    print(comp[["family", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1", "test_logloss",
                "test_route_continuity_clf", "infer_ms_per_trip"]].round(4).to_string())
    for p in make_all_plots(comp):
        print("saved", p)
