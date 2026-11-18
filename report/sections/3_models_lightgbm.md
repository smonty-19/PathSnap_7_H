<!-- DRAFT by Sanchita (AI-assisted). Numbers come from results/model_comparison.csv and results/model_runs/lightgbm.json.
     Rewrite in your own words before submission and keep the AI-use disclosure in the report. -->
# 3.2 LightGBM (gradient boosting)

**Why this model.** LightGBM builds trees one after another, each correcting the errors of the previous ones, and grows them leaf-wise.
It learns interactions between our 21 geometric, directional, kinematic and topological features without manual crossing, it is trained on
binary log loss so its output is a usable probability for the HMM emissions, and it fits in about 6 s, which makes Bayesian tuning affordable.

**Configuration (one fixed setting, no grid or random search).** 400 trees, learning rate 0.05, 63 leaves, `min_child_samples` 20,
row and feature subsampling 0.8, `class_weight="balanced"` (about 14% of candidate rows are positive), seed 42.

**Protocol.** 5-fold GroupKFold by trip on 78 training trips (trips 33 and 54 excluded because their reference route lies more than 50 m
from the GPS track), refit on all 78, evaluated on the 10 student_test trips. Primary metric: observation-level Top-1 accuracy.

| Metric | CV (mean ± std) | student_test |
|---|---|---|
| Top-1 | 0.8737 ± 0.0272 | 0.8827 |
| Macro F1 | 0.9114 ± 0.0130 | 0.9133 |
| Log loss | 0.1240 | 0.1149 |
| Route continuity (classifier argmax) | n/a | 0.9547 |

Fit time 6.5 s, inference about 43 ms per trip, model file about 1 MB.

**Over/underfitting.** Train Top-1 is 0.9175 against a CV Top-1 of 0.8737, a gap of 0.044 (mild overfitting). CV and student_test agree
(0.874 and 0.883), so the model generalises to unseen trips. Fold-to-fold Top-1 ranges from 0.834 to 0.899, which reflects how hard individual trips are.

**Feature importance (share of total gain).** `dist_minus_min` 39.4%, `dist_rank` 15.6%, `heading_cos_x_speed` 14.8%, `heading_diff` 8.9%.
The model mainly asks whether a candidate is clearly closer than the alternatives and whether the travel direction matches the arc direction.

**Limitations.** More hyperparameters than the other models, and it can overfit with large trees (see section 4 for tuning).
The five models differ by only 0.9 points of CV Top-1, which is smaller than the fold standard deviation, so the ranking is suggestive rather than significant.
Figures: `results/plots/confusion_lightgbm.png`, `results/plots/lgbm_importance.png`.
