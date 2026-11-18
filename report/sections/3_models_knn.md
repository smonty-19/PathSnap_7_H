<!-- DRAFT (AI-assisted) for Monty's KNN subsection. Numbers come from results/model_comparison.csv. Rewrite in your own words. -->
# 3.2 K-Nearest Neighbours (instance-based)

**Why.** Candidate arcs that look alike in feature space (similar distance, heading and topology) tend to share a label, so a similarity vote is a natural
non-parametric baseline. Features are standardised first, because KNN is distance based and would otherwise be dominated by large-scale features such as `arc_length_m`.

**Configuration.** k = 25, distance-weighted vote, Euclidean distance, `StandardScaler` inside the pipeline, trained on a 25% sample of observations
(147,016 rows) because prediction cost grows with the training-set size. One fixed setting, no grid or random search. There is no loss function: the output is a
distance-weighted majority vote.

| Metric | CV (mean ± std) | student_test |
|---|---|---|
| Top-1 | 0.8649 ± 0.0320 | 0.8626 |
| Macro F1 | 0.9188 ± 0.0153 | 0.9162 |
| Log loss | 0.1572 | 0.1435 |

**Over/underfitting.** Train Top-1 is 0.991 against a CV Top-1 of 0.865: strong overfitting (every training point is its own neighbour). Inference takes about
1.4 s per trip, so KNN is not suitable for the live API. It had the lowest Top-1 of the five models.
