<!-- DRAFT by Sanchita (AI-assisted). Numbers come from results/optuna/*, results/tuning_comparison.csv and results/model_comparison.csv.
     Rewrite in your own words before submission. -->
# 4. Bayesian hyperparameter optimisation

**Method.** Optuna with the Tree-structured Parzen Estimator (TPE) sampler. After each trial TPE splits past trials into a good and a bad group,
fits a density to each, and proposes the point that maximises the ratio good/bad, so it learns from history. Grid and random search were not
used anywhere (organiser rule). The first 8 trials are random start-up trials; a MedianPruner stops trials whose running score is below the median.

**Objective.** Mean 3-fold GroupKFold (by trip) observation-level Top-1 on the 78 clean training trips. Top-1 is the primary metric because the route
is built from the arc chosen for each point. Three folds are used inside the search for speed; the tuned model is then re-scored with the same 5-fold
harness and on student_test as the untuned model. Evaluator trips were never used.

**Search space.** `n_estimators` 100 to 800 (log), `learning_rate` 0.01 to 0.3 (log), `num_leaves` 15 to 255 (log), `min_child_samples` 10 to 300 (log),
`subsample` and `colsample_bytree` 0.5 to 1.0, `reg_alpha` and `reg_lambda` 1e-3 to 10 (log), `scale_pos_weight` 1 to 10 (log).
Log scales are used where a parameter spans orders of magnitude; `num_leaves` is the main overfitting control for leaf-wise growth.

**Run.** 33 trials (23 completed, 10 pruned) in 731 s. Best 3-fold CV Top-1 0.8735.
Best parameters: 472 trees, learning rate 0.0204, 150 leaves, `min_child_samples` 14, subsample 0.67, colsample 0.77, `reg_alpha` 0.0022,
`reg_lambda` 0.078, `scale_pos_weight` 1.83. Record: `results/optuna/` (`lgbm.db`, `trials.csv`, `best_params.json`, plots).

| | Untuned | Tuned |
|---|---|---|
| CV Top-1 | 0.8737 | 0.8740 |
| student_test Top-1 | 0.8827 | 0.8847 |
| CV macro F1 | 0.9114 | 0.9249 |
| student_test macro F1 | 0.9133 | 0.9297 |
| student_test log loss | 0.1149 | 0.0881 |
| Inference (ms per trip) | 42 | 76 |

**Interpretation.** Top-1 barely changes (+0.002 on student_test, inside the fold noise of about 0.027). Macro F1 and log loss improve clearly.
The search preferred a lower learning rate with more trees, larger trees, and a much smaller positive weight than `balanced`, which removed the
over-prediction of positives and gave better-calibrated probabilities. That matters because the probabilities feed the HMM emissions.
Cost: about 12 minutes of search and roughly 2x slower inference, still well within live-API needs.
The tuned model is exported as `artifacts/final_model.joblib` with `artifacts/model_meta.json`.
