# `models/`

**Owner (only this person pushes here):** **Shared folder, one file per owner**

Training scripts. Every script calls `models.common.run_experiment(...)`. Only edit **your own files**.

| File | What it is |
|---|---|
| `common.py` | **Sanchita**: training harness (GroupKFold CV, metrics, saving) |
| `train_lgbm.py` | **Sanchita**: LightGBM baseline |
| `tune_lgbm_optuna.py` | **Sanchita**: Bayesian optimisation (Optuna TPE) |
| `train_logreg.py` | **Patil**: Logistic Regression |
| `train_knn.py` | **Patil**: KNN (distance-weighted) |
| `train_gnb.py` | **Patil** (optional): Gaussian Naive Bayes |
| `train_rf.py` | **Sheetal**: Random Forest |
| `train_mlp.py` | **Sheetal**: MLP + real train/val loss curves |
