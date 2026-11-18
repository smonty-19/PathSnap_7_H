"""Model 3: Random Forest (bagging ensemble family). Owner: Sheetal."""
from sklearn.ensemble import RandomForestClassifier

from models.common import run_experiment, SEED

cfg = dict(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features="sqrt",
           class_weight="balanced_subsample")
est = RandomForestClassifier(**cfg, n_jobs=-1, random_state=SEED)

if __name__ == "__main__":
    run_experiment("random_forest", "Ensemble (bagging)", est, cfg)
