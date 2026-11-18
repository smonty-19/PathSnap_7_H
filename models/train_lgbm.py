"""Model 5: LightGBM (gradient boosting family), untuned baseline. Owner: Sanchita."""
from lightgbm import LGBMClassifier

from models.common import run_experiment, SEED

cfg = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced")
est = LGBMClassifier(**cfg, random_state=SEED, n_jobs=-1, verbose=-1)

if __name__ == "__main__":
    run_experiment("lightgbm", "Boosting", est, cfg)
