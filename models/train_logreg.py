"""Model 1: Logistic Regression (linear / probabilistic family). Owner: Patil."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common import run_experiment, SEED

cfg = dict(C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, solver="lbfgs")
est = make_pipeline(StandardScaler(), LogisticRegression(**cfg, random_state=SEED))

if __name__ == "__main__":
    run_experiment("logreg", "Linear/probabilistic", est, cfg)
