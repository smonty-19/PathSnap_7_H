"""Model 2: distance-weighted KNN (instance-based family). Owner: Monty.
KNN prediction cost grows with the training set, so it is fitted on a 25% sample of observations."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common import run_experiment

cfg = dict(n_neighbors=25, weights="distance", metric="minkowski", p=2, train_sample_frac=0.25)
est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))

if __name__ == "__main__":
    run_experiment("knn", "Instance-based", est, cfg, train_sample_frac=0.25)
