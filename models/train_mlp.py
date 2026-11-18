"""Model 4: Multi-layer perceptron (neural network family). Owner: Sheetal.
Trained on a 40% sample of observations to keep CV time reasonable."""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common import run_experiment, SEED

cfg = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3,
           batch_size=512, max_iter=60, early_stopping=True, n_iter_no_change=5, train_sample_frac=0.4)
est = make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3,
    batch_size=512, max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=SEED))

if __name__ == "__main__":
    run_experiment("mlp", "Neural network", est, cfg, train_sample_frac=0.4)
