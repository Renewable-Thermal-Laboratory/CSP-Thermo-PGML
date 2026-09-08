"""Shared model wrapper so pickled models load from any entry point."""

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MLPModel:
    """Standardize -> MLP; picklable wrapper with fit/predict."""

    def __init__(self):
        self.sc = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="relu",
                                alpha=1e-4, learning_rate_init=1e-3, batch_size=512,
                                max_iter=60, early_stopping=True, n_iter_no_change=8,
                                random_state=0)

    def fit(self, X, y):
        self.net.fit(self.sc.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.net.predict(self.sc.transform(X))
