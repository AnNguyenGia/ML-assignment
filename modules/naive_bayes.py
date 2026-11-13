import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.prior_ = np.zeros(n_classes)

        global_variance = np.var(X, axis=0)
        epsilon = self.var_smoothing * np.max(global_variance)

        for index, class_label in enumerate(self.classes_):
            class_features = X[y == class_label]
            self.mean_[index] = class_features.mean(axis=0)
            self.var_[index] = class_features.var(axis=0) + epsilon
            self.prior_[index] = class_features.shape[0] / X.shape[0]
        return self

    def _log_gaussian(self, X):
        log_constant = -0.5 * np.log(2 * np.pi * self.var_)
        log_probability = log_constant - 0.5 * ((X[:, None, :] - self.mean_) ** 2) / self.var_
        return log_probability.sum(axis=2)

    def predict(self, X):
        X = np.asarray(X)
        log_likelihood = self._log_gaussian(X)
        log_prior = np.log(self.prior_)
        return self.classes_[np.argmax(log_prior + log_likelihood, axis=1)]