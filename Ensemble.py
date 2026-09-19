"""Bagging sample and equal-weight, calibrated reconstruction scores."""
import numpy as np


def bootstrap_indices(n_samples, seed):
    if n_samples < 1:
        raise ValueError("Training data must not be empty.")
    return np.random.default_rng(seed).integers(0, n_samples, size=n_samples)


class Ensemble:
    def __init__(self, models):
        if not models:
            raise ValueError("Supply at least one model.")
        self.models = list(models)
        self.scales = None
        self.threshold = None

    def member_scores(self, X, batch_size=4096):
        """One row per flow, one column per member; no training here."""
        if len(X) == 0 or batch_size < 1:
            raise ValueError("Need nonempty data and positive batch size.")
        scores = np.empty((len(X), len(self.models)))
        for column, model in enumerate(self.models):
            model.set_training(False)
            for start in range(0, len(X), batch_size):
                batch = X[start:start + batch_size]
                reconstruction = model.forward(batch)
                scores[start:start + len(batch), column] = np.mean(
                    (batch - reconstruction) ** 2, axis=1)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite reconstruction scores.")
        return scores

    def calibrate(self, benign_scores, percentile=95):
        """Fit only on benign validation scores, never test scores."""
        scores = np.asarray(benign_scores)
        if (scores.ndim != 2 or scores.shape[0] == 0
                or scores.shape[1] != len(self.models)
                or not np.isfinite(scores).all() or np.any(scores < 0)):
            raise ValueError("Invalid benign calibration scores.")
        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100.")
        self.scales = np.maximum(np.percentile(scores, percentile, axis=0), 1e-12)
        combined = self.combine(scores)
        self.threshold = float(np.percentile(combined, percentile))

    def combine(self, member_scores):
        if self.scales is None:
            raise ValueError("Calibrate or load calibration first.")
        return np.mean(member_scores / self.scales, axis=1)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Calibrate or load calibration first.")
        scores = self.combine(self.member_scores(X))
        return (scores > self.threshold).astype(int), scores

    def save_calibration(self, filename):
        if self.threshold is None:
            raise ValueError("Calibrate first.")
        np.savez_compressed(filename, scales=self.scales, threshold=self.threshold)

    def load_calibration(self, filename):
        # Models must be supplied in the SAME order as during calibration.
        with np.load(filename, allow_pickle=False) as saved:
            scales = saved['scales'].copy()
            threshold = float(saved['threshold'])
        if (scales.shape != (len(self.models),) or not np.isfinite(scales).all()
                or np.any(scales <= 0) or not np.isfinite(threshold)):
            raise ValueError("Invalid or incompatible calibration.")
        self.scales, self.threshold = scales, threshold
