"""Standardisation strictement ajustée sur les batteries d'entraînement."""
import numpy as np


class TrainScaler:
    def __init__(self):
        self.parameters = None

    def fit(self, train):
        # Les fenêtres se chevauchent : on ajuste sur leurs observations
        # exclusivement TRAIN. Jamais sur validation ni test.
        specs = (("x", (0, 1, 2)), ("aux", (0, 1)), ("y", (0,)))
        self.parameters = {}
        for key, axes in specs:
            value = np.asarray(train[key], dtype=np.float64)
            mean = value.mean(axis=axes, keepdims=True)
            std = value.std(axis=axes, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            self.parameters[key] = (mean.astype(np.float32), std.astype(np.float32))
        return self

    def transform(self, batch):
        if self.parameters is None:
            raise RuntimeError("fit(train) doit preceder transform")
        copy = dict(batch)
        for key, (mean, std) in self.parameters.items():
            copy[key] = ((batch[key] - mean) / std).astype(np.float32)
        return copy

    def inverse_y(self, y):
        mean, std = self.parameters["y"]
        return np.asarray(y) * std.reshape(1, 2) + mean.reshape(1, 2)

    def to_dict(self):
        return {key: {"mean": m.ravel().tolist(), "std": s.ravel().tolist()}
                for key, (m, s) in self.parameters.items()}

    @classmethod
    def from_dict(cls, value):
        scalar = cls()
        shapes = {"x": (1, 1, 1, 3), "aux": (1, 1, 2), "y": (1, 2)}
        scalar.parameters = {k: (np.asarray(v["mean"], dtype=np.float32).reshape(shapes[k]),
                                  np.asarray(v["std"], dtype=np.float32).reshape(shapes[k]))
                             for k, v in value.items()}
        return scalar
