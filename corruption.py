import numpy as np

class MakeNoise:
    def __init__(self, corruption_rate=0.1, seed=42):
        if not 0 <= corruption_rate < 1:
            raise ValueError("corruption_rate must be between 0 and 1.")

        self.corruption_rate = corruption_rate
        self.rng = np.random.default_rng(seed)

    def corrupt(self, X):
        # One random number for each input value.
        random_values = self.rng.random(X.shape)

        # True keeps a value; False replaces it with zero.
        mask = random_values >= self.corruption_rate

        return X * mask