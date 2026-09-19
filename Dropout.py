import numpy as np


class Dropout:

    def __init__(self, rate=0.1, seed=None):
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be >= 0 and < 1.")

        self.rate = rate
        self.trainable = False
        self.training = False
        self.mask = None
        self.rng = np.random.default_rng(seed)

    def forward(self, input_data):
        if not self.training:
            self.mask = None
            return input_data

        keep_probability = 1 - self.rate

        self.mask = (
            self.rng.random(input_data.shape) < keep_probability
        ).astype(input_data.dtype)

        self.mask /= keep_probability

        return input_data * self.mask

    def backward(self, output_gradient):
        if self.mask is None:
            return output_gradient

        return output_gradient * self.mask