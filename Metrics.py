import numpy as np

# This class calculates the reconstruction error for autoencoders per sample.
class ReconstructionError:

    def calculate(self, y_true, y_pred):

        return np.mean(
            (y_pred - y_true) ** 2,
            axis=1
        )