import numpy as np
from Autoencoder import Autoencoder


class TopKSparseAutoencoder(Autoencoder):

    def __init__(self, encoder, decoder, k=4):
        super().__init__(encoder, decoder)

        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = int(k)
        self.mask = None

    def forward(self, input_data):
        raw_latent = self.encoder.forward(input_data)

        if raw_latent.ndim != 2:
            raise ValueError(
                "Expected a batch with shape (samples, latent_dimensions)."
            )

        latent_dimensions = raw_latent.shape[1]

        if self.k > latent_dimensions:
            raise ValueError(
                f"k={self.k} exceeds the {latent_dimensions} latent neurons."
            )

        # Select the k largest magnitudes for EACH sample.
        # Stable sorting resolves equal magnitudes by their original order.
        indices = np.argsort(
            -np.abs(raw_latent),
            axis=1,
            kind="stable",
        )[:, :self.k]

        # Store which positions were selected for backpropagation.
        self.mask = np.zeros_like(raw_latent)

        np.put_along_axis(
            self.mask,
            indices,
            1.0,
            axis=1,
        )

        self.latent = raw_latent * self.mask

        self.reconstruction = self.decoder.forward(self.latent)

        return self.reconstruction

    def backward(self, output_gradient):
        latent_gradient = self.decoder.backward(output_gradient)

        # Only selected positions receive a gradient.
        latent_gradient = latent_gradient * self.mask

        return self.encoder.backward(latent_gradient)

    def regularization_loss(self):
        # Sparsity is imposed by selection, not an L1 penalty.
        return 0.0