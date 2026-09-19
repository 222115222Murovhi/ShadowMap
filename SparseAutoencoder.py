import numpy as np
from Autoencoder import Autoencoder


class SparseAutoencoder(Autoencoder):

    def __init__(
        self,
        encoder,
        decoder,
        sparsity_weight=1e-3,
        decoder_weight_penalty=1e-4,
    ):
        super().__init__(encoder, decoder)

        self.sparsity_weight = sparsity_weight
        self.decoder_weight_penalty = decoder_weight_penalty

        # In our architecture, the decoder starts with a DenseLayer.
        self.first_decoder_layer = self.decoder.layers[0]

    def regularization_loss(self):
        sparsity_loss = (
            self.sparsity_weight * np.mean(np.abs(self.latent))
        )

        weight_loss = (
            self.decoder_weight_penalty
            * np.mean(self.first_decoder_layer.weights ** 2)
        )

        return sparsity_loss + weight_loss

    def backward(self, output_gradient):
        # Reconstruction-loss gradient arriving at the bottleneck.
        latent_gradient = self.decoder.backward(output_gradient)

        # Add the L2 penalty gradient to the decoder weights.
        layer = self.first_decoder_layer

        layer.weights_gradient += (
            2 * self.decoder_weight_penalty
            * layer.weights / layer.weights.size
        )

        # Add the L1 penalty gradient to the latent activations.
        latent_gradient += (
            self.sparsity_weight
            * np.sign(self.latent) / self.latent.size
        )

        return self.encoder.backward(latent_gradient)