import numpy as np

class Autoencoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

        self.latent = None
        self.reconstruction = None

    def forward(self, input_data):
        self.latent = self.encoder.forward(input_data)
        self.reconstruction = self.decoder.forward(self.latent)
        return self.reconstruction

    def backward(self, output_gradient):
        decoder_gradient = self.decoder.backward(output_gradient)
        encoder_gradient = self.encoder.backward(decoder_gradient)
        return encoder_gradient

    def get_parameters(self):

        return {
            "encoder": self.encoder.get_parameters(),
            "decoder": self.decoder.get_parameters()
        }

    def set_parameters(self, parameters):

        self.encoder.set_parameters(
            parameters["encoder"]
        )

        self.decoder.set_parameters(
            parameters["decoder"]
        )

    def set_training(self, training):
        self.encoder.set_training(training)
        self.decoder.set_training(training)
        
    def regularization_loss(self):
        return 0.0
    
    def save(self, filename="autoencoder.npz"):
        parameters = self.get_parameters()
        saved_arrays = {}

        for section in ("encoder", "decoder"):
            for index, layer_parameters in enumerate(parameters[section]):
                saved_arrays[f"{section}_{index}_weights"] = (
                    layer_parameters["weights"]
                )
                saved_arrays[f"{section}_{index}_biases"] = (
                    layer_parameters["biases"]
                )

        np.savez_compressed(filename, **saved_arrays)
        print(f"Model saved to {filename}")


    def load(self, filename="autoencoder.npz"):
        parameters = self.get_parameters()
        with np.load(filename, allow_pickle=False) as saved_arrays:
            for section in ("encoder", "decoder"):
                for index, layer_parameters in enumerate(parameters[section]):
                    for name in ("weights", "biases"):
                        key = f"{section}_{index}_{name}"
                        value = saved_arrays[key]
                        if value.shape != layer_parameters[name].shape:
                            raise ValueError(f"Shape mismatch for {key}")
                        layer_parameters[name] = value.copy()
        self.set_parameters(parameters)
        self.set_training(False)
        print(f"Model loaded from {filename}")
