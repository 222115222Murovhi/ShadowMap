import numpy as np


class DenseLayer:

    def __init__(self, input_size, neurons):

        # He (Kaiming) Weight Initialization
        self.weights = np.random.randn(input_size, neurons) * np.sqrt(2.0 / input_size)

        # Biases start at zero
        self.biases = np.zeros((1, neurons))

        # Cached input from forward pass
        self.input_data = None

        # Cached output (optional, useful for debugging)
        self.output = None

        self.trainable=True

        # Gradient placeholders
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input_data):

        self.input_data = input_data

        self.output = input_data @ self.weights + self.biases

        return self.output

    def backward(self, output_gradient):

        # Number of training examples in this batch
        batch_size = self.input_data.shape[0]

        # Gradient with respect to weights
        self.weights_gradient = (
            self.input_data.T @ output_gradient
        ) 

        # Gradient with respect to biases
        self.biases_gradient = (
            np.sum(output_gradient, axis=0, keepdims=True)
        ) 

        # Gradient passed to previous layer
        input_gradient = output_gradient @ self.weights.T

        return input_gradient

    def get_parameters(self):

        return {
            "weights": self.weights.copy(),
            "biases": self.biases.copy()
        }

    def set_parameters(self, parameters):


        self.weights = parameters["weights"].copy()
        self.biases = parameters["biases"].copy()