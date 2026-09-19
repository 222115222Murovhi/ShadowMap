from abc import ABC, abstractmethod

import numpy

class Optimizer(ABC):

    @abstractmethod
    def update(self, layer):
        pass

class gradient_descent(Optimizer):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.weights_gradient
        layer.biases -= self.learning_rate * layer.biases_gradient

class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9,
                 beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}
        self.t = {}

    def update(self, layer):
        # Initialise moments and timestep for this layer.
        if layer not in self.m_weights:
            self.m_weights[layer] = numpy.zeros_like(layer.weights)
            self.v_weights[layer] = numpy.zeros_like(layer.weights)
            self.m_biases[layer] = numpy.zeros_like(layer.biases)
            self.v_biases[layer] = numpy.zeros_like(layer.biases)
            self.t[layer] = 0

        self.t[layer] += 1
        t = self.t[layer]

        # Update weight moments.
        self.m_weights[layer] = (
            self.beta1 * self.m_weights[layer]
            + (1 - self.beta1) * layer.weights_gradient
        )
        self.v_weights[layer] = (
            self.beta2 * self.v_weights[layer]
            + (1 - self.beta2) * layer.weights_gradient ** 2
        )

        # Correct initial bias in the moments.
        m_hat_weights = self.m_weights[layer] / (1 - self.beta1 ** t)
        v_hat_weights = self.v_weights[layer] / (1 - self.beta2 ** t)

        layer.weights -= (
            self.learning_rate * m_hat_weights
            / (numpy.sqrt(v_hat_weights) + self.epsilon)
        )

        # Update bias moments.
        self.m_biases[layer] = (
            self.beta1 * self.m_biases[layer]
            + (1 - self.beta1) * layer.biases_gradient
        )
        self.v_biases[layer] = (
            self.beta2 * self.v_biases[layer]
            + (1 - self.beta2) * layer.biases_gradient ** 2
        )

        m_hat_biases = self.m_biases[layer] / (1 - self.beta1 ** t)
        v_hat_biases = self.v_biases[layer] / (1 - self.beta2 ** t)

        layer.biases -= (
            self.learning_rate * m_hat_biases
            / (numpy.sqrt(v_hat_biases) + self.epsilon)
        )