from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    def __init__(self):
        self.trainable=False

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, output_gradient):
        pass

class ReLU(Activation):
    
    def __init__(self):
        self.input_data = None
        self.trainable=False

    def forward(self, input):
        self.input_data = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        relu_gradient = (self.input_data > 0).astype(float)
        return output_gradient * relu_gradient