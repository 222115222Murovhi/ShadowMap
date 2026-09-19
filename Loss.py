from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self, y_pred, y_true):
        pass

class MSE(Loss):

    def forward(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

