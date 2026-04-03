
import numpy as np
from abc import ABC, abstractmethod
from numpy.lib.stride_tricks import sliding_window_view

class Layer(ABC):

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dA):
        pass

class Softmax(Layer):

    def forward(self, X):
        # stabilité numérique
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def backward(self, dY):
        # Jacobien complet (coûteux mais correct)
        m, n = self.out.shape
        dX = np.zeros_like(dY)

        for i in range(m):
            y = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - y @ y.T
            dX[i] = jacobian @ dY[i]

        return dX

class Linear(Layer):

    def forward(self, X, *args):
        return X

    def backward(self, dA):
        return dA


class ReLU(Layer):

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)


class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0) + self.alpha * np.minimum(X, 0)

    def backward(self, dA):
        dx = np.ones_like(self.X)
        dx[self.X < 0] = self.alpha
        return dA * dx


class Sigmoide(Layer):

    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)


class Tanh(Layer):
    
    def forward(self, X):
        self.A = np.tanh(X)
        return self.A

    def backward(self, dA):
        return dA * (1 - self.A**2)
    

def add_padding(X, padding):
    # X : (B, C, H, W)

    B, C, H, W = X.shape
    out = np.zeros((B, C, H + padding, W + padding), dtype=X.dtype)

    out[:, :, :H, :W] = X
    return out