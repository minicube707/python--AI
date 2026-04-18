import cupy as cp

from .Layer import Layer

class Softmax_GPU(Layer):

    def __init__(self):
        self.class_ = "Softmax"
        
    def forward(self, X):
        # stabilité numérique
        X_shifted = X - cp.max(X, axis=1, keepdims=True)
        exp_X = cp.exp(X_shifted)
        self.out = exp_X / cp.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def backward(self, dY):
        dot = cp.sum(dY * self.out, axis=1, keepdims=True)
        dX = self.out * (dY - dot)
        return dX


class ReLU_GPU(Layer):

    def __init__(self):
        self.class_ = "ReLU"
        
    def forward(self, X):
        self.X = X
        return cp.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)


class LeakyReLU_GPU(Layer):

    def __init__(self):
        self.class_ = "LeakyReLU"
        
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, X):
        self.X = X
        return cp.maximum(X, 0) + self.alpha * cp.minimum(X, 0)

    def backward(self, dA):
        dx = cp.ones_like(self.X)
        dx[self.X < 0] = self.alpha
        return dA * dx


class Sigmoide_GPU(Layer):

    def __init__(self):
        self.class_ = "sigmoide"
        
    def forward(self, X):
        self.A = 1 / (1 + cp.exp(-X))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)


class Tanh_GPU(Layer):
    
    def __init__(self):
        self.class_ = "Tanh"
        
    def forward(self, X):
        self.A = cp.tanh(X)
        return self.A

    def backward(self, dA):
        return dA * (1 - self.A**2)

def add_padding_GPU(X, padding):
    # X : (B, C, H, W)

    if padding == 0:
        return X

    B, C, H, W = X.shape

    # Distribution of padding
    pad_top = padding // 2
    pad_bottom = padding - pad_top

    pad_left = padding // 2
    pad_right = padding - pad_left

    # Creation of the padded tensor
    out = cp.zeros(
        (B, C, H + pad_top + pad_bottom, W + pad_left + pad_right),
        dtype=X.dtype
    )

    # Placement of X in the right place
    out[:, :, pad_top:pad_top + H, pad_left:pad_left + W] = X

    return out
