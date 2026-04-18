
from .Layer import Layer

from .Mathematical_function_CPU import ReLU_CPU, LeakyReLU_CPU, Sigmoide_CPU, Tanh_CPU, Softmax_CPU
from .Mathematical_function_GPU import ReLU_GPU, LeakyReLU_GPU, Sigmoide_GPU, Tanh_GPU, Softmax_GPU

class Linear(Layer):

    def forward(self, X, *args):
        return X

    def backward(self, dA):
        return dA
    
class Softmax:
    
    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return Softmax_CPU()
        
        elif support == "gpu":
            return Softmax_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")


class ReLU:

    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return ReLU_CPU()
        
        elif support == "gpu":
            return ReLU_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")


class LeakyReLU:

    @staticmethod
    def add_layer(alpha, support):
        
        support = support.lower()

        if support == "cpu":
            return LeakyReLU_CPU(alpha)
        
        elif support == "gpu":
            return LeakyReLU_GPU(alpha)
        
        else:
            raise ValueError(f"Unknown support: {support}")


class Sigmoide:

    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return Sigmoide_CPU()
        
        elif support == "gpu":
            return Sigmoide_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")

    
class Tanh:

    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return Tanh_CPU()
        
        elif support == "gpu":
            return Tanh_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")


def remove_padding(X, padding):
    # X : (B, C, H, W)

    if padding == 0:
        return X

    pad_top = padding // 2
    pad_bottom = padding - pad_top

    pad_left = padding // 2
    pad_right = padding - pad_left

    B, C, H, W = X.shape

    return X[
        :,
        :,
        pad_top:H - pad_bottom,
        pad_left:W - pad_right
    ]