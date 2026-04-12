
from .Layer import Layer

from .Layers_CPU import MaxPooling_CPU, Convolution_CPU, BatchNorm_CPU, Dropout_CPU, Dense_CPU
from .Layers_GPU import MaxPooling_GPU, Convolution_GPU, BatchNorm_GPU, Dropout_GPU, Dense_GPU


class Flatten(Layer):

    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dZ):
        return dZ.reshape(self.input_shape)
    
class Block(Layer):

    def __init__(self, dense, batchnorm, activation, dropout):

        self.dense = dense
        self.batchnorm = batchnorm
        self.activation = activation
        self.dropout = dropout

    def forward(self, X, training=True):

        Z = self.dense.forward(X)
        Z = self.batchnorm.forward(Z, training)
        A = self.activation.forward(Z)
        A = self.dropout.forward(A, training)

        return A

    def backward(self, dZ):

        dA = self.dropout.backward(dZ)
        dZ = self.activation.backward(dA)
        dZ = self.batchnorm.backward(dZ)
        dZ = self.dense.backward(dZ)

        return dZ
    
class MaxPooling:

    @staticmethod
    def add_layer(k_size, stride, padding, support):
        
        support = support.lower()

        if support == "cpu":
            return MaxPooling_CPU(k_size, stride, padding)
        
        elif support == "gpu":
            return MaxPooling_GPU(k_size, stride, padding)
        
        else:
            raise ValueError(f"Unknown support: {support}")


class Convolution:

    @staticmethod
    def add_layer(nb_kernel, nb_layer, k_size, stride, o_size, padding, support):
        
        support = support.lower()

        if support == "cpu":
            return Convolution_CPU(nb_kernel, nb_layer, k_size, stride, o_size, padding)
        
        elif support == "gpu":
            return Convolution_GPU(nb_kernel, nb_layer, k_size, stride, o_size, padding)
        
        else:
            raise ValueError(f"Unknown support: {support}")


class BatchNorm:
    
    @staticmethod
    def add_layer(n_features, support):
        
        support = support.lower()

        if support == "cpu":
            return BatchNorm_CPU(n_features)
        
        elif support == "gpu":
            return BatchNorm_GPU(n_features)
        
        else:
            raise ValueError(f"Unknown support: {support}")


class Dropout:
    
    @staticmethod
    def add_layer(dropout_per, support):
        
        support = support.lower()

        if support == "cpu":
            return Dropout_CPU(dropout_per)
        
        elif support == "gpu":
            return Dropout_GPU(dropout_per)
        
        else:
            raise ValueError(f"Unknown support: {support}")
        

class Dense:
    
    @staticmethod
    def add_layer(nb_activation, nb_neuron, support):
        
        support = support.lower()

        if support == "cpu":
            return Dense_CPU(nb_activation, nb_neuron)
        
        elif support == "gpu":
            return Dense_GPU(nb_activation, nb_neuron)
        
        else:
            raise ValueError(f"Unknown support: {support}")
        