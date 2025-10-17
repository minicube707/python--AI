
import numpy as np
from scipy.signal import correlate2d

def tanh(X):
    return np.tanh(X)

def dx_tanh(X):
    return (1 - X**2)
            
"""
============================
==========Fonction==========
============================
"""
"""
sigmoïde:
=========DESCRIPTION=========
Apply the sigmoide function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def sigmoide(X):
    return 1/(1 + np.exp(-X))


"""
relu:
=========DESCRIPTION=========
Apply the relu function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def relu(X, alpha):
    return np.where(X < 0, alpha*X, X)


"""
dx_sigmoïde:
=========DESCRIPTION=========
Apply the derivate sigmoide function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def dx_sigmoide(X):
    return X * (1 - X)

"""
dx_relu:
=========DESCRIPTION=========
Apply the derivative relu function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def dx_relu(X, alpha):
    return np.where(X < 0, alpha, 1)


"""
max_pooling:
=========DESCRIPTION=========
Return the max of each row of the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def max_pooling(X):
    a = np.int8(np.sqrt(X.shape[1]))
    return np.max(X, axis=2).reshape((X.shape[0], a, a))


"""
softmax:
=========DESCRIPTION=========
Apply the softmax function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def softmax(X):

    X = np.clip(X, -64, 64)
    X_max = np.max(X, axis=1, keepdims=True)
    e_x = np.exp(X - X_max)
    
    return e_x / np.sum(e_x, axis=1, keepdims=True)
