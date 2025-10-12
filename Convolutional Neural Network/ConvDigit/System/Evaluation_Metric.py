
import  numpy as np

from .Mathematical_function import softmax
from .Propagation import forward_propagation

"""
============================
Evaluation Metrics Function
============================
"""

def dx_log_loss(y_true, y_pred):
    return - 1 / y_true.size * np.sum(y_true - y_pred)

def activation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, dimensions_DNN, C_CNN, C_DNN, alpha):
    _, activation_DNN = forward_propagation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN, dimensions_DNN, C_DNN, alpha)
    A = softmax(activation_DNN["A" + str(C_DNN)].T)
    return A.T

def log_loss(y_true, y_pred):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return - 1 / y_true.size * np.sum(y_true * np.log(y_pred))

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.argmax(y_true) == np.argmax(y_pred)

def confidence_score(y_pred):
    return (np.max(y_pred))