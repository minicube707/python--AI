
import numpy as np

from .Mathematical_function import softmax
from .Deep_Neuron_Network import foward_propagation_DNN
"""
============================
Evaluation Metrics Function
============================
"""

def dx_log_loss(y_true, y_pred):
    return - np.mean(np.sum(y_true - y_pred))

def activation(X, parametres_DNN, dimensions_DNN, C_DNN, alpha):
    activation_DNN = foward_propagation_DNN(X, parametres_DNN, dimensions_DNN, C_DNN, alpha)
    A = softmax(activation_DNN["A" + str(C_DNN)])
    return A.flatten()

def log_loss(y_true, y_pred):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return - np.mean(np.sum(y_true * np.log(y_pred + epsilon)))


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.argmax(y_true) == np.argmax(y_pred)

def confidence_score(y_true, y_pred):
    return (y_pred[np.argmax(y_true)])