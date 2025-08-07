
import  numpy as np
from Sklearn_tools import accuracy_score
from Mathematical_function import softmax
from Propagation import forward_propagation

"""
============================
Evaluation Metrics Function
============================
"""

def dx_log_loss(y_true, y_pred):
    y_pred = np.mean(y_pred, axis=1)
    return -1/y_true.size * np.sum((y_true)/(y_pred) - (1 - y_true)/(1 - y_pred))

def learning_progression(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN):
    _, activation_DNN = forward_propagation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
    A = softmax(activation_DNN["A" + str(C_CNN)].T)
    return A.T

def predict(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN):
    _, activation_DNN = forward_propagation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
    Af = activation_DNN["A" + str(C_CNN)]
    return Af >= 0.5

def log_loss(A, y):

    A = np.mean(A, axis=1)
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  1/y.size * np.sum( -y * np.log(A + epsilon) - (1-y)*np.log(1-A + epsilon))

def verification(X, y, activation, loss, accu, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN):

    loss = np.append(loss, log_loss(activation, y))
    y_pred = predict(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
    y_pred = np.mean(y_pred, axis=1)
    current_accuracy = accuracy_score(y.flatten(), y_pred.flatten()) 
    accu = np.append(accu, current_accuracy)
    return loss, accu
