
import  numpy as np
from Deep_Neuron_Network import foward_propagation_DNN
from Sklearn_tools import accuracy_score
from Mathematical_function import softmax

"""
============================
Evaluation Metrics Function
============================
"""

def dx_log_loss(y_true, y_pred):
    return -1/y_true.size * np.sum((y_true)/(y_pred) - (1 - y_true)/(1 - y_pred))

def learning_progression(X, parametres):
    activations = foward_propagation_DNN(X, parametres)
    C = len(parametres) // 2
    A = softmax(activations["A" + str(C)].T)
    return A.T

def predict(X, parametres):
    activations = foward_propagation_DNN(X, parametres)
    C = len(parametres) // 2
    Af = activations["A" + str(C)]
    return Af >= 0.5

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  1/y.size * np.sum( -y * np.log(A + epsilon) - (1-y)*np.log(1-A + epsilon))

def verification(X, y, activation, parametres, loss, accu):

    loss = np.append(loss, log_loss(activation, y))
    y_pred = predict(X, parametres)
    current_accuracy = accuracy_score(y.flatten(), y_pred.flatten()) 
    accu = np.append(accu, current_accuracy)
    return loss, accu
