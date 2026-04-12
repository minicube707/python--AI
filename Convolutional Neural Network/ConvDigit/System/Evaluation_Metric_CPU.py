
import numpy as np

from .Layer import Layer

"""
============================
Evaluation Metrics Function
============================
"""

class BinaryCrossEntropy_CPU(Layer):

    def __init__(self):
        self.class_ = "BinaryCrossEntropy"
        
    def forward(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true
        
        
        eps = 1e-12
        self.y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        return - np.mean(y_true * np.log(self.y_pred_clipped) + (1 - y_true) * np.log(1 - self.y_pred_clipped))

    def backward(self):
        m = self.y_true.shape[0]
        return -(self.y_true / self.y_pred_clipped - (1 - self.y_true) / (1 - self.y_pred_clipped)) / m

class CrossEntropyLoss_CPU(Layer):

    def __init__(self):
        self.class_ = "CrossEntropyLoss"
        
    def forward(self, y_true, y_pred):

        self.y_pred = y_pred
        self.y_true = y_true

        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]

    def backward(self):
        m = self.y_pred.shape[0]
        return - (self.y_true / self.y_pred) / m
    

class MSE_CPU(Layer):

    def __init__(self):
        self.class_ = "MSE"
        
    def forward(self, y_true, y_pred):

        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true
        return  np.mean(self.diff ** 2)

    def backward(self):
        return 2 * (self.diff) / self.y_true.shape[0]
    

def accuracy_score_cpu(y_true, y_pred):

    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
        
    if y_true.ndim == 1:
        y_pred_labels = (y_pred >= 0.5).astype(int)
        return np.mean(y_true == y_pred_labels)
    
    else:  
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_true_labels == y_pred_labels)


def confidence_score_cpu(y_true, y_pred):

    if y_true.ndim == 1:
        true_class_probs = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        
    else:
        true_class_probs = y_pred[np.arange(y_true.shape[0]), np.argmax(y_true, axis=1)]

    return np.mean(true_class_probs)