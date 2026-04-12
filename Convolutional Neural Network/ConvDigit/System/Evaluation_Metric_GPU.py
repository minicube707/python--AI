
import cupy as cp

from .Layer import Layer

"""
============================
Evaluation Metrics Function
============================
"""

class BinaryCrossEntropy_GPU(Layer):

    def __init__(self):
        self.class_ = "BinaryCrossEntropy"
        
    def forward(self, y_true, y_pred):
        
        self.y_pred = y_pred
        self.y_true = y_true
        
        eps = 1e-12
        self.y_pred_clipped = cp.clip(y_pred, eps, 1 - eps)
        return -cp.mean(y_true * cp.log(self.y_pred_clipped) + (1 - y_true) * cp.log(1 - self.y_pred_clipped))

    def backward(self):
        m = self.y_true.shape[0]
        return - (self.y_true / self.y_pred_clipped - (1 - self.y_true) / (1 - self.y_pred_clipped)) / m
    

class CrossEntropyLoss_GPU(Layer):

    def __init__(self):
        self.class_ = "CrossEntropyLoss"
        
    def forward(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true

        eps = 1e-12
        self.y_pred_clipped = cp.clip(y_pred, eps, 1 - eps)

        return -cp.sum(y_true * cp.log(self.y_pred_clipped)) / y_pred.shape[0]

    def backward(self):
        m = self.y_pred.shape[0]
        return - (self.y_true / self.y_pred_clipped) / m


class MSE_GPU(Layer):

    def __init__(self):
        self.class_ = "MSE"
        
    def forward(self, y_true, y_pred):
        self.diff = y_pred - y_true
        return cp.mean(self.diff ** 2)

    def backward(self):
        return 2 * self.diff / self.diff.shape[0]
    

def accuracy_score_gpu(y_true, y_pred):
   
    if y_true.ndim == 1:
        acc = cp.mean(y_true == (y_pred >= 0.5))
    
    else:
        acc = cp.mean(cp.argmax(y_true, axis=1) == cp.argmax(y_pred, axis=1))

    return acc.item()


def confidence_score_gpu(y_true, y_pred):
    
    if y_true.ndim == 1:
        probs = cp.where(y_true == 1, y_pred, 1 - y_pred)
        
    else:
        probs = y_pred[
            cp.arange(y_pred.shape[0]),
            cp.argmax(y_true, axis=1)
        ]

    return cp.mean(probs).item()