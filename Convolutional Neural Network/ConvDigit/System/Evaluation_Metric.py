
import  numpy as np

"""
============================
Evaluation Metrics Function
============================
"""

class CrossEntropyLoss:

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true
        
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        return loss

    def backward(self):
        m = self.y_pred.shape[0]
        return -(self.y_true / self.y_pred) / m
    

class MSE:

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
    

def log_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))


def dx_log_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean((y_true / y_pred - (1-y_true) / (1-y_pred)))


def accuracy_score(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(y_true_labels == y_pred_labels)


def confidence_score(y_true, y_pred):
    true_class_probs = y_pred[np.arange(y_true.shape[0]), np.argmax(y_true, axis=1)]
    return np.mean(true_class_probs)