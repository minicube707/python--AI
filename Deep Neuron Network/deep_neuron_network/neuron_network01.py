
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred - (1 - y_true)/(1 - y_pred)

def algebre(x, a, b):
    return a * x + b

def sigmoide(X):
    return 1/(1 + np.exp(-X))

X = 1
y = 1

learning_rate = 0.1
nb_iteraton = 5000

W = np.random.rand(1) * 2 - 1
B = np.random.rand(1) * 2 - 1

log = []
dx_log = []

print("")
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for _ in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    #Backpropagation
    dA = A - y
    dW = X * dA
    db = dA
    W -= dW * learning_rate
    B -= db * learning_rate

print("")
print("W: ", W)
print("B: ", B)

print("Loss final ", log[-1])
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

plt.figure()
plt.plot(log)
plt.show()

plt.figure()
plt.plot(dx_log)
plt.show()

