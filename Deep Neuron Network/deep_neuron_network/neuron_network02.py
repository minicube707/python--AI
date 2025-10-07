
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

#INITIALISATION
X = np.array([0, 1])
y = np.array([0, 1])

learning_rate = 0.001
nb_iteraton = 30_000

W = np.random.rand(1) * 2 - 1
B = np.random.rand(1) * 2 - 1

log = []
dx_log = []


#PREMIER PASSAGE
print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.size):

        #Foreward propagation
        Z = algebre(X[i], W, B)
        A = sigmoide(Z)

        if (j % 50 == 0):
            log.append(log_loss(A, y[i]))
            dx_log.append(dx_log_loss(y[i], A))

        #Backpropagation
        dA = A - y[i]
        dz = dA * (1 - dA)
        dW = X[i] * dz
        db = dz
        W -= dW * learning_rate
        B -= db * learning_rate


print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log[-1])
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes

# Premier plot
axes[0].plot(log)
axes[0].set_title("log")

# Deuxième plot
axes[1].plot(dx_log)
axes[1].set_title("dx_log")

# Afficher le tout
plt.tight_layout()
plt.show()


#DEUXIEME PASSAGE
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for j in tqdm(range(3*nb_iteraton)):
    
    for i in range(X.size):

        #Foreward propagation
        Z = algebre(X[i], W, B)
        A = sigmoide(Z)

        if (j % 50 == 0):
            log.append(log_loss(A, y[i]))
            dx_log.append(dx_log_loss(y[i], A))

        #Backpropagation
        dA = A - y[i]
        dW = X[i] * dA
        db = dA
        W -= dW * learning_rate
        B -= db * learning_rate


print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log[-1])
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes

# Premier plot
axes[0].plot(log)
axes[0].set_title("log")

# Deuxième plot
axes[1].plot(dx_log)
axes[1].set_title("dx_log")

# Afficher le tout
plt.tight_layout()
plt.show()