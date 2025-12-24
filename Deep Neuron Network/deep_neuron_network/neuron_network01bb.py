
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return y_true/y_pred + (1 - y_true)/(1 - y_pred)

def algebre(x, a, b):
    return a * x + b

def sigmoide(X):
    return 1/(1 + np.exp(-X))

#INITIALISATION
X = 1
y = 1

learning_rate = 0.1
nb_iteraton = 5000

W = np.random.rand(1) * 2 - 1
B = np.random.rand(1) * 2 - 1

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W_log, B_log = [], []

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

for _ in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dA = y - A      #dL/dZ
    dW = X * dA     #dL/dW
    db = dA         #dL/dB
    W += dW * learning_rate
    B += db * learning_rate

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()


# Créer une figure
plt.figure(figsize=(12, 8))
plt.plot(W_log, label='W')
plt.plot(B_log, label='B')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()

#DEUXIEME PASSAGE
y = 0
print("")
print("DEUXIEME apprentissage")
print("X: ", X)
print("y: ", y)
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

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    #Backpropagation
    dZ = y - A          #dL/dZ
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W += dW * learning_rate
    B += db * learning_rate

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()

# Créer une figure
plt.figure(figsize=(12, 8))
plt.plot(W_log, label='W')
plt.plot(B_log, label='B')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()