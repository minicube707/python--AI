
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

def forward_propagation(X, W1, B1, W2, B2):

    Z1 = algebre(X, W1, B1)     #Z1 = X*W1 + B1
    A1 = sigmoide(Z1)

    Z2 = algebre(A1, W2, B2)    #Z2 = A1*W2 + B2
    A2 = sigmoide(Z2)

    return A1, A2

def backward_propagation(X, y, A1, A2, W1, B1, W2, B2, learning_rate):

    dZ2 = A2 - y                #dL/dZ2
    dW2 = dZ2 * A1              #dL/dW2
    db2 = dZ2                   #dL/db2

    dA1 = dZ2 * W2              #dL/dA1
    dZ1 = dA1 * A1 * (1 - A1)   #dL/dZ1
    dW1 = dZ1 * X               #dL/dW1
    db1 = dZ1                   #dL/db1

    W1 -= dW1 * learning_rate
    B1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    B2 -= db2 * learning_rate
    return W1, B1, W2, B2

#INITIALISATION
X = np.array([0, 1])
y = np.array([1, 1])

learning_rate = 0.01
nb_iteraton = 30_000

W1 = np.random.rand(1) * 2 - 1
B1 = np.random.rand(1) * 2 - 1
W2 = np.random.rand(1) * 2 - 1
B2 = np.random.rand(1) * 2 - 1

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W1_log, B1_log = [], []
W2_log, B2_log = [], []

#PREMIER PASSAGE
print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W1:':<6} {W1[0]:>10.6f}   {'B1:':<6} {B1[0]:>10.6f}")
print(f"{'W2:':<6} {W2[0]:>10.6f}   {'B2:':<6} {B2[0]:>10.6f}")

A1, A2 = forward_propagation(X, W1, B1, W2, B2)

print("Loss", log_loss(A2, y))
print("ACTIVATION", A2)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.size):

        #Foreward propagation
        A1, A2 = forward_propagation(X[i], W1, B1, W2, B2)

        if (j % 50 == 0):
            log.append(log_loss(A2, y[i]))
            dx_log.append(dx_log_loss(y[i], A2))

        # Sauvegarde des poids et biais
        W1_log.append(W1.copy())
        B1_log.append(B1.copy())
        W2_log.append(W2.copy())
        B2_log.append(B2.copy())

        #Backpropagation
        W1, B1, W2, B2 = backward_propagation(X[i], y[i], A1, A2, W1, B1, W2, B2, learning_rate)

A1, A2 = forward_propagation(X, W1, B1, W2, B2)
print("")
print(f"{'W1:':<6} {W1[0]:>10.6f}   {'B1:':<6} {B1[0]:>10.6f}")
print(f"{'W2:':<6} {W2[0]:>10.6f}   {'B2:':<6} {B2[0]:>10.6f}")
print("Loss final ", log[-1])
print("y: ", y)
print("ACTIVATION final", A2)

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
plt.plot(W1_log, label='W1')
plt.plot(B1_log, label='B1')
plt.plot(W2_log, label='W2')
plt.plot(B2_log, label='B2')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()

#DEUXIEME PASSAGE
y = np.array([0, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W1:':<6} {W1[0]:>10.6f}   {'B1:':<6} {B1[0]:>10.6f}")
print(f"{'W2:':<6} {W2[0]:>10.6f}   {'B2:':<6} {B2[0]:>10.6f}")

A1, A2 = forward_propagation(X, W1, B1, W2, B2)

print("Loss", log_loss(A2, y))
print("ACTIVATION", A2)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.size):

        #Foreward propagation
        A1, A2 = forward_propagation(X[i], W1, B1, W2, B2)

        if (j % 50 == 0):
            log.append(log_loss(A2, y[i]))
            dx_log.append(dx_log_loss(y[i], A2))

        # Sauvegarde des poids et biais
        W1_log.append(W1.copy())
        B1_log.append(B1.copy())
        W2_log.append(W2.copy())
        B2_log.append(B2.copy())

        #Backpropagation
        W1, B1, W2, B2 = backward_propagation(X[i], y[i], A1, A2, W1, B1, W2, B2, learning_rate)


A1, A2 = forward_propagation(X, W1, B1, W2, B2)
print("")
print(f"{'W1:':<6} {W1[0]:>10.6f}   {'B1:':<6} {B1[0]:>10.6f}")
print(f"{'W2:':<6} {W2[0]:>10.6f}   {'B2:':<6} {B2[0]:>10.6f}")
print("Loss final ", log_loss(A2, y))
print("y: ", y)

print("ACTIVATION final", A2)

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
plt.plot(W1_log, label='W1')
plt.plot(B1_log, label='B1')
plt.plot(W2_log, label='W2')
plt.plot(B2_log, label='B2')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()