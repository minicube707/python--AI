
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred - (1 - y_true)/(1 - y_pred)

def algebre(x, a, b):
    return a* x  + b

def sigmoide(X):
    return 1/(1 + np.exp(-X))

def initialisation():
    W11 = np.random.rand(1) * 2 - 1
    B11 = np.random.rand(1) * 2 - 1
    W12 = np.random.rand(1) * 2 - 1
    B12 = np.random.rand(1) * 2 - 1
    W21 = np.random.rand(1) * 2 - 1
    W22 = np.random.rand(1) * 2 - 1
    B21 = np.random.rand(1) * 2 - 1
    return W11, B11, W12, B12, W21, W22, B21

def forward_propagation(X, W11, B11, W12, B12, W21, W22, B21):

    Z11 = algebre(X, W11, B11)     #Z11 = X * W11 + B11
    A11 = sigmoide(Z11)

    Z12 = algebre(X, W12, B12)     #Z12 = X * W12 + B12
    A12 = sigmoide(Z12)

    Z21 = A11 * W21 + A12 * W22 + B21
    A21 = sigmoide(Z21)

    return A11, A12, A21

def backward_propagation(X, y, A11, A12, A21, W11, B11, W12, B12, W21, W22, B21, learning_rate):

    dZ21 = A21 - y                  #dL/dZ21
    dW22 = dZ21 * A12               #dL/dW22
    dW21 = dZ21 * A11               #dL/dW21
    db21 = dZ21                     #dL/db21


    dA12 = dZ21 * W22               #dL/dA12
    dA11 = dZ21 * W21               #dL/dA11

    dZ12 = dA12 * A12 * (1 - A12)   #dL/dZ12
    dZ11 = dA11 * A11 * (1 - A11)   #dL/dZ11

    dW12 = dZ12 * X                 #dL/dW12
    dW11 = dZ11 * X                 #dL/dW11
    db12 = dZ12                     #dL/db12
    db11 = dZ11                     #dL/db11

    W11 -= dW11 * learning_rate
    B11 -= db11 * learning_rate
    W12 -= dW12 * learning_rate
    B12 -= db12 * learning_rate
    W21 -= dW21 * learning_rate
    W22 -= dW22 * learning_rate
    B21 -= db21 * learning_rate


    return W11, B11, W12, B12, W21, W22, B21

#INITIALISATION
X = np.array([0, 1])
y = np.array([0, 1])

learning_rate = 0.01
nb_iteraton = 30_000

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W11_log, B11_log = [], []
W12_log, B12_log = [], []
W21_log, W22_log, B21_log = [], [], []

#PREMIER PASSAGE
W11, B11, W12, B12, W21, W22, B21 = initialisation()
A11, A12, A21 = forward_propagation(X, W11, B11, W12, B12, W21, W22, B21)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}")
print(f"{'B21:':<6} {B21[0]:>10.6f}")
print("Loss", log_loss(A21, y))
print("ACTIVATION", A21)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.size):

        #Foreward propagation
        A11, A12, A21 = forward_propagation(X[i], W11, B11, W12, B12, W21, W22, B21)

        if (j % 50 == 0):
            log.append(log_loss(A21, y[i]))
            dx_log.append(dx_log_loss(y[i], A21))

        # Sauvegarde des poids et biais
        W11_log.append(W11.copy())
        B11_log.append(B11.copy())
        W12_log.append(W12.copy())
        B12_log.append(B12.copy())
        W21_log.append(W21.copy())
        W22_log.append(W22.copy())
        B21_log.append(B21.copy())

        #Backpropagation
        W11, B11, W12, B12, W21, W22, B21 = backward_propagation(X[i], y[i], A11, A12, A21, W11, B11, W12, B12, W21, W22, B21, learning_rate)


A11, A12, A21 = forward_propagation(X, W11, B11, W12, B12, W21, W22, B21)

print("")
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}")
print(f"{'B21:':<6} {B21[0]:>10.6f}")
print("Loss final ", log_loss(A21, y))
print("y: ", y)
print("ACTIVATION final", A21)

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
plt.plot(W11_log, label='W11')
plt.plot(B11_log, label='B11')
plt.plot(W12_log, label='W12')
plt.plot(B12_log, label='B12')
plt.plot(W21_log, label='W21')
plt.plot(W22_log, label='W22')
plt.plot(B21_log, label='B21')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()


#DEUXIEME PASSAGE
W11, B11, W12, B12, W21, W22, B21 = initialisation()
A11, A12, A21 = forward_propagation(X, W11, B11, W12, B12, W21, W22, B21)
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}")
print(f"{'B21:':<6} {B21[0]:>10.6f}")
print("Loss", log_loss(A21, y))
print("ACTIVATION", A21)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.size):

        #Foreward propagation
        A11, A12, A21 = forward_propagation(X[i], W11, B11, W12, B12, W21, W22, B21)

        if (j % 50 == 0):
            log.append(log_loss(A21, y[i]))
            dx_log.append(dx_log_loss(y[i], A21))

        # Sauvegarde des poids et biais
        W11_log.append(W11.copy())
        B11_log.append(B11.copy())
        W12_log.append(W12.copy())
        B12_log.append(B12.copy())
        W21_log.append(W21.copy())
        W22_log.append(W22.copy())
        B21_log.append(B21.copy())

        #Backpropagation
        W11, B11, W12, B12, W21, W22, B21 = backward_propagation(X[i], y[i], A11, A12, A21, W11, B11, W12, B12, W21, W22, B21, learning_rate)

A11, A12, A21 = forward_propagation(X, W11, B11, W12, B12, W21, W22, B21)
print("")
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}")
print(f"{'B21:':<6} {B21[0]:>10.6f}")
print("Loss final ", log_loss(A21, y))
print("y: ", y)
print("ACTIVATION final", A21)

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
plt.plot(W11_log, label='W11')
plt.plot(B11_log, label='B11')
plt.plot(W12_log, label='W12')
plt.plot(B12_log, label='B12')
plt.plot(W21_log, label='W21')
plt.plot(W22_log, label='W22')
plt.plot(B21_log, label='B21')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()