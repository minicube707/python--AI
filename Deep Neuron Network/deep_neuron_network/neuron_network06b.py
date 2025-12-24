
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred + (1 - y_true)/(1 - y_pred)

def algebre(x, a, b):
    return a* x  + b

def sigmoide(X):
    return 1/(1 + np.exp(-X))

def relu(X, alpha):
    return np.where(X < 0, alpha*X, X)

def dx_relu(X, alpha):
    return np.where(X < 0, alpha, 1)

def initialisation():
    W11 = np.random.rand(1) * 2 - 1
    B11 = np.random.rand(1) * 2 - 1
    W12 = np.random.rand(1) * 2 - 1
    B12 = np.random.rand(1) * 2 - 1

    W21 = np.random.rand(1) * 2 - 1
    W22 = np.random.rand(1) * 2 - 1
    W23 = np.random.rand(1) * 2 - 1
    W24 = np.random.rand(1) * 2 - 1
    B21 = np.random.rand(1) * 2 - 1
    B22 = np.random.rand(1) * 2 - 1

    W31 = np.random.rand(1) * 2 - 1
    W32 = np.random.rand(1) * 2 - 1
    B31 = np.random.rand(1) * 2 - 1

    return W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31

def forward_propagation(X, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha):

    #Layer1
    Z11 = algebre(X, W11, B11)     #Z11 = X * W11 + B11
    A11 = relu(Z11, alpha)
    Z12 = algebre(X, W12, B12)     #Z12 = X * W12 + B12
    A12 = relu(Z12, alpha)

    #Layer2
    Z21 = A11 * W21 + A12 * W22 + B21
    A21 = relu(Z21, alpha)
    Z22 = A11 * W23 + A12 * W24 + B22
    A22 = relu(Z22, alpha)

    #Layer3
    Z31 = A21 * W31 + A22 * W32 + B31
    A31 = sigmoide(Z31)

    return A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31

def backward_propagation(X, y, A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, learning_rate, alpha):

    #Layer3
    dZ31 = A31 - y                  #dL/dZ31
    dW31 = dZ31 * A21               #dL/dW31
    dW32 = dZ31 * A22               #dL/dW32
    db31 = dZ31                     #dL/db31

    #Layer2
    dA21 = dZ31 * W31                       #dL/dA21
    dA22 = dZ31 * W32                       #dL/dA22
    dZ21 = dA21 * dx_relu(Z21, alpha)       #dL/dZ21
    dZ22 = dA22 * dx_relu(Z22, alpha)       #dL/dZ22
    dW21 = dZ21 * A11                       #dL/dW21
    dW22 = dZ21 * A12                       #dL/dW22
    dW23 = dZ22 * A11                       #dL/dW23
    dW24 = dZ22 * A12                       #dL/dW24
    db21 = dZ21                             #dL/db21
    db22 = dZ22                             #dL/db22

    #Layer1
    dA11 = dZ21 * W21 + dZ22 * W22          #dL/dA11
    dA12 = dZ21 * W23 + dZ22 * W24          #dL/dA12
    dZ11 = dA11 * dx_relu(Z11, alpha)       #dL/dZ11
    dZ12 = dA12 * dx_relu(Z12, alpha)       #dL/dZ12
    dW11 = dZ11 * X                         #dL/dW11
    dW12 = dZ12 * X                         #dL/dW12
    db11 = dZ11                             #dL/db11
    db12 = dZ12                             #dL/db12

    #Layer1
    W11 -= dW11 * learning_rate
    W12 -= dW12 * learning_rate
    B11 -= db11 * learning_rate
    B12 -= db12 * learning_rate

    #Layer2
    W21 -= dW21 * learning_rate
    W22 -= dW22 * learning_rate
    W23 -= dW23 * learning_rate
    W24 -= dW24 * learning_rate
    B21 -= db21 * learning_rate
    B22 -= db22 * learning_rate

    #Layer3
    W31 -= dW31 * learning_rate
    W32 -= dW32 * learning_rate
    B31 -= db31 * learning_rate

    return W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31

#INITIALISATION
X = np.array([0, 1])
y = np.array([0, 1])

learning_rate = 0.1
nb_iteraton = 15_000
alpha = 0.02

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W11_log, B11_log = [], []
W12_log, B12_log = [], []
W21_log, W22_log, W23_log, W24_log, B21_log, B22_log = [], [], [], [], [], []
W31_log, W32_log, B31_log = [], [], []

#PREMIER PASSAGE
W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31 = initialisation()
A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31 = forward_propagation(X, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W24:':<6} {W24[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W31:':<6} {W31[0]:>10.6f}   {'W32:':<6} {W32[0]:>10.6f}   {'B31:':<6} {B31[0]:>10.6f}")
print("Loss", log_loss(A31, y))
print("ACTIVATION", A31)
print("")

for j in tqdm(range(nb_iteraton)):
    
    sum_log = 0
    sum_dx_log = 0

    for i in range(X.size):

        #Foreward propagation
        A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31 = forward_propagation(X[i], W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha)

        sum_log += log_loss(A31, y[i])
        sum_dx_log += dx_log_loss(y[i], A31)

        # Sauvegarde des poids et biais
        W11_log.append(W11.copy())
        B11_log.append(B11.copy())
        W12_log.append(W12.copy())
        B12_log.append(B12.copy())
        W21_log.append(W21.copy())
        W22_log.append(W22.copy())
        B21_log.append(B21.copy())
        W23_log.append(W23.copy())
        W24_log.append(W24.copy())
        B22_log.append(B22.copy())
        W31_log.append(W31.copy())
        W32_log.append(W32.copy())
        B31_log.append(B31.copy())

        #Backpropagation
        W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31 = backward_propagation(X[i], y[i], A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, learning_rate, alpha)

    log.append(sum_log)
    dx_log.append(sum_dx_log)

#Prediction final
A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31 = forward_propagation(X, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha)

print("")
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W24:':<6} {W24[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W31:':<6} {W31[0]:>10.6f}   {'W32:':<6} {W32[0]:>10.6f}   {'B31:':<6} {B31[0]:>10.6f}")
print("Loss final ", log_loss(A31, y))
print("y: ", y)
print("ACTIVATION final", A31)

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
plt.plot(W23_log, label='W23')
plt.plot(W24_log, label='W24')
plt.plot(B22_log, label='B22')
plt.plot(W31_log, label='W31')
plt.plot(W32_log, label='W32')
plt.plot(B31_log, label='B31')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()


#DEUXIEME PASSAGE
A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31 = forward_propagation(X, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha)
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W24:':<6} {W24[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W31:':<6} {W31[0]:>10.6f}   {'W32:':<6} {W32[0]:>10.6f}   {'B31:':<6} {B31[0]:>10.6f}")
print("Loss", log_loss(A31, y))
print("ACTIVATION", A31)
print("")

for j in tqdm(range(nb_iteraton)):
    
    sum_log = 0
    sum_dx_log = 0

    for i in range(X.size):

        #Foreward propagation
        A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31 = forward_propagation(X[i], W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha)

        sum_log += log_loss(A31, y[i])
        sum_dx_log += dx_log_loss(y[i], A31)

        # Sauvegarde des poids et biais
        W11_log.append(W11.copy())
        B11_log.append(B11.copy())
        W12_log.append(W12.copy())
        B12_log.append(B12.copy())
        W21_log.append(W21.copy())
        W22_log.append(W22.copy())
        B21_log.append(B21.copy())
        W23_log.append(W23.copy())
        W24_log.append(W24.copy())
        B22_log.append(B22.copy())
        W31_log.append(W31.copy())
        W32_log.append(W32.copy())
        B31_log.append(B31.copy())

        #Backpropagation
        W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31 = backward_propagation(X[i], y[i], A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, learning_rate, alpha)

    log.append(sum_log)
    dx_log.append(sum_dx_log)

#Prediction final
A11, A12, A21, A22, A31, Z11, Z12, Z21, Z22, Z31 = forward_propagation(X, W11, W12, B11, B12, W21, W22, W23, W24, B21, B22, W31, W32, B31, alpha)


print("")
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")
print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W22:':<6} {W22[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W24:':<6} {W24[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W31:':<6} {W31[0]:>10.6f}   {'W32:':<6} {W32[0]:>10.6f}   {'B31:':<6} {B31[0]:>10.6f}")
print("Loss final ", log_loss(A31, y))
print("y: ", y)
print("ACTIVATION final", A31)

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
plt.plot(W23_log, label='W23')
plt.plot(W24_log, label='W24')
plt.plot(B22_log, label='B22')
plt.plot(W31_log, label='W31')
plt.plot(W32_log, label='W32')
plt.plot(B31_log, label='B31')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()