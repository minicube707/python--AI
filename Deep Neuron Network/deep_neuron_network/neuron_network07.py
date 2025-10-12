
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A21, A22, A23, A24, y):
    S1, S2, S3, S4 = sofmax(A21, A22, A23, A24)
    epsilon = 1e-15 #Pour empecher les log(0) = -inf

    R = 0
    R += - y[0] * np.log(S1 + epsilon)
    R += - y[1] * np.log(S2 + epsilon)
    R += - y[2] * np.log(S3 + epsilon)
    R += - y[3] * np.log(S4 + epsilon)
    return  R

def dx_log_loss(y_true, A21, A22, A23, A24):
    S1, S2, S3, S4 = sofmax(A21, A22, A23, A24)

    R = 0
    R += - y_true[0] / S1 
    R += - y_true[1] / S2
    R += - y_true[2] / S3 
    R += - y_true[3] / S4 
    return R


def sigmoide(X):
    return 1/(1 + np.exp(-X))

def sofmax(A21, A22, A23, A24):
    sum = np.exp(A21) + np.exp(A22) + np.exp(A23) + np.exp(A24)
    a = np.exp(A21) / sum
    b = np.exp(A22) / sum
    c = np.exp(A23) / sum
    d = np.exp(A24) / sum
    return a, b, c, d

def initialisation():
    W11 = np.random.rand(1) * 2 - 1
    W13 = np.random.rand(1) * 2 - 1
    B11 = np.random.rand(1) * 2 - 1

    W12 = np.random.rand(1) * 2 - 1
    W14 = np.random.rand(1) * 2 - 1
    B12 = np.random.rand(1) * 2 - 1

    W21 = np.random.rand(1) * 2 - 1
    W25 = np.random.rand(1) * 2 - 1
    B21 = np.random.rand(1) * 2 - 1

    W22 = np.random.rand(1) * 2 - 1
    W26 = np.random.rand(1) * 2 - 1
    B22 = np.random.rand(1) * 2 - 1
    
    W23 = np.random.rand(1) * 2 - 1
    W27 = np.random.rand(1) * 2 - 1
    B23 = np.random.rand(1) * 2 - 1

    W24 = np.random.rand(1) * 2 - 1
    W28 = np.random.rand(1) * 2 - 1
    B24 = np.random.rand(1) * 2 - 1

    return W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24

def forward_propagation(X, W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24):

        X1 = X[0]
        X2 = X[1]

        Z11 = X1 * W11 + X2 * W13 + B11
        A11 = sigmoide(Z11)
        Z12 = X1 * W12 + X2 * W14 + B12
        A12 = sigmoide(Z12)

        Z21 = A11 * W21 + A12 * W25 + B21
        A21 = sigmoide(Z21)
        Z22 = A11 * W22 + A12 * W26 + B22
        A22 = sigmoide(Z22)
        Z23 = A11 * W23 + A12 * W27 + B23
        A23 = sigmoide(Z23)
        Z24 = A11 * W24 + A12 * W28 + B24
        A24 = sigmoide(Z24)

        return A11, A12, A21, A22, A23, A24

def backward_propagation(X, y, A11, A12, A21, A22, A23, A24, W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24, learning_rate):

    X1 = X[0]
    X2 = X[1]

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    y4 = y[3]

    dZ21 = A21 - y1   #dL/dZ21
    dZ22 = A22 - y2   #dL/dZ22
    dZ23 = A23 - y3   #dL/dZ23
    dZ24 = A24 - y4   #dL/dZ24

    dW21 = dZ21 * A11               #dL/dW21
    dW22 = dZ22 * A11               #dL/dW22
    dW23 = dZ23 * A11               #dL/dW23
    dW24 = dZ24 * A11               #dL/dW24

    dW25 = dZ21 * A12               #dL/dW25
    dW26 = dZ22 * A12               #dL/dW26
    dW27 = dZ23 * A12               #dL/dW27
    dW28 = dZ24 * A12               #dL/dW28

    db21 = dZ21                     #dL/db21
    db22 = dZ22                     #dL/db22
    db23 = dZ23                     #dL/db23
    db24 = dZ24                     #dL/db24

    dA11 = dZ21 * W21 + dZ22 * W22 + dZ23 * W23 + dZ24 * W24    #dL/dA11
    dA12 = dZ21 * W25 + dZ22 * W26 + dZ23 * W27 + dZ24 * W28    #dL/dA12

    dZ12 = dA12 * A12 * (1 - A12)   #dL/dZ12
    dZ11 = dA11 * A11 * (1 - A11)   #dL/dZ11

    dW11 = dZ11 * X1                 #dL/dW11
    dW12 = dZ12 * X1                 #dL/dW12
    dW13 = dZ11 * X2                 #dL/dW13
    dW14 = dZ12 * X2                 #dL/dW14

    db11 = dZ11                     #dL/db11
    db12 = dZ12                     #dL/db12
    
    W11 -= dW11 * learning_rate
    W13 -= dW13 * learning_rate
    B11 -= db11 * learning_rate

    W12 -= dW12 * learning_rate
    W14 -= dW14 * learning_rate
    B12 -= db12 * learning_rate

    W21 -= dW21 * learning_rate
    W25 -= dW25 * learning_rate
    B21 -= db21 * learning_rate

    W22 -= dW22 * learning_rate
    W26 -= dW26 * learning_rate
    B22 -= db22 * learning_rate

    W23 -= dW23 * learning_rate
    W27 -= dW27 * learning_rate
    B23 -= db23 * learning_rate

    W24 -= dW24 * learning_rate
    W28 -= dW28 * learning_rate
    B24 -= db24 * learning_rate

    return W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24

#INITIALISATION
X = np.array([[0, 0],
             [0, 1],
             [1, 0], 
             [1, 1]])

y = np.array([[0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 0, 0, 0]])

learning_rate = 1
nb_iteraton = 15_000

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W11_log, W13_log, B11_log = [], [], []
W12_log, W14_log, B12_log = [], [], []

W21_log, W25_log, B21_log = [], [], []
W22_log, W26_log, B22_log = [], [], []
W23_log, W27_log, B23_log = [], [], []
W24_log, W28_log, B24_log = [], [], []

#PREMIER PASSAGE
W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24 = initialisation()
A11, A12, A21, A22, A23, A24 = forward_propagation(X[0], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'W13:':<6} {W13[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'W14:':<6} {W14[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")

print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W25:':<6} {W25[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W22:':<6} {W22[0]:>10.6f}   {'W26:':<6} {W26[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W27:':<6} {W27[0]:>10.6f}   {'B23:':<6} {B23[0]:>10.6f}")
print(f"{'W24:':<6} {W24[0]:>10.6f}   {'W28:':<6} {W28[0]:>10.6f}   {'B24:':<6} {B24[0]:>10.6f}")

print("Loss", log_loss(A21, A22, A23, A24, y[0]))
print("ACTIVATION", sofmax(A21, A22, A23, A24))
print("")

for j in tqdm(range(nb_iteraton//2)):
    
    for i in range(4):

        #Foreward propagation
        A11, A12, A21, A22, A23, A24 = forward_propagation(X[i], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)

        if (j % 50 == 0):
            log.append(log_loss(A21, A22, A23,A24, y[i]))
            dx_log.append(dx_log_loss(y[i], A21, A22, A23, A24))

        # Sauvegarde des poids et biais
        W11_log.append(W11.copy())
        W13_log.append(W13.copy())
        B11_log.append(B11.copy())

        W12_log.append(W12.copy())
        W14_log.append(W14.copy())
        B12_log.append(B12.copy())

        W21_log.append(W21.copy())
        W25_log.append(W25.copy())
        B21_log.append(B21.copy())

        W22_log.append(W22.copy())
        W26_log.append(W26.copy())
        B22_log.append(B22.copy())

        W23_log.append(W23.copy())
        W27_log.append(W27.copy())
        B23_log.append(B23.copy())

        W24_log.append(W24.copy())
        W28_log.append(W28.copy())
        B24_log.append(B24.copy())

        #Backpropagation
        W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24 = backward_propagation(X[i], y[i], A11, A12, A21, A22, A23, A24, W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24, learning_rate)


A11, A12, A21, A22, A23, A24 = forward_propagation(X[0], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)

print("")
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'W13:':<6} {W13[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'W14:':<6} {W14[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")

print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W25:':<6} {W25[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W22:':<6} {W22[0]:>10.6f}   {'W26:':<6} {W26[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W27:':<6} {W27[0]:>10.6f}   {'B23:':<6} {B23[0]:>10.6f}")
print(f"{'W24:':<6} {W24[0]:>10.6f}   {'W28:':<6} {W28[0]:>10.6f}   {'B24:':<6} {B24[0]:>10.6f}")


A11, A12, A21, A22, A23, A24 = forward_propagation(X[0], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[0]))
print("y: ", y[0])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

A11, A12, A21, A22, A23, A24 = forward_propagation(X[1], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[1]))
print("y: ", y[1])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

A11, A12, A21, A22, A23, A24 = forward_propagation(X[2], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[2]))
print("y: ", y[2])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

A11, A12, A21, A22, A23, A24 = forward_propagation(X[3], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[3]))
print("y: ", y[3])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

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
plt.plot(W13_log, label='W13')
plt.plot(B11_log, label='B11')


plt.plot(W12_log, label='W12')
plt.plot(W14_log, label='W14')
plt.plot(B12_log, label='B12')

plt.plot(W21_log, label='W21')
plt.plot(W25_log, label='W25')
plt.plot(B21_log, label='B21')

plt.plot(W22_log, label='W22')
plt.plot(W26_log, label='W26')
plt.plot(B22_log, label='B22')

plt.plot(W23_log, label='W23')
plt.plot(W27_log, label='W27')
plt.plot(B23_log, label='B23')

plt.plot(W24_log, label='W24')
plt.plot(W28_log, label='W28')
plt.plot(B24_log, label='B24')

plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()


#DEUXIEME PASSAGE
A11, A12, A21, A22, A23, A24 = forward_propagation(X[0], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)

y = np.array([[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'W13:':<6} {W13[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'W14:':<6} {W14[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")

print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W25:':<6} {W25[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W22:':<6} {W22[0]:>10.6f}   {'W26:':<6} {W26[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W27:':<6} {W27[0]:>10.6f}   {'B23:':<6} {B23[0]:>10.6f}")
print(f"{'W24:':<6} {W24[0]:>10.6f}   {'W28:':<6} {W28[0]:>10.6f}   {'B24:':<6} {B24[0]:>10.6f}")

print("Loss", log_loss(A21, A22, A23, A24, y[0]))
print("ACTIVATION", sofmax(A21, A22, A23, A24))
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(4):

        #Foreward propagation
        A11, A12, A21, A22, A23, A24 = forward_propagation(X[i], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)

        if (j % 50 == 0):
            log.append(log_loss(A21, A22, A23, A24, y[i]))
            dx_log.append(dx_log_loss(y[i], A21, A22, A23, A24))

        # Sauvegarde des poids et biais
        W11_log.append(W11.copy())
        W13_log.append(W13.copy())
        B11_log.append(B11.copy())

        W12_log.append(W12.copy())
        W14_log.append(W14.copy())
        B12_log.append(B12.copy())

        W21_log.append(W21.copy())
        W25_log.append(W25.copy())
        B21_log.append(B21.copy())

        W22_log.append(W22.copy())
        W26_log.append(W26.copy())
        B22_log.append(B22.copy())

        W23_log.append(W23.copy())
        W27_log.append(W27.copy())
        B23_log.append(B23.copy())

        W24_log.append(W24.copy())
        W28_log.append(W28.copy())
        B24_log.append(B24.copy())

        #Backpropagation
        W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24 = backward_propagation(X[i], y[i], A11, A12, A21, A22, A23, A24, W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24, learning_rate)

A11, A12, A21, A22, A23, A241 = forward_propagation(X[0], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("")
print(f"{'W11:':<6} {W11[0]:>10.6f}   {'W13:':<6} {W13[0]:>10.6f}   {'B11:':<6} {B11[0]:>10.6f}")
print(f"{'W12:':<6} {W12[0]:>10.6f}   {'W14:':<6} {W14[0]:>10.6f}   {'B12:':<6} {B12[0]:>10.6f}")

print(f"{'W21:':<6} {W21[0]:>10.6f}   {'W25:':<6} {W25[0]:>10.6f}   {'B21:':<6} {B21[0]:>10.6f}")
print(f"{'W22:':<6} {W22[0]:>10.6f}   {'W26:':<6} {W26[0]:>10.6f}   {'B22:':<6} {B22[0]:>10.6f}")
print(f"{'W23:':<6} {W23[0]:>10.6f}   {'W27:':<6} {W27[0]:>10.6f}   {'B23:':<6} {B23[0]:>10.6f}")
print(f"{'W24:':<6} {W24[0]:>10.6f}   {'W28:':<6} {W28[0]:>10.6f}   {'B24:':<6} {B24[0]:>10.6f}")

A11, A12, A21, A22, A23, A24 = forward_propagation(X[0], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[0]))
print("y: ", y[0])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

A11, A12, A21, A22, A23, A24 = forward_propagation(X[1], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[1]))
print("y: ", y[1])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

A11, A12, A21, A22, A23, A24 = forward_propagation(X[2], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[2]))
print("y: ", y[2])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

A11, A12, A21, A22, A23, A24 = forward_propagation(X[3], W11, W13, B11, W12, W14, B12, W21, W25, B21, W22, W26, B22, W23, W27, B23, W24, W28, B24)
print("Loss final ", log_loss(A21, A22, A23, A24, y[3]))
print("y: ", y[3])
print("ACTIVATION final", sofmax(A21, A22, A23, A24))

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
plt.plot(W13_log, label='W13')
plt.plot(B11_log, label='B11')


plt.plot(W12_log, label='W12')
plt.plot(W14_log, label='W14')
plt.plot(B12_log, label='B12')

plt.plot(W21_log, label='W21')
plt.plot(W25_log, label='W25')
plt.plot(B21_log, label='B21')

plt.plot(W22_log, label='W22')
plt.plot(W26_log, label='W26')
plt.plot(B22_log, label='B22')

plt.plot(W23_log, label='W23')
plt.plot(W27_log, label='W27')
plt.plot(B23_log, label='B23')

plt.plot(W24_log, label='W24')
plt.plot(W28_log, label='W28')
plt.plot(B24_log, label='B24')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()