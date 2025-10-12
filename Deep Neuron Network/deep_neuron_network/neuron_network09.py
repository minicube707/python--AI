
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred - (1 - y_true)/(1 - y_pred)

def sigmoide(X):
    return 1/(1 + np.exp(-X))

def initialisation(X, y):

    nb_neuron1 = 4
    nb_activation1 = X.size

    nb_neuron2 = 4
    nb_activation2 = nb_neuron1

    nb_neuron3 = y.size
    nb_activation3 = nb_neuron2

    W1 = np.random.rand(nb_activation1, nb_neuron1) * 2 - 1
    B1 = np.random.rand(1, nb_neuron1) * 2 - 1

    W2 = np.random.rand(nb_activation2, nb_neuron2) * 2 - 1
    B2 = np.random.rand(1, nb_neuron2) * 2 - 1

    W3 = np.random.rand(nb_activation3, nb_neuron3) * 2 - 1
    B3 = np.random.rand(1, nb_neuron3) * 2 - 1

    return W1, B1, W2, B2, W3, B3

def forward_propagation(X,  W1, B1, W2, B2, W3, B3):

    Z1 = np.dot(X, W1) + B1
    A1 = sigmoide(Z1)

    Z2 = np.dot(A1, W2) + B2
    A2 = sigmoide(Z2)

    Z3 = np.dot(A2, W3) + B3
    A3 = sigmoide(Z3)

    return A1, A2, A3

def backward_propagation(X, y, A1, A2, A3, W1, B1, W2, B2, W3, B3, learning_rate):

    X = X.reshape((-1, X.size))

    dZ3 = A3 - y                                    #dL/dZ3
    dW3 = np.dot(A2.T, dZ3)                         #dL/dW3 
    db3 = np.mean(dZ3, axis=0, keepdims=True)       #dL/db3

    dA2 = np.dot(dZ3, W3.T)     #dL/dA2
    dZ2 = dA2 * A2 * (1 - A2)   #dL/dZ2
    
    dW2 = np.dot(A1.T, dZ2)                          #dL/dW2
    db2 = np.mean(dZ2, axis=0, keepdims=True)        #dL/db2

    dA1 = np.dot(dZ2, W2.T)     #dL/dA1
    dZ1 = dA1 * A1 * (1 - A1)   #dL/dZ1
    
    dW1 = np.dot(X.T, dZ1)                          #dL/dW1
    db1 = np.mean(dZ1, axis=0, keepdims=True)       #dL/db1

    W1 -= dW1 * learning_rate
    B1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    B2 -= db2 * learning_rate
    W3 -= dW3 * learning_rate
    B3 -= db3 * learning_rate

    return  W1, B1, W2, B2, W3, B3

def show_evolution(historique_W1, historique_b1, historique_W2, historique_b2, historique_W3, historique_b3):
    
    # 1. Convertir la liste en array proprement
    historique_W1 = np.array(historique_W1)
    historique_b1 = np.array(historique_b1)
    historique_W2 = np.array(historique_W2)
    historique_b2 = np.array(historique_b2)
    historique_W3 = np.array(historique_W3)
    historique_b3 = np.array(historique_b3)

    # 2. Aplatir chaque matrice en vecteur (si W1 est multidimensionnel)
    # Cela donne une forme (nb_epochs, nombre_total_de_valeurs)
    historique_W1 = historique_W1.reshape(historique_W1.shape[0], -1)
    historique_b1 = historique_b1.reshape(historique_b1.shape[0], -1)
    historique_W2 = historique_W2.reshape(historique_W2.shape[0], -1)
    historique_b2 = historique_b2.reshape(historique_b2.shape[0], -1)
    historique_W3 = historique_W3.reshape(historique_W3.shape[0], -1)
    historique_b3 = historique_b3.reshape(historique_b3.shape[0], -1)

    # 3. Tracer chaque valeur individuelle
    plt.figure(figsize=(10, 6))
    for i in range(historique_W1.shape[1]):
        plt.plot(historique_W1[:, i], label=f"W1{i+1}")
    for i in range(historique_b1.shape[1]):
        plt.plot(historique_b1[:, i], label=f"B1{i+1}")
    for i in range(historique_W2.shape[1]):
        plt.plot(historique_W2[:, i], label=f"W2{i+1}")
    for i in range(historique_b2.shape[1]):
        plt.plot(historique_b2[:, i], label=f"B2{i+1}")
    for i in range(historique_W3.shape[1]):
        plt.plot(historique_W3[:, i], label=f"W3{i+1}")
    for i in range(historique_b3.shape[1]):
        plt.plot(historique_b3[:, i], label=f"B3{i+1}")
    plt.xlabel("Époque")
    plt.ylabel("Valeur")
    plt.title("Évolution des valeurs de la matrice W1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#INITIALISATION
X = np.array([[0, 1],
              [1, 0]])
y = np.array([0, 1])

learning_rate = 0.01
nb_iteraton = 30_000

log = []
dx_log = []

# Pour stocker les poids, biais ou autres au fil des époques
historique_W1 = []
historique_b1 = []
historique_W2 = []
historique_b2 = []
historique_W3 = []
historique_b3 = []

#PREMIER PASSAGE
W1, B1, W2, B2, W3, B3 = initialisation(X[0], y[0])
A1, A2, A3 = forward_propagation(X[0], W1, B1, W2, B2, W3, B3)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss", log_loss(A3, y))
print("ACTIVATION", A3)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        A1, A2, A3 = forward_propagation(X[i], W1, B1, W2, B2, W3, B3)

        if (j % 50 == 0):
            log.append(log_loss(A3.flatten(), y[i]))
            dx_log.append(dx_log_loss(y[i], A3.flatten()))
            
            # Enregistrement des valeurs à chaque époque
            historique_W1.append(W1.copy())     # .copy() est important pour éviter des effets de référence
            historique_b1.append(B1.copy())
            historique_W2.append(W2.copy())     # .copy() est important pour éviter des effets de référence
            historique_b2.append(B2.copy())
            historique_W3.append(W3.copy())     # .copy() est important pour éviter des effets de référence
            historique_b3.append(B3.copy())

        #Backpropagation
        W1, B1, W2, B2, W3, B3 = backward_propagation(X[i], y[i], A1, A2, A3, W1, B1, W2, B2, W3, B3, learning_rate)


A1, A2, A3 = forward_propagation(X[0], W1, B1, W2, B2, W3, B3)
print("Loss final ", log_loss(A3, y[0]))
print("y: ", y[0])
print("ACTIVATION final", A3)
A1, A2, A3 = forward_propagation(X[1], W1, B1, W2, B2, W3, B3)
print("Loss final ", log_loss(A3, y[1]))
print("y: ", y[1])
print("ACTIVATION final", A3)

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()

show_evolution(historique_W1, historique_b1, historique_W2, historique_b2, historique_W3, historique_b3)

#DEUXIEME PASSAGE
A1, A2, A3 = forward_propagation(X[0], W1, B1, W2, B2, W3, B3)
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss", log_loss(A2, y[0]))
print("ACTIVATION", A2)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        A1, A2, A3 = forward_propagation(X[i], W1, B1, W2, B2, W3, B3)

        if (j % 50 == 0):
            log.append(log_loss(A3.flatten(), y[i]))
            dx_log.append(dx_log_loss(y[i], A3.flatten()))
            
            # Enregistrement des valeurs à chaque époque
            historique_W1.append(W1.copy())     # .copy() est important pour éviter des effets de référence
            historique_b1.append(B1.copy())
            historique_W2.append(W2.copy())     # .copy() est important pour éviter des effets de référence
            historique_b2.append(B2.copy())
            historique_W3.append(W3.copy())     # .copy() est important pour éviter des effets de référence
            historique_b3.append(B3.copy())

        #Backpropagation
        W1, B1, W2, B2, W3, B3 = backward_propagation(X[i], y[i], A1, A2, A3, W1, B1, W2, B2, W3, B3, learning_rate)

A1, A2, A3 = forward_propagation(X[0], W1, B1, W2, B2, W3, B3)
print("Loss final ", log_loss(A3, y[0]))
print("y: ", y[0])
print("ACTIVATION final", A3)
A1, A2, A3 = forward_propagation(X[1], W1, B1, W2, B2, W3, B3)
print("Loss final ", log_loss(A3, y[1]))
print("y: ", y[1])
print("ACTIVATION final", A3)

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()


show_evolution(historique_W1, historique_b1, historique_W2, historique_b2, historique_W3, historique_b3)