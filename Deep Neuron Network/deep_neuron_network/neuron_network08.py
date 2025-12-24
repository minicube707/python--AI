
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred + (1 - y_true)/(1 - y_pred)


def sigmoide(X):
    return 1/(1 + np.exp(-X))

def initialisation():
    nb_neuron1 = 2
    nb_activation1 = 1

    nb_neuron2 = 1
    nb_activation2 = nb_neuron1

    W1 = np.random.rand(nb_activation1, nb_neuron1) * 2 - 1
    B1 = np.random.rand(nb_activation1, nb_neuron1) * 2 - 1

    W2 = np.random.rand(nb_activation2, nb_neuron2) * 2 - 1
    B2 = np.random.rand(nb_activation1, nb_neuron2) * 2 - 1

    print("W1\n",W1)
    print("B1\n",B1)
    print("W2\n",W2)
    print("B2\n",B2)
    return W1, B1, W2, B2

def forward_propagation(X, W1, B1, W2, B2):

    Z1 = np.dot(X, W1) + B1
    A1 = sigmoide(Z1)

    Z2 = np.dot(A1, W2) + B2
    A2 = sigmoide(Z2)

    return A1, A2

def backward_propagation(X, y, A1, A2, W1, B1, W2, B2, learning_rate):

    dZ2 = A2 - y                 #dL/dZ2
    dW2 = np.dot(dZ2, A1).T        #dL/dW2 
    db2 = dZ2                    #dL/db2

    dA1 = np.dot(dZ2, W2.T)     #dL/dA1
    dZ1 = dA1 * A1 * (1 - A1)   #dL/dZ1
    
    dW1 = np.dot(dZ1, X)        #dL/dW1
    db1 = dZ1                   #dL/db1

    W1 -= dW1 * learning_rate
    B1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    B2 -= db2 * learning_rate

    return W1, B1, W2, B2

def show_evolution(historique_W1, historique_b1, historique_W2, historique_b2):
    
    # 1. Convertir la liste en array proprement
    historique_W1 = np.array(historique_W1)
    historique_b1 = np.array(historique_b1)
    historique_W2 = np.array(historique_W2)
    historique_b2 = np.array(historique_b2)

    # 2. Aplatir chaque matrice en vecteur (si W1 est multidimensionnel)
    # Cela donne une forme (nb_epochs, nombre_total_de_valeurs)
    historique_W1 = historique_W1.reshape(historique_W1.shape[0], -1)
    historique_b1 = historique_b1.reshape(historique_b1.shape[0], -1)
    historique_W2 = historique_W2.reshape(historique_W2.shape[0], -1)
    historique_b2 = historique_b2.reshape(historique_b2.shape[0], -1)

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
    plt.xlabel("Époque")
    plt.ylabel("Valeur")
    plt.title("Évolution des valeurs de la matrice W1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#INITIALISATION
X = np.array([0, 1])
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

#PREMIER PASSAGE
W1, B1, W2, B2 = initialisation()
A1, A2 = forward_propagation(X[0], W1, B1, W2, B2)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss", log_loss(A2, y))
print("ACTIVATION", A2)
print("")

for j in tqdm(range(nb_iteraton)):
    
    sum_log = 0
    sum_dx_log = 0

    for i in range(X.size):

        #Foreward propagation
        A1, A2 = forward_propagation(X[i], W1, B1, W2, B2)

        sum_log += log_loss(A2.flatten(), y[i])
        sum_dx_log += dx_log_loss(y[i], A2.flatten())

        if (j % 50 == 0):
            # Enregistrement des valeurs à chaque époque
            historique_W1.append(W1.copy())     # .copy() est important pour éviter des effets de référence
            historique_b1.append(B1.copy())
            historique_W2.append(W2.copy())     # .copy() est important pour éviter des effets de référence
            historique_b2.append(B2.copy())

        #Backpropagation
        W1, B1, W2, B2 = backward_propagation(X[i], y[i], A1, A2, W1, B1, W2, B2, learning_rate)

    log.append(sum_log)
    dx_log.append(sum_dx_log)

#Prediction final 
A1, A2 = forward_propagation(X[0], W1, B1, W2, B2)
print("Loss final ", log_loss(A2, y[0]))
print("y: ", y[0])
print("ACTIVATION final", A2)

#Prediction final 
A1, A2 = forward_propagation(X[1], W1, B1, W2, B2)
print("Loss final ", log_loss(A2, y[1]))
print("y: ", y[1])
print("ACTIVATION final", A2)

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()

show_evolution(historique_W1, historique_b1, historique_W2, historique_b2)

#DEUXIEME PASSAGE
A1, A2 = forward_propagation(X[0], W1, B1, W2, B2)
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss", log_loss(A2, y[0]))
print("ACTIVATION", A2)
print("")

for j in tqdm(range(nb_iteraton)):
    
    sum_log = 0
    sum_dx_log = 0

    for i in range(X.size):

        #Foreward propagation
        A1, A2 = forward_propagation(X[i], W1, B1, W2, B2)

        sum_log += log_loss(A2.flatten(), y[i])
        sum_dx_log += dx_log_loss(y[i], A2.flatten())

        if (j % 50 == 0):
            # Enregistrement des valeurs à chaque époque
            historique_W1.append(W1.copy())     # .copy() est important pour éviter des effets de référence
            historique_b1.append(B1.copy())
            historique_W2.append(W2.copy())     # .copy() est important pour éviter des effets de référence
            historique_b2.append(B2.copy())

        #Backpropagation
        W1, B1, W2, B2 = backward_propagation(X[i], y[i], A1, A2, W1, B1, W2, B2, learning_rate)

    log.append(sum_log)
    dx_log.append(sum_dx_log)

   
#Prediction final 
A1, A2 = forward_propagation(X[0], W1, B1, W2, B2)

print("Loss final ", log_loss(A2, y[0]))
print("y: ", y[0])
print("ACTIVATION final", A2)

#Prediction final 
A1, A2 = forward_propagation(X[1], W1, B1, W2, B2)
print("Loss final ", log_loss(A2, y[1]))
print("y: ", y[1])
print("ACTIVATION final", A2)

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()


show_evolution(historique_W1, historique_b1, historique_W2, historique_b2)