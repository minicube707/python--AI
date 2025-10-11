
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - 1/y.size * y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return - 1/y_true.size * y_true/(y_pred + epsilon) - (1 - y_true)/(1 - y_pred + epsilon)

def algebre(x, a, b):
    return a * x  + b

def sigmoide(X):
    X = np.clip(X, -100, 100)
    return 1/(1 + np.exp(-X))

def tanh(X):
    return np.tanh(X)

def relu(X, alpha):
    return np.where(X < 0, alpha*X, X)

def dx_relu(X, alpha):
    return np.where(X < 0, alpha, 1)

def initialisation(X, y, dimension):

    parametres ={}
    C = len(dimension)

    dimension[str(C)] = (y.size, dimension[str(C)][1])
    nb_activation = X.size

    for i in range(1, C+1):
        nb_neuron = dimension[str(i)][0]
        parametres["W" + str(i)] = np.random.rand(nb_activation, nb_neuron) * 2 -1
        parametres["B" + str(i)] = np.random.rand(1, nb_neuron) * 2 -1
        nb_activation = nb_neuron
        
        print("W" + str(i), ":", parametres["W" + str(i)].shape, dimension[str(i)][1])
        print("B" + str(i), ":", parametres["B" + str(i)].shape)

    print("")
    for c in range(1, C+1):
        print(dimension[str(c)][0], end="")
        if c < C:
            print("->", end="")
    print("\n")

    return parametres

def forward_propagation(X,  parametres, dimension, alpha):

    if X.ndim == 1:
        X = X.reshape(1, -1)

    activation = {"A0" : X}
    Z_value = {}
    C = len(dimension)

    for i in range(1, C+1):
        Z = np.dot(activation["A" + str(i-1)], parametres["W" + str(i)]) + parametres["B" + str(i)]
        Z_value["Z" + str(i)] = Z

        if dimension[str(i)][1] == "sigmoide":
            activation["A" + str(i)] =  sigmoide(Z)
        elif dimension[str(i)][1] == "tanh":
            activation["A" + str(i)] =  tanh(Z)
        elif dimension[str(i)][1] == "relu":
            activation["A" + str(i)] =  relu(Z, alpha)

    return activation, Z_value

def backward_propagation(y, activation, Z_value, parametres, dimension, alpha):

    if y.ndim == 1:
        m = y.size  
    else:
        m = y.shape[1]    
    
    C = len(dimension)
    gradients = {}  

    if dimension[str(C)][1] == "sigmoide":
        dZ = activation["A" + str(C)] - y     
    elif dimension[str(C)][1] == "tanh":
        dL = log_loss(activation["A" + str(C)], y)
        dZ = dL * (1 - activation["A" + str(C)]**2)
    elif dimension[str(C)][1] == "relu":
        dL = log_loss(activation["A" + str(C)], y)
        dZ = dL * dx_relu(Z_value["Z" + str(C)], alpha)

    for i in reversed(range(1, C+1)):
        gradients["dW" + str(i)] = 1/m * np.dot(activation["A" + str(i-1)].T, dZ)
        gradients["dB" + str(i)] = 1/m * np.mean(dZ, axis=0, keepdims=True)    
        dA = np.dot(dZ, parametres["W" + str(i)].T)
        dA = np.clip(dA, -100, 100)

        if i > 1:
            if dimension[str(i)][1] == "sigmoide":
                dZ = dA * activation["A" + str(i-1)] * (1 - activation["A" + str(i-1)])
            elif dimension[str(i)][1] == "tanh":
                dZ = dA * (1 - activation["A" + str(i-1)]**2)
            elif dimension[str(i)][1] == "relu":
                dZ = dA * dx_relu(Z_value["Z" + str(i-1)], alpha)

    return gradients      

def update(gradients, parametres, learning_rate):
    
    C = len(dimension)

    for c in range(1, C+1):
        parametres["W" + str(c)] = parametres["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        parametres["B" + str(c)] = parametres["B" + str(c)] - learning_rate * gradients["dB" + str(c)]
    
    return parametres

def grah(log, dx_log):

    log = np.array(log)
    dx_log = np.array(dx_log)

    # Créer une figure avec deux sous-graphes côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
    # Courbes et légende dynamiques pour log
    for i in range(log.shape[1]):
        axes[0].plot(log[:, i], label=f"Logloss {i+1}")
    axes[0].set_title("Log")
    axes[0].legend()

    # Courbes et légende dynamiques pour dx_log
    for i in range(dx_log.shape[1]):
        axes[1].plot(dx_log[:, i], label=f"dLogloss {i+1}")
    axes[1].set_title("dLog")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

#INITIALISATION
X = np.array([0, 1])
y = np.array([0, 1])

learning_rate = 0.01
nb_iteraton = 30_000

dimension = {
    "1" : (4, "relu"),
    "2" : (16, "tanh"),
    "3" : (16, "tanh"),
    "4" : (0, "sigmoide")
}

log = []
dx_log = []

C = len(dimension)

alpha = 0

#PREMIER PASSAGE
parametres = initialisation(X, y, dimension)
activation, _ = forward_propagation(X, parametres, dimension, alpha)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss      ", log_loss(activation["A" + str(C)], y))
print("ACTIVATION", activation["A" + str(C)])
print("ERREEUR   ", activation["A" + str(C)] - y)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        activation, Z_value = forward_propagation(X, parametres, dimension, alpha)

        if (j % 50 == 0):
            log.append(log_loss(activation["A" + str(C)].flatten(), y))
            dx_log.append(dx_log_loss(y, activation["A" + str(C)].flatten()))


        #Backpropagation
        gradients = backward_propagation(y, activation, Z_value, parametres, dimension, alpha)
        parametres = update(gradients, parametres, learning_rate)


activation, _ = forward_propagation(X, parametres, dimension, alpha)
print("y: ", y)
print("Loss      ", log_loss(activation["A" + str(C)], y))
print("ACTIVATION", activation["A" + str(C)])
print("ERREEUR   ", activation["A" + str(C)] - y)

grah(log, dx_log)


#DEUXIEME PASSAGE
activation, _ = forward_propagation(X, parametres, dimension, alpha)
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss      ", log_loss(activation["A" + str(C)], y))
print("ACTIVATION", activation["A" + str(C)])
print("ERREEUR   ", activation["A" + str(C)] - y)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        activation, Z_value = forward_propagation(X, parametres, dimension, alpha)

        if (j % 50 == 0):
            log.append(log_loss(activation["A" + str(C)].flatten(), y))
            dx_log.append(dx_log_loss(y, activation["A" + str(C)].flatten()))
            

        #Backpropagation
        gradients = backward_propagation(y, activation, Z_value, parametres, dimension, alpha)
        parametres = update(gradients, parametres, learning_rate)

activation, _ = forward_propagation(X, parametres, dimension, alpha)
print("y: ", y)
print("Loss      ", log_loss(activation["A" + str(C)], y))
print("ACTIVATION", activation["A" + str(C)])
print("ERREEUR   ", activation["A" + str(C)] - y)

grah(log, dx_log)

