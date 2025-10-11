
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred - (1 - y_true)/(1 - y_pred)

def algebre(x, a, b):
    return a * x  + b

def sigmoide(X):
    return 1/(1 + np.exp(-X))

def initialisation(X, y, dimension):

    parametres ={}
    C = len(dimension)

    dimension[str(C)] = y.size
    nb_activation = X.size

    for i in range(1, C+1):
        nb_neuron = dimension[str(i)]
        parametres["W" + str(i)] = np.random.rand(nb_activation, nb_neuron) * 2 -1
        parametres["B" + str(i)] = np.random.rand(1, nb_neuron) * 2 -1
        nb_activation = nb_neuron
        
        print("W" + str(i), ":", parametres["W" + str(i)].shape)
        print("B" + str(i), ":", parametres["B" + str(i)].shape)

    for c in range(1, C+1):
        print(dimension[str(c)], end="")
        if c < C:
            print("->", end="")
    print("\n")

    return parametres

def forward_propagation(X,  parametres):

    X = X.reshape((-1, X.size))
    activation = {"A0" : X}
    C = len(dimension)

    for i in range(1, C+1):
        Z = np.dot(activation["A" + str(i-1)], parametres["W" + str(i)]) + parametres["B" + str(i)]
        activation["A" + str(i)] =  sigmoide(Z)

    return activation

def backward_propagation(y, activation, parametres):

    C = len(dimension)
    gradients = {}  

    dZ = activation["A" + str(C)] - y     

    for i in reversed(range(1, C+1)):
        gradients["dW" + str(i)] = np.dot(activation["A" + str(i-1)].T, dZ)
        gradients["dB" + str(i)] = np.mean(dZ, axis=0, keepdims=True)    
        dA = np.dot(dZ, parametres["W" + str(i)].T)

        if i > 1:
            dZ = dA * activation["A" + str(i-1)] * (1 - activation["A" + str(i-1)])   

    return gradients      

def update(gradients, parametres, learning_rate):
    
    C = len(dimension)

    for c in range(1, C+1):
        parametres["W" + str(c)] = parametres["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        parametres["B" + str(c)] = parametres["B" + str(c)] - learning_rate * gradients["dB" + str(c)]
    
    return parametres


#INITIALISATION
X = np.array([0, 1, 0.5])
y = np.array([0, 1, 0.5])

learning_rate = 0.01
nb_iteraton = 30_000

dimension = {
    "1" : 4,
    "2" : 4,
    "3" : 16,
    "4" : 0
}

log = []
dx_log = []

C = len(dimension)

#PREMIER PASSAGE
parametres = initialisation(X[0], y[0], dimension)
activation = forward_propagation(X[0], parametres)

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss", log_loss(activation["A" + str(C)], y))
print("ACTIVATION", activation["A" + str(C)])
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        activation = forward_propagation(X[i], parametres)

        if (j % 50 == 0):
            log.append(log_loss(activation["A" + str(C)].flatten(), y[i]))
            dx_log.append(dx_log_loss(y[i], activation["A" + str(C)].flatten()))


        #Backpropagation
        gradients = backward_propagation(y[i], activation, parametres)
        parametres = update(gradients, parametres, learning_rate)


activation = forward_propagation(X[0], parametres)
print("Loss final ", log_loss(activation["A" + str(C)], y[0]))
print("y: ", y[0])
print("ACTIVATION final", activation["A" + str(C)])
activation = forward_propagation(X[1], parametres)
print("Loss final ", log_loss(activation["A" + str(C)], y[1]))
print("y: ", y[1])
print("ACTIVATION final", activation["A" + str(C)])

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()


#DEUXIEME PASSAGE
activation = forward_propagation(X[0], parametres)
y = np.array([0.5, 1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)
print("Loss", log_loss(activation["A" + str(C)], y[0]))
print("ACTIVATION", activation["A" + str(C)])
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        activation = forward_propagation(X[i], parametres)

        if (j % 50 == 0):
            log.append(log_loss(activation["A" + str(C)].flatten(), y[i]))
            dx_log.append(dx_log_loss(y[i], activation["A" + str(C)].flatten()))
            

        #Backpropagation
        gradients = backward_propagation(y[i], activation, parametres)
        parametres = update(gradients, parametres, learning_rate)

activation = forward_propagation(X[0], parametres)
print("Loss final ", log_loss(activation["A" + str(C)], y[0]))
print("y: ", y[0])
print("ACTIVATION final", activation["A" + str(C)])
activation = forward_propagation(X[1], parametres)
print("Loss final ", log_loss(activation["A" + str(C)], y[1]))
print("y: ", y[1])
print("ACTIVATION final", activation["A" + str(C)])

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()

