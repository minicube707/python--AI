
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

np.set_printoptions(precision=4, suppress=True)

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - 1/y.shape[1] * y * np.log(A + epsilon)

def dx_log_loss(y_true, y_pred):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return - 1/y_true.shape[1]  * y_true/(y_pred + epsilon)

def sigmoide(X):
    X = np.clip(X, -100, 100)
    return 1/(1 + np.exp(-X))

def dx_sigmoide(X):
    return X * (1 - X)

def tanh(X):
    return np.tanh(X)

def dx_tanh(X):
    return (1 - X**2)

def relu(X, alpha):
    return np.where(X < 0, alpha*X, X)

def dx_relu(X, alpha):
    return np.where(X < 0, alpha, 1)

def softmax(X):
    res = np.array([])
    for i in range(X.shape[0]):
        x = np.clip(X[i,:], -100, 100)
        res = np.append(res, np.exp(x) / np.sum(np.exp(x)))
         
    return res.reshape((X.shape))

def initialisation(X, y, dimension):

    parametres ={}
    C = len(dimension)

    dimension[str(C)] = (y.shape[1], dimension[str(C)][1])
    nb_activation = X.shape[1]

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
    C = len(dimension)

    for i in range(1, C+1):
        Z = np.dot(activation["A" + str(i-1)], parametres["W" + str(i)]) + parametres["B" + str(i)]
        activation["Z" + str(i)] = Z

        if dimension[str(i)][1] == "sigmoide":
            activation["A" + str(i)] =  sigmoide(Z)
        elif dimension[str(i)][1] == "tanh":
            activation["A" + str(i)] =  tanh(Z)
        elif dimension[str(i)][1] == "relu":
            activation["A" + str(i)] =  relu(Z, alpha)

    return activation

def backward_propagation(y, activation, parametres, dimension, alpha):

    if y.ndim == 1:
        m = y.size  
    else:
        m = y.shape[1]    
    
    C = len(dimension)
    gradients = {}  

    dZ = softmax(activation["A" + str(C)]) - y     

    for i in reversed(range(1, C+1)):
        gradients["dW" + str(i)] = 1/m * np.dot(activation["A" + str(i-1)].T, dZ)
        gradients["dB" + str(i)] = 1/m * np.mean(dZ, axis=0, keepdims=True)    
        dA = np.dot(dZ, parametres["W" + str(i)].T)
        dA = np.clip(dA, -100, 100)

        if i > 1:
            if dimension[str(i)][1] == "sigmoide":
                dZ = dA * dx_sigmoide(activation["A" + str(i-1)])
            elif dimension[str(i)][1] == "tanh":
                dZ = dA * dx_tanh(activation["A" + str(i-1)])
            elif dimension[str(i)][1] == "relu":
                dZ = dA * dx_relu(activation["Z" + str(i-1)], alpha)

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
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1],
             ])

y = np.arange(8)

transformer=LabelBinarizer()
transformer.fit(y)
y = transformer.transform(y.reshape((-1, 1)))

learning_rate = 0.01
nb_iteraton = 30_000

dimension = {
    "1" : (4, "relu"),
    "2" : (4, "relu"),
    "3" : (4, "relu"),
    "4" : (4, "relu"),
    "5" : (0, "relu")
}

log = []
dx_log = []

C = len(dimension)

alpha = 0.01

#PREMIER PASSAGE
parametres = initialisation(X, y, dimension)
activation = forward_propagation(X, parametres, dimension, alpha)
res = softmax(activation["A" + str(C)])

print("")
print("Premier apprentissage")
print("X\n", X)
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        activation = forward_propagation(X, parametres, dimension, alpha)
        res = softmax(activation["A" + str(C)])

        if (j % 50 == 0):
            log.append(log_loss(res, y))
            dx_log.append(dx_log_loss(y, res))


        #Backpropagation
        gradients = backward_propagation(y, activation, parametres, dimension, alpha)
        parametres = update(gradients, parametres, learning_rate)


activation = forward_propagation(X, parametres, dimension, alpha)
res = softmax(activation["A" + str(C)])
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)

grah(log, dx_log)


#DEUXIEME PASSAGE
activation = forward_propagation(X, parametres, dimension, alpha)
y = np.rot90(y)

print("")
print("Deuxieme apprentissage")
print("X\n", X)
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)
print("")

for j in tqdm(range(nb_iteraton)):
    
    for i in range(X.shape[0]):

        #Foreward propagation
        activation = forward_propagation(X, parametres, dimension, alpha)
        res = softmax(activation["A" + str(C)])

        if (j % 50 == 0):
            log.append(log_loss(res, y))
            dx_log.append(dx_log_loss(y, res))
            

        #Backpropagation
        gradients = backward_propagation(y, activation, parametres, dimension, alpha)
        parametres = update(gradients, parametres, learning_rate)

res = forward_propagation(X, parametres, dimension, alpha)
res = softmax(activation["A" + str(C)])
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)

grah(log, dx_log)

