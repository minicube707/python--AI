
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from itertools import product

np.set_printoptions(precision=2, suppress=True)

np.set_printoptions(
    threshold=np.inf,       # Affiche tout
    linewidth=200,          # Largeur max avant saut de ligne
    edgeitems=10,           # Combien d’éléments afficher en début/fin si tronqué
    precision=3,            # Nombre de décimales
    suppress=True           # Ne pas utiliser la notation scientifique
)

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  np.mean(- y * np.log(A + epsilon), axis=1)

def dx_log_loss(y_true, y_pred):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  np.mean(- y_true/(y_pred + epsilon), axis=1)

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

    parameters = {}
    parameters_grad = {}

    C = len(dimension)

    dimension[str(C)] = (y.shape[1], dimension[str(C)][1], dimension[str(C)][2])
    nb_activation = X.shape[1]

    for i in range(1, C+1):
        nb_neuron = dimension[str(i)][0]

        w_shpape = (nb_activation, nb_neuron)
        b_shape = (1, nb_neuron)

        parameters["W" + str(i)] = np.random.rand(*w_shpape) * 2 -1
        parameters["B" + str(i)] = np.random.rand(*b_shape) * 2 -1

        nb_activation = nb_neuron
        
        parameters_grad["wm" + str(i)] = np.zeros(w_shpape).astype(np.float32)
        parameters_grad["wv" + str(i)] = np.zeros(w_shpape).astype(np.float32)

        parameters_grad["bm" + str(i)] = np.zeros(b_shape).astype(np.float32)
        parameters_grad["bv" + str(i)] = np.zeros(b_shape).astype(np.float32)

    return parameters, parameters_grad

def adam_weight(param, grad, m, v, lr, beta1, beta2, t, eps=1e-8):

    grad = np.clip(grad, -2, 2)

    # Update moments
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)

    # Bias correction
    bias_corr1 = max(1 - beta1**(t + 1), 1e-12)
    bias_corr2 = max(1 - beta2**(t + 1), 1e-12)

    m_hat = m / bias_corr1
    v_hat = np.maximum(v / bias_corr2, 1e-12)
    
    # Update parameter
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param, m, v

def grah(log, dx_log):

    log = np.array(log)
    dx_log = np.array(dx_log)

    # Créer une figure avec deux sous-graphes côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
    # Courbes et légende dynamiques pour log
    axes[0].plot(log, label=f"Logloss")
    axes[0].set_title("Log")
    axes[0].legend()

    # Courbes et légende dynamiques pour dx_log
    axes[1].plot(dx_log, label=f"dLogloss")
    axes[1].set_title("dLog")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


class DNN():

    def __init__(self, X, y, dimensions):
        parameters, parameters_grad = initialisation(X, y, dimensions)

        self.dimensions = dimensions
        self.parameters = parameters
        self.parameters_grad = parameters_grad
        self.activation = {}
        self.gradients = {}
        self.C_DNN = len(dimensions)

    def forward_propagation(self, X, alpha, training):

        dimensions = self.dimensions
        parameters = self.parameters
    
        if X.ndim == 1:
            X = X.reshape(1, -1)

        activation = {"A0" : X}
        C = self.C_DNN

        for i in range(1, C+1):

            Z = np.dot(activation["A" + str(i-1)], parameters["W" + str(i)]) + parameters["B" + str(i)]
            activation["Z" + str(i)] = Z

            if dimensions[str(i)][1] == "sigmoide":
                A = sigmoide(Z)

            elif dimensions[str(i)][1] == "tanh":
                A = tanh(Z)
            
            elif dimensions[str(i)][1] == "relu":
                A = relu(Z, alpha)

            dropout_per = dimensions[str(i)][2]
            if training and dropout_per > 0:
                M = (np.random.rand(*A.shape) > dropout_per).astype(X.dtype)
                A = M * A / (1 - dropout_per)

            else:
                M = np.ones_like(A)


            activation["A" + str(i)] = A
            activation["M" + str(i)] = M

        self.activation = activation

    def backward_propagation(self, y, alpha, training):
        
        activation = self.activation
        dimensions = self.dimensions
        parameters = self.parameters

        if y.ndim == 1:
            m = y.size  
        else:
            m = y.shape[1]    
        
        C = self.C_DNN
        gradients = {}  

        dZ = softmax(activation["A" + str(C)]) - y     

        for i in reversed(range(1, C+1)):
            gradients["dW" + str(i)] = 1/m * np.dot(activation["A" + str(i-1)].T, dZ)
            gradients["dB" + str(i)] = 1/m * np.mean(dZ, axis=0, keepdims=True)

            dA = np.dot(dZ, parameters["W" + str(i)].T)
            dA = np.clip(dA, -100, 100)

            if i > 1:
                if dimensions[str(i)][1] == "sigmoide":
                    dZ = dA * dx_sigmoide(activation["A" + str(i-1)])

                elif dimensions[str(i)][1] == "tanh":
                    dZ = dA * dx_tanh(activation["A" + str(i-1)])

                elif dimensions[str(i)][1] == "relu":
                    dZ = dA * dx_relu(activation["Z" + str(i-1)], alpha)

                dropout_per = dimensions[str(i)][2]
                if training and dropout_per > 0:
                    M = activation[f"M{i-1}"]
                    dZ = dZ * M / (1 - dropout_per)

        self.gradients = gradients 


    def update(self, lr, beta1, beta2, t):

        parameters = self.parameters
        C_DNN = self.C_DNN
        gradients = self.gradients
        parameters_grad = self.parameters_grad

        for c in range(1, C_DNN + 1):
                
            # ----- Kernel -----
            parameters[f"W{c}"], parameters_grad[f"wm{c}"], parameters_grad[f"wv{c}"] = adam_weight(
                parameters[f"W{c}"],
                gradients[f"dW{c}"],
                parameters_grad[f"wm{c}"],
                parameters_grad[f"wv{c}"],
                lr, beta1, beta2, t
            )

            # ----- Bias -----
            parameters[f"B{c}"], parameters_grad[f"bm{c}"], parameters_grad[f"bv{c}"] = adam_weight(
                parameters[f"B{c}"],
                gradients[f"dB{c}"],
                parameters_grad[f"bm{c}"],
                parameters_grad[f"bv{c}"],
                lr, beta1, beta2, t
            )

        self.parameters = parameters

    def print_info(self):

        dimensions = self.dimensions
        parameters = self.parameters
        C_DNN = self.C_DNN

        print("")
        print("============================")
        print("    INITIALISATION DNN")
        print("============================")

        print("\nDétail de la convolution :")
        print("Nb activation")
        for c in range(1, C_DNN + 1):
            print(dimensions[str(c)][0], end="")
            if c < C_DNN:
                print("->", end="")
        print("")

        print("")
        for c in range(1, C+1):
            print("W" + str(c), ":", parameters["W" + str(c)].shape)
            print("B" + str(c), ":", parameters["B" + str(c)].shape)
        
        print("")
        print("nb neuron, function, dropout")
        for keys, values in dimensions.items():
            print(keys, values)


#INITIALISATION
# Génération de toutes les combinaisons de 3 bits (0 et 1)
n = 4
combinations = list(product([0, 1], repeat=n))

# Conversion en tableau numpy
X = np.array(combinations)
y = np.arange(np.power(2, n))

transformer=LabelBinarizer()
transformer.fit(y)
y = transformer.transform(y.reshape((-1, 1)))

learning_rate = 0.001
nb_iteraton = 10_000
alpha = 0.01
beta1 = 0.9
beta2 = 0.99

dimensions = {
    "1" : (16, "relu", 0.1),
    "2" : (8, "relu", 0.1),
    "3" : (0, "relu", 0.1)
}

log = []
dx_log = []
C = len(dimensions)

model = DNN(X, y, dimensions)

model.print_info()

#PREMIER PASSAGE
model.forward_propagation(X, alpha, False)
res = softmax(model.activation["A" + str(C)])

print("")
print("Premier apprentissage")
print("X\n", X)
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)
print("")

for j in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    model.forward_propagation(X, alpha, True)
    res = softmax(model.activation["A" + str(C)])

    if (j % 50 == 0):
        model.forward_propagation(X, alpha, False)
        res = softmax(model.activation["A" + str(C)])
        log.append(log_loss(res, y))
        dx_log.append(dx_log_loss(y, res))


    #Backpropagation
    model.backward_propagation(y, alpha, True)
    model.update(learning_rate, beta1, beta2, j)


model.forward_propagation(X, alpha, False)
res = softmax(model.activation["A" + str(C)])
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)

grah(log, dx_log)


#DEUXIEME PASSAGE
model.forward_propagation(X, alpha, False)
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
    
    #Foreward propagation
    model.forward_propagation(X, alpha, True)
    res = softmax(model.activation["A" + str(C)])

    if (j % 50 == 0):
        model.forward_propagation(X, alpha, False)
        res = softmax(model.activation["A" + str(C)])
        log.append(log_loss(res, y))
        dx_log.append(dx_log_loss(y, res))
        

    #Backpropagation
    model.backward_propagation(y, alpha, True)
    model.update(learning_rate, beta1, beta2, j)


model.forward_propagation(X, alpha, False)
res = softmax(model.activation["A" + str(C)])
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)

grah(log, dx_log)

