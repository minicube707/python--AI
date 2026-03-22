
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from itertools import product
from abc import ABC, abstractmethod

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

def softmax(X):
    res = np.array([])
    for i in range(X.shape[0]):
        x = np.clip(X[i,:], -100, 100)
        res = np.append(res, np.exp(x) / np.sum(np.exp(x)))
         
    return res.reshape((X.shape))


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


class Layer(ABC):
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dA):
        pass


class ReLU(Layer):
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)


class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0) + self.alpha * np.minimum(X, 0)

    def backward(self, dA):
        dx = np.ones_like(self.X)
        dx[self.X < 0] = self.alpha
        return dA * dx


class Sigmoide(Layer):

    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)
    
class Tanh(Layer):
    
    def forward(self, X):
        self.A = np.tanh(X)
        return self.A

    def backward(self, dA):
        return dA * (1 - self.A**2)

class Dropout(Layer):

    def __init__(self, dropout_per):
        self.dropout_per = dropout_per
        self.training = False

    def forward(self, A, training):
        
        self.training = training

        if training or self.dropout_per > 0:
            self.training = training
            self.M = (np.random.rand(*A.shape) > self.dropout_per).astype(A.dtype)
            return  self.M * A / (1 - self.dropout_per)
        
        else:
            return A
    
    def backward(self, dZ):

        if self.training:
            return dZ * self.M / (1 - self.dropout_per)
        else:
            return dZ

class Dense(Layer):

    def __init__(self, nb_activation, nb_neuron):
        w_shpape = (nb_activation, nb_neuron)
        b_shape = (1, nb_neuron)

        #Parameters
        self.W = np.random.randn(*w_shpape) * 0.01
        self.b = np.zeros(b_shape)
        
        #Gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.Wm = np.zeros(w_shpape)
        self.Wv = np.zeros(w_shpape)

        self.bm = np.zeros(b_shape)
        self.bv = np.zeros(b_shape)


    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.W.T)

        self.dW = dW
        self.db = db

        return dA

class Block(Layer):

    def __init__(self, dense, activation, dropout):
        self.dense = dense
        self.activation = activation
        self.dropout = dropout

    def forward(self, X, training=True):
        Z = self.dense.forward(X)
        A = self.activation.forward(Z)
        A = self.dropout.forward(A, training)

        return A

    def backward(self, dZ):

        dA = self.dropout.backward(dZ)
        dZ = self.activation.backward(dA)
        dZ = self.dense.backward(dZ)

        return dZ
    
class DNN():
    
    def __init__(self, X, y, dimensions, alpha):

        self.dimensions = dimensions
        self.layers = []
        self.C_DNN = len(dimensions)
        self.y_pred = None

        DNN.initialisation(self, X, y, alpha)
        

    def initialisation(self, X, y, alpha):

        dimensions = self.dimensions
        C_DNN = self.C_DNN
        
        dimensions[str(C_DNN)] = (y.shape[1], dimensions[str(C_DNN)][1], dimensions[str(C_DNN)][2])
        nb_activation = X.shape[1]

        for i in range(1, C_DNN + 1):

            nb_neuron, activation_function, dropout_per = dimensions[str(i)]

            #Dense
            dense = Dense(nb_activation, nb_neuron)

            #Activation
            if activation_function == "sigmoide":
                activation = Sigmoide()
            
            elif activation_function == "tanh":
                activation = Tanh()
            
            elif activation_function == "relu":
                activation = ReLU()

            elif activation_function == "leaky relu":
                activation = LeakyReLU(alpha)

            #Droout
            dropout = Dropout(dropout_per)

            self.layers.append(Block(dense, activation, dropout))

            nb_activation = nb_neuron
            

    def forward_propagation(self, X, training):

        for block in self.layers:
            X = block.forward(X, training)

        self.y_pred = X

    def backward_propagation(self, y):
        
        if y.ndim == 1:
            m = y.size  
        else:
            m = y.shape[1]    

        dZ = softmax(self.y_pred) - y     

        for block in reversed(self.layers):
            dZ = block.backward(dZ)

    def adam_weight(param, grad, m, v, lr, beta1, beta2, t, eps=1e-8):

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

    def update(self, lr, beta1, beta2, t):

        for block in self.layers:
            
            dense =  block.dense

            # ----- Kernel -----
            dense.W, dense.Wm, dense.Wv = DNN.adam_weight(
                dense.W, 
                dense.dW, 
                dense.Wm, 
                dense.Wv, 
                lr, beta1, beta2, t
            )

            # ----- Bias -----
            dense.b, dense.bm, dense.bv = DNN.adam_weight(
                dense.b, 
                dense.db, 
                dense.bm, 
                dense.bv, 
                lr, beta1, beta2, t
            )



    def print_info(self):

        dimensions = self.dimensions
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
        for block in self.layers:
            print("W" + str(c), ":", block.dense.W.shape)
            print("B" + str(c), ":", block.dense.b.shape)
         

        print("")
        print("nb neuron, function, dropout")
        for keys, values in dimensions.items():
            print(keys, values)
        print("")


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
    "1" : (16, "leaky relu", 0.0),
    "2" : (16, "leaky relu", 0.1),
    "3" : (0, "leaky relu", 0.0)
}

log = []
dx_log = []

model = DNN(X, y, dimensions, alpha)
model.print_info()

#PREMIER PASSAGE
model.forward_propagation(X, alpha, False)
res = softmax(model.y_pred)

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
    model.forward_propagation(X, True)
    res = softmax(model.y_pred)

    if (j % 50 == 0):
        model.forward_propagation(X, False)
        res = softmax(model.y_pred)
        log.append(log_loss(res, y))
        dx_log.append(dx_log_loss(y, res))


    #Backpropagation
    else:
        model.backward_propagation(y, True)
        model.update(learning_rate, beta1, beta2, j)


model.forward_propagation(X, False)
res = softmax(model.y_pred)
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)

grah(log, dx_log)


#DEUXIEME PASSAGE
model.forward_propagation(X, False)
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
    model.forward_propagation(X, True)
    res = softmax(model.y_pred)

    if (j % 50 == 0):
        model.forward_propagation(X, False)
        res = softmax(model.y_pred)
        log.append(log_loss(res, y))
        dx_log.append(dx_log_loss(y, res))
        

    #Backpropagation
    model.backward_propagation(y)
    model.update(learning_rate, beta1, beta2, j)

model.forward_propagation(X, False)
res = softmax(model.y_pred)
print("y\n", y)
print("Loss\n", log_loss(res, y))
print("ACTIVATION\n", res)
print("ERREEUR\n", res - y)

grah(log, dx_log)

