
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#Fonction
def sigmoïde(X):
    return 1/(1 + np.exp(-X))

def relu(X):
    return np.where(X < 0, 0, X)


def ouput_shape(input_size, k_size, padding, stride):
    return np.int8((input_size - k_size + 2*padding)/stride +1)


def initialisation(X, dimension, padding, stride):

    parametres = {}
    input_size =  X.shape[0]
    print("\nDétail de la convolution")
    for i in range(1, len(dimension)+1):
        
        print(f"{input_size}->", end="")
        
        k_size = dimension[str(i)][0]
        o_size = ouput_shape(input_size, k_size, padding, stride)
        input_size = o_size

        parametres["K" + str(i)] = np.random.randn(k_size**2, 1)
        parametres["b" + str(i)] = np.random.randn(o_size**2, 1)
        parametres["f" + str(i)] = dimension[str(i)][1]

    print(input_size,"\n")
    return parametres

def foward_propagation(X, parametres):

    activation = {"A0" : X}
    C = len(parametres) // 3

    for c in range(1, C+1):
        Z = activation["A" + str(c-1)].dot(parametres["K" + str(c)]) + parametres["b" + str(c)]
        Z = Z.reshape((np.int8(np.sqrt(Z.size)), np.int8(np.sqrt(Z.size))))
        Z = np.clip(Z, -20, 20)

        if c < C:
            Z = reshape(Z, parametres["K" + str(c+1)])

        if parametres["f" + str(c)] == "relu":
            activation["A" + str(c)] =  relu(Z)

        elif parametres["f" + str(c)] == "sigmoide":
            activation["A" + str(c)] =  sigmoïde(Z)
        
        else:
            raise NameError(f"ERROR: Function parameter '{parametres['f' + str(c)]}' is not defined. Please correct with 'relu' or 'sigmoide'.")

    return activation

def convolution(dZ, K, k_size_sqrt):
    new_dZ = np.pad(dZ, pad_width=k_size_sqrt-1, mode='constant', constant_values=0)
    next_dZ = np.zeros((dZ.shape[0]+k_size_sqrt-1, dZ.shape[1]+k_size_sqrt-1))

    for i in range(dZ.shape[0]+k_size_sqrt-1):
        for j in range(dZ.shape[1]+k_size_sqrt-1): 
            next_dZ[i, j] = np.dot(new_dZ[i:i + k_size_sqrt, j:j + k_size_sqrt].flatten(), K[::-1].flatten())

    return next_dZ

def back_propagation(activation, parametres, y):

    C = len(parametres) // 3

    dZ = activation["A" + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):
        dK = np.zeros((activation["A" + str(c-1)].shape[1], 1))
        for i in range(activation["A" + str(c-1)].shape[1]):
            dK[i, 0] = np.dot(activation["A" + str(c-1)][:, i], dZ.flatten())

        gradients["dK" + str(c)] = dK
        gradients["db" + str(c)] = dZ.reshape((dZ.size, -1))

        if c > 1:
            k_size_sqrt = np.int8(np.sqrt(parametres["K" + str(c)].shape[0]))
            dZ = convolution(dZ, parametres["K" + str(c)], k_size_sqrt)
            

    return gradients


def log_loss(A, y):
     A = A.reshape(y.shape)
     epsilon = 1e-15 #Pour empecher les log(0) = -inf
     return  -1/y.size * np.sum( y*np.log(A+ epsilon) + (1-y)*np.log(1-A+ epsilon))



def update(gradients, parametres, learning_rate):
    
    C = len(parametres) // 3

    for c in range(1, C+1):
        parametres["K" + str(c)] = parametres["K" + str(c)] - learning_rate * gradients["dK" + str(c)]
        parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]
    
    return parametres


def reshape(X, K):

    k_size = K.size
    k_size_sqrt = np.int8(np.sqrt(k_size))
    x_size = X.size
    x_size_sqrt = np.int8(np.sqrt(x_size))

    new_X = np.array([])
    
    for i in range(0, X.shape[0]-k_size_sqrt+1):
        for j in range(0, X.shape[1]-k_size_sqrt+1):
            new_X = np.append(new_X, X[i:i + k_size_sqrt, j:j + k_size_sqrt])

    return new_X.reshape((x_size_sqrt-k_size_sqrt+1)**2, k_size)

def accuracy_score(y_pred, y_true):
    y_true = np.round(y_true, 3)
    y_pred = np.round(y_pred, 3)
    return np.count_nonzero(y_pred == y_true)  / y_true.size

def dx_log_loss(y_pred, y_true):
    return -1/y_true.size * np.sum((y_true)/(y_pred) - (1 - y_true)/(1 - y_pred))


#Initialisation
learning_rate = 0.01
nb_iteration = 10_000

x_shape = 28
X = np.random.rand(x_shape, x_shape)

dimensions = {}
dimensions = {"1": (7, "relu"), "2": (7, "relu"), "3": (7, "relu"), "4": (7, "sigmoide")}
parametres = initialisation(X, dimensions, 0, 1)

var = 0
for i in dimensions.values():
    var += i[0]-1

y_shape = x_shape - var
y = np.random.rand(y_shape, y_shape)

"""print("\nData\n",X)
print("\nLabel\n",y)"""

K = parametres["K1"]
X = reshape(X, K)

for keys, values in dimensions.items():
    print(keys, values)

"""print("\nData\n",X)
for keys, values in parametres.items():
    print(keys)
    print(values)"""

l_array = np.array([])
a_array = np.array([])
b_array = np.array([])
C = len(parametres) // 3
for _ in tqdm(range(nb_iteration)):

    activations = foward_propagation(X, parametres)
    gradients = back_propagation(activations, parametres, y)
    parametres = update(gradients, parametres, learning_rate)

    l_array = np.append(l_array, log_loss(activations["A" + str(C)], y))
    a_array = np.append(a_array, accuracy_score(activations["A" + str(C)].flatten(), y.flatten()))
    b_array = np.append(b_array, dx_log_loss(activations["A" + str(C)], y))


"""print("\nFinal activation\n",activations["A" + str(C)])
for keys, values in parametres.items():
    print("")
    print(keys)
    print(values)"""

plt.figure(figsize=(12,4))
plt.subplot(1, 3, 1)
plt.plot(l_array, label="Cost function")
plt.title("Fonction Cout")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(a_array, label="Accuracy du train_set")
plt.title("L'Acccuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(b_array, label="Variation de l'apprentisage")
plt.title("Deriver de la fonction cout")
plt.legend()

plt.show()

plt.figure(figsize=(16,8))
for i in range(1,len(dimensions)+1):
    plt.subplot(1,len(dimensions), i)
    plt.imshow(parametres["K" + str(i)].reshape((np.int8(np.sqrt(parametres["K" + str(i)].size)), -1)), cmap="gray")
    plt.title(f"Kernel{i}",)
    plt.tight_layout()
    plt.axis("off")
    plt.colorbar()
plt.show() 

plt.figure(figsize=(16,8))
for i in range(1,len(dimensions)+1):
    plt.subplot(1,len(dimensions), i)
    plt.imshow(parametres["b" + str(i)].reshape((np.int8(np.sqrt(parametres["b" + str(i)].size)), -1)), cmap="gray")
    plt.title(f"Biais{i}")
    plt.tight_layout()
    plt.axis("off")
    plt.colorbar()
plt.show() 

plt.figure(figsize=(16,8))

plt.subplot(1,2, 1)
plt.title("Y prediction")
plt.imshow(activations["A" + str(C)].reshape((np.int8(np.sqrt(activations["A" + str(C)].size)), -1)), cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.colorbar()

plt.subplot(1,2, 2)
plt.title("Y")
plt.imshow(y, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.colorbar()
plt.show() 


