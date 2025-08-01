
import  numpy as np
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
    for i in range(1, len(dimension)+1):

        k_size = dimension[str(i)][0]
        o_size = ouput_shape(input_size, k_size, padding, stride)
        input_size = o_size

        parametres["K" + str(i)] = np.random.randn(k_size**2, 1)
        parametres["b" + str(i)] = np.random.randn(o_size**2, 1)
        parametres["f" + str(i)] = dimension[str(i)][1]

    return parametres

def foward_propagation(X, parametres):

    activation = {"A0" : X}
    C = len(parametres) // 3

    for c in range(1, C+1):
        Z = activation["A" + str(c-1)].dot(parametres["K" + str(c)]) + parametres["b" + str(c)]
        Z = Z.reshape((np.int8(np.sqrt(Z.size)), np.int8(np.sqrt(Z.size))))

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
    next_dZ = np.zeros((dZ.shape[0]+1, dZ.shape[1]+1))

    for i in range(dZ.shape[0]+1):
        for j in range(dZ.shape[1]+1):
            next_dZ[i, j] = np.dot(new_dZ[i:i + k_size_sqrt, j:j + k_size_sqrt].flatten(), K[::-1].flatten())

    return next_dZ

def back_propagation(activation, parametres, y):

    C = len(parametres) // 3

    dZ = activation["A" + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):
        dK = np.zeros(parametres["K" + str(c)].shape)
        for i in range(X.shape[1]):
            dK[i, 0] = np.dot(activation["A" + str(c-1)][:, i], dZ.flatten())

        gradients["dK" + str(c)] = dK
        gradients["db" + str(c)] = dZ.reshape((dZ.size, -1))

        if c > 1:
            k_size_sqrt = np.int8(np.sqrt(parametres["K" + str(c)].shape[0]))
            dZ = convolution(dZ, parametres["K" + str(c)], k_size_sqrt)
            

    return gradients


def log_loss(A, y):
     A = A.reshape(y.shape)
     return  -1/y.size * np.sum( y*np.log(A) + (1-y)*np.log(1-A))



def update(gradients, parametres, learning_rate):
    
    C = len(parametres) // 3

    for c in range(1, C+1):
        parametres["K" + str(c)] = parametres["K" + str(c)] - learning_rate * gradients["dK" + str(c)]
        parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]
    
    return parametres


def reshape(X, K):

    k_size = K.size
    k_size_sqrt = np.int8(np.sqrt(k_size))
    new_X = np.array([])
    
    for i in range(0, X.shape[0]-1):
        for j in range(0, X.shape[1]-1):
            new_X = np.append(new_X, X[i:i + k_size_sqrt, j:j + k_size_sqrt])

    
    return new_X.reshape(((X.shape[0]-1)*(X.shape[1]-1)), k_size)

X = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]])

y = np.array([[0, 1, 0], 
              [1, 0, 1],
              [0, 1, 0]])

#Initialisation
learning_rate = 0.01
nb_iteration = 5000


dimensions = {}
dimensions = {"1" : (2, "relu"), "2" : (2, "sigmoide")}
parametres = initialisation(X, dimensions, 0, 1)

K = parametres["K1"]
X = reshape(X, K)

print("\nData\n",X)
for keys, values in parametres.items():
    print(keys)
    print(values)

l_array = np.array([])
C = len(parametres) // 3
for _ in range(nb_iteration):

    activations = foward_propagation(X, parametres)
    gradients = back_propagation(activations, parametres, y)
    parametres = update(gradients, parametres, learning_rate)

    l_array = np.append(l_array, log_loss(activations["A" + str(C)], y))


print("\nFinal activation\n",activations["A" + str(C)])
for keys, values in parametres.items():
    print(keys)
    print(values)

plt.figure()
plt.title("Evolution du cout en fonction des itérations")
plt.plot(l_array)
plt.show()

plt.figure(figsize=(16,8))
for i in range(1,3):
    plt.subplot(1,2, i)
    plt.imshow(parametres["K" + str(i)].reshape((2, 2)), cmap="gray")
    plt.title("Kernel")
    plt.tight_layout()
    plt.axis("off")
    plt.colorbar()
plt.show() 

plt.figure(figsize=(16,8))
for i in range(1,3):
    plt.subplot(1,2, i)
    plt.imshow(parametres["b" + str(i)].reshape((np.int8(np.sqrt(parametres["b" + str(i)].size)), -1)), cmap="gray")
    plt.title("Biais")
    plt.tight_layout()
    plt.axis("off")
    plt.colorbar()
plt.show() 

plt.figure()
plt.title("Y prediction")
plt.imshow(activations["A" + str(C)].reshape((np.int8(np.sqrt(activations["A" + str(C)].size)), -1)), cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()
