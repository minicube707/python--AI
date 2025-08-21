
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#Fonction
def sigmoïde(X):
    return 1/(1 + np.exp(-X))

def relu(X):
    return np.where(X < 0, 0, X)

def dx_sigmoïde(X):
    A = sigmoïde(X)
    return A * (1 - A)

def dx_relu(X):
    return np.where(X < 0, 0, 1)

def max(X):
    a = np.int8(np.sqrt(X.shape[0]))
    return np.max(X, axis=1).reshape((a, a))

def correlate(A, K, b, x_size):
    Z = A.dot(K) + b
    Z = Z.reshape((x_size, x_size))
    Z = np.clip(Z, -88, 88)
    return Z

def convolution(dZ, K, k_size_sqrt):
    new_dZ = np.pad(dZ, pad_width=k_size_sqrt-1, mode='constant', constant_values=0)
    next_dZ = np.zeros((dZ.shape[0]+k_size_sqrt-1, dZ.shape[1]+k_size_sqrt-1))

    for i in range(dZ.shape[0]+k_size_sqrt-1):
        for j in range(dZ.shape[1]+k_size_sqrt-1): 
            next_dZ[i, j] = np.dot(new_dZ[i:i + k_size_sqrt, j:j + k_size_sqrt].flatten(), K[::-1].flatten())

    return next_dZ

def ouput_shape(input_size, k_size, padding, stride):
    return np.int8((input_size - k_size + 2*padding)/stride +1)


#Fonction dui CNN
def show_information(X, tuple_size, dimensions):
    print("\nDétail de la convolution")
    print(f"{X.shape[0]}->", end="")
    for i in range(len(tuple_size)):
        print(f"{tuple_size[i]}", end="")
        if i < len(tuple_size)-1:
            print("->", end="")
    print("")

    for keys, values in dimensions.items():
        print(keys, values)


def initialisation(X, dimension, padding):

    parametres = {}
    parametres_grad = {}
    list_size = []
    input_size =  X.shape[0]
    previ_input_size = input_size
    for i in range(1, len(dimension)+1):
        
        k_size = dimension[str(i)][0]
        stride = dimension[str(i)][1]
        type_layer = dimension[str(i)][2]
        fonction = dimension[str(i)][3]

        o_size = ouput_shape(input_size, k_size, padding, stride)
        previ_input_size = input_size
        input_size = o_size

        if input_size < 1:
            show_information(X, list_size, dimension)
            raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
        
        if type_layer == "kernel":
            list_size.append(input_size)
            parametres["K" + str(i)] = np.random.randn(k_size**2, 1)
            parametres["b" + str(i)] = np.random.randn(o_size**2, 1)
            parametres["l" + str(i)] = type_layer
            parametres["f" + str(i)] = fonction

            parametres_grad["m" + str(i)] = np.zeros((k_size**2, 1))
            parametres_grad["v" + str(i)] = np.zeros((k_size**2, 1))

        elif type_layer == "pooling":

            if previ_input_size%input_size !=0:
                show_information(X, list_size, dimension)
                raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
            
            list_size.append(input_size)
            parametres["K" + str(i)] = k_size**2
            parametres["b" + str(i)] = None
            parametres["l" + str(i)] = type_layer
            parametres["f" + str(i)] = fonction

        else:
            raise NameError(f"ERROR: Layer parameter '{type_layer}' is not defined. Please correct with 'pooling' or 'kernel'.")
        
    return parametres, parametres_grad, tuple(list_size)

def function_activation(A, K, b, mode, type_layer, k_size, x_size, stride):

    if type_layer == "kernel":
        Z = correlate(A, K, b, x_size)

    else:
        Z = A

    if mode == "relu":
        Z = relu(Z)

    elif mode == "sigmoide":
       Z = sigmoïde(Z)
    
    elif mode == "max":
        Z = max(Z)
        
    else:
        raise NameError(f"ERROR: Function parameter '{mode}' is not defined. Please correct with 'relu', 'sigmoide' or 'max'.")
    
    if k_size != None:
        Z = reshape(Z, k_size , x_size, stride)  

    return Z

def foward_propagation(X, parametres, tuple_size, dimensions):

    activation = {"A0" : X}
    C = len(parametres) // 4

    
    for c in range(1, C+1):
        A = activation["A" + str(c-1)]
        K = parametres["K" + str(c)]
        b = parametres["b" + str(c)]
        mode = parametres["f" + str(c)]
        type_layer = parametres["l" + str(c)]
        x_size = tuple_size[c-1]
        
        k_size = None
        stride = None
        if c < C:
            k_size = dimensions[str(c+1)][0]
            stride = dimensions[str(c+1)][1]
        
        activation["A" + str(c)] = function_activation(A, K, b, mode, type_layer, k_size, x_size, stride)

    return activation

def back_propagation(activation, parametres, dimensions, y):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(parametres) // 4
    dZ = activation["A" + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):

        if parametres["l" + str(c)] == "pooling":

            
            # Trouver les indices des valeurs maximales dans array1 (par ligne)
            max_indices = np.argmax(activation["A" + str(c-1)], axis=1)

            # Créer un tableau résultant avec des zéros
            result = np.zeros_like(activation["A" + str(c-1)])

            # Remplacer les valeurs maximales par les éléments de array2
            result[np.arange(activation["A" + str(c-1)].shape[0]), max_indices] = dZ.reshape((1, dZ.size))

            dZ = deshape(result, dimensions[str(c)][0], dimensions[str(c)][1])

        elif parametres["l" + str(c)] == "kernel":
            dK = np.zeros(parametres["K" + str(c)].shape)
            for i in range(activation["A" + str(c-1)].shape[1]):
                dK[i, 0] = np.dot(activation["A" + str(c-1)][:, i], dZ.flatten())

            gradients["dK" + str(c)] = dK
            gradients["db" + str(c)] = dZ.reshape((dZ.size, -1))

            if c > 1:
                activation_fonction = parametres["f" + str(c)]
                A = activation["A" + str(c)]
                dim = dimensions[str(c)]

                # Chose the correct derivative
                if activation_fonction == "relu":
                    dA = dx_relu(A)
                elif activation_fonction == "sigmoide":
                    dA = dx_sigmoïde(A)

                dA = deshape(dA, dim[0], dim[1])
                dZ *= dA

                # Apply convolution
                dZ = convolution(dZ, parametres["K" + str(c)], dim[0])

    return gradients

def update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2):
    
    C = len(parametres) // 4
    epsilon = 1e-8 #Pour empecher les log(0) = /0

    #Adam (Adaptativ Momentum)
    for c in range(1, C+1):
        if parametres["l" + str(c)] == "kernel":

            #Update moment
            parametres_grad["m" + str(c)] = beta1 * parametres_grad["m" + str(c)] + (1 - beta1) * gradients["dK" + str(c)]     # Première estimation des moments (moyenne des gradients)
            parametres_grad["v" + str(c)] = beta2 * parametres_grad["v" + str(c)] + (1 - beta2) * gradients["dK" + str(c)]**2  # Deuxième estimation des moments (moyenne des carrés des gradients)

            #Biais correction
            m_hat = parametres_grad["m" + str(c)] / (1 - beta1**(c+1))
            v_hat = parametres_grad["v" + str(c)] / (1 - beta2**(c+1))

            #Update weights
            parametres["K" + str(c)] = parametres["K" + str(c)] - (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
            parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]
    
    return parametres


#Shape fonction 
#Allow to pass nxn grid to axb with a the number of placement and b the size of the kernel. To do cross product
def reshape(X, k_size_sqrt, x_size_sqrt, stride):

    k_size = k_size_sqrt**2
    new_X = np.array([])
    
    for i in range(0, X.shape[0]-k_size_sqrt+1, stride):
        for j in range(0, X.shape[1]-k_size_sqrt+1, stride):
            new_X = np.append(new_X, X[i:i + k_size_sqrt, j:j + k_size_sqrt])

    o_size = ouput_shape(x_size_sqrt, k_size_sqrt, 0, stride)
    return new_X.reshape((o_size)**2, k_size)

#Is the inverse function of reshape. Allow to pass axb to nxn
def deshape(X, k_size_sqrt, stride):

    input_size = np.int8(np.sqrt(X.size))
    new_X = np.array([])
    
    step1 = input_size//stride
    step2 = k_size_sqrt
    for i in range(0, X.shape[0], step1):
        for j in range(0, X.shape[1], step2):
            new_X = np.append(new_X, X[i:i + step1, j:j + step2])

    return new_X.reshape((input_size,input_size))



#Evaluation Metrics Function
def mean_square_error(A, y):
    A = A.reshape(y.shape)
    return  1/(2*len(y))*np.sum((A-y)**2)

def accuracy_score(y_pred, y_true):
    y_true = np.round(y_true, 2)
    y_pred = np.round(y_pred, 2)
    return np.count_nonzero(y_pred == y_true) / y_true.size

def dx_mean_square_error(A, y):
    A = A.reshape(y.shape)
    return  -1/(len(y))*np.sum(A-y)


#Initialisation
learning_rate = 0.005
beta1 = 0.9
beta2 = 0.99
nb_iteration = 20_000


x_shape = 26
X = np.random.rand(x_shape, x_shape)

dimensions = {}
dimensions = {"1" :(3, 1, "kernel", "relu"),
              "2" :(2, 2, "pooling", "max"),
              "3" :(3, 1, "kernel", "relu"),
              "4" :(2, 2, "pooling", "max"),
              "5" :(3, 1, "kernel", "sigmoide")}

parametres, parametres_grad, tuple_size = initialisation(X, dimensions, 0)

show_information(X, tuple_size, dimensions)


input_size = X.shape[0]
for val in dimensions.values():
    o_size = ouput_shape(input_size, val[0], 0, val[1])
    input_size = o_size

y_shape = o_size
y = np.random.rand(y_shape, y_shape)

"""print("\nData\n",X)
print("\nLabel\n",y)"""

X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1])


"""print("\nData\n",X)
for keys, values in parametres_grad.items():
    print(keys)
    print(values)"""

l_array = np.array([])
a_array = np.array([])
b_array = np.array([])
C = len(parametres) // 4

#Here 
#the activation are in different shape, that allow the cross product for more efficy
#the kernel are vector to do cross product
#the gradient are vector
for _ in tqdm(range(nb_iteration)):

    activations = foward_propagation(X, parametres, tuple_size, dimensions)
    gradients = back_propagation(activations, parametres, dimensions, y)
    parametres = update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2)
    
    l_array = np.append(l_array, mean_square_error(activations["A" + str(C)], y))
    a_array = np.append(a_array, accuracy_score(activations["A" + str(C)].flatten(), y.flatten()))
    b_array = np.append(b_array, dx_mean_square_error(activations["A" + str(C)], y))


"""print("\nFinal activation\n",activations["A" + str(C)])
for keys, values in activations.items():
    print("")
    print(keys)
    print(values)

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

kernel_count = 0
keys_with_kernel = []
for key, value in dimensions.items():
    # Si "kernel" est dans la valeur
    if "kernel" in value:
        kernel_count += 1
        keys_with_kernel.append(key)

plt.figure(figsize=(16,8))
for i in range(1,kernel_count+1):

    a = keys_with_kernel[i-1]
    plt.subplot(1,kernel_count, i)
    plt.imshow(parametres["K" + str(a)].reshape((np.int8(np.sqrt(parametres["K" + str(a)].size)), -1)), cmap="gray")
    plt.title(f"Kernel {a}",)
    plt.tight_layout()
    plt.axis("off")
    plt.colorbar()
plt.show() 

plt.figure(figsize=(16,8))
for i in range(1,kernel_count+1):

    a = keys_with_kernel[i-1]
    plt.subplot(1,kernel_count, i)

    plt.imshow(parametres["b" + str(a)].reshape((np.int8(np.sqrt(parametres["b" + str(a)].size)), -1)), cmap="gray")
    plt.title(f"Biais {a}")
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


