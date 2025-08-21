
import  numpy as np
from    tqdm import tqdm
import  matplotlib.pyplot as plt

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
    a = np.int8(np.sqrt(X.shape[1]))
    return np.max(X, axis=2).reshape((1, a, a))

def correlate(A, K, b, x_size):

    Z = A[0].dot(K[0])
    for i in range(1, A.shape[0]):
        Z = np.add(Z, A[i].dot(K[i]))

    Z = np.add(Z, b)

    Z = Z.reshape((1, x_size, x_size))
    Z = np.clip(Z, -88, 88)
    return Z

def convolution(dZ, K, k_size_sqrt):
    new_dZ = np.pad(dZ, pad_width=((0, 0), (k_size_sqrt - 1, k_size_sqrt - 1), (k_size_sqrt - 1, k_size_sqrt - 1)), mode='constant', constant_values=0)
    next_dZ = np.zeros((dZ.shape[0], dZ.shape[1]+k_size_sqrt-1, dZ.shape[2]+k_size_sqrt-1))

    for k in range(next_dZ.shape[0]):
        for i in range(next_dZ.shape[1]):
            for j in range(next_dZ.shape[2]): 
                next_dZ[k, i, j] = np.dot(new_dZ[k, i:i + k_size_sqrt, j:j + k_size_sqrt].flatten(), K[k][::-1].flatten())

    return next_dZ

def ouput_shape(input_size, k_size, stride, padding):
    return np.int8((input_size - k_size + padding) / stride +1)




#Fonction du CNN
def show_information(x_shape0, tuple_size, dimensions):
    print("\nDétail de la convolution")
    print(f"{x_shape0}({dimensions['1'][2]})->", end="")

    for i in range(len(tuple_size)):
        if i < len(tuple_size)-1:
            print(f"{tuple_size[i] - dimensions[str(i+2)][2]}", end="")
            print(f"({dimensions[str(i+2)][2]})", end="")
            print("->", end="")

    print(f"{tuple_size[i]}")  
    print("")

    for keys, values in dimensions.items():
        print(keys, values)


def error_initialisation(x_shape1, list_size, dimension, input_size, previ_input_size, type_layer, fonction, stride):

    if input_size < 1:
        show_information(x_shape1, list_size, dimension)
        raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
    if previ_input_size%input_size !=0 and stride != 1:
        show_information(x_shape1, list_size, dimension)
        raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
    
    if type_layer not in ["kernel", "pooling"]:
        show_information(x_shape1, list_size, dimension)
        raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pooling' or 'kernel'.")
    
    if fonction not in ["relu", "sigmoide", "max"]:
        show_information(x_shape1, list_size, dimension)
        raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu' or 'sigmoide', 'max'.")


def initialisation_extraction(dimension, i):
    #Kernel size, stride, padding, nb_kernel, type layer, function

    k_size = dimension[str(i)][0]
    stride = dimension[str(i)][1]
    padding = dimension[str(i)][2]
    nb_kernel = dimension[str(i)][3]
    type_layer = dimension[str(i)][4]
    fonction = dimension[str(i)][5]

    return k_size, stride, padding, nb_kernel, type_layer, fonction

def initialisation_pooling(parametres, k_size, type_layer, fonction, i):

    parametres["K" + str(i)] = k_size**2
    parametres["b" + str(i)] = None
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction
    
    return parametres

def initialisation_kernel(parametres, parametres_grad, k_size, o_size, nb_kernel, type_layer, fonction, i):

    parametres["K" + str(i)] = np.random.randn(nb_kernel, k_size**2, 1)
    parametres["b" + str(i)] = np.random.randn(nb_kernel, o_size**2, 1)
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction

    parametres_grad["m" + str(i)] = np.zeros((nb_kernel, k_size**2, 1))
    parametres_grad["v" + str(i)] = np.zeros((nb_kernel, k_size**2, 1))

    return parametres, parametres_grad

def initialisation_calcul(x_shape1, dimension, padding_mode):

    list_size = []
    input_size =  x_shape1
    previ_input_size = input_size

    for i in range(1, len(dimension)+1):

        k_size, stride, padding, nb_kernel, type_layer, fonction = initialisation_extraction(dimension, i)

        #If the input doesn't match perfectly with the kernel and padding is in mode auto-correction, the system correct the mistake and add the right padding
        if input_size % stride != 0 and padding_mode == "auto":
            padding = stride - input_size % stride
            dimension[str(i)] = k_size, stride, padding, nb_kernel, type_layer, fonction
            list_size[-1] = input_size + padding

        o_size = ouput_shape(input_size, k_size, stride, padding)
        previ_input_size = input_size + padding
        input_size = o_size

        list_size.append(input_size)
        error_initialisation(x_shape1, list_size, dimension, input_size, previ_input_size, type_layer, fonction, stride)

    return dimension, list_size

def initialisation_affectation(dimension, list_size):

    parametres = {}
    parametres_grad = {}
    for i in range(1, len(dimension)+1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimension, i)
        o_size = list_size[i-1]

        if type_layer == "kernel":
            parametres, parametres_grad = initialisation_kernel(parametres, parametres_grad, k_size, o_size, nb_kernel, type_layer, fonction, i)

        elif type_layer == "pooling":
            parametres = initialisation_pooling(parametres, k_size, type_layer, fonction, i)

    return parametres, parametres_grad


def initialisation(x_shape1, dimension, padding_mode):

    dimension, list_size = initialisation_calcul(x_shape1, dimension, padding_mode)
    parametres, parametres_grad = initialisation_affectation(dimension, list_size)

    return parametres, parametres_grad, dimension, tuple(list_size)

def function_activation(A, K, b, mode, type_layer, k_size, x_size, stride, padding):

    
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
    
    if padding != None:
        Z = add_padding(Z, padding)
    if k_size != None:
        Z = reshape(Z, k_size , x_size, stride, padding)  

    return Z

def foward_propagation(X, parametres, tuple_size, dimensions):

    activation = {"A0" : X}
    C = len(dimensions.keys())

    for c in range(1, C+1):
        A = activation["A" + str(c-1)]
        K = parametres["K" + str(c)]
        b = parametres["b" + str(c)]
        mode = parametres["f" + str(c)]
        type_layer = parametres["l" + str(c)]
        x_size = tuple_size[c-1]
        
        k_size = None
        stride = 1
        padding = 0

        if c < C:
            k_size = dimensions[str(c+1)][0]
            stride = dimensions[str(c+1)][1]
            
        if c+1 < C:
           padding = dimensions[str(c+2)][2] 

        activation["A" + str(c)] = function_activation(A, K, b, mode, type_layer, k_size, x_size, stride, padding)

    return activation

def back_propagation(activation, parametres, dimensions, y, tuple_size):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(dimensions.keys())
    dZ = activation["A" + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):
        
        #Remove the padding
        dZ = dZ[:,:tuple_size[c-1], :tuple_size[c-1]]

        if parametres["l" + str(c)] == "pooling":
            
            # Trouve les valeurs maximales et leurs indices le long de l'axe 2
            max_dZ = dZ.reshape(dZ.shape[0], dZ.size//dZ.shape[0])
            max_indices = np.argmax(activation["A" + str(c-1)], axis=2)

            # Initialise le résultat avec des zéros
            result = np.zeros_like(activation["A" + str(c-1)])

            # Utilise un indexage avancé pour placer les valeurs maximales
            batch_indices = np.arange(activation["A" + str(c-1)].shape[0])[:, None]
            row_indices = np.arange(activation["A" + str(c-1)].shape[1])[None, :]
            result[batch_indices, row_indices, max_indices] = max_dZ

            # Affichage
            dZ = deshape(result, dimensions[str(c)][0], dimensions[str(c)][1])
            
            
        elif parametres["l" + str(c)] == "kernel":
            
            #Create a table for each dx of the kernel
            dK = np.zeros(parametres["K" + str(c)].shape)

            for i in range(activation["A" + str(c-1)].shape[0]):        #For each layer
                for j in range(activation["A" + str(c-1)].shape[2]):    #For each weight
                    dK[i, j, 0] = np.dot(activation["A" + str(c-1)][i, :, j], dZ.flatten())
            
            #Add the result in the dictionnaire
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

def update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2, C):
    
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
def reshape(X, k_size_sqrt, x_size_sqrt, stride, padding):

    k_size = k_size_sqrt**2
    new_X = np.array([])
    
    for k in range(X.shape[0]):
        for i in range(0, X.shape[1]-k_size_sqrt+1, stride):
            for j in range(0, X.shape[2]-k_size_sqrt+1, stride):
                new_X = np.append(new_X, X[k, i:i + k_size_sqrt, j:j + k_size_sqrt])

    o_size = ouput_shape(x_size_sqrt, k_size_sqrt, stride, padding)
    return new_X.reshape(X.shape[0], (o_size)**2, k_size)

#Is the inverse function of reshape. Allow to pass axb to nxn
def deshape(X, k_size_sqrt, stride):

    input_size = np.int8(np.sqrt(X.shape[1]*X.shape[2]))
    new_X = np.array([])
    
    step1 = input_size//stride
    step2 = k_size_sqrt

    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1], step1):
            for k in range(0, X.shape[2], step2):
                new_X = np.append(new_X, X[i, j:j + step1, k:k + step2])

    new_X = new_X.reshape((1, input_size ,input_size))
    return new_X

#Add zeros to the bottom right corner to fit perfectly with the kernel
def add_padding(X, padding):
    return np.pad(X, pad_width=((0, 0), (0, padding), (0, padding)), mode='constant', constant_values=0)



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



def main():
    #Initialisation
    learning_rate = 0.005
    beta1 = 0.9
    beta2 = 0.99
    nb_iteration = 20_000


    x_shape = 51
    X = np.random.rand(x_shape, x_shape)

    if len(X.shape) == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])

    dimensions = {}
    #Kernel size, stride, padding, nb_kernel, type layer, function
    dimensions = {"1" :(3, 1, 0, 1, "kernel", "relu"),
                  "2" :(2, 2, 0, 1, "pooling", "max"),
                  "3" :(3, 1, 0, 1, "kernel", "relu"),
                  "4" :(2, 2, 0, 1, "pooling", "max"),
                  "5" :(5, 1, 0, 1, "kernel", "sigmoide")}

    x_shape1 = X.shape[1]
    padding_mode = "auto"
    parametres, parametres_grad, dimensions, tuple_size = initialisation(x_shape1, dimensions, padding_mode)

    show_information(x_shape1, tuple_size, dimensions)


    input_size = x_shape1
    for val in dimensions.values():
        o_size = ouput_shape(input_size, val[0], val[1], val[2])
        input_size = o_size

    y_shape = o_size
    y = np.random.rand(y_shape, y_shape)

    """print("\nData\n",X)
    print("\nLabel\n",y)"""

    X = add_padding(X, dimensions["2"][2])
    X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1], dimensions["2"][2])

    """print("\nData\n",X)
    for keys, values in parametres_grad.items():
        print(keys)
        print(values)"""

    l_array = np.array([])
    a_array = np.array([])
    b_array = np.array([])
    C = len(dimensions.keys())

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector
    for _ in tqdm(range(nb_iteration)):

        activations = foward_propagation(X, parametres, tuple_size, dimensions)
        gradients = back_propagation(activations, parametres, dimensions, y, tuple_size)
        parametres = update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2, C)

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

main()