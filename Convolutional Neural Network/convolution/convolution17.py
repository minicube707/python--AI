
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
============================
========Documentation=======
============================

Le but de ce CNN est de transforme les activation en matrice
Allow to pass AxNxN grid to AxBxC with A the number of layer, B the number of pixel and C the size of the kernel. To do cross product
The kernel are shaped AxBxC A the number of layer, B the size of the kernel (ex:3*3=9) and C = 1.
The biais are shaped AxBxC A the number of layer, B the size of the output (ex:3*3=9) and C = 1. To do cross product

A : Activation in memory (Always in line format)
K : Kernel
b : bias 
X : input
Z : New activaton
dZ : derivative of the activation
y : label
"""



"""
============================
==========Fonction==========
============================
"""
def sigmoïde(X):
    return 1/(1 + np.exp(-X))

def relu(X):
    return np.where(X < 0, 0, X)

def max(X):
    a = np.int8(np.sqrt(X.shape[1]))
    return np.max(X, axis=2).reshape((1, a, a))

def correlate(A, K, b, x_size):

    #First cross product
    Z = A[0].dot(K[0])
    
    #Other cross product if the number of layer is GT 1
    for i in range(1, A.shape[0]):
        Z = np.add(Z, A[i].dot(K[i]))

    Z = Z.reshape((1, x_size, x_size))
    Z = np.clip(Z, -88, 88)
    return Z


def convolution(dZ, K, k_size_sqrt):
    #new_dz is intput with a pas to do the the cross product with all value
    new_dZ = np.pad(dZ, pad_width=((0, 0), (k_size_sqrt - 1, k_size_sqrt - 1), (k_size_sqrt - 1, k_size_sqrt - 1)), mode='constant', constant_values=0)

    #next_dz is the output
    next_dZ = np.zeros((dZ.shape[0], dZ.shape[1]+k_size_sqrt-1, dZ.shape[2]+k_size_sqrt-1))

    for k in range(next_dZ.shape[0]):
        for i in range(next_dZ.shape[1]):
            for j in range(next_dZ.shape[2]): 
                next_dZ[k, i, j] = np.dot(new_dZ[k, i:i + k_size_sqrt, j:j + k_size_sqrt].flatten(), K[k][::-1].flatten())

    return next_dZ

def ouput_shape(input_size, k_size, stride, padding):
    return np.int8((input_size - k_size + padding) / stride +1)




"""
============================
======Fonction du CNN=======
============================
"""
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

    print("\nkernel size, stride, padding, nb_kernel, type layer, function")
    for keys, values in dimensions.items():
        print(keys, values)

"""
error_initialisation:
=========DESCRIPTION=========
Print message if an error is dectecedd

=========INPUT=========
int         x_shape1 :          taile en largueur de l'input
list        list_size :         list of all activation shape with padding
dict        dimensions :         all the information on how is built the CNN
int         input_size :        the size in row of the input activation 
int         previ_input_size :  the size in row of the previous input activation
string      type_layer :        the type of layer 
string      fonction :          the type of function
int         stride :            how many pixel the kernel move 

=========OUTPUT=========
void
"""
def error_initialisation(x_shape1, list_size, dimensions, input_size, previ_input_size, type_layer, fonction, stride):

    if input_size < 1:
        show_information(x_shape1, list_size, dimensions)
        raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
    if previ_input_size % input_size != 0 and stride != 1:
        show_information(x_shape1, list_size, dimensions)
        raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
    
    if type_layer not in ["kernel", "pooling"]:
        show_information(x_shape1, list_size, dimensions)
        raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pooling' or 'kernel'.")
    
    if fonction not in ["relu", "sigmoide", "max"]:
        show_information(x_shape1, list_size, dimensions)
        raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu' or 'sigmoide', 'max'.")


"""
initialisation_extraction:
=========DESCRIPTION=========
Extrait all the information inside the dict

=========INPUT=========
dict    dimensions :    all the information on how is built the CNN
int     i :             the stage of the CNN

=========OUTPUT=========
int     k_size :        the size in row of kernel
int     stride :        how many pixel the kernel move  
int     padding :       how many pixel we add to the activation 
int     nb_kernel :     how many kernel
string  type_layer :    the type of layer 
string  fonction :      the type of function
"""
def initialisation_extraction(dimensions, i):
    #Kernel size, stride, padding, nb_kernel, type layer, function

    k_size = dimensions[str(i)][0]
    stride = dimensions[str(i)][1]
    padding = dimensions[str(i)][2]
    nb_kernel = dimensions[str(i)][3]
    type_layer = dimensions[str(i)][4]
    fonction = dimensions[str(i)][5]

    return k_size, stride, padding, nb_kernel, type_layer, fonction


"""
initialisation_pooling:
=========DESCRIPTION=========
Set the value for pooling operation

=========INPUT=========
dict    parametres :    dictionary to fill with the pooling operation
int     k_size :        the size in row of the kernel
string  type_layer :    the type of layer 
string  fonction :      the type of function
int     i :             the stage of the CNN

=========OUTPUT=========
dict    parametres :    containt all the information for the pooling operation
"""
def initialisation_pooling(parametres, k_size, type_layer, fonction, i):

    parametres["K" + str(i)] = k_size**2
    parametres["b" + str(i)] = None
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction
    
    return parametres


"""
initialisation_kernel:
=========DESCRIPTION=========
Set the value for kernel operation, the update operation

=========INPUT=========
dict    parametres :        dictionary to fill with the kernel information
dict    parametres_grad :   dictionary to fill with the update information
int     k_size :            the size in row of the kernel
int     o_size :            the size in row of the output
int     nb_kernel :         the number of kernel
string  type_layer :        the type of layer 
string  fonction :          the type of function
int     i :                 the stage of the CNN

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
"""
def initialisation_kernel(parametres, parametres_grad, k_size, o_size, nb_kernel, type_layer, fonction, i):

    parametres["K" + str(i)] = np.random.randn(nb_kernel, k_size**2, 1)
    parametres["b" + str(i)] = np.random.randn(nb_kernel, o_size**2, 1)
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction

    parametres_grad["m" + str(i)] = np.zeros((nb_kernel, k_size**2, 1))
    parametres_grad["v" + str(i)] = np.zeros((nb_kernel, k_size**2, 1))

    return parametres, parametres_grad


"""
initialisation_calcul:
=========DESCRIPTION=========
Preproce the information to built the CNN

=========INPUT=========
int     x_shape1 :      the size in row of the 2nd  (2/3) of the input
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    dimension :     all the information on how is built the CNN
list    list_size :     list of all activation shape with padding
"""
def initialisation_calcul(x_shape1, dimensions, padding_mode):

    list_size = []
    input_size =  x_shape1
    previ_input_size = input_size

    for i in range(1, len(dimensions)+1):

        k_size, stride, padding, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)

        #If the input doesn't match perfectly with the kernel and padding and is in mode auto-correction, the system correct the mistake and add the right padding
        if input_size % stride != 0 and padding_mode == "auto":
            padding = stride - input_size % stride
            dimensions[str(i)] = k_size, stride, padding, nb_kernel, type_layer, fonction
            list_size[-1] = input_size + padding

        o_size = ouput_shape(input_size, k_size, stride, padding)
        previ_input_size = input_size + padding
        input_size = o_size

        list_size.append(input_size)
        error_initialisation(x_shape1, list_size, dimensions, input_size, previ_input_size, type_layer, fonction, stride)

    return dimensions, list_size

"""
initialisation_affectation:
=========DESCRIPTION=========
Set all the value to built the CNN

=========INPUT=========
dict    dimensions :    all the information on how is built the CNN
list    list_size :     list of all activation shape with padding

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
"""
def initialisation_affectation(dimensions, list_size):

    parametres = {}
    parametres_grad = {}
    for i in range(1, len(dimensions)+1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)
        o_size = list_size[i-1]

        if type_layer == "kernel":
            parametres, parametres_grad = initialisation_kernel(parametres, parametres_grad, k_size, o_size, nb_kernel, type_layer, fonction, i)

        elif type_layer == "pooling":
            parametres = initialisation_pooling(parametres, k_size, type_layer, fonction, i)

    return parametres, parametres_grad


"""
initialisation:
=========DESCRIPTION=========
Set all the value to built the CNN

=========INPUT=========
int     x_shape1 :      the size in row of the 2nd  (2/3) of the input
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
dict    dimension :         all the information on how is built the CNN
tuple   list_size:          tuple of all activation shape with padding
"""
def initialisation(x_shape1, dimensions, padding_mode):

    dimensions, list_size = initialisation_calcul(x_shape1, dimensions, padding_mode)
    parametres, parametres_grad = initialisation_affectation(dimensions, list_size)

    return parametres, parametres_grad, dimensions, tuple(list_size)


"""
function_activation:
=========DESCRIPTION=========
Function that centrelize all the use to process the CNN

=========INPUT=========
numpy.array     A :             the activation matrice
numpy.array     K :             the kernel matrice           
numpy.array     b :             the biais matrice           
string          mode :          the type of activation function we use
string          type_layer :    the type of layer 
int             k_size :        the size in row of the kernel
int             x_size :        the size in row of the activation matrice
int             stride :        how many pixel the kernel move  
int             padding :       how many pixel we add to the activation 

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def function_activation(A, K, b, mode, type_layer, k_size, x_size, stride, padding):

    #Activation are in line format
    if type_layer == "kernel":
        Z = correlate(A, K, b, x_size)

    else:
        Z = A

    #Kernel part
    if mode == "relu":
        Z = relu(Z)

    elif mode == "sigmoide":
       Z = sigmoïde(Z)
    
    #Pooling part
    elif mode == "max":
        Z = max(Z)

    #Activation are in square format
    if padding != None:
        Z = add_padding(Z, padding)
    if k_size != None:
        Z = reshape(Z, k_size , x_size, stride, padding)  

    #Activation are in line format
    return Z


"""
foward_propagation:
=========DESCRIPTION=========
Pass the input into the activation functions for the foreward propagation

=========INPUT=========
numpy.array     X :             the features,input of the CNN
dict            parametres :    containt all the information for the kernel operation
tuple           tuple_size :    tuple of all activation shape with padding
dict            dimensions :    all the information on how is built the CNN

=========OUTPUT=========
dict            activation :     containt all the activation during the foreward propagation
"""
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


        #This part is to get data for the reshape
        #There is no information for the last reshape
        if c < C:
            k_size = dimensions[str(c+1)][0]
            stride = dimensions[str(c+1)][1]
        
        #The information for the padding is at the next step
        if c+1 < C:
           padding = dimensions[str(c+2)][2] 

        activation["A" + str(c)] = function_activation(A, K, b, mode, type_layer, k_size, x_size, stride, padding)

    return activation


"""
back_propagation:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            parametres :    containt all the information for the kernel operation
dict            dimensions :    all the information on how is built the CNN
numpy.array     y :             the target, the objectif of the CNN
tuple           tuple_size :    tuple of all activation shape with padding

=========OUTPUT=========
dict           gradients :     containt all the gradient need for the update
"""
def back_propagation(activation, parametres, dimensions, y, tuple_size):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(dimensions.keys())
    dZ = activation["A" + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):
        
        #Remove the padding
        #Activation are in square format
        dZ = dZ[:,:tuple_size[c-1], :tuple_size[c-1]]
        
        if parametres["l" + str(c)] == "pooling":
            
            # Trouve les valeurs maximales et leurs indices le long de l'axe 2
            #Reshape dz to (A,BxC)
            max_dZ = dZ.reshape(dZ.shape[0], dZ.size//dZ.shape[0])

            #Get the max value, before the operation max in foreword propagation
            max_indices = np.argmax(activation["A" + str(c-1)], axis=2)

            # Initialise le résultat avec des zéros
            result = np.zeros_like(activation["A" + str(c-1)])

            # Utilise un indexage avancé pour placer les valeurs maximales
            batch_indices = np.arange(activation["A" + str(c-1)].shape[0])[:, None]
            row_indices = np.arange(activation["A" + str(c-1)].shape[1])[None, :]

            #Use a mask, everywhere is 0, exept where the max value while be take
            result[batch_indices, row_indices, max_indices] = max_dZ

            # Affichage
            dZ = deshape(result, dimensions[str(c)][0], dimensions[str(c)][1])
            
            
        elif parametres["l" + str(c)] == "kernel":
            
            #Create a table for each dx of the kernel
            dK = np.zeros((activation["A" + str(c-1)].shape[0], activation["A" + str(c-1)].shape[2], 1))

            for i in range(activation["A" + str(c-1)].shape[0]):        #For each layer
                for j in range(activation["A" + str(c-1)].shape[2]):    #For each weight
                    dK[i, j, 0] = np.dot(activation["A" + str(c-1)][i, :, j], dZ.flatten())
            
            #Add the result in the dictionary
            gradients["dK" + str(c)] = dK
            gradients["db" + str(c)] = dZ.reshape((dZ.size, -1))
            
            if c > 1:
                dZ = convolution(dZ, parametres["K" + str(c)], dimensions[str(c)][0])

    return gradients


"""
update:
=========DESCRIPTION=========
Update the kernel and the biais, to improve the accuracy of the CNN

=========INPUT=========
dict            gradients :         containt all the gradient need for the update
dict            parametres :        containt all the information for the kernel operation
dict            parametres_grad :   containt all the information for the update operation
float           learning_rate :     constante to slow down the update of the parametre
float           beta1 :             constante for Adam
float           beta2 :             constante for Adam
int             C :                 constante the number of stage in CNN

=========OUTPUT=========
dict            parametres :        containt all the information for the kernel operation
"""
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


"""
============================
=======Shape fonction ======
============================
"""
"""
reshape:
=========DESCRIPTION=========
Allow to pass nxn grid to axb with a the number of placement and b the size of the kernel. To do cross product

=========INPUT=========
numpy.array     X :             the activation matrice
int             k_size_sqrt :   the size in row of the kernel
int             x_size_sqrt :   the size in row of the activation matrice
int             stride :        how many pixel the kernel move  
int             padding :       how many pixel we add to the activation 

=========OUTPUT=========
numpy.array      :             the activation matrice
"""
def reshape(X, k_size_sqrt, x_size_sqrt, stride, padding):

    k_size = k_size_sqrt**2
    new_X = np.array([])
    
    for k in range(X.shape[0]):
        for i in range(0, X.shape[1]-k_size_sqrt+1, stride):
            for j in range(0, X.shape[2]-k_size_sqrt+1, stride):
                new_X = np.append(new_X, X[k, i:i + k_size_sqrt, j:j + k_size_sqrt])

    o_size = ouput_shape(x_size_sqrt, k_size_sqrt, stride, padding)
    return new_X.reshape(X.shape[0], (o_size)**2, k_size)


"""
deshape:
=========DESCRIPTION=========
#Is the inverse function of reshape. Allow to pass ABxC to AxnNxN

=========INPUT=========
numpy.array     X :             the activation matrice
int             k_size_sqrt :   the size in row of the kernel
int             stride :        how many pixel the kernel move  

=========OUTPUT=========
numpy.array      :             the activation matrice
"""
def deshape(X, k_size_sqrt, stride):

    input_size = np.int8(np.sqrt(X.shape[1]*X.shape[2]))
    new_X = np.array([])
    
    step1 = input_size//stride
    step2 = k_size_sqrt

    for i in range(0, X.shape[0], step1):
        for j in range(0, X.shape[1], step1):
            for k in range(0, X.shape[2], step2):
                new_X = np.append(new_X, X[i, j:j + step1, k:k + step2])

    new_X = new_X.reshape((1, input_size ,input_size))
    return new_X


"""
deadd_paddinghape:
=========DESCRIPTION=========
Add zeros to the bottom right corner to fit perfectly with the kernel

=========INPUT=========
numpy.array     X :             the activation matrice
int             padding :       how many pixel we add to the activation 

=========OUTPUT=========
numpy.array      :             the activation matrice
"""
def add_padding(X, padding):
    return np.pad(X, pad_width=((0, 0), (0, padding), (0, padding)), mode='constant', constant_values=0)


"""
============================
Evaluation Metrics Function
============================
"""
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
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    nb_iteration = 20_000


    x_shape = 10
    X = np.random.rand(x_shape, x_shape)

    if len(X.shape) == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])

    dimensions = {}
    #Kernel size, stride, padding, nb_kernel, type layer, function
    dimensions = {"1" :(3, 1, 0, 1, "kernel", "relu"),
                  "2" :(2, 2, 0, 1, "pooling", "max"),
                  "3" :(2, 1, 0, 1, "kernel", "sigmoide")}

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
    plt.title("Fonction Cout en fonction des itérations")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(a_array, label="Accuracy du train_set")
    plt.title("L'acccuracy en fonction des itérations")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(b_array, label="Variation de l'apprentisage")
    plt.title("Deriver de l'acccuracy en fonction des itérations")
    plt.legend()

    plt.show()

    layer_kernel_count = 0
    kernel_count = 0
    keys_with_kernel = []
    for key, value in dimensions.items():
        # Si "kernel" est dans la valeur
        if "kernel" in value:
            layer_kernel_count += 1
            kernel_count += value[3]
            keys_with_kernel.append(key)


    plt.figure(figsize=(16, 8))
    # Compteur global pour les sous-graphes
    subplot_index = 1

    for i in range(1, layer_kernel_count + 1):
        a = keys_with_kernel[i - 1]  # Nom/clé de la couche, ex: 1, 2, etc.

        kernels = parametres["K" + str(a)]
        num_kernels = kernels.shape[0]

        for x in range(num_kernels):
            # Calculer la taille de l'image du noyau (supposée carrée)
            kernel_size = int(np.sqrt(kernels[x].size))
            plt.subplot(1, layer_kernel_count + num_kernels, subplot_index)
            plt.imshow(kernels[x].reshape(kernel_size, kernel_size), cmap="gray")
            plt.title(f"Kernel {a} : {x}")
            plt.axis("off")
            plt.colorbar()
            subplot_index += 1

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(16, 8))
    # Compteur global pour les sous-graphes
    subplot_index = 1

    for i in range(1, layer_kernel_count + 1):
        a = keys_with_kernel[i - 1]  # Nom/clé de la couche, ex: 1, 2, etc.

        kernels = parametres["b" + str(a)]
        num_kernels = kernels.shape[0]

        for x in range(num_kernels):
            # Calculer la taille de l'image du noyau (supposée carrée)
            kernel_size = int(np.sqrt(kernels[x].size))
            plt.subplot(1, layer_kernel_count + num_kernels, subplot_index)
            plt.imshow(kernels[x].reshape(kernel_size, kernel_size), cmap="gray")
            plt.title(f"Biais {a} : {x}")
            plt.axis("off")
            plt.colorbar()
            subplot_index += 1

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(16, 8))

    # ---- Affichage de la prédiction ----
    plt.subplot(1, 2, 1)
    plt.title("Y prediction")

    A = activations["A" + str(C)]
    size = int(np.sqrt(A.size))  # On suppose que c'est une image carrée
    plt.imshow(A.reshape(size, size), cmap="gray")
    plt.axis("off")
    plt.colorbar()

    # ---- Affichage de la vérité réelle ----
    plt.subplot(1, 2, 2)
    plt.title("Y")
    plt.imshow(y, cmap="gray")
    plt.axis("off")
    plt.colorbar()

    # ---- Mise en page finale ----
    plt.tight_layout()
    plt.show()

main()
