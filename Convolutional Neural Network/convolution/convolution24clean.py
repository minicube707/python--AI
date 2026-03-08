
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from numpy.lib.stride_tricks import as_strided

#Allow to show all tab with numpy
np.set_printoptions(linewidth=200, threshold=np.inf)

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
"""
sigmoïde:
=========DESCRIPTION=========
Apply the sigmoide function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def sigmoide(X):
    return 1/(1 + np.exp(-X))


"""
relu:
=========DESCRIPTION=========
Apply the relu function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def relu(X, alpha):
    return np.maximum(X, 0) + alpha * np.minimum(X, 0)


"""
dx_sigmoïde:
=========DESCRIPTION=========
Apply the derivate sigmoide function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def dx_sigmoide(X):
    return X * (1 - X)

"""
dx_relu:
=========DESCRIPTION=========
Apply the derivative relu function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def dx_relu(X, alpha):
    dx = np.ones_like(X)
    dx[X < 0] = alpha
    return dx


def tanh(X):
    return np.tanh(X)


def dx_tanh(X):
    return (1 - X**2)
            

"""
max:
=========DESCRIPTION=========
Return the max of each row of the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def max_pooling(X, x_size):
    n, m, _ = X.shape
    return np.max(X, axis=2).reshape(n, x_size, x_size)




"""
=========DESCRIPTION=========
Perform a correlation between two arrays (activation and kernel).

=========INPUT=========
A (np.ndarray): Activation matrix (shape: [in_channels, ...])
K (np.ndarray): Kernel matrix (shape: [out_channels, kernel_size])
b (np.ndarray): Bias vector (shape: [out_channels])
x_size (int): Size of the spatial dimension of the activation

=========OUTPUT=========
Z_concat (np.ndarray): Next activation array (shape: [out_channels, x_size, x_size])
"""
def correlate(A, K, b, x_size):
    """
    A: (L_A, NB_Dot_Product, K_Size)
    K: (NB_K, L_A, K_Size, one)
    b: (NB_K,)
    x_size: int, dimension spatiale finale
    """

    # On étend A pour avoir forme compatible
    # A : (1, L_A, NB_Dot_Product, K_Size)
    A_expanded = A[np.newaxis, :, :, :]  # ajout axe filtre NB_K

    # K : (NB_K, L_A, K_Size, one)
    # On veut multiplier A_expanded et K le long de K_Size

    # Pour la multiplication matricielle batch on peut utiliser einsum:
    # on veut multiplier pour chaque filtre i et chaque canal j :
    # A_expanded shape: (1, L_A, NB_Dot_Product, K_Size)
    # K shape:          (NB_K, L_A, K_Size, one)
    #
    # Produit sur K_Size: pour chaque (i, j), calculer (NB_Dot_Product, K_Size) dot (K_Size, one)
    # Résultat: (NB_K, L_A, NB_Dot_Product, one)
    
    prod = np.einsum('nadk,nako->nado', A_expanded, K)
    # prod shape: (NB_K, L_A, NB_Dot_Product, one)

    # Somme sur les canaux (L_A)
    Z = np.sum(prod, axis=1)  # shape (NB_K, NB_Dot_Product, one)

    # Ajout biais, reshape pour broadcasting
    Z += b

    # reshape en output spatiale
    Z = Z.reshape((Z.shape[0], x_size, x_size))

    # Clipping pour stabilité numérique
    Z = np.clip(Z, -88, 88)

    return Z



"""
convolution:
=========DESCRIPTION=========
Do the full convolution of two arrays

=========INPUT=========
numpy.array     dZ :            the derivated of the previous activation (what should be the activation)
numpy.array     K :             the kernel matrice

=========OUTPUT=========
numpy.array    next_dZ :       Array containe the derivated for the next layer
"""
def convolution(dZ, K, k_size, dZ_dim1, dZ_dim2):
     
    # Sortie (nb_kernel, nb_layers, k_size, k_size)
    K_dim0, K_dim1, K_dim2, K_dim3 = K.shape

    K = K.reshape(K_dim0, K_dim1, k_size, k_size)
    output = np.zeros((K_dim1, dZ_dim1 + k_size - 1, dZ_dim2 + k_size - 1))
    
    # Convolution pleine pour chaque filtre et chaque canal
    for i, dz in enumerate(dZ):  # nb_filters
        kernels = K[i]

        for c, kernel in enumerate(kernels): # nb_layers (canaux de sortie
            output[c] += correlate2d(dz, kernel, mode="full")

    return output

"""
ouput_shape:
=========DESCRIPTION=========
Calcul the ouput of a given array

=========INPUT=========
int             input_size  :   the size in row of the activation matrice
int             k_size :        the size in row of the kernel
int             stride :        how many pixel the kernel move  
int             padding :       how many pixel we add to the border of the activation

=========OUTPUT=========
int             the number of pixel in row for the ouput
"""
def calcul_output_shape(input_size, k_size, stride, padding):
    return np.int8((input_size - k_size + padding) / stride +1)


"""
============================
======Fonction du CNN=======
============================
"""
"""
show_information:
=========DESCRIPTION=========
Print the final setup of the CNN

=========INPUT=========
tuple   list_size_activation:       tuple of all activation shape with number of activation and padding
dict        dimensions :            all the information on how is built the CNN

=========OUTPUT=========
void
"""
def show_information(dimensions, input_size):

    print("\n============================")
    print("    INITIALISATION CNN")
    print("============================")

    print("\nDétail de la convolution :")
    print("Nb activation")
    print(f"{input_size[0]}", end="")
    print("->", end="")
    for i in range(1, len(dimensions)+1):

        if i < len(dimensions):
            print(f"{dimensions[str(i)][3]}", end="")
            print("->", end="")

    print(f"{dimensions[str(i)][3]}")  

    print("\nPadding")
    outpu_shape = input_size[1]
    for i in range(len(dimensions)):
        
        
        if i < len(dimensions):
            print(f"{outpu_shape}", end="")
            print(f"({dimensions[str(i+1)][2]})", end="")
            print("->", end="")

        outpu_shape = calcul_output_shape(outpu_shape, dimensions[str(i+1)][0], dimensions[str(i+1)][1], dimensions[str(i+1)][2])

    print(f"{outpu_shape}")  

    print("\nkernel size, stride, padding, nb_kernel, type layer, function")
    for keys, values in dimensions.items():
        print(keys, values)

"""
error_initialisation:
=========DESCRIPTION=========
Print message if an error is decteced

=========INPUT=========
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
def error_initialisation(dimensions, input_size, previ_input_size, type_layer, fonction, stride):

    if input_size < 1:
        show_information(dimensions, input_size)
        raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
    if previ_input_size % input_size != 0 and stride != 1:
        show_information(dimensions, input_size)
        raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
    
    if type_layer not in ["kernel", "pooling"]:
        show_information(dimensions, input_size)
        raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pooling' or 'kernel'.")
    
    if fonction not in ["relu", "sigmoide", "max", "tanh"]:
        show_information(dimensions, input_size)
        raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu', 'sigmoide', 'max' ou 'tanh'.")



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
int     padding :       how many pixel we add to the border of the activation
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
def initialisation_kernel(parametres, parametres_grad, k_size, fonction, i, list_size_activation):

    nb_kernel = list_size_activation[i][0]
    nb_layer =  list_size_activation[i-1][0]
    o_size = list_size_activation[i][1]

    k_shape = (nb_kernel, nb_layer, k_size**2, 1)

    if fonction == "relu":
        std = np.sqrt(2 / (nb_layer * k_size**2))
        K = np.random.randn(*k_shape).astype(np.float32) * std

    elif fonction == "tanh" or  fonction == "sigmoide":
        limit = np.sqrt(6 / (nb_layer + nb_kernel))
        K = (np.random.rand(*k_shape).astype(np.float32) * 2 - 1) * limit

    else:
        # Default to small random values
        K = np.random.randn(*k_shape).astype(np.float32) * 0.01

    b_shape = (nb_kernel, np.int64(o_size)**2, 1) #np.int64 avoid overflow with o_size**2
    b = np.zeros(b_shape).astype(np.float32)  # Bias souvent initialisé à 0

    parametres["K" + str(i)] = K
    parametres["b" + str(i)] = b

    parametres_grad["m" + str(i)] = np.zeros(k_shape).astype(np.float32)
    parametres_grad["v" + str(i)] = np.zeros(k_shape).astype(np.float32)

    return parametres, parametres_grad


"""
initialisation_calcul:
=========DESCRIPTION=========
Preproce the information to built the CNN

=========INPUT=========
int     x_shape1 :      the shape of the input
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    dimension :     all the information on how is built the CNN
list    list_size_activation :     list of all activation shape with number of activation and padding
"""
def initialisation_calcul(x_shape, dimensions, padding_mode):

    list_size_activaton = []
    list_size_activaton.append((x_shape[0], x_shape[1]))
    nb_activation  = x_shape[0]
    input_size =  x_shape[1]
    previ_input_size = input_size
    
    for i in range(1, len(dimensions)+1):

        k_size, stride, padding, nb_channel, type_layer, fonction = initialisation_extraction(dimensions, i)

        #If the input doesn't match perfectly with the kernel and padding and is in mode auto-correction, the system correct the mistake and add the right padding
        if input_size % stride != 0 and padding_mode == "auto":
            padding = stride - input_size % stride
            list_size_activaton[-1] = (list_size_activaton[-1][0], input_size + padding)

        if (dimensions[str(i)][4] == "kernel"):
            #Add the modificaton to the dict
            dimensions[str(i)] = k_size, stride, padding, nb_channel, type_layer, fonction
        
        else:
            nb_channel = nb_activation
            nb_channel = list_size_activaton[-1][0]
            dimensions[str(i)] = k_size, stride, padding, nb_channel, type_layer, fonction

        o_size = calcul_output_shape(input_size, k_size, stride, padding)
        previ_input_size = input_size + padding
        input_size = o_size
        nb_activation = dimensions[str(i)][3]
        
        list_size_activaton.append((nb_channel, input_size))
        error_initialisation(dimensions, input_size, previ_input_size, type_layer, fonction, stride)

    return dimensions, list_size_activaton

"""
initialisation_affectation:
=========DESCRIPTION=========
Set all the value to built the CNN

=========INPUT=========
dict    dimensions :    all the information on how is built the CNN
list    list_size_activation :     list of all activation shape with number of activation and padding

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
"""
def initialisation_affectation(dimensions, x_shape, list_size_activation):

    parametres = {}
    parametres_grad = {}
    tuple_mode_info = ()

    nb_layer = x_shape[0]
    o_size = np.int32(x_shape[1])
    C = len(dimensions)

    for i in range(1, C + 1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)
        o_size = np.int32(calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2]))

        if (i < C):
            o_size += dimensions[str(i+1)][2]

        # function: 0 = relu, 1 = sigmoide, 2 = tanh, 3 = max
        if fonction == "relu":
            fonction = 0

        elif fonction == "sigmoide":
            fonction = 1

        elif fonction == "tanh":
            fonction = 2

        elif fonction == "max":
            fonction = 3

        # mode: 0 = kernel, 1 = pooling
        if type_layer == "kernel":

            #Tuple_mode_info: mode, function, output_size, kernel_size, stride, padding
            tuple_mode_info += ((0, fonction, o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2]),)
            parametres, parametres_grad = initialisation_kernel(
                parametres, parametres_grad, k_size, fonction, i, list_size_activation)

        elif type_layer == "pooling":
            #Tuple_mode_info: mode, function, output_size, kernel_size, stride, padding
            tuple_mode_info += ((1, fonction, o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2]),)

        nb_layer = nb_kernel

    return parametres, parametres_grad, tuple_mode_info


"""
initialisation:
=========DESCRIPTION=========
Set all the value to built the CNN

=========INPUT=========
int     x_shape :       the shape of the input
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
dict    dimension :         all the information on how is built the CNN
tuple   list_size_activation:          tuple of all activation shape with number of activation and padding
"""
def initialisation(x_shape, dimensions, padding_mode):

    dimensions, list_size_activation = initialisation_calcul(x_shape, dimensions, padding_mode)
    parametres, parametres_grad, tuple_mode_info = initialisation_affectation(dimensions, x_shape, list_size_activation)

    return parametres, parametres_grad, tuple_mode_info


"""
pooling_activation:
=========DESCRIPTION=========
Activation of pooling

=========INPUT=========
numpy.array     A :                 the activation matrice

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def pooling_activation(A, x_size):
    Z = max_pooling(A, x_size)
    return Z


"""
kernel_activation:
=========DESCRIPTION=========
Activation of kernel

=========INPUT=========
numpy.array     A :                 the activation matrice
numpy.array     K :                 the kernel matrice           
numpy.array     b :                 the biais matrice   
int             x_size :            the size in row of the activation matrice        
string          mode :              the type of activation function we use

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def kernel_activation(X, K, b, x_size, mode, alpha):

    Z = correlate(X, K, b, x_size)

    # mode: 0 = relu, 1 = sigmoide, 2 = tanh
    if mode == 0:
        A = relu(Z, alpha)
    elif mode == 1:
        A = sigmoide(Z)
    elif mode == 2:
        A = tanh(Z)
    return A, Z



"""
foward_propagation:
=========DESCRIPTION=========
Pass the input into the activation functions for the foreward propagation

=========INPUT=========
numpy.array     X :                             the features,input of the CNN
dict            parametres :                    containt all the information for the kernel operation
tuple           tuple_mode_info:                tuple of all activation shape with number of activation and padding

=========OUTPUT=========
dict            activation :     containt all the activation during the foreward propagation
"""
def foward_propagation(X, parametres, tuple_mode_info, alpha):

    activation = {"A0" : X}
    C = len(tuple_mode_info)
    input_shape = X.shape[1]

    for c in range(1, C+1):
        A = activation["A" + str(c-1)]

        current_tuple = tuple_mode_info[c-1]
        type_layer, mode, x_size = current_tuple[:3]

        k_size = -1
        stride = 1
        padding = 0

        #This part is to get data for the reshape
        #There is no information for the last reshape
        if c < C:
            k_size, stride = tuple_mode_info[c][3:5]
        
        #The information for the padding is at the next step
        if c+1 < C:
            padding = tuple_mode_info[c+1][5]

        if type_layer == 0:

            K = parametres["K" + str(c)]
            b = parametres["b" + str(c)]
            A, Z = kernel_activation(A, K, b, x_size, mode, alpha)
            

        else:
            A = pooling_activation(A, x_size)
            Z = None

        #Activation are in square format
        A = add_padding(A, padding)

        if k_size != -1:
            A = reshape(A, k_size, stride, padding)  

        activation["A" + str(c)] = A 
        activation["Z" + str(c)] = Z

    return activation

"""
back_propagation_pooling:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got for the layer pooling

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            dimensions :    all the information on how is built the CNN
numpy.array     DZ :            the derivated of the previous activation (what should be the activation)

=========OUTPUT=========
numpy.array     DZ :            the derivated of this activation for the next step of backpropagation
"""
def back_propagation_pooling(A, o_size, k_size, stride, dZ):
    
    # Trouve les valeurs maximales et leurs indices le long de l'axe 2
    #Reshape dz to (A,BxC)
    max_dZ = dZ.reshape(dZ.shape[0], -1)

    #Get the max value, before the operation max in foreword propagation
    max_indices = np.argmax(A, axis=2)

    # Initialise le résultat avec des zéros
    result = np.zeros_like(A)

    # Utilise un indexage avancé pour placer les valeurs maximales
    batch_indices = np.arange(A.shape[0])[:, None]
    row_indices = np.arange(A.shape[1])[None, :]

    #Use a mask, everywhere is 0, exept where the max value while be take
    result[batch_indices, row_indices, max_indices] = max_dZ

    # Affichage
    dZ = deshape(result, k_size, stride, o_size*2)

    return dZ


"""
back_propagation_kernel:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got for the layer kernel

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            parametres :    containt all the information for the kernel operation
dict            dimensions :    all the information on how is built the CNN
dict            gradients  :    containt all the information for the update
numpy.array     dZ :            the derivated of the previous activation (what should be the activation)
int             c  :            which stage we are in backpropagatioin 

=========OUTPUT=========
dict            gradients :     containt all the gradient need for the update
numpy.array     DZ :            the derivated of this activation for the next step of backpropagation
"""
def back_propagation_kernel(activation, K, activation_fonction, k_size, gradients, dZ, c, alpha):
        
    #Create a table for each dx of the kernel
    L_A, NB_Dot_Product, K_Size = activation["A" + str(c-1)].shape
    NB_K, L_K, K_Size, one  = K.shape
    dZ_dim0, dZ_dim1, dZ_dim2 = dZ.shape

    dK = np.zeros((NB_K, L_K, K_Size, one))
    
    #For each kernel
    for i in range(NB_K):

        #For each activation
        for j in range(L_A):
            
            #For each weight
            for k in range(K_Size):
                
                dK[i, j, k] = np.dot(activation["A" + str(c-1)][j, :, k], dZ[i].flatten())


    #Add the result in the dictionary
    gradients["dK" + str(c)] = dK
    gradients["db" + str(c)]  = dZ.reshape((dZ_dim0, dZ_dim1 * dZ_dim2, 1))
            
    if c > 1:

        # Chose the correct derivative
        if activation_fonction == 0:
            dA = dx_relu(activation["Z" + str(c)], alpha)
        elif activation_fonction == 1:
            dA = dx_sigmoide(activation["A" + str(c)])
        elif activation_fonction == 2:
            dA = dx_tanh(activation["A" + str(c)])

        dZ *= dA

        # Apply convolution
        dZ = convolution(dZ, K, k_size, dZ_dim1, dZ_dim2)

    return gradients, dZ


"""
back_propagation:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got

=========INPUT=========
dict            activation :                    containt all the activation during the foreward propagation
dict            parametres :                    containt all the information for the kernel operation
dict            dimensions :                    all the information on how is built the CNN
numpy.array     y :                             the target, the objectif of the CNN
tuple           list_size_activation:           tuple of all activation shape with number of activation and padding

=========OUTPUT=========
dict           gradients :     containt all the gradient need for the update
"""
def back_propagation_CNN(activation, parametres, gradients, tuple_mode_info, y, alpha):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(tuple_mode_info)
    dZ = activation["A" + str(C)] - y
    
    for c in reversed(range(1, C+1)):

        #Remove the padding
        #Activation are in square format
        size = tuple_mode_info[c-1][2]
        dZ = dZ[:, :size, :size]
        
        mode = tuple_mode_info[c-1][0]
        if mode == 1:
           dZ = back_propagation_pooling(
                activation["A" + str(c-1)],
                tuple_mode_info[c-1][2],
                tuple_mode_info[c-1][3],
                tuple_mode_info[c-1][4],
                dZ
            )

        elif mode == 0:
            gradients, dZ = back_propagation_kernel(
                activation, 
                parametres["K" + str(c)], 
                tuple_mode_info[c-1][1], 
                tuple_mode_info[c-1][3],  
                gradients, dZ, c, alpha)


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
def update(gradients, parametres, parametres_grad, tuple_mode_info, learning_rate, beta1, beta2, C):
        
    epsilon = 1e-8 #Pour empecher les log(0) = /0
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2

    one_minus_expo_beta1 = (1 - beta1 + epsilon)
    one_minus_expo_beta2 = (1 - beta2 + epsilon)

    #Adam (Adaptativ Momentum)
    for c in range(1, C+1):
        if tuple_mode_info[c-1][0] == 0:

            #Update moment
            parametres_grad["m" + str(c)] = beta1 * parametres_grad["m" + str(c)] + one_minus_beta1 * gradients["dK" + str(c)]     # Première estimation des moments (moyenne des gradients)
            parametres_grad["v" + str(c)] = beta2 * parametres_grad["v" + str(c)] + one_minus_beta2 * gradients["dK" + str(c)]**2  # Deuxième estimation des moments (moyenne des carrés des gradients)

            #Biais correction
            m_hat = parametres_grad["m" + str(c)] / one_minus_expo_beta1
            v_hat = parametres_grad["v" + str(c)] / one_minus_expo_beta2

            #Update weights
            parametres["K" + str(c)] = parametres["K" + str(c)] - (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
            parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]


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
int             stride :        how many pixel the kernel move  
int             padding :       how many pixel we add to the border of the activation

=========OUTPUT=========
numpy.array      :             the activation matrice
"""
def reshape(X, k_size_sqrt, stride, padding):
    """
    X: (batch, H, W)
    k_size_sqrt: taille du kernel (ex: 2 pour 2x2)
    stride: pas du kernel
    padding: int
    """

    batch, H, W = X.shape
    k = k_size_sqrt
    out_h = (H - k)//stride + 1
    out_w = (W - k)//stride + 1

    # Calcul des strides pour créer les patches
    shape = (batch, out_h, out_w, k, k)
    strides = (X.strides[0], X.strides[1]*stride, X.strides[2]*stride, X.strides[1], X.strides[2])

    patches = as_strided(X, shape=shape, strides=strides)

    # Reformat pour im2col (batch, out_h*out_w, k*k)
    patches = patches.reshape(batch, out_h*out_w, k*k)

    return patches


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
def deshape(X, k, stride, input_size):
    """
    X: (batch, n_patches, k*k)
    Retourne: (batch, input_size, input_size)
    """
    batch, n_patches, k_flat = X.shape
    H = W = input_size

    # Dimensions de sortie par patch
    out_h = (H - k)//stride + 1
    out_w = (W - k)//stride + 1

    # Reshape X pour séparer le patch 2D
    patches = X.reshape(batch, out_h, out_w, k, k)

    # Calcul des strides pour placer les patches
    new_X = np.zeros((batch, H, W), dtype=X.dtype)

    # Accumulation vectorisée avec einsum
    # On fait: pour chaque patch, on ajoute k*k éléments au bon endroit
    for i in range(k):
        for j in range(k):
            new_X[:, i:i + stride*out_h:stride, j:j + stride*out_w:stride] += patches[:, :, :, i, j]

    return new_X


"""
add_padding:
=========DESCRIPTION=========
Add zeros to the bottom right corner to fit perfectly with the kernel

=========INPUT=========
numpy.array     X :             the activation matrice
int             padding :       how many pixel we add to the border of the activation

=========OUTPUT=========
numpy.array      :             the activation matrice
"""
def add_padding(X, padding):

    if padding == 0:
        return X 
    
    return np.pad(X, pad_width=((0, 0), (0, padding), (0, padding)), mode='constant', constant_values=0)


"""
============================
Evaluation Metrics Function
============================
"""

def dx_log_loss(y_pred, y_true):
    epsilon = 1e-15
    return - np.mean(np.sum((y_true / y_pred + epsilon) - (1 - y_true) / (1 - y_pred + epsilon)))

def log_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return  - np.mean(np.sum(y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)))

def accuracy_score(y_pred, y_true):
    y_true = np.round(y_true, 1)
    y_pred = np.round(y_pred, 1)
    return np.count_nonzero(y_pred == y_true) / y_true.size


def display_kernel(array_4d, type, stage, max_par_fig=12):
    if not isinstance(array_4d, np.ndarray) or array_4d.ndim != 4:
        raise ValueError("Entrée invalide : un array NumPy à 4 dimensions est requis (nb_kernels, nb_layers, height, width).")

    nb_kernels, nb_layers, h, w = array_4d.shape

    for kernel_idx in range(nb_kernels):
        total_layers = nb_layers

        for start in range(0, total_layers, max_par_fig):
            end = min(start + max_par_fig, total_layers)
            batch = array_4d[kernel_idx, start:end]

            n = batch.shape[0]
            cols = min(4, n)
            rows = (n + cols - 1) // cols

            plt.figure(figsize=(cols * 4, rows * 3))
            for i in range(n):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(batch[i], cmap='gray')
                plt.title(f'{type} K{kernel_idx} L{start + i}')
                plt.axis('off')
                plt.colorbar()

            plt.suptitle(f'Stage {stage} | Kernel {kernel_idx} (Layers {start} à {end - 1})', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

"""
display_layer:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
numpy.array     array_3d :      the activation matrice
string          type     :      string to inform if is the kernel matrice or biais matrice
string          stage    :      string to inform the stage of the in CNN      
=========OUTPUT=========
void
"""
def display_biais(array_3d, type, stage, max_par_fig=12):

    
    if not isinstance(array_3d, np.ndarray) or array_3d.ndim != 3:
        raise ValueError("Entrée invalide : un array NumPy à 3 dimensions est requis.")
    
    total = array_3d.shape[0]
    
    for start in range(0, total, max_par_fig):
        end = min(start + max_par_fig, total)
        batch = array_3d[start:end]

        n = batch.shape[0]
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(cols * 4, rows * 3))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(batch[i], cmap='gray')
            plt.title(f'{type} Couche {stage}: {start + i}')
            plt.axis('off')
            plt.colorbar()

        plt.suptitle(f'{type} - {stage} (couches {start} à {end - 1})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Laisser de l’espace pour le suptitle
        plt.show()

"""
display_kernel_and_biais:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
dict    parametres :    containt all the information for the pooling operation

=========OUTPUT=========
void
"""
def display_kernel_and_biais(parametres):
    for key, value in parametres.items():
        if isinstance(value, np.ndarray):


            if key.startswith('K'):
                sqrt = np.int8(np.sqrt(value.shape[2]))
                K = value.reshape(value.shape[0], value.shape[1], sqrt, sqrt)
                display_kernel(K, "Kernel", key[-1])

            elif key.startswith('b'):
                sqrt = np.int8(np.sqrt(value.shape[1]))
                B = value.reshape(value.shape[0], sqrt, sqrt)
                display_biais(B, "Biais", key[-1])


"""
display_comparaison_layer:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
numpy.array     y :             the target
numpy.array     y_pred :        the prediction of the model

=========OUTPUT=========
void
"""
def display_comparaison_layer(y, y_pred, max_par_fig=12):
    """
    Affiche chaque couche de deux tableaux 3D (y et y_pred) côte à côte,
    répartis sur plusieurs figures si nécessaire (max_par_fig par figure).
    """

    if y.shape != y_pred.shape or y.ndim != 3:
        raise ValueError("y et y_pred doivent être des arrays 3D de même forme (D, H, W)")

    total_couches = y.shape[0]

    for start in range(0, total_couches, max_par_fig):
        end = min(start + max_par_fig, total_couches)
        n = end - start

        cols = min(4, n)  # 4 paires par ligne
        rows = np.int8(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols, 3 * rows))

        # Assurer que axes est 2D même pour une seule ligne
        if rows == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n):
            layer_idx = start + i
            row = i // cols
            col = i % cols

            ax_y = axes[row, col * 2]
            ax_pred = axes[row, col * 2 + 1]

            im1 = ax_y.imshow(y[layer_idx], cmap='gray')
            ax_y.set_title(f'Y - Couche {layer_idx}')
            ax_y.axis('off')
            fig.colorbar(im1, ax=ax_y, fraction=0.046, pad=0.04)

            im2 = ax_pred.imshow(y_pred[layer_idx], cmap='gray')
            ax_pred.set_title(f'Prediction - Couche {layer_idx}')
            ax_pred.axis('off')
            fig.colorbar(im2, ax=ax_pred, fraction=0.046, pad=0.04)

        # Masquer les axes inutilisés
        total_axes = rows * cols * 2
        for j in range(n * 2, total_axes):
            row = j // (cols * 2)
            col = j % (cols * 2)
            axes[row, col].axis('off')

        plt.suptitle(f'Couches {start} à {end - 1}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

"""
display_info_learning:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
numpy.array     l_array :       list containt the loss during the trainnig
numpy.array     a_array:        list containt the accuracy during the trainnig
numpy.array     d_array:        list containt the derivative of loss during the trainnig

=========OUTPUT=========
void
"""
def display_info_learning(l_array, a_array, d_array):
    plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    plt.plot(l_array, label="Cost function")
    plt.title("Fonction Cout")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(a_array, label="Accuracy du train_set")
    plt.title("L'acccuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(d_array, label="Variation de l'apprentisage")
    plt.title("Deriver de la fonction cout")
    plt.legend()

    plt.show()


def main():
    #Initialisation
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    alpha = 0.001
    nb_iteration = 1_000

    x_shape = 28
    input_shape = (1, x_shape, x_shape)

    #X = np.random.rand(x_shape, x_shape)
    X = np.random.rand(x_shape * x_shape).reshape(x_shape, x_shape)

    if len(X.shape) == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])

    dimensions = {}
    #Kernel size, stride, padding, nb_kernel, type layer, function
    dimensions = {
        "1": (5, 1, 0, 32, "kernel", "relu"),
        "2": (2, 2, 0, 1, "pooling", "max"),
        "3": (3, 1, 0, 64, "kernel", "relu"),
        "4": (2, 2, 0, 1, "pooling", "max"),
        "5": (3, 1, 0, 64, "kernel", "relu")
    }
    
    padding_mode = "auto"
    parametres, parametres_grad, tuple_mode_info = initialisation (
    input_shape, dimensions, padding_mode)

    show_information(dimensions, input_shape)

    input_size = X.shape[1]
    for val in dimensions.values():
        o_size = calcul_output_shape(input_size, val[0], val[1], val[2])
        input_size = o_size

    C_CNN = len(dimensions.keys())
    y_shape = o_size
    y = np.random.rand(dimensions[str(C_CNN)][3], y_shape, y_shape)

    if len(dimensions) > 1:
        X = add_padding(X, dimensions["2"][2])
        X = reshape(X, dimensions["1"][0], dimensions["1"][1], dimensions["2"][2])

    else:
         X = reshape(X, dimensions["1"][0], dimensions["1"][1], 0)

    l_array = np.array([])
    a_array = np.array([])
    d_array = np.array([])

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector
    gradients = {}

    for i in tqdm(range(nb_iteration)):
        
        activations = foward_propagation(X, parametres, tuple_mode_info, alpha)
        back_propagation_CNN(activations, parametres, gradients, tuple_mode_info, y, alpha)
        update(gradients, parametres, parametres_grad, tuple_mode_info, learning_rate, beta1, beta2, C_CNN)

        last_activation = activations["A" + str(C_CNN)]
        l_array = np.append(l_array, log_loss(last_activation, y))
        a_array = np.append(a_array, accuracy_score(last_activation.flatten(), y.flatten()))
        d_array = np.append(d_array, dx_log_loss(last_activation, y))

    print("Final accuracy ", a_array[-1])

    #Display info of during the learing
    display_info_learning(l_array, a_array, d_array)

    #Display kernel & biais
    #display_kernel_and_biais(parametres)

    #Display target vs prediction
    y_pred = activations["A" + str(C_CNN)]
    display_comparaison_layer(y, y_pred)
    
main()