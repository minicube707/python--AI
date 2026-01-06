
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

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
    return np.where(X < 0, alpha*X, X)


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
    return np.where(X < 0, alpha, 1)


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
def max_pooling(X):
    a = np.int8(np.sqrt(X.shape[1]))
    return np.max(X, axis=2).reshape((X.shape[0], a, a))



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
int             k_size_sqrt :   the size in row of the kernel

=========OUTPUT=========
numpy.array    next_dZ :       Array containe the derivated for the next layer
"""
def convolution(dZ, K, k_size_sqrt):
     
    # Sortie (nb_layers, 4, 4)
    root = np.int8(np.sqrt(K.shape[2] ))
    K = K.reshape(K.shape[0], K.shape[1], root, root)
    output = np.zeros((K.shape[1], dZ.shape[1] + K.shape[2] - 1, dZ.shape[2] + K.shape[3] - 1))

    # Convolution pleine pour chaque filtre et chaque canal
    for i in range(K.shape[0]):  # nb_filters
        for c in range(K.shape[1]):  # nb_layers (canaux de sortie)
            output[c] += correlate2d(dZ[i], K[i, c], mode='full')

    return (output)

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
create_tuple_size:
=========DESCRIPTION=========
Create a tuple with all the original shape of the activation (no more calculation)

=========INPUT=========
tuple           X_shape  :      tuple with the dimension of the input
dict            dimensions :    all the information on how is built the CNN

=========OUTPUT=========
tuple           tuple containe all the dimension of the input and activation
"""
def create_tuple_size(X_shape, dimensions):

    tuple_size = []

    outout_shape = X_shape[1]
    for i in range(len(dimensions)):
        outout_shape = calcul_output_shape(outout_shape, dimensions[str(i+1)][0], dimensions[str(i+1)][1], dimensions[str(i+1)][2])
        tuple_size.append(outout_shape)

    return (tuple_size)


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
def initialisation_kernel(parametres, parametres_grad, k_size, type_layer, fonction, i, nb_kernel, nb_layer, o_size):
    shape = (nb_kernel, nb_layer, k_size**2, 1)

    if fonction == "relu":
        std = np.sqrt(2 / (nb_layer * k_size**2))
        K = np.random.randn(*shape).astype(np.float32) * std

    elif fonction == "tanh" or  fonction == "sigmoide":
        limit = np.sqrt(6 / (nb_layer + nb_kernel))
        K = (np.random.rand(*shape).astype(np.float32) * 2 - 1) * limit

    else:
        # Default to small random values
        K = np.random.randn(*shape).astype(np.float32) * 0.01

    b_shape = (nb_kernel, o_size**2, 1)
    b = np.zeros(b_shape).astype(np.float32)  # Bias souvent initialisé à 0

    parametres["K" + str(i)] = K
    parametres["b" + str(i)] = b
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction

    parametres_grad["m" + str(i)] = np.zeros(shape).astype(np.float32)
    parametres_grad["v" + str(i)] = np.zeros(shape).astype(np.float32)

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

    nb_layer = x_shape[0]
    o_size = x_shape[1]

    for i in range(1, len(dimensions)+1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)
        o_size = calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2])

        if type_layer == "kernel":
            parametres, parametres_grad = initialisation_kernel(parametres, parametres_grad, k_size, type_layer, fonction, i, nb_kernel, nb_layer, o_size)

        elif type_layer == "pooling":
            parametres = initialisation_pooling(parametres, k_size, type_layer, fonction, i)

        nb_layer = nb_kernel

    return parametres, parametres_grad


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
    parametres, parametres_grad = initialisation_affectation(dimensions, x_shape, list_size_activation)

    return parametres, parametres_grad, dimensions


"""
pooling_activation:
=========DESCRIPTION=========
Activation of pooling

=========INPUT=========
numpy.array     A :                 the activation matrice

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def pooling_activation(A):
    Z = max_pooling(A)
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

    if mode == "relu":
        A = relu(Z, alpha)
    elif mode == "sigmoide":
        A = sigmoide(Z)
    elif mode == "tanh":
        A = tanh(Z)
    return A, Z 


"""
function_activation:
=========DESCRIPTION=========
Function that centrelize all the use to process the CNN

=========INPUT=========
numpy.array     A :                 the activation matrice
numpy.array     K :                 the kernel matrice           
numpy.array     b :                 the biais matrice           
string          mode :              the type of activation function we use
string          type_layer :        the type of layer 
int             k_size :            the size in row of the kernel
int             x_size :            the size in row of the activation matrice
int             stride :            how many pixel the kernel move  
int             padding :           how many pixel we add to the border of the activation

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def function_activation(X, K, b, mode, type_layer, k_size, x_size, stride, padding, alpha):

    #Activation are in line format
    if type_layer == "kernel":
        A, Z = kernel_activation(X, K, b, x_size, mode, alpha)
    else:
        A = pooling_activation(X)
        Z = None
        
    #Activation are in square format
    if padding != None:
        A = add_padding(A, padding)
    if k_size != None:
        A = reshape(A, k_size , x_size, stride, padding)  

    #Activation are in line format
    return A, Z


"""
foward_propagation:
=========DESCRIPTION=========
Pass the input into the activation functions for the foreward propagation

=========INPUT=========
numpy.array     X :                             the features,input of the CNN
dict            parametres :                    containt all the information for the kernel operation
tuple           list_size_activation:           tuple of all activation shape with number of activation and padding
dict            dimensions :                    all the information on how is built the CNN

=========OUTPUT=========
dict            activation :     containt all the activation during the foreward propagation
"""
def foward_propagation(X, parametres, tuple_size_activation, dimensions, alpha):

    activation = {"A0" : X}
    C = len(dimensions.keys())
    input_shape = X.shape[1]

    for c in range(1, C+1):
        A = activation["A" + str(c-1)]
        K = parametres["K" + str(c)]
        b = parametres["b" + str(c)]
        mode = parametres["f" + str(c)]
        type_layer = parametres["l" + str(c)]
        x_size = tuple_size_activation[c-1]

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

        activation["A" + str(c)], activation["Z" + str(c)] = function_activation(A, K, b, mode, type_layer, k_size, x_size, stride, padding, alpha)
        
    return activation

"""
back_propagation_pooling:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got for the layer pooling

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            dimensions :    all the information on how is built the CNN
numpy.array     DZ :            the derivated of the previous activation (what should be the activation)
int             c  :            which stage we are in backpropagatioin 

=========OUTPUT=========
numpy.array     DZ :            the derivated of this activation for the next step of backpropagation
"""
def back_propagation_pooling(activation, dimensions, dZ, c):
    
    # Trouve les valeurs maximales et leurs indices le long de l'axe 2
    #Reshape dz to (A,BxC)
    max_dZ = dZ.reshape(dZ.shape[0], -1)

    #Get the max value, before the operation max in foreword propagation
    max_indices = np.argmax(activation["A" + str(c-1)], axis=2)

    # Initialise le résultat avec des zéros
    result = np.zeros_like(activation["A" + str(c-1)])

    # Utilise un indexage avancé pour placer les valeurs maximales
    batch_indices = np.arange(activation["A" + str(c-1)].shape[0])[:, None]
    row_indices = np.arange(activation["A" + str(c-1)].shape[1])[None, :]

    #Use a mask, everywhere is 0, exept where the max value while be take
    result[batch_indices, row_indices, max_indices] = max_dZ
    
    k_size = dimensions[str(c)][0]
    stride = dimensions[str(c)][1]
    dZ = deshape(result, k_size, stride)
    
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
def back_propagation_kernel(activation, parametres, dimensions, gradients, dZ, c, alpha):
        
    #Create a table for each dx of the kernel
    L_A, NB_Dot_Product, K_Size = activation["A" + str(c-1)].shape
    NB_K, L_K, K_Size, one  = parametres["K" + str(c)].shape

    dK = np.zeros(parametres["K" + str(c)].shape)

    #For each kernel
    for i in range(NB_K):

        #For each activation
        for j in range(L_A):
            
            #For each weight
            for k in range(K_Size):
                
                dK[i, j, k] = np.dot(activation["A" + str(c-1)][j, :, k], dZ[i].flatten())
    
    #Add the result in the dictionary
    gradients["dK" + str(c)] = dK
    gradients["db" + str(c)]  = dZ.reshape((dZ.shape[0], dZ.shape[1] * dZ.shape[2], 1))
            
    if c > 1:
        activation_fonction = parametres["f" + str(c)]

        # Chose the correct derivative
        if activation_fonction == "relu":
            dA = dx_relu(activation["Z" + str(c)], alpha)
        elif activation_fonction == "sigmoide":
            dA = dx_sigmoide(activation["A" + str(c)])
        elif activation_fonction == "tanh":
            dA = dx_tanh(activation["A" + str(c)])

        dZ *= dA

        # Apply convolution
        dZ = convolution(dZ, parametres["K" + str(c)], dimensions[str(c)][0])

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
def back_propagation_CNN(activation, parametres, dimensions, y, tuple_size_activation, alpha):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(dimensions.keys())
    gradients = {}
    dZ = activation["A" + str(C)] - y
    
    
    for c in reversed(range(1, C+1)):
        #Remove the padding
        #Activation are in square format
        dZ = dZ[:,:tuple_size_activation[c-1], :tuple_size_activation[c-1]]
        
        if parametres["l" + str(c)] == "pooling":
           dZ = back_propagation_pooling(activation, dimensions, dZ, c) 

        elif parametres["l" + str(c)] == "kernel":
            gradients, dZ = back_propagation_kernel(activation, parametres, dimensions, gradients, dZ, c, alpha)

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
int             padding :       how many pixel we add to the border of the activation

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

    o_size = calcul_output_shape(x_size_sqrt, k_size_sqrt, stride, padding)
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
    """
    Reconstitue une matrice A à partir de blocs X (chaque ligne = bloc aplati)
    k_size_sqrt : taille racine du bloc (par exemple 2 pour un bloc 2x2)
    stride : entier, nombre de pas entre chaque bloc (en lignes et colonnes)
    """

    def predire_taille_A(B, h, w, stride):
        nb_layer, n_blocks, elements_par_bloc = B.shape

        # Essayer de deviner la grille de blocs (en supposant qu'elle est rectangulaire)
        n_rows_blocks = (B.shape[1]) // ((B.shape[1])**0.5)
        n_rows_blocks = int(round(n_rows_blocks))
        n_cols_blocks = B.shape[1] // n_rows_blocks

        height_A = (n_rows_blocks - 1) * stride + h
        width_A = (n_cols_blocks - 1) * stride + w

        return (nb_layer, height_A, width_A)

    h = w = k_size_sqrt  # taille du bloc

    # Taille de la matrice à reconstruire
    A_shape = predire_taille_A(X, h, w, stride)
    A_rec = np.zeros(A_shape, dtype=float)
    counts = np.zeros_like(A_rec)

    for l in range(A_shape[0]):  # Pour chaque couche
        k = 0  # Important : réinitialiser k pour chaque couche
        for i in range(0, A_shape[1] - h + 1, stride):
            for j in range(0, A_shape[2] - w + 1, stride):
                if k >= X.shape[1]:  # Vérifie le nombre de blocs par couche
                    break
                block = X[l, k].reshape(h, w)
                A_rec[l, i:i+h, j:j+w] += block
                counts[l, i:i+h, j:j+w] += 1
                k += 1

    # Moyenne des valeurs partagées, éviter division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        A_rec = np.divide(A_rec, counts, where=counts != 0)
        A_rec[counts == 0] = 0  # ou np.nan si tu veux marquer les trous

    return A_rec



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
def display_comparaison_layer(A, Z=None, max_par_fig=12):
    """
    Affiche chaque couche du tableau 3D A, et optionnellement Z si fourni,
    côte à côte. S'adapte si Z est None.
    """
    if A.ndim != 3:
        raise ValueError("A doit être un array 3D (D, H, W)")

    if Z is not None:
        if Z.shape != A.shape:
            raise ValueError("A et Z doivent avoir la même forme si Z est fourni")
        mode_paire = True
    else:
        mode_paire = False

    total_couches = A.shape[0]

    for start in range(0, total_couches, max_par_fig):
        end = min(start + max_par_fig, total_couches)
        n = end - start

        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        total_subplots = cols * rows

        fig_cols = cols * 2 if mode_paire else cols
        fig, axes = plt.subplots(rows, fig_cols, figsize=(4 * cols, 3 * rows))
        
        # Assurer que axes est toujours 2D
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        if fig_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(n):
            layer_idx = start + i
            row = i // cols
            col = i % cols

            # Affichage de A
            ax_a = axes[row, col * 2] if mode_paire else axes[row, col]
            im_a = ax_a.imshow(A[layer_idx], cmap='gray')
            ax_a.set_title(f"A - Couche {layer_idx}")
            ax_a.axis('off')
            fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

            # Affichage de Z si présent
            if mode_paire:
                ax_z = axes[row, col * 2 + 1]
                im_z = ax_z.imshow(Z[layer_idx], cmap='gray')
                ax_z.set_title(f"Z - Couche {layer_idx}")
                ax_z.axis('off')
                fig.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04)

        # Masquer les axes inutilisés
        for j in range(n, total_subplots):
            row = j // cols
            col = j % cols
            if mode_paire:
                axes[row, col * 2].axis('off')
                axes[row, col * 2 + 1].axis('off')
            else:
                axes[row, col].axis('off')

        plt.suptitle(f'Couches {start} à {end - 1}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def display_activation(X, y, parametres_CNN, dimensions_CNN, alpha):

    # Affichage côte à côte
    plt.figure(figsize=(10, 5))

    # Afficher l'image X
    plt.subplot(1, 2, 1)
    plt.imshow(deshape(X, dimensions_CNN["1"][0],  dimensions_CNN["1"][1])[0], cmap='gray')
    plt.axis('off')
    plt.title("Image X")

    # Afficher l'image y
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.axis('off')
    plt.title("Image y")

    plt.show()

    C_CNN = len(dimensions_CNN.keys())

    tuple_size_activation = create_tuple_size((1, 28, 28), dimensions_CNN)
    activations_CNN = foward_propagation(X, parametres_CNN, tuple_size_activation, dimensions_CNN, alpha)

    for i in range(1, len(dimensions_CNN)):
        display_comparaison_layer(deshape(activations_CNN["A" +str(i)], dimensions_CNN[str(i+1)][0],  dimensions_CNN[str(i+1)][1]),
                                   activations_CNN["Z" +str(i)])


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
    #X = np.random.rand(x_shape * x_shape).reshape(x_shape, x_shape)
    #Create a cross to calibrate the model 
    X = np.zeros((x_shape, x_shape))
    X[:, 8:16] = 1
    X[8:16, :] = 1

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
    parametres, parametres_grad, dimensions = initialisation (
    input_shape, dimensions, padding_mode)
    tuple_size_activation = create_tuple_size(input_shape, dimensions)

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
        X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1], dimensions["2"][2])

    else:
         X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1], 0)

    l_array = np.array([])
    a_array = np.array([])
    d_array = np.array([])

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    for _ in tqdm(range(nb_iteration)):
        
        activations = foward_propagation(X, parametres, tuple_size_activation, dimensions, alpha)
        gradients = back_propagation_CNN(activations, parametres, dimensions, y, tuple_size_activation, alpha)
        parametres = update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2, C_CNN)

        l_array = np.append(l_array, log_loss(activations["A" + str(C_CNN)], y))
        a_array = np.append(a_array, accuracy_score(activations["A" + str(C_CNN)].flatten(), y.flatten()))
        d_array = np.append(d_array, dx_log_loss(activations["A" + str(C_CNN)], y))

    print("Final accuracy ", a_array[-1])
    print("Final loss ", l_array[-1])

    #Display info of during the learing
    display_info_learning(l_array, a_array, d_array)

    #Display kernel & biais
    #display_kernel_and_biais(parametres)

    #Display target vs prediction
    y_pred = activations["A" + str(C_CNN)]
    display_comparaison_layer(y, y_pred)

    display_activation(X, y, parametres, dimensions, alpha)
    
main()