
import  numpy as np

from .Mathematical_function import relu, sigmoide, max_pooling, convolution, correlate, dx_relu, dx_sigmoide, tanh, dx_tanh


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


def create_tuple_size(X_shape, dimensions):

    tuple_size = []

    outout_shape = X_shape[1]
    for i in range(len(dimensions)):
        outout_shape = calcul_output_shape(outout_shape, dimensions[str(i+1)][0], dimensions[str(i+1)][1], dimensions[str(i+1)][2])
        tuple_size.append(outout_shape)

    return (tuple_size)


"""
calcul_output_shape:
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
def show_information_CNN(dimensions, input_size):

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
        show_information_CNN(dimensions, input_size)
        raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
    if previ_input_size % input_size != 0 and stride != 1:
        show_information_CNN(dimensions, input_size)
        raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
    
    if type_layer not in ["kernel", "pooling"]:
        show_information_CNN(dimensions, input_size)
        raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pooling' or 'kernel'.")
    
    if fonction not in ["relu", "sigmoide", "max", "tanh"]:
        show_information_CNN(dimensions, input_size)
        raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu', 'sigmoide', 'max' ou 'tanh'.")





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
def foward_propagation_CNN(X, parametres, tuple_size_activation, dimensions, alpha):

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
    result = np.zeros_like(activation["A" + str(c-1)], dtype=np.int16)

    # Utilise un indexage avancé pour placer les valeurs maximales
    batch_indices = np.arange(activation["A" + str(c-1)].shape[0])[:, None]
    row_indices = np.arange(activation["A" + str(c-1)].shape[1])[None, :]

    #Use a mask, everywhere is 0, exept where the max value while be take
    result[batch_indices, row_indices, max_indices] = max_dZ

    # Affichage
    dZ = deshape(result, dimensions[str(c)][0], dimensions[str(c)][1])

    return dZ


"""
back_propagation_kernel:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got for the layer kernel

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            parametres :    containt all the information for the kernel operation
dict            parametres :    containt all the information for the kernel operation
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
    gradients["db" + str(c)] = dZ.reshape((dZ.shape[0], dZ.shape[1] * dZ.shape[2], 1))
            
    if c > 1:
        activation_fonction = parametres["f" + str(c)]
        dim = dimensions[str(c)]

        # Chose the correct derivative
        if activation_fonction == "relu":
            dA = dx_relu(activation["Z" + str(c)], alpha)
        elif activation_fonction == "sigmoide":
            dA = dx_sigmoide(activation["A" + str(c)])
        elif activation_fonction == "tanh":
            dA = dx_tanh(activation["A" + str(c)])

        dA = deshape(dA, dim[0], dim[1])
        dZ *= dA

        # Apply convolution
        dZ = convolution(dZ, parametres["K" + str(c)], dim[0])

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
def back_propagation_CNN(activation, parametres, dimensions, dZ, tuple_size_activation, alpha):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(dimensions.keys())
    gradients = {}
    dZ = dZ.reshape(activation["A" +str(C)].shape)
    
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
def update_CNN(gradients, parametres, parametres_grad, learning_rate, beta1, beta2, C):
        
    epsilon = 1e-8 #Pour empecher les log(0) = /0
    #Adam (Adaptativ Momentum)
    for c in range(1, C+1):
        if parametres["l" + str(c)] == "kernel":

            #Update moment
            grad = np.clip(gradients["dK" + str(c)], -1e3, 1e3)
            parametres_grad["m" + str(c)] = beta1 * parametres_grad["m" + str(c)] + (1 - beta1) * gradients["dK" + str(c)]     # Première estimation des moments (moyenne des gradients)
            parametres_grad["v" + str(c)] = beta2 * parametres_grad["v" + str(c)] + (1 - beta2) * grad**2  # Deuxième estimation des moments (moyenne des carrés des gradients)

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

    input_size = int(np.sqrt(X.shape[1]*X.shape[2]))
    new_X = np.array([], dtype=np.int16)
    
    step1 = input_size // stride
    step2 = k_size_sqrt

    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1], step1):
            for k in range(0, X.shape[2], step2):
                new_X = np.append(new_X, X[i, j:j + step1, k:k + step2].astype(np.uint16))

    new_X = new_X.reshape((X.shape[0], input_size ,input_size))
    return new_X










