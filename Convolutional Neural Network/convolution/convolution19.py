
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
def sigmoïde(X):
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
def relu(X):
    return np.where(X < 0, 0, X)

"""
dx_sigmoïde:
=========DESCRIPTION=========
Apply the derivate sigmoide function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def dx_sigmoïde(X):
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
def dx_relu(X):
    return np.where(X < 0, 0, 1)


"""
max:
=========DESCRIPTION=========
Return the max of each row of the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def max(X):
    a = np.int8(np.sqrt(X.shape[1]))
    return np.max(X, axis=2).reshape((X.shape[0], a, a))


"""
correlate:
=========DESCRIPTION=========
Do the correlate of two arrays

=========INPUT=========
numpy.array     A :                 the activation matrice
numpy.array     K :                 the kernel matrice           
numpy.array     b :                 the biais matrice   
int             x_size :            the size in row of the activation matrice  

=========OUTPUT=========
numpy.array     Z_concat :          array containe the next activation
"""
def correlate(A, K, b, x_size):

    # Liste pour stocker chaque couche transformée
    layers = []
    nb_channel = K.shape[0] // A.shape[0]

    #For each activation
    for j in range(A.shape[0]):

        #Do the calcul for each kernel of the channel
        for i in range(nb_channel): 
            k = i + (j * nb_channel)

            Z = A[j].dot(K[k]) + b[k]
            Z = Z.reshape((1, x_size, x_size))
            layers.append(Z)

    # Concaténation des couches le long de l'axe des canaux (ici axe 0)
    Z_concat = np.concatenate(layers, axis=0)
    # Clipping des valeurs
    Z_concat = np.clip(Z_concat, -88, 88)

    return Z_concat

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

    #new_dz is intput with a pas to do the the cross product with all value
    new_dZ = np.pad(dZ, pad_width=((0, 0), (k_size_sqrt - 1, k_size_sqrt - 1), (k_size_sqrt - 1, k_size_sqrt - 1)), mode='constant', constant_values=0)

    #next_dz is the output
    next_dZ = np.zeros((dZ.shape[0], dZ.shape[1]+k_size_sqrt-1, dZ.shape[2]+k_size_sqrt-1))

    #FOR EACH LAYER
    for a in range(next_dZ.shape[0]):

        #FOR SELCTION COLOMN
        for b in range(next_dZ.shape[1]):

            #FOR SELCTION ROW
            for c in range(next_dZ.shape[2]): 

                #DO THE CONVOLUTION
                next_dZ[a, b, c] = np.dot(new_dZ[a, b:b + k_size_sqrt, c:c + k_size_sqrt].flatten(), K[a][::-1].flatten())

    return next_dZ

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
def ouput_shape(input_size, k_size, stride, padding):
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
def show_information(tuple_size_activation, dimensions):

    print("\nDétail de la convolution :")
    print("Nb activation")
    for i in range(len(tuple_size_activation)):

        if i < len(tuple_size_activation)-1:
            print(f"{tuple_size_activation[i][0]}", end="")
            print("->", end="")

    print(f"{tuple_size_activation[i][0]}")  
    print("")

    print("Padding")
    for i in range(len(tuple_size_activation)):

        if i < len(tuple_size_activation)-1:
            print(f"{tuple_size_activation[i][1] - dimensions[str(i+1)][2]}", end="")
            print(f"({dimensions[str(i+1)][2]})", end="")
            print("->", end="")

    print(f"{tuple_size_activation[i][1]}")  
    print("")

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
def error_initialisation(list_size, dimensions, input_size, previ_input_size, type_layer, fonction, stride):

    if input_size < 1:
        show_information(list_size, dimensions)
        raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
    if previ_input_size % input_size != 0 and stride != 1:
        show_information(list_size, dimensions)
        raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
    
    if type_layer not in ["kernel", "pooling"]:
        show_information(list_size, dimensions)
        raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pooling' or 'kernel'.")
    
    if fonction not in ["relu", "sigmoide", "max"]:
        show_information(list_size, dimensions)
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
    input_size =  x_shape[1]
    previ_input_size = input_size

    for i in range(1, len(dimensions)+1):

        k_size, stride, padding, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)

        #If the input doesn't match perfectly with the kernel and padding and is in mode auto-correction, the system correct the mistake and add the right padding
        if input_size % stride != 0 and padding_mode == "auto":
            padding = stride - input_size % stride
            list_size_activaton[-1] = (list_size_activaton[-1][0], input_size + padding)

        if (dimensions[str(i)][4] == "kernel"):
            #Modify the number of kernel depending on the number of activation
            new_nb_kernel = nb_kernel * list_size_activaton[i - 1][0]

            #Add the modificaton to the dict
            dimensions[str(i)] = k_size, stride, padding, nb_kernel, type_layer, fonction
        
        else:
            dimensions[str(i)] = k_size, stride, padding, 1, type_layer, fonction

        o_size = ouput_shape(input_size, k_size, stride, padding)
        previ_input_size = input_size + padding
        input_size = o_size

        list_size_activaton.append((new_nb_kernel, input_size))
        error_initialisation(list_size_activaton, dimensions, input_size, previ_input_size, type_layer, fonction, stride)

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
def initialisation_affectation(dimensions, list_size_activation):

    parametres = {}
    parametres_grad = {}
    for i in range(1, len(dimensions)+1):
        k_size, _, _, _, type_layer, fonction = initialisation_extraction(dimensions, i)
        nb_kernel = list_size_activation[i][0]
        o_size = list_size_activation[i][1]

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
    parametres, parametres_grad = initialisation_affectation(dimensions, list_size_activation)

    return parametres, parametres_grad, dimensions, tuple(list_size_activation)


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
    Z = max(A)
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
def kernel_activation(A, K, b, x_size, mode):

    Z = correlate(A, K, b, x_size)

    if mode == "relu":
        Z = relu(Z)
    elif mode == "sigmoide":
        Z = sigmoïde(Z)
    
    return(Z)


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
def function_activation(A, K, b, mode, type_layer, k_size, x_size, stride, padding):

    #Activation are in line format
    if type_layer == "kernel":
        Z = kernel_activation(A, K, b, x_size, mode)
    else:
        Z = pooling_activation(A)

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
numpy.array     X :                             the features,input of the CNN
dict            parametres :                    containt all the information for the kernel operation
tuple           list_size_activation:           tuple of all activation shape with number of activation and padding
dict            dimensions :                    all the information on how is built the CNN

=========OUTPUT=========
dict            activation :     containt all the activation during the foreward propagation
"""
def foward_propagation(X, parametres, tuple_size_activation, dimensions):

    activation = {"A0" : X}
    C = len(dimensions.keys())
    
    for c in range(1, C+1):
        A = activation["A" + str(c-1)]
        K = parametres["K" + str(c)]
        b = parametres["b" + str(c)]
        mode = parametres["f" + str(c)]
        type_layer = parametres["l" + str(c)]
        x_size = tuple_size_activation[c][1]

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

    #Merge the gradient for the pooling
    #Get the number of kernel of the previous stage
    nb_kernel  = dimensions[str(c+1)][3]

    #The new ouput is the old one divided by the number of kernel
    new_rows = max_dZ.shape[0] // nb_kernel
    
    #For each colomn get the max. The 3d array is merge to 2d array. We merge the array of the numbers the numbers of output
    max_dZ = max_dZ.reshape(new_rows, nb_kernel, -1).max(axis=1)

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
def back_propagation_kernel(activation, parametres, dimensions, gradients, dZ, c):
        
    #Create a table for each dx of the kernel
    nb_activation = activation["A" + str(c-1)].shape[0]
    nb_kernel = dimensions[str(c)][3]
    nb_weight = activation["A" + str(c-1)].shape[2]
    dK = np.zeros(parametres["K" + str(c)].shape)

    for i in range(nb_activation):        #For each activation
        for j in range(nb_kernel):        #For each kernel of the channel
            l = j + (i * nb_kernel)

            for k in range(nb_weight):          #For each weight

                dK[i, k, 0] = np.dot(activation["A" + str(c-1)][i, :, k], dZ[l].flatten())

    #Add the result in the dictionary
    gradients["dK" + str(c)] = dK
    gradients["db" + str(c)] = dZ.reshape((dZ.shape[0], dZ.shape[1] * dZ.shape[2], 1))
    
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
def back_propagation(activation, parametres, dimensions, y, tuple_size_activation):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = len(dimensions.keys())
    dZ = activation["A" + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C+1)):

        #Remove the padding
        #Activation are in square format        
        dZ = dZ[:,:tuple_size_activation[c][1], :tuple_size_activation[c][1]]

        if parametres["l" + str(c)] == "pooling":
           dZ = back_propagation_pooling(activation, dimensions, dZ, c) 

        elif parametres["l" + str(c)] == "kernel":
            gradients, dZ = back_propagation_kernel(activation, parametres, dimensions, gradients, dZ, c)

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
    
    step1 = input_size // stride
    step2 = k_size_sqrt

    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1], step1):
            for k in range(0, X.shape[2], step2):
                new_X = np.append(new_X, X[i, j:j + step1, k:k + step2])

    new_X = new_X.reshape((X.shape[0], input_size ,input_size))
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
    return np.pad(X, pad_width=((0, 0), (0, padding), (0, padding)), mode='constant', constant_values=0)


"""
============================
Evaluation Metrics Function
============================
"""
def dx_log_loss(y_pred, y_true):
    epsilon = 1e-15
    return -1 / y_true.size * np.sum((y_true / y_pred + epsilon) - (1 - y_true) / (1 - y_pred + epsilon))

def log_loss(y_pred, y_true):

    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  (1/y_true.size) * np.sum( -y_true * np.log(y_pred + epsilon) - (1 - y_true) * np.log(1 - y_pred + epsilon))

def accuracy_score(y_pred, y_true):
    y_true = np.round(y_true, 2)
    y_pred = np.round(y_pred, 2)
    return np.count_nonzero(y_pred == y_true) / y_true.size


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
def display_layer(array_3d, type, stage, max_par_fig=12):

    
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

            sqrt = np.int8(np.sqrt(value.shape[1]))
            value = value.reshape(value.shape[0], sqrt, sqrt)
            if key.startswith('K'):
                display_layer(value, "Kernel", key[-1])

            elif key.startswith('b'):
                display_layer(value, "Biais", key[-1])


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

            ax_y.imshow(y[layer_idx], cmap='gray')
            ax_y.set_title(f'Y - Couche {layer_idx}')
            ax_y.axis('off')

            ax_pred.imshow(y_pred[layer_idx], cmap='gray')
            ax_pred.set_title(f'Prediction - Couche {layer_idx}')
            ax_pred.axis('off')

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
    learning_rate = 0.005
    beta1 = 0.9
    beta2 = 0.99
    nb_iteration = 10_000


    x_shape = 21
    X = np.random.rand(x_shape, x_shape)

    if len(X.shape) == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])

    dimensions = {}
    #Kernel size, stride, padding, nb_kernel, type layer, function
    dimensions = {"1" :(3, 1, 0, 2, "kernel", "relu"),
                  "2" :(2, 2, 0, 1, "pooling", "max"),
                  "3" :(2, 1, 0, 4, "kernel", "relu"),
                  "4" :(2, 2, 0, 1, "pooling", "max"),
                  "5" :(2, 1, 0, 3, "kernel", "sigmoide")}

    padding_mode = "auto"
    parametres, parametres_grad, dimensions, tuple_size_activation = initialisation(X.shape, dimensions, padding_mode)
    show_information(tuple_size_activation, dimensions)


    input_size = X.shape[1]
    for val in dimensions.values():
        o_size = ouput_shape(input_size, val[0], val[1], val[2])
        input_size = o_size

    y_shape = o_size
    y = np.random.rand(24, y_shape, y_shape)

    X = add_padding(X, dimensions["2"][2])
    X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1], dimensions["2"][2])

    l_array = np.array([])
    a_array = np.array([])
    d_array = np.array([])
    C = len(dimensions.keys())

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector


    for _ in tqdm(range(nb_iteration)):

        activations = foward_propagation(X, parametres, tuple_size_activation, dimensions)
        gradients = back_propagation(activations, parametres, dimensions, y, tuple_size_activation)
        parametres = update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2, C)

        l_array = np.append(l_array, log_loss(activations["A" + str(C)], y))
        a_array = np.append(a_array, accuracy_score(activations["A" + str(C)].flatten(), y.flatten()))
        d_array = np.append(d_array, dx_log_loss(activations["A" + str(C)], y))


    #Displau info of during the learing
    display_info_learning(l_array, a_array, d_array)

    #Display kernel & biais
    display_kernel_and_biais(parametres)

    #Display target vs prediction
    y_pred = activations["A" + str(C)]
    display_comparaison_layer(y, y_pred)
    
main()
