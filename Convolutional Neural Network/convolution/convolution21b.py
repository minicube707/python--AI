
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit

#Allow to show all tab with numpy
np.set_printoptions(linewidth=200, threshold=np.inf)

"""
============================
========Documentation=======
============================
A : Activation in memory
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
max:
=========DESCRIPTION=========
Return the max of each row of the activation function

=========INPUT=========
X (np.ndarray)      : Activation matrix (shape: [layer, height, width])
K (np.ndarray)      : Kernel matrix (shape: [layer, height, width])
stride (int)        : the number of pixels the kernel move

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def max(X, K, stride):
    
    len_kernel = K
    len_activation = X.shape[1]

    new_size = ouput_shape(len_activation, len_kernel, stride, 0)

    Z = np.zeros((X.shape[0], new_size, new_size))
    for a in range(X.shape[0]):
        for b in range (0, len_activation, stride):
            for c in range (0, len_activation, stride):
            
                Z[a, b // stride, c // stride] = np.max(X[a, b:b + len_kernel, c:c + len_kernel])

    return Z


"""
=========DESCRIPTION=========
Perform a correlation between two arrays (activation and kernel).

=========INPUT=========
A (np.ndarray)      : Activation matrix (shape: [layer, height, width])
K (np.ndarray)      : Kernel matrix (shape: [layer, height, width])
b (np.ndarray)      : Biais matrix (shape: [layer, height, width])
stride (int)        : the number of pixels the kernel move
padding (int)       : the number of pixels add to the border of the activation

=========OUTPUT=========
Z_concat (np.ndarray): Next activation array (shape: [out_channels, x_size, x_size])
"""
def correlate(A, K, b, stride, padding):

    o_size = ouput_shape(A.shape[1], K.shape[1], stride, padding)
    Z = np.zeros((K.shape[0], o_size, o_size))

    nb_layer_activation = A.shape[0]
    nb_layer_kernel = K.shape[0]
    channel_by_kernel = nb_layer_kernel // nb_layer_activation
    len_kernel = K.shape[1]
    len_activation = A.shape[1]

    for i in range(nb_layer_activation):
        for j in range(channel_by_kernel):
            for k in range (0, len_activation - len_kernel + 1, stride):
                for l in range (0, len_activation - len_kernel + 1, stride):
                    
                    Z[j + (i * channel_by_kernel), k , l] = np.dot(A[i, k:k + len_kernel, l:l + len_kernel].flatten(), K[j].flatten())
    
    return Z + b


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

    parametres["K" + str(i)] = k_size
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

    parametres["K" + str(i)] = np.random.rand(nb_kernel, k_size, k_size).astype(np.float16) * 2 - 1
    parametres["b" + str(i)] = np.random.rand(nb_kernel, o_size, o_size).astype(np.float16) * 2 - 1
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction

    parametres_grad["m" + str(i)] = np.zeros((nb_kernel, k_size, k_size)).astype(np.float16)
    parametres_grad["v" + str(i)] = np.zeros((nb_kernel, k_size, k_size)).astype(np.float16)

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
A (np.ndarray)      : Activation matrix (shape: [layer, height, width])
K (np.ndarray)      : Kernel matrix (shape: [layer, height, width])
stride (int)        : the number of pixels the kernel move

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def pooling_activation(A, K, stride):
    Z = max(A, K, stride)
    return Z


"""
kernel_activation:
=========DESCRIPTION=========
Activation of kernel

=========INPUT=========
A (np.ndarray)      : Activation matrix (shape: [layer, height, width])
K (np.ndarray)      : Kernel matrix (shape: [layer, height, width])
b (np.ndarray)      : Biais matrix (shape: [layer, height, width])
stride (int)        : the number of pixels the kernel move
padding (int)       : the number of pixels add to the border of the activation      
mode (string)       : the type of activation function we use

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def kernel_activation(A, K, b, stride, padding , mode):

    Z = correlate(A, K, b, stride, padding)

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
A (np.ndarray)      : Activation matrix (shape: [layer, height, width])
K (np.ndarray)      : Kernel matrix (shape: [layer, height, width])
b (np.ndarray)      : Biais matrix (shape: [layer, height, width])
stride (int)        : the number of pixels the kernel move
padding (int)       : the number of pixels add to the border of the activation      
mode (string)       : the type of activation function we use
type_layer (string) : the type of layer 
padding (int)       : the number of pixels add to the border of the activation      
mode (string)       : the type of activation function we use

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def function_activation(A, K, b, mode, type_layer, stride, padding):

    #Activation are in line format
    if type_layer == "kernel":
        Z = kernel_activation(A, K, b, stride, padding , mode)
    else:
        Z = pooling_activation(A, K, stride)

    #Activation are in square format
    if padding != None:
        Z = add_padding(Z, padding)  

    #Activation are in line format
    return Z


"""
foward_propagation:
=========DESCRIPTION=========
Pass the input into the activation functions for the foreward propagation

=========INPUT=========
numpy.array     X :                             the features,input of the CNN
dict            parametres :                    containt all the information for the kernel operation
dict            dimensions :                    all the information on how is built the CNN

=========OUTPUT=========
dict            activation :     containt all the activation during the foreward propagation
"""
def foward_propagation(X, parametres, dimensions):

    activation = {"A0" : X}
    C = len(dimensions.keys())
    
    for c in range(1, C+1):
        A = activation["A" + str(c-1)]
        K = parametres["K" + str(c)]
        b = parametres["b" + str(c)]
        mode = parametres["f" + str(c)]
        type_layer = parametres["l" + str(c)]
        stride = dimensions[str(c)][1]

        #The information for the padding is at the next step
        if c+1 < C:
           padding = dimensions[str(c+2)][2] 

        activation["A" + str(c)] = function_activation(A, K, b, mode, type_layer, stride, padding)
        
    return activation

"""
back_propagation_pooling:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got for the layer pooling

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            parametres :    containt all the information for the kernel operation
dict            dimensions :    all the information on how is built the CNN
numpy.array     DZ :            the derivated of the previous activation (what should be the activation)
int             c  :            which stage we are in backpropagatioin 

=========OUTPUT=========
numpy.array     DZ :            the derivated of this activation for the next step of backpropagation
"""
def back_propagation_pooling(activation, parametres, dimensions, dZ, c):

    A_prev = activation["A" + str(c - 1)]
    stride = dimensions[str(c)][1]
    k_size = parametres["K" + str(c - 1)].shape[1]  # suppose carré : kH == kW

    L, H, W = A_prev.shape
    out_H, out_W = dZ.shape[1], dZ.shape[2]

    new_dz = np.zeros_like(A_prev).astype(np.float32)

    for l in range(L):  # pour chaque canal
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + k_size
                w_end = w_start + k_size

                patch = A_prev[l, h_start:h_end, w_start:w_end]
                max_index = np.unravel_index(np.argmax(patch), patch.shape)

                # Position du max dans la fenêtre
                new_dz[l, h_start + max_index[0], w_start + max_index[1]] = dZ[l, i, j]

    return new_dz


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
def back_propagation_kernel(activation, parametres, dimensions, gradients, dZ, c):
    
    L_A, H_A, W_A = activation["A" + str(c-1)].shape
    kL, kH, kW = parametres["K" + str(c)].shape
    channel_by_kernel = kL // L_A
    stride = dimensions[str(c)][1]

    dK = np.zeros(parametres["K" + str(c)].shape)
    
    for i in range(L_A):
        for j in range(channel_by_kernel):
            for k in range(0, kH, stride):
                for l in range(0, kW, stride):
                    
                    dK[j + (i * channel_by_kernel)] += activation["A" + str(c-1)][i, k:k + kH, l:l + kH] * dZ[j, k, l]

    #Add the result in the dictionary
    gradients["dK" + str(c)] = dK
    gradients["db" + str(c)] = dZ
            
    if c > 1:
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
           dZ = back_propagation_pooling(activation, parametres, dimensions, dZ, c) 

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
def mean_square_error(y_pred, y):
    return  1 / (2 * len(y)) * np.sum((y_pred - y)**2)

def accuracy_score(y_pred, y_true):
    y_true = np.round(y_true, 1)
    y_pred = np.round(y_pred, 1)
    return np.count_nonzero(y_pred == y_true) / y_true.size

def dx_mean_square_error(y_pred, y):
    return  -1 / (len(y)) * np.sum(y_pred - y)


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
    y = np.random.rand(tuple_size_activation[-1][0], y_shape, y_shape)

    if len(dimensions) > 1:
        X = add_padding(X, dimensions["2"][2])

    l_array = np.array([])
    a_array = np.array([])
    d_array = np.array([])
    C = len(dimensions.keys())

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    for _ in tqdm(range(nb_iteration)):

        activations = foward_propagation(X, parametres, dimensions)
        gradients = back_propagation(activations, parametres, dimensions, y, tuple_size_activation)
        parametres = update(gradients, parametres, parametres_grad, learning_rate, beta1, beta2, C)
        
        l_array = np.append(l_array, mean_square_error(activations["A" + str(C)], y))
        a_array = np.append(a_array, accuracy_score(activations["A" + str(C)].flatten(), y.flatten()))
        d_array = np.append(d_array, dx_mean_square_error(activations["A" + str(C)], y))

    print("Final accuracy ", a_array[-1])

    #Displau info of during the learing
    display_info_learning(l_array, a_array, d_array)

    #Display kernel & biais
    #display_kernel_and_biais(parametres)

    #Display target vs prediction
    y_pred = activations["A" + str(C)]
    display_comparaison_layer(y, y_pred)
    
main()
