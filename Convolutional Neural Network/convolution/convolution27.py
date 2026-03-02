
import  numpy as np
from numba import njit, types
from numba.typed import Dict, List
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
@njit(fastmath=True)
def sigmoide(X):
    out = np.empty_like(X)
    for i in range(X.size):
        out.flat[i] = 1.0 / (1.0 + np.exp(-X.flat[i]))
    return out


"""
relu:
=========DESCRIPTION=========
Apply the relu function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
@njit(fastmath=True)
def relu(X, alpha):
    out = np.empty_like(X)
    for i in range(X.size):
        x = X.flat[i]
        out.flat[i] = x if x >= 0 else alpha * x
    return out


"""
dx_sigmoïde:
=========DESCRIPTION=========
Apply the derivate sigmoide function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
@njit(fastmath=True)
def dx_sigmoide(X):
    out = np.empty_like(X)
    for i in range(X.size):
        x = X.flat[i]
        out.flat[i] = x * (1.0 - x)
    return out

"""
dx_relu:
=========DESCRIPTION=========
Apply the derivative relu function at the activation function
=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
@njit(fastmath=True)
def dx_relu(X, alpha):
    out = np.empty_like(X)
    for i in range(X.size):
        out.flat[i] = 1.0 if X.flat[i] >= 0 else alpha
    return out


@njit(fastmath=True)
def tanh(X):
    out = np.empty_like(X)
    for i in range(X.size):
        out.flat[i] = np.tanh(X.flat[i])
    return out


@njit(fastmath=True)
def dx_tanh(X):
    out = np.empty_like(X)
    for i in range(X.size):
        x = X.flat[i]
        out.flat[i] = 1.0 - x * x
    return out
            

"""
max:
=========DESCRIPTION=========
Return the max of each row of the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
@njit(fastmath=True)
def max_pooling(X):
    n = X.shape[0]
    m = X.shape[1]
    k = X.shape[2]

    a = int(np.sqrt(m))

    out = np.empty((n, m), dtype=X.dtype)

    # max sur axis=2
    for i in range(n):
        for j in range(m):
            max_val = X[i, j, 0]
            for t in range(1, k):
                val = X[i, j, t]
                if val > max_val:
                    max_val = val
            out[i, j] = max_val

    return out.reshape(n, a, a)



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
@njit(fastmath=True)
def correlate(A, K, b, x_size):

    L_A, NB_Dot_Product, K_Size = A.shape
    NB_K, L_K, K_Size2, one = K.shape

    Z = np.zeros((NB_K, NB_Dot_Product, one), dtype=A.dtype)

    A = np.ascontiguousarray(A)
    K = np.ascontiguousarray(K)

    for i in range(NB_K):
        for j in range(L_A):
            Z[i] += np.dot(A[j], K[i, j])

        # ajout biais
        Z[i] += b[i]

    # reshape final
    Z = Z.reshape(NB_K, x_size, x_size)

    # clip manuel (plus rapide que np.clip en numba)
    for i in range(NB_K):
        for x in range(x_size):
            for y in range(x_size):
                val = Z[i, x, y]
                if val > 88:
                    Z[i, x, y] = 88
                elif val < -88:
                    Z[i, x, y] = -88

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
@njit(fastmath=True)
def convolution(dZ, K):

    nb_filters, out_h, out_w = dZ.shape
    nb_kernels, nb_layers, k_flat, one = K.shape

    root = int(np.sqrt(k_flat))
    K = np.ascontiguousarray(K)
    K = K.reshape(nb_kernels, nb_layers, root, root)

    out_height = out_h + root - 1
    out_width = out_w + root - 1

    output = np.zeros((nb_layers, out_height, out_width), dtype=dZ.dtype)

    for i in range(nb_filters):
        for c in range(nb_layers):
            for y in range(out_h):
                for x in range(out_w):
                    val = dZ[i, y, x]
                    for ky in range(root):
                        for kx in range(root):
                            output[c, y+ky, x+kx] += val * K[i, c, ky, kx]

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
    return int((input_size - k_size + padding) / stride + 1)


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
def initialisation_pooling(
        parametres_K, parametres_B, 
        gradients_dK, gradients_db,
        m_list, v_list,
        k_size, i):

    K = np.empty((1, 1, 1, 1)).astype(np.float32)
    B = np.empty((1, 1, 1)).astype(np.float32)
    
    parametres_K.append(K)
    parametres_B.append(B)

    dK = np.zeros(K.shape, dtype=np.float32)
    db = np.zeros(B.shape, dtype=np.float32)

    gradients_dK.append(dK)
    gradients_db.append(db)

    m = np.zeros((1, 1, 1, 1), dtype=np.float32)
    v = np.zeros((1, 1, 1, 1), dtype=np.float32)

    m_list.append(m)
    v_list.append(v)

    return  parametres_K, parametres_B, m_list, v_list, gradients_dK, gradients_db

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
#Get the function from convolution23.py
def initialisation_kernel(
        parametres_K, parametres_B, 
        m_list, v_list, 
        gradients_dK, gradients_db,
        list_size_activation, k_size, i):

    nb_kernel = list_size_activation[i][0]
    nb_layer =  list_size_activation[i-1][0]
    o_size = list_size_activation[i][1]

    #Set every kernel to zero, exept the center to 1
    K = np.zeros((nb_kernel, nb_layer, k_size**2, 1), dtype=np.float32)
    center_index = (k_size**2) // 2  # ex: pour 3x3 → 9//2 = 4
    K[:, :, center_index, 0] = 1
    parametres_K.append(K)

    #Put biais to zero to vanish them
    B = np.zeros((nb_kernel, np.int64(o_size)**2, 1), dtype=np.float32) #np.int64 avoid overflow with o_size**2
    parametres_B.append(B)

    m = np.zeros((nb_kernel, nb_layer, k_size**2, 1), dtype=np.float32)
    v = np.zeros((nb_kernel, nb_layer, k_size**2, 1), dtype=np.float32)

    m_list.append(m)
    v_list.append(v)

    dK = np.zeros(K.shape, dtype=np.float32)
    db = np.zeros(B.shape, dtype=np.float32)

    gradients_dK.append(dK)
    gradients_db.append(db)

    return  parametres_K, parametres_B, m_list, v_list, gradients_dK, gradients_db

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

    array_3dtype = types.float32[:, :, :]
    array_4dtype = types.float32[:, :, :, :]

    parametres_K = List.empty_list(array_4dtype)
    parametres_B = List.empty_list(array_3dtype)
    
    gradients_dK = List.empty_list(array_4dtype)
    gradients_db = List.empty_list(array_3dtype)

    tuple_type = types.UniTuple(types.int32, 6)
    tuple_mode_info = List.empty_list(tuple_type)

    m_list = List.empty_list(array_4dtype)
    v_list = List.empty_list(array_4dtype)
    
    nb_layer = x_shape[0]
    o_size = np.int32(x_shape[1])

    for i in range(1, len(dimensions)+1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)
        o_size = np.int32(calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2]))

        # function: 0 = relu, 1 = sigmoide, 2 = tanh
        if fonction == "relu":
            fonction = 0
        elif fonction == "sigmoide":
            fonction = 1
        else:
            fonction = 2

        # mode: 0 = kernel, 1 = pooling
        if type_layer == "kernel":

            #Tuple_mode_info: mode, function, output_size, kernel_size, stride, padding
            tuple_mode_info.append((
                np.int32(0),
                np.int32(fonction),
                o_size,
                np.int32(dimensions[str(i)][0]),
                np.int32(dimensions[str(i)][1]),
                np.int32(dimensions[str(i)][2])
            ))

            parametres_K, parametres_B, m_list, v_list, gradients_dK, gradients_db = initialisation_kernel(
                parametres_K, parametres_B, 
                m_list, v_list, 
                gradients_dK, gradients_db,
                list_size_activation, k_size, i)
            
        elif type_layer == "pooling":
            #Tuple_mode_info: mode, function, output_size, kernel_size, stride, padding
            tuple_mode_info.append((
                np.int32(1),
                np.int32(fonction),
                o_size,
                np.int32(dimensions[str(i)][0]),
                np.int32(dimensions[str(i)][1]),
                np.int32(dimensions[str(i)][2])
            ))

            parametres_K, parametres_B, m_list, v_list, gradients_dK, gradients_db = initialisation_pooling(
                parametres_K, parametres_B, 
                gradients_dK, gradients_db,
                m_list, v_list,
                k_size, i)

        nb_layer = nb_kernel

    return parametres_K, parametres_B, tuple_mode_info, m_list, v_list, gradients_dK, gradients_db


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
    parametres_K, parametres_B, tuple_mode_info, m_list, v_list, gradients_dK, gradients_db = initialisation_affectation(
        dimensions, x_shape, list_size_activation)

    return parametres_K, parametres_B, tuple_mode_info, m_list, v_list, dimensions, gradients_dK, gradients_db


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
@njit
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
@njit
def function_activation(X, K, b, mode, type_layer, k_size, x_size, stride, padding, alpha):

    #Activation are in line format
    # mode: 0 = kernel, 1 = pooling)
    if type_layer == 0:
        A, Z = kernel_activation(X, K, b, x_size, mode, alpha)
    else:
        A = max_pooling(X)
        Z = np.zeros((A.shape[0], 1, 1), dtype=A.dtype)   # If Numba doesn't like None, we create an array of zeros
        
    #Activation are in square format
    if padding > 0:
        A = add_padding(A, padding)
    if k_size > 0:
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

=========OUTPUT=========
dict            activation :     containt all the activation during the foreward propagation
"""
@njit
def forward_propagation(X, parametres_K, parametres_B, tuple_mode_info, alpha):

    C = len(tuple_mode_info)
    array3d_type = types.float32[:, :, :]

    # Liste des activations A
    activations_A = List.empty_list(array3d_type)
    activations_Z = List.empty_list(array3d_type)

    activations_A.append(X.astype(np.float32))  # A0
    activations_Z.append(X.astype(np.float32))  # Z0 Useless, but usefull to keep align with A

    for c in range(C):

        A = activations_A[c]
        K = parametres_K[c]
        b = parametres_B[c]

        type_layer = tuple_mode_info[c][0]
        mode       = tuple_mode_info[c][1]
        x_size     = tuple_mode_info[c][2]

        k_size  = 0
        stride  = 1
        padding = 0

        if c + 1 < C:
            k_size = tuple_mode_info[c+1][3]
            stride = tuple_mode_info[c+1][4]

        if c + 2 < C:
            padding = tuple_mode_info[c+2][5]

        A_next, Z_next = function_activation(
            A, K, b,
            mode,
            type_layer,
            k_size,
            x_size,
            stride,
            padding,
            alpha
        )

        activations_A.append(A_next)
        activations_Z.append(Z_next)

    return activations_A, activations_Z

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
@njit
def back_propagation_pooling(A, k_size, stride, dZ):

    batch = A.shape[0]
    rows = A.shape[1]
    cols = A.shape[2]

    # numba compatible manual reshape
    max_dZ = np.zeros((batch, rows), dtype=A.dtype)
    
    # Boucle explicite pour "reshaper" dZ
    for b in range(batch):
        for r in range(rows):
            # Suppose dZ[b, r, :] correspond à max_dZ[b, r]
            # Ici, on prend juste la première valeur si dZ est 3D
            max_dZ[b, r] = dZ[b, r, 0]

    result = np.zeros_like(A)

    for b in range(batch):
        for r in range(rows):

            # Find argmax manually (axis=2)
            max_idx = 0
            max_val = A[b, r, 0]

            for c in range(1, cols):
                if A[b, r, c] > max_val:
                    max_val = A[b, r, c]
                    max_idx = c

            # Assign the corresponding value
            result[b, r, max_idx] = max_dZ[b, r]

    dZ_out = deshape(result, k_size, stride)

    return dZ_out


"""
back_propagation_kernel:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got for the layer kernel

=========INPUT=========
dict            activation :    containt all the activation during the foreward propagation
dict            parametres :    containt all the information for the kernel operation
dict            gradients  :    containt all the information for the update
numpy.array     dZ :            the derivated of the previous activation (what should be the activation)
int             c  :            which stage we are in backpropagatioin 

=========OUTPUT=========
dict            gradients :     containt all the gradient need for the update
numpy.array     DZ :            the derivated of this activation for the next step of backpropagation
"""
@njit(parallel=True, fastmath=True)
def back_propagation_kernel(
    activations_A,
    activations_Z,
    K,
    fonction,
    k_size, stride,
    gradients_dK,
    gradients_db,
    dZ,
    c,
    alpha
):

    L_A, NB_Dot_Product, K_Size = activations_A[c-1].shape
    NB_K, L_K, K_Size, one = K.shape

    dK = np.zeros_like(K)

    # --------- Compute dK ----------
    for i in range(NB_K):            # each kernel
        for j in range(L_A):         # each activation
            for k in range(K_Size):  # each weight
                
                acc = 0.0
                for p in range(NB_Dot_Product):
                    for x in range(dZ.shape[1]):
                        for y in range(dZ.shape[2]):
                            idx = x * dZ.shape[2] + y
                            acc += activations_A[c-1][j, p, k] * dZ[i, x, y]
                
                dK[i, j, k, 0] = acc

    gradients_dK[c-1] = dK.astype(np.float32)
    gradients_db[c-1] = dZ.reshape(dZ.shape[0], dZ.shape[1]*dZ.shape[2], 1).astype(np.float32)

    # --------- Backprop to previous layer ----------
    if c > 1:

        if fonction == 0:
            dA = dx_relu(activations_Z[c], alpha)
        elif fonction == 1:
            dA = dx_sigmoide(activations_A[c])
        else:
            dA = dx_tanh(activations_A[c])

        dZ = dZ * dA

        dZ = convolution(dZ, K)

    return dZ


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
@njit
def back_propagation_CNN(
    activations_A, activations_Z, 
    parametres_K, parametres_B,
    gradients_dK, gradients_db,
    tuple_mode_info, y, alpha):

    #Here the derivative activation are in shape nxn, then they are modify to work effectively with code
    C = np.int32(len(tuple_mode_info))
    dZ = activations_A[C] - y
    
    index_gradient = 0
    c = C
    while c > 0:

        size = tuple_mode_info[c-1][2]
        dZ = np.ascontiguousarray(dZ[:, :size, :size])

        mode = tuple_mode_info[c-1][0]

        if mode == 1:

            dZ = back_propagation_pooling(
                activations_A[c-1],
                tuple_mode_info[c-1][3],
                tuple_mode_info[c-1][4],
                dZ
            )

        else:

            dZ = back_propagation_kernel(
                activations_A,
                activations_Z,
                parametres_K[c-1],
                tuple_mode_info[c-1][1],
                tuple_mode_info[c-1][3],
                tuple_mode_info[c-1][4],
                gradients_dK,
                gradients_db,
                dZ,
                c,
                alpha
            )
        c -= 1

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
@njit
def update(
    parametres_K, parametres_B,
    gradients_dK, gradients_db,
    m_list, v_list,
    tuple_mode_info,
    learning_rate, beta1, beta2, C
):
    epsilon = np.float32(1e-8)

    for c in range(C):

        if tuple_mode_info[c][0] == 0:  # kernel layer
            
            # update moments
            m_list[c] = beta1 * m_list[c] + (1 - beta1) * gradients_dK[c]
            v_list[c] = beta2 * v_list[c].astype(np.float64) + (1 - beta2) * (gradients_dK[c].astype(np.float64) ** 2)
            v_list[c] = v_list[c].astype(np.float32)

            # biais correction
            m_hat = m_list[c] / (1 - np.float32(beta1**(c+1)))
            v_hat = v_list[c] / (1 - np.float32(beta2**(c+1)))

            # update weights
            parametres_K[c] = parametres_K[c] - (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
            parametres_B[c] = parametres_B[c] - learning_rate * gradients_db[c]

    return parametres_K, parametres_B


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
@njit(fastmath=True)
def reshape(X, k_size_sqrt, x_size_sqrt, stride, padding):

    k_size = k_size_sqrt * k_size_sqrt
    n_samples, h, w = X.shape

    # Calcul out size
    o_size = (x_size_sqrt - k_size_sqrt + padding) // stride + 1
    n_patches = o_size * o_size

    # Preallocation
    new_X = np.empty((n_samples, n_patches, k_size), dtype=X.dtype)

    for k in range(n_samples):
        patch_idx = 0
        for i in range(0, h - k_size_sqrt + 1, stride):
            for j in range(0, w - k_size_sqrt + 1, stride):
                patch = X[k, i:i + k_size_sqrt, j:j + k_size_sqrt].ravel()
                for t in range(k_size):
                    new_X[k, patch_idx, t] = patch[t]
                patch_idx += 1

    return new_X


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
@njit
def deshape(X, k_size_sqrt, stride):
    n_layers, n_blocks, block_size = X.shape
    h = w = k_size_sqrt

    # Guess the grid of blocks
    n_rows_blocks = int(round(n_blocks ** 0.5))
    n_cols_blocks = n_blocks // n_rows_blocks

    height_A = (n_rows_blocks - 1) * stride + h
    width_A  = (n_cols_blocks - 1) * stride + w

    A_rec = np.zeros((n_layers, height_A, width_A), dtype=X.dtype)
    counts = np.zeros((n_layers, height_A, width_A), dtype=X.dtype)

    for l in range(n_layers):
        k = 0
        for i_block in range(n_rows_blocks):
            for j_block in range(n_cols_blocks):
                if k >= n_blocks:
                    continue
                # Get the block flat and transformed it in 2D
                block = X[l, k]
                for y in range(h):
                    for x in range(w):
                        A_rec[l, i_block*stride + y, j_block*stride + x] += block[y*w + x]
                        counts[l, i_block*stride + y, j_block*stride + x] += 1
                k += 1

    # Mean of recoveries
    for l in range(n_layers):
        for i in range(height_A):
            for j in range(width_A):
                if counts[l, i, j] > 0:
                    A_rec[l, i, j] /= counts[l, i, j]
                else:
                    A_rec[l, i, j] = 0.0

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
@njit
def add_padding(X, padding):
    n_layers, h, w = X.shape
    new_h = h + padding
    new_w = w + padding

    # Nouveau tableau rempli de 0
    X_padded = np.zeros((n_layers, new_h, new_w), dtype=X.dtype)

    # Copier les valeurs originales
    for l in range(n_layers):
        for i in range(h):
            for j in range(w):
                X_padded[l, i, j] = X[l, i, j]

    return X_padded


"""
============================
Evaluation Metrics Function
============================
"""

@njit
def dx_log_loss(y_pred, y_true):

    epsilon = np.float32(1e-15)
    n, h, w = y_pred.shape
    grad_sum = np.float32(0.0)
    
    for i in range(n):
        for j in range(h):
            for k in range(w):
                p = y_pred[i, j, k]
                t = y_true[i, j, k]
                g = - (t / (p + epsilon) - (1 - t) / (1 - p + epsilon))
                grad_sum += g
    
    return grad_sum / (n * h * w)


@njit
def log_loss(y_pred, y_true):

    epsilon = np.float32(1e-15)
    loss = np.float32(0.0)

    n, h, w = y_pred.shape  # unpack les 3 dimensions

    for i in range(n):
        for j in range(h):
            for k in range(w):
                p = y_pred[i, j, k]
                # clip to avoid log(0)
                if p < epsilon:
                    p = epsilon
                elif p > 1 - epsilon:
                    p = 1 - epsilon
                loss += y_true[i, j, k] * np.log(p) + (1 - y_true[i, j, k]) * np.log(1 - p)
    
    return -loss / (n * h * w)


@njit
def accuracy_score(y_pred, y_true):
    n = y_pred.shape[0]
    correct = 0
    
    for i in range(n):
        yp = np.round(y_pred[i])
        yt = np.round(y_true[i])
        if yp == yt:
            correct += 1
                
    return correct / n


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
def display_comparaison_layer(A, Z, max_par_fig=12):
    """
    Affiche chaque couche du tableau 3D A, et optionnellement Z si fourni,
    côte à côte. S'adapte si Z est None.
    """
    if A.ndim != 3:
        raise ValueError("A doit être un array 3D (D, H, W)")
    
    if np.any(Z) > 0:
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


def display_activation(X, y, parametres_K, parametres_B, tuple_mode_info, alpha):

    C_CNN = len(tuple_mode_info)

    # Affichage côte à côte
    plt.figure(figsize=(10, 5))

    # Afficher l'image X
    plt.subplot(1, 2, 1)
    img = deshape(X, tuple_mode_info[0][3], tuple_mode_info[0][4])
    img = np.transpose(img, (1, 2, 0))  # (3,28,28) → (28,28,3)

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title("Image X")

    # Afficher l'image y
    plt.subplot(1, 2, 2)
    img = np.transpose(y, (1, 2, 0))
    img = np.sum(img, axis=2)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title("Image y")

    plt.show()

    activations_A, activations_Z = forward_propagation(X, parametres_K, parametres_B, tuple_mode_info, alpha)

    for i in range(1, C_CNN):
        new_A = deshape(activations_A[i], tuple_mode_info[i][3], tuple_mode_info[i][4])
        new_Z = activations_Z[i]

        display_comparaison_layer(new_A, new_Z)


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

@njit
def train_loop(X, y, parametres_K, parametres_B,
               gradients_dK, gradients_db,
               m_list, v_list, tuple_mode_info,
               alpha, learning_rate, beta1, beta2,
               C_CNN, nb_iteration):

    # Préallocation
    l_array = np.zeros(nb_iteration, dtype=np.float32)
    a_array = np.zeros(nb_iteration, dtype=np.float32)
    d_array = np.zeros(nb_iteration, dtype=np.float32)

    for it in range(nb_iteration):

        # Forward
        activations_A, activations_Z = forward_propagation(
            X, parametres_K, parametres_B, tuple_mode_info, alpha)

        # Backward
        dZ = back_propagation_CNN(
            activations_A, activations_Z,
            parametres_K, parametres_B,
            gradients_dK, gradients_db,
            tuple_mode_info, y, alpha)

        # Update
        """parametres_K, parametres_B = update(
            parametres_K, parametres_B,
            gradients_dK, gradients_db,
            m_list, v_list,
            tuple_mode_info,
            learning_rate, beta1, beta2, C_CNN)"""

        # Metrics
        last_activation = activations_A[C_CNN]
        l_array[it] = log_loss(last_activation, y)
        a_array[it] = accuracy_score(last_activation.flatten(), y.flatten())
        d_array[it] = dx_log_loss(last_activation, y)

    return parametres_K, parametres_B, l_array, a_array, d_array, activations_A, activations_Z

def main():
    #Initialisation
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    alpha = 0.001
    nb_iteration = 1

    x_shape = 28
    input_shape = (3, x_shape, x_shape)

    #X = np.random.rand(x_shape, x_shape)
    #X = np.random.rand(x_shape * x_shape).reshape(x_shape, x_shape)
    #Create a cross to calibrate the model 

    # Création du volume RGB (3 couches)
    X = np.zeros((3, x_shape, x_shape))

    tiers = x_shape // 3                         # Calcul des positions
    middle = x_shape // 2

    X[0, tiers // 2, :] = 1                      # Couche 0 (R) → ligne dans le tiers supérieur
    X[1, middle, :] = 1                         # Couche 1 (G) → ligne au milieu
    X[2, x_shape - tiers // 2 - 1, :] = 1        # Couche 2 (B) → ligne dans le tiers inférieur

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
    parametres_K, parametres_B, tuple_mode_info, m_list, v_list, dimensions, gradients_dK, gradients_db = initialisation (
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
        X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1], dimensions["2"][2])

    else:
         X = reshape(X, dimensions["1"][0], x_shape, dimensions["1"][1], 0)


    learning_rate = np.float32(learning_rate)
    beta1 = np.float32(beta1)
    beta2 = np.float32(beta2)
    epsilon = np.float32(1e-8)

    # Paramètres et gradients
    parametres_K = [np.ascontiguousarray(x, dtype=np.float32) for x in parametres_K]
    parametres_B = [np.ascontiguousarray(x, dtype=np.float32) for x in parametres_B]
    gradients_dK = [np.ascontiguousarray(x, dtype=np.float32) for x in gradients_dK]
    gradients_db = [np.ascontiguousarray(x, dtype=np.float32) for x in gradients_db]

    m_list = [np.ascontiguousarray(x, dtype=np.float32) for x in m_list]
    v_list = [np.ascontiguousarray(x, dtype=np.float32) for x in v_list]

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    alpha = np.float32(alpha)

    parametres_K, parametres_B, l_array, a_array, d_array, activations_A, activations_Z = train_loop(
        X, y, 
        parametres_K, parametres_B,
        gradients_dK, gradients_db,
        m_list, v_list, 
        tuple_mode_info,
        alpha, learning_rate, 
        beta1, beta2,
        C_CNN, nb_iteration)
    
    print("Final accuracy ", a_array[-1])
    print("Final loss ", l_array[-1])

    #Display info of during the learing
    display_info_learning(l_array, a_array, d_array)

    #Display kernel & biais
    #display_kernel_and_biais(parametres)

    #Display target vs prediction
    y_pred = activations_A[C_CNN]
    #display_comparaison_layer(y, y_pred)

    display_activation(X, y_pred, parametres_K, parametres_B, tuple_mode_info, alpha)
    
main()