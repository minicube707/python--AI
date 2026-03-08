
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, correlate
from numpy.lib.stride_tricks import sliding_window_view

#Allow to show all tab with numpy
np.set_printoptions(linewidth=200, threshold=np.inf)

"""
============================
========Documentation=======
============================

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
def max_pooling(X, k_size, stride):
    windows = np.lib.stride_tricks.sliding_window_view(X, (k_size, k_size), axis=(1, 2))
    windows = windows[:, ::stride, ::stride, :, :]
    return windows.max(axis=(-1, -2))



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
def op_correlate(A, K, B):
    # A : (C, H, W)
    # K : (N, C, Kh, Kw)
    # B : (N, Hout, Wout)

    windows = sliding_window_view(A, K.shape[-2:], axis=(1, 2))
    # windows : (C, Hout, Wout, Kh, Kw)

    out = np.tensordot(K, windows, axes=([1,2,3],[0,3,4]))
    # out : (N, Hout, Wout)

    return out + B

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
def convolution(dZ, K):
    # dZ : (F, H, W)
    # K  : (F, C, Kh, Kw)

    F, H, W = dZ.shape
    _, C, Kh, Kw = K.shape

    pad_h = Kh - 1
    pad_w = Kw - 1

    # padding pour simuler mode='full'
    padded = np.pad(dZ, ((0,0),(pad_h,pad_h),(pad_w,pad_w)))

    # extraction des fenêtres
    windows = sliding_window_view(padded, (Kh, Kw), axis=(1,2))
    # shape : (F, H+Kh-1, W+Kw-1, Kh, Kw)

    # produit tensoriel
    out = np.tensordot(K, windows, axes=([0,2,3],[0,3,4]))
    # shape : (C, H+Kh-1, W+Kw-1)

    return out

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
string  fonction :          the type of function
int     i :                 the stage of the CNN

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
"""
def initialisation_kernel(parametres, parametres_grad, k_size, type_layer, fonction, i, nb_kernel, nb_layer, o_size, padding):

    k_shape = (nb_kernel, nb_layer, k_size, k_size)

    # function: 0 = relu, 1 = sigmoide, 2 = tanh, 3 = max
    if fonction == 0:
        std = np.sqrt(2 / (nb_layer * k_size**2))
        K = np.random.randn(*k_shape).astype(np.float32) * std

    elif fonction == 1 or  fonction == 2:
        limit = np.sqrt(6 / (nb_layer + nb_kernel))
        K = (np.random.rand(*k_shape).astype(np.float32) * 2 - 1) * limit

    else:
        # Default to small random values
        K = np.random.randn(*k_shape).astype(np.float32) * 0.01

    b_shape = (nb_kernel, o_size + padding, o_size + padding)
    b = np.zeros(b_shape).astype(np.float32)  # Bias souvent initialisé à 0

    parametres["K" + str(i)] = K
    parametres["b" + str(i)] = b

    parametres_grad["km" + str(i)] = np.zeros(k_shape).astype(np.float32)
    parametres_grad["kv" + str(i)] = np.zeros(k_shape).astype(np.float32)

    parametres_grad["bm" + str(i)] = np.zeros(b_shape).astype(np.float32)
    parametres_grad["bv" + str(i)] = np.zeros(b_shape).astype(np.float32)

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
    o_size = x_shape[1]
    C = len(dimensions)

    for i in range(1, C +1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)
        o_size = calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2])

        padding = 0
        if (i < C):
           padding = dimensions[str(i+1)][2]

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
            parametres, parametres_grad = initialisation_kernel(parametres, parametres_grad, k_size, 0, fonction, i, nb_kernel, nb_layer, o_size, padding)

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

    return parametres, parametres_grad, dimensions, tuple_mode_info


"""
pooling_activation:
=========DESCRIPTION=========
Activation of pooling

=========INPUT=========
numpy.array     A :                 the activation matrice

=========OUTPUT=========
numpy.array     Z   : the resultat of the activation matrice after pass throw the activation function
"""
def pooling_activation(A, k_size, stride):
    Z = max_pooling(A, k_size, stride)
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
def kernel_activation(X, K, b, mode, alpha):

    Z = op_correlate(X, K, b)

    # function: 0 = relu, 1 = sigmoide, 2 = tanh, 3 = max
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
def function_activation(X, K, b, mode, type_layer, k_size, stride, padding, alpha):

    #Activation are in line format
    # function: 0 = relu, 1 = sigmoide, 2 = tanh, 3 = max

    if type_layer == 0:
        A, Z = kernel_activation(X, K, b, mode, alpha)

    else:
        A = pooling_activation(X, k_size, stride)
        Z = None
        
    #Activation are in square format
    if padding != None:
        A = add_padding(A, padding) 

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
def foward_propagation(X, parameters, tuple_mode_info, alpha):

    activations = {"A0" : X}
    C = len(tuple_mode_info)
    K = None
    b = None

    for c in range(1, C+1):
        A_prev = activations[f"A{c-1}"]

        #Tuple_mode_info: mode, function, output_size, kernel_size, stride, padding
        type_layer, mode = tuple_mode_info[c-1][:2]
        k_size, stride = tuple_mode_info[c-1][3:5]

        if type_layer == 0:
            K = parameters[f"K{c}"]
            b = parameters[f"b{c}"]
       
        #The information for the padding is at the next step
        padding = 0 
        if c+1 < C:
           padding = tuple_mode_info[c+1][5]

        A, Z = function_activation(
            A_prev, K, b,
            mode, type_layer,
            k_size,
            stride, padding,
            alpha
        )

        activations[f"A{c}"] = A
        activations[f"Z{c}"] = Z
        

    return activations

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
def back_propagation_pooling(activation, k_size, stride, dZ, c):

    A_prev = activation[f"A{c-1}"]
    m, H_prev, W_prev = A_prev.shape

    windows = sliding_window_view(A_prev, (k_size, k_size), axis=(1, 2))
    windows = windows[:, ::stride, ::stride, :, :]   # (m, H_out, W_out, k, k)

    max_vals = windows.max(axis=(-1, -2), keepdims=True)
    mask = windows == max_vals

    new_dZ = np.zeros_like(A_prev)

    for i in range(mask.shape[0]):
        for h in range(mask.shape[1]):
            for w in range(mask.shape[2]):
                h_start = h * stride
                w_start = w * stride
                new_dZ[i,
                       h_start:h_start+k_size,
                       w_start:w_start+k_size] += mask[i,h,w] * dZ[i,h,w]

    return new_dZ


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
def back_propagation_kernel(activation, parameters, gradients, activation_function, dZ, c, alpha):

    K = parameters[f"K{c}"]
    A_prev = activation[f"A{c}"]

    NB_K, L_K, Kh, Kw = K.shape

    # Gradient des kernels
    dK = correlate(dZ, A_prev, mode="valid")

    gradients[f"dK{c}"] = dK
    gradients[f"db{c}"] = dZ

    if c > 1:
        
        if activation_function == 0:
            dA = dx_relu(activation[f"Z{c}"], alpha)

        elif activation_function == 1:
            dA = dx_sigmoide(activation[f"A{c}"])

        elif activation_function == 2:
            dA = dx_tanh(activation[f"A{c}"])

        dZ = dZ * dA

        # propagation vers la couche précédente
        dZ = convolution(dZ, K)

    return gradients, dZ


"""
back_propagation:
=========DESCRIPTION=========
Evalaute the difference between the target and the resultat got

=========INPUT=========
dict            activation :                    containt all the activation during the foreward propagation
dict            parametres :                    containt all the information for the kernel operation
numpy.array     y :                             the target, the objectif of the CNN
tuple           list_size_activation:           tuple of all activation shape with number of activation and padding

=========OUTPUT=========
dict           gradients :     containt all the gradient need for the update
"""
def back_propagation_CNN(activation, parameters, y, layer_info, alpha):

    C = len(layer_info)
    gradients = {}

    dZ = activation[f"A{C}"] - y

    for c in range(C, 0, -1):

        type_layer, activation_function, size, k_size, stride = layer_info[c-1][:5]

        # pooling
        if type_layer == 1:  

            dZ = dZ[:, :size, :size]

            dZ = back_propagation_pooling(
                activation,
                k_size,
                stride,
                dZ,
                c
            )

        # convolution
        else:  

            gradients, dZ = back_propagation_kernel(
                activation,
                parameters,
                gradients,
                activation_function,
                dZ,
                c,
                alpha
            )

    return gradients

def adam_weight(param, grad, m, v, lr, beta1, beta2, t, eps=1e-8):

    # Update moments
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)

    # Bias correction
    bias_corr1 = max(1 - beta1**(t + 1), 1e-12)
    bias_corr2 = max(1 - beta2**(t + 1), 1e-12)

    m_hat = m / bias_corr1
    v_hat = np.maximum(v / bias_corr2, 1e-12)

    # Update parameter
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param, m, v


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
def update(gradients, parameters, parameters_grad, tuple_mode_info, lr, beta1, beta2, C, t):

    for c in range(1, C+1):

        if tuple_mode_info[c-1][0] == 0:

            # ----- Kernel -----
            parameters[f"K{c}"], parameters_grad[f"km{c}"], parameters_grad[f"kv{c}"] = adam_weight(
                parameters[f"K{c}"],
                gradients[f"dK{c}"],
                parameters_grad[f"km{c}"],
                parameters_grad[f"kv{c}"],
                lr, beta1, beta2, t
            )

            # ----- Bias -----
            parameters[f"b{c}"], parameters_grad[f"bm{c}"], parameters_grad[f"bv{c}"] = adam_weight(
                parameters[f"b{c}"],
                gradients[f"db{c}"],
                parameters_grad[f"bm{c}"],
                parameters_grad[f"bv{c}"],
                lr, beta1, beta2, t
            )

    return parameters


"""
============================
=======Shape fonction ======
============================
"""

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
    C, H, W = X.shape
    out = np.zeros((C, H + padding, W + padding), dtype=X.dtype)
    out[:, :H, :W] = X
    return out


"""
============================
Evaluation Metrics Function
============================
"""

def log_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


def dx_log_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean((y_true/y_pred - (1-y_true)/(1-y_pred)) / y_true.size)


def accuracy_score(y_pred, y_true):
    y_pred = (y_pred >= 0.1)
    y_true = (y_true >= 0.1)
    return np.mean(y_pred == y_true)


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
                display_kernel(value, "Kernel", key[-1])

            elif key.startswith('b'):
                display_biais(value, "Biais", key[-1])


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
def display_comparaison_layer(A, Z=None, max_par_fig=12, label_A="A", label_Z="Z"):
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
            ax_a.set_title(f"{label_A} - Couche {layer_idx}")
            ax_a.axis('off')
            fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

            # Affichage de Z si présent
            if mode_paire:
                ax_z = axes[row, col * 2 + 1]
                im_z = ax_z.imshow(Z[layer_idx], cmap='gray')
                ax_z.set_title(f"{label_Z} - Couche {layer_idx}")
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


def display_activation(X, y, parametres_CNN, dimensions_CNN, tuple_mode_info, alpha):

    # Affichage côte à côte
    plt.figure(figsize=(10, 5))

    # Afficher l'image X
    plt.subplot(1, 2, 1)
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')
    plt.title("Image X")

    # Afficher l'image y
    plt.subplot(1, 2, 2)
    y_reduced = np.sum(y, axis=0)
    plt.imshow(y_reduced, cmap='gray')
    plt.axis('off')
    plt.title("Image y")

    plt.show()

    C_CNN = len(dimensions_CNN.keys())

    activations_CNN = foward_propagation(X, parametres_CNN, tuple_mode_info, alpha)

    for i in range(1, len(dimensions_CNN)):     
        display_comparaison_layer(activations_CNN["A" +str(i)], activations_CNN["Z" +str(i)])
        

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
    nb_iteration = 2_000

    x_shape = 28
    input_shape = (1, x_shape, x_shape)

    X = np.random.rand(x_shape * x_shape).reshape(x_shape, x_shape)
    #X = np.zeros((x_shape, x_shape))
    #X[:, 8:16] = 1
    #X[8:16, :] = 1

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
    parametres, parametres_grad, dimensions, tuple_mode_info = initialisation (
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

    l_array = np.array([])
    a_array = np.array([])
    d_array = np.array([])

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    for t in tqdm(range(nb_iteration)):
        
        activations = foward_propagation(X, parametres, tuple_mode_info, alpha)
        gradients = back_propagation_CNN(activations, parametres, y, tuple_mode_info, alpha)
        parametres = update(gradients, parametres, parametres_grad, tuple_mode_info, learning_rate, beta1, beta2, C_CNN, t)

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
    display_comparaison_layer(y, y_pred, label_A="Y", label_Z="Y Pred")
    
    display_activation(X, y, parametres, dimensions, tuple_mode_info, alpha)

main()