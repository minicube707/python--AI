
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

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
    # X : (batch, channels, height, width)
    
    windows = np.lib.stride_tricks.sliding_window_view(X, (k_size, k_size), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    
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
def op_correlate(A, K, stride):
    # A : (B, C, H, W)
    # K : (N, C, Kh, Kw)

    B, C, H, W = A.shape
    N, _, Kh, Kw   = K.shape

    windows = np.lib.stride_tricks.sliding_window_view(
        A, (Kh, Kw), axis=(2, 3)
        )
    
    windows = windows[:, :, ::stride, ::stride, :, :]
    
    H_out, W_out = windows.shape[2], windows.shape[3]
    
    # (N, B, H_out, W_out)
    out = np.tensordot(
        K,
        windows,
        axes=([1, 2, 3], [1, 4, 5])
    )  

    # → (B, N, H_out, W_out)
    out = np.moveaxis(out, 0, 1)

    return out
    
def grad_kernel(A_prev, dZ, K, stride):
    # A_prev : (B, C, H, W)
    # dZ     : (B, N, Hout, Wout)
    
    N, _, Kh, Kw   = K.shape

    windows = np.lib.stride_tricks.sliding_window_view(
    A_prev, (Kh, Kw), axis=(2, 3)
    )
    windows = windows[:, :, ::stride, ::stride, :, :]

    dK = np.tensordot(
        dZ,
        windows,
        axes=([0, 2, 3], [0, 2, 3])
    )

    return dK


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
    # dZ : (B, F, H, W)
    # K  : (F, C, Kh, Kw)

    B, F, H, W = dZ.shape
    _, C, Kh, Kw = K.shape

    pad_h = Kh - 1
    pad_w = Kw - 1

    padded = np.pad(dZ, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)))

    # (B, F, H+Kh-1, W+Kw-1, Kh, Kw)
    windows = sliding_window_view(padded, (Kh, Kw), axis=(2,3))

    # (C, B, H+Kh-1, W+Kw-1)
    out = np.tensordot(K, windows, axes=([0,2,3],[1,4,5]))

    # (B, C, H+Kh-1, W+Kw-1)
    return np.moveaxis(out, 0, 1)
    

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
    outpu_shape = input_size[2]
    for i in range(len(dimensions)):
        
        
        if i < len(dimensions):
            print(f"{outpu_shape}", end="")
            print(f"({dimensions[str(i+1)][2]})", end="")
            print("->", end="")

        outpu_shape = calcul_output_shape(outpu_shape, dimensions[str(i+1)][0], dimensions[str(i+1)][1], dimensions[str(i+1)][2])

    print(f"{outpu_shape}")  

    print("\nkernel size, stride, padding, nb_kernel, type layer, function, dropout")
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
def error_initialisation(dimensions, nb_activation, input_size, previ_input_size, type_layer, fonction, stride, dropout):

    if input_size < 1:
        show_information(dimensions, (nb_activation, input_size, input_size))
        raise ValueError(f"ERROR: The current dimensions is {input_size}. Dimension can't be negatif")
        
    if previ_input_size % input_size != 0 and stride != 1:
        show_information(dimensions, (nb_activation, input_size, input_size))
        raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
    
    if type_layer not in ["conv", "pool"]:
        show_information(dimensions, (nb_activation, input_size, input_size))
        raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pool' or 'conv'.")
    
    if fonction not in ["relu", "sigmoide", "max", "tanh"]:
        show_information(dimensions, (nb_activation, input_size, input_size))
        raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu', 'sigmoide', 'max' ou 'tanh'.")

    if ( not (0 <= dropout <= 1)):
        show_information(dimensions, (nb_activation, input_size, input_size))
        raise NameError(f"ERROR: dropout percent should be betwenn 0 and 1.")
    
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
    #Kernel size, stride, padding, nb_kernel, type layer, function, dropout

    k_size = dimensions[str(i)][0]
    stride = dimensions[str(i)][1]
    padding = dimensions[str(i)][2]
    nb_kernel = dimensions[str(i)][3]
    type_layer = dimensions[str(i)][4]
    fonction = dimensions[str(i)][5]
    dropout = dimensions[str(i)][6]

    return k_size, stride, padding, nb_kernel, type_layer, fonction, dropout


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
def initialisation_kernel(parametres, parametres_grad, k_size, fonction, i, nb_kernel, nb_layer, o_size):

    k_shape = (nb_kernel, nb_layer, k_size, k_size)

    if fonction == "relu":
        std = np.sqrt(2 / (nb_layer * k_size**2))
        K = np.random.randn(*k_shape).astype(np.float32) * std

    elif fonction == "sigmoide" or  fonction == "tanh":
        limit = np.sqrt(6 / (nb_layer + nb_kernel))
        K = (np.random.rand(*k_shape).astype(np.float32) * 2 - 1) * limit

    else:
        # Default to small random values
        K = np.random.randn(*k_shape).astype(np.float32) * 0.01

    b_shape = (nb_kernel, o_size, o_size)
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
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    dimension :     all the information on how is built the CNNing
"""
def initialisation_calcul(x_shape, dimensions, padding_mode):

    nb_channel = x_shape[0]
    input_size = x_shape[1]

    previ_input_size = input_size
    previ_channel = nb_channel

    
    for i in range(1, len(dimensions)+1):

        k_size, stride, padding, nb_kernel, type_layer, fonction, dropout = initialisation_extraction(dimensions, i)

        #Add padding
        if input_size % stride != 0 and padding_mode == "auto":
            padding = int(stride - input_size % stride)
            dimensions[str(i)] = (k_size, stride, padding, nb_kernel, type_layer, fonction, dropout)
            
        if type_layer == "conv":
            nb_channel = nb_kernel
            previ_channel = nb_channel

        #Conserve the nb of channel
        elif type_layer == "pool":
            dimensions[str(i)] = (k_size, stride, padding, previ_channel, type_layer, fonction, dropout)

        o_size = calcul_output_shape(input_size, k_size, stride, padding)
        input_size = o_size
        previ_input_size = input_size

        error_initialisation(dimensions, nb_channel, input_size, previ_input_size, type_layer, fonction, stride, dropout)

    return dimensions

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
def initialisation_affectation(dimensions, x_shape):

    parametres = {}
    parametres_grad = {}

    nb_layer = x_shape[0]
    o_size = x_shape[1]
    C = len(dimensions)

    for i in range(1, C +1):
        k_size, _, padding, nb_kernel, type_layer, fonction, dropout = initialisation_extraction(dimensions, i)
        o_size = calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2])
    
        if type_layer == "conv":

            if (i < C):
                o_size = o_size + padding

            parametres, parametres_grad = initialisation_kernel(parametres, parametres_grad, k_size, fonction, i, nb_kernel, nb_layer, o_size)

        elif type_layer == "pool":
            pass

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
"""
def initialisation(x_shape, dimensions, padding_mode):

    dimensions = initialisation_calcul(x_shape, dimensions, padding_mode)
    parametres, parametres_grad = initialisation_affectation(dimensions, x_shape)

    return parametres, parametres_grad


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
def kernel_activation(X, K, b, mode, alpha, stride):

    Z = op_correlate(X, K, stride)
    Z += b
    
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
def function_activation(X, K, b, mode, type_layer, k_size, stride, padding, dropout_per, training, alpha):

    # Padding
    if padding > 0:
        X = add_padding(X, padding)

    if type_layer == "conv":
        A, Z = kernel_activation(X, K, b, mode, alpha, stride)

    elif type_layer == "pool":
        A = pooling_activation(X, k_size, stride)
        Z = None
    
    if training:
        M = (np.random.rand(*A.shape) > dropout_per).astype(X.dtype)
        A = M * A / (1 - dropout_per)

    else:
        M = np.ones_like(A)

    return A, Z, M


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
def back_propagation_pooling(A_prev, k_size, stride, dZ, c):
    """
    A_prev : (B, C, H, W)
    dZ     : (B, C, H_out, W_out)
    """
    B, C, H, W = A_prev.shape
    H_out, W_out = dZ.shape[2], dZ.shape[3]

    # Sliding windows
    windows = sliding_window_view(A_prev, (k_size, k_size), axis=(2,3))
    # shape: (B, C, H_out, W_out, k, k)

    windows = windows[:, :, ::stride, ::stride, :, :]

    max_vals = windows.max(axis=(-1, -2), keepdims=True)
    mask = windows == max_vals  # shape: (B, C, H_out, W_out, k, k)

    # On broadcast dZ sur les k,k
    dZ_expanded = dZ[:, :, :, :, None, None]
    dA_prev = mask * dZ_expanded
    
    dA_prev_full = np.zeros_like(A_prev)

    H_out, W_out = dZ.shape[2], dZ.shape[3]

    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end   = h_start + k_size
            w_start = w * stride
            w_end   = w_start + k_size
            dA_prev_full[:, :, h_start:h_end, w_start:w_end] += dA_prev[:, :, h, w, :, :]

    return dA_prev_full


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
def back_propagation_kernel(activation, parameters, gradients, activation_function, stride, dZ, c, alpha):

    K = parameters[f"K{c}"]
    A_prev = activation[f"A{c-1}"]

    B, N, H_out, W_out = dZ.shape
    _, C, Kh, Kw  = K.shape

    #For each kernel
    dK = grad_kernel(A_prev, dZ, K, stride)

    gradients[f"dK{c}"] = dK
    gradients[f"db{c}"] = np.sum(dZ, axis=0)

    if c > 1:
        
        if activation_function == "relu":
            dA = dx_relu(activation[f"Z{c}"], alpha)

        elif activation_function == "sigmoide":
            dA = dx_sigmoide(activation[f"A{c}"])

        elif activation_function == "tanh":
            dA = dx_tanh(activation[f"A{c}"])

        dZ = dZ * dA
        
        # propagation vers la couche précédente
        dZ = convolution(dZ, K)

        dZ_expanded = dZ[:, :, :, :, None, None]
        new_dZ = np.zeros_like(A_prev)

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * stride
                h_end   = h_start + Kh

                w_start = w * stride
                w_end   = w_start + Kw

                new_dZ[:, :, h_start:h_end, w_start:w_end] += dZ_expanded[:, :, h, w]

        dZ = new_dZ

    return gradients, dZ


def adam_weight(param, grad, m, v, lr, beta1, beta2, t, eps=1e-8):

    grad = np.clip(grad, -2, 2)

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
    # X : (B, C, H, W)

    B, C, H, W = X.shape
    out = np.zeros((B, C, H + padding, W + padding), dtype=X.dtype)

    out[:, :, :H, :W] = X
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
    y_pred = (y_pred >= 0.01)
    y_true = (y_true >= 0.01)
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
                display_kernel(value, "Conv", key[-1])

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
    Affiche chaque couche du tableau 4D A (B, D, H, W), et optionnellement Z (B, D, H, W),
    côte à côte. S'adapte si Z est None.
    """
    if A.ndim != 4:
        raise ValueError("A doit être un array 4D (B, D, H, W)")

    B, D, H, W = A.shape

    if Z is not None:
        if Z.shape != A.shape:
            raise ValueError("A et Z doivent avoir la même forme si Z est fourni")
        mode_paire = True
    else:
        mode_paire = False

    for b in range(B):
        print(f"Batch {b}")
        total_couches = D

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
                im_a = ax_a.imshow(A[b, layer_idx], cmap='gray')
                ax_a.set_title(f"{label_A} - Couche {layer_idx}")
                ax_a.axis('off')
                fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

                # Affichage de Z si présent
                if mode_paire:
                    ax_z = axes[row, col * 2 + 1]
                    im_z = ax_z.imshow(Z[b, layer_idx], cmap='gray')
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

            plt.suptitle(f'Batch {b} - Couches {start} à {end - 1}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


def display_activation(X, y, activations_CNN, parametres_CNN, dimensions_CNN, alpha):

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
    for i in range(1, C_CNN):     
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

class CNN():

    def __init__(self, dimensions, input_shape, padding_mode, ):

        parametres, parameters_grad = initialisation (input_shape, dimensions, padding_mode)
        show_information(dimensions, input_shape)

        self.parametres = parametres
        self.parameters_grad = parameters_grad
        self.activations = {}
        self.gradient = {}

        self.dimensions = dimensions
        self.C_CNN = len(dimensions)


    def foward_propagation(self, X, alpha, training):

        activations = {"A0" : X}
        dimensions = self.dimensions
        parameters = self.parametres
        C_CNN = self.C_CNN

        for c in range(1, C_CNN + 1):
            A_prev = activations[f"A{c-1}"]

            type_layer, mode, dropout_per = dimensions[f"{c}"][4:7]
            k_size, stride, padding = dimensions[f"{c}"][:3]

            if type_layer == "conv":
                K = parameters[f"K{c}"]
                b = parameters[f"b{c}"]
        
            #The information for the padding is at the next step
            A, Z, M = function_activation(
                A_prev, K, b,
                mode, type_layer,
                k_size,
                stride, padding,
                dropout_per, training,
                alpha
            )

            activations[f"A{c}"] = A
            activations[f"Z{c}"] = Z
            activations[f"M{c}"] = M

        self.activations = activations

    
    def back_propagation_CNN(self, y, alpha, training):

        activations = self.activations
        dimensions = self.dimensions
        parameters = self.parametres
        C_CNN = self.C_CNN
        gradients = {}

        dZ = activations[f"A{C_CNN}"] - y

        for c in range(C_CNN, 0, -1):

            k_size, stride, padding, _, type_layer, activation_function, dropout_per = dimensions[f"{c}"]
            A = activations[f"A{c-1}"]

            # Appliquer le dropout si fourni
            if training:
                dZ = dZ * activations[f"M{c}"] / (1 - dropout_per)

            # Padding
            if padding > 0:
                A = add_padding(A, padding)

            # pooling
            if type_layer == "pool":
                dZ = back_propagation_pooling(
                    A,
                    k_size,
                    stride,
                    dZ,
                    c
                )

            # convolution
            elif type_layer == "conv":
                gradients, dZ = back_propagation_kernel(
                    activations,
                    parameters,
                    gradients,
                    activation_function,
                    stride,
                    dZ,
                    c,
                    alpha
                )
            
            # Removal of padding
            if padding > 0:
                dZ = dZ[:, :, :-padding, :-padding]

        self.gradient =  gradients


    def update(self, lr, beta1, beta2, t):

        dimensions = self.dimensions
        parameters = self.parametres
        C_CNN = self.C_CNN
        gradients = self.gradient
        parameters_grad = self.parameters_grad

        for c in range(1, C_CNN + 1):

            if dimensions[f"{c}"][4] == "conv":

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

        self.parametres = parameters


def main():
    #Initialisation
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    alpha = 0.001
    nb_iteration = 1000

    x_shape = 28
    input_shape = (1, x_shape, x_shape)

    X1 = np.random.rand(*input_shape)
    X2 = np.random.rand(*input_shape)
    X = np.stack([X1, X2], axis=0)  # batch=2

    #X = np.zeros((x_shape, x_shape))
    #X[:, 8:16] = 1
    #X[8:16, :] = 1

    if len(X.shape) == 2:
        X = X.reshape(1, 1, X.shape[0], X.shape[1])

    dimensions = {}
    #Kernel size, stride, padding, nb_kernel, type layer, function, dropout
    dimensions = {
        "1": (5, 1, 0, 32, "conv", "relu", 0.0),
        "2": (2, 2, 0, 1, "pool", "max", 0.0),
        "3": (3, 1, 0, 64, "conv", "relu", 0.1),
        "4": (2, 2, 0, 1, "pool", "max", 0.1),
        "5": (3, 1, 0, 64, "conv", "relu", 0.1)
    }
    
    padding_mode = "auto"

    model = CNN(dimensions, input_shape, padding_mode)
    
    input_size = X.shape[2]
    for val in dimensions.values():
        o_size = calcul_output_shape(input_size, val[0], val[1], val[2])
        input_size = o_size

    C_CNN = len(dimensions.keys())
    y_shape = o_size
    y1 = np.random.rand(dimensions[str(C_CNN)][3], y_shape, y_shape)
    y2 = np.random.rand(dimensions[str(C_CNN)][3], y_shape, y_shape)
    y = np.stack([y1, y2], axis=0)  # batch=2

    l_array = np.array([])
    a_array = np.array([])
    d_array = np.array([])

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    for t in tqdm(range(nb_iteration)):
        
        model.foward_propagation(X, alpha, True)
        model.back_propagation_CNN(y, alpha, True)
        model.update(learning_rate, beta1, beta2, t)

        model.foward_propagation(X, alpha, False)
        activations = model.activations
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
    model.foward_propagation(X, alpha, False)
    activations = model.activations

    y_pred = activations["A" + str(C_CNN)]
    display_comparaison_layer(y, y_pred, label_A="Y", label_Z="Y Pred")
    
    parametres = model.activations
    display_activation(X[0], y[0], activations, parametres, dimensions, alpha)

main()