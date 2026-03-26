
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC, abstractmethod

#Allow to show all tab with numpy
np.set_printoptions(linewidth=200, threshold=np.inf)


class Layer(ABC):

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dA):
        pass


class Softmax(Layer):

    def forward(self, X):
        # stabilité numérique
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def backward(self, dY):
        # Jacobien complet (coûteux mais correct)
        m, n = self.out.shape
        dX = np.zeros_like(dY)

        for i in range(m):
            y = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - y @ y.T
            dX[i] = jacobian @ dY[i]

        return dX

class Linear(Layer):

    def forward(self, X, *args):
        return X

    def backward(self, dA):
        return dA


class ReLU(Layer):

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)


class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0) + self.alpha * np.minimum(X, 0)

    def backward(self, dA):
        dx = np.ones_like(self.X)
        dx[self.X < 0] = self.alpha
        return dA * dx


class Sigmoide(Layer):

    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)


class Tanh(Layer):
    
    def forward(self, X):
        self.A = np.tanh(X)
        return self.A

    def backward(self, dA):
        return dA * (1 - self.A**2)
            

class MaxPooling(Layer):
    
    def __init__(self, k_size, stride, padding):
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.X = None

    def forward(self, X):
        
        padding = self.padding

        # Padding
        if padding > 0:
            X = add_padding(X, padding)

        self.X = X

        k = self.k_size
        s = self.stride
        
        windows = np.lib.stride_tricks.sliding_window_view(
            X, (k, k), axis=(2, 3)
        )
        windows = windows[:, :, ::s, ::s, :, :]

        self.windows = windows 
        return windows.max(axis=(-1, -2))

    def backward(self, dA):
        k = self.k_size
        s = self.stride
        X = self.X
        windows = self.windows
        padding = self.padding

        # mask des max
        max_vals = windows.max(axis=(-1, -2), keepdims=True)
        mask = (windows == max_vals)

        # On broadcast dZ sur les k,k
        dZ_expanded = dA[:, :, :, :, None, None]
        dA_prev = mask * dZ_expanded

        dA_prev_full = np.zeros_like(X)

        H_out, W_out = dA.shape[2], dA.shape[3]

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * s
                h_end   = h_start + k
                w_start = w * s
                w_end   = w_start + k
                dA_prev_full[:, :, h_start:h_end, w_start:w_end] += dA_prev[:, :, h, w, :, :]

        # Removal of padding
        if padding > 0:
            dA_prev_full = dA_prev_full[:, :, :-padding, :-padding]

        return dA_prev_full


class Dropout(Layer):

    def __init__(self, dropout_per):
        self.dropout_per = dropout_per
        self.training = False

    def forward(self, A, training):
        
        self.training = training
        if training:
            self.M = (np.random.rand(*A.shape) > self.dropout_per).astype(A.dtype)
            return  self.M * A / (1 - self.dropout_per)
        
        else:
            return A
    
    def backward(self, dZ):
        
        training = self.training

        if training:
            return dZ * self.M / (1 - self.dropout_per)
        
        else:
            return dZ

class BatchNorm(Layer):

    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.training = False

        self.gamma = np.ones((1, n_features))
        self.beta  = np.zeros((1, n_features))
        
        self.running_mean = np.zeros((1, n_features))
        self.running_var  = np.ones((1, n_features))
    
    def forward(self, X, training):

        self.training = training

        # ===== Detect DNN vs CNN =====
        if X.ndim == 2:
            axes = (0,)
            reshape = (1, -1)
            m = X.shape[0]

        elif X.ndim == 4:
            axes = (0, 2, 3)
            reshape = (1, -1, 1, 1)
            m = X.shape[0] * X.shape[2] * X.shape[3]

        else:
            raise ValueError("Unsupported input shape")

        gamma = self.gamma.reshape(reshape)
        beta  = self.beta.reshape(reshape)

        if self.training:
            self.mu  = np.mean(X, axis=axes, keepdims=True)
            self.var = np.var(X, axis=axes, keepdims=True)

            self.X_centered = X - self.mu
            self.var_eps = self.var + self.eps
            self.std_inv = 1.0 / np.sqrt(self.var_eps)

            self.X_hat = self.X_centered * self.std_inv

            # running stats (always in (1, C))
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * self.mu.reshape(1, -1)
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * self.var.reshape(1, -1)
            )

        else:
            mu  = self.running_mean.reshape(reshape)
            var = self.running_var.reshape(reshape)
            self.X_hat = (X - mu) / np.sqrt(var + self.eps)

        return gamma * self.X_hat + beta


    def backward(self, dY):

        if not self.training:
            raise RuntimeError("Backward called in inference mode")

        # ===== Detect DNN vs CNN =====
        if dY.ndim == 2:
            axes = (0,)
            reshape = (1, -1)
            m = dY.shape[0]

        elif dY.ndim == 4:
            axes = (0, 2, 3)
            reshape = (1, -1, 1, 1)
            m = dY.shape[0] * dY.shape[2] * dY.shape[3]

        else:
            raise ValueError("Unsupported input shape")

        gamma = self.gamma.reshape(reshape)

        # ===== Gradients =====
        dX_hat = dY * gamma

        dvar = np.sum(
            dX_hat * self.X_centered * -0.5 * self.var_eps**(-1.5),
            axis=axes, keepdims=True
        )

        dmu = (
            np.sum(dX_hat * -self.std_inv, axis=axes, keepdims=True)
            + dvar * np.sum(-2 * self.X_centered, axis=axes, keepdims=True) / m
        )

        dX = (
            dX_hat * self.std_inv
            + dvar * 2 * self.X_centered / m
            + dmu / m
        )

        # gamma / beta gradients (always in (1, C))
        self.dgamma = np.sum(dY * self.X_hat, axis=axes, keepdims=True).reshape(1, -1)
        self.dbeta  = np.sum(dY, axis=axes, keepdims=True).reshape(1, -1)

        return dX

    def get_params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

class Convolution(Layer):

    def __init__(self, nb_kernel, nb_layer, k_size, stride, o_size, padding):
        
        k_shape = (nb_kernel, nb_layer, k_size, k_size)
        b_shape = (nb_kernel, o_size, o_size)
        
        self.K = np.random.randn(*k_shape) * 0.01
        self.b = np.zeros(b_shape)
        self.X = None

        self.dK = np.zeros(k_shape)
        self.db = np.zeros(b_shape)
        self.stride = stride
        self.padding = padding
        self.windows = None

    def forward(self, X):
        stride  = self.stride
        padding = self.padding

        B, C, H, W     = X.shape
        N, _, Kh, Kw   = self.K.shape

        # Padding
        if padding > 0:
            X = add_padding(X, padding)

        self.X = X  # stocke l'entrée paddée

        # Extraction des fenêtres (cross-correlation)
        windows = np.lib.stride_tricks.sliding_window_view(
            X, (Kh, Kw), axis=(2, 3)
        )
        windows = windows[:, :, ::stride, ::stride, :, :]
        self.windows = windows

        # Dimensions de sortie
        H_out, W_out = windows.shape[2], windows.shape[3]

        # Convolution (produit tensoriel)
        out = np.tensordot(
            self.K,
            windows,
            axes=([1, 2, 3], [1, 4, 5])
        )  # (N, B, H_out, W_out)

        out = np.moveaxis(out, 0, 1)  # → (B, N, H_out, W_out)

        # Ajout du biais
        out += self.b
        
        return out

    def backward(self, dZ):
        stride  = self.stride
        padding = self.padding

        B, N, H_out, W_out = dZ.shape
        _, C, Kh, Kw       = self.K.shape

        X  = self.X
        windows = self.windows

        # ========================
        # Parameter Gradients
        # ========================

        # dK
        self.dK = np.tensordot(
            dZ,
            windows,
            axes=([0, 2, 3], [0, 2, 3])
        )

        # db
        self.db = np.sum(dZ, axis=0)

        # ========================
        # Gradient entry
        # ========================

        # Propagation via convolution
        dZ = convolution(dZ, self.K)

        # Expansion to distribute across windows
        dZ_expanded = dZ[:, :, :, :, None, None]

        dX = np.zeros_like(X)

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * stride
                h_end   = h_start + Kh

                w_start = w * stride
                w_end   = w_start + Kw

                dX[:, :, h_start:h_end, w_start:w_end] += dZ_expanded[:, :, h, w]

        # Removal of padding
        if padding > 0:
            dX = dX[:, :, :-padding, :-padding]

        return dX
    
    def get_params(self):
        return [(self.K, self.dK), (self.b, self.db)]
    
    def get_activations(self):
        return self.A
    
class Block(Layer):

    def __init__(self, dense, batchnorm, activation, dropout):

        self.dense = dense
        self.batchnorm = batchnorm
        self.activation = activation
        self.dropout = dropout

    def forward(self, X, training=True):

        Z = self.dense.forward(X)
        Z = self.batchnorm.forward(Z, training)
        A = self.activation.forward(Z)
        A = self.dropout.forward(A, training)

        return A

    def backward(self, dZ):

        dA = self.dropout.backward(dZ)
        dZ = self.activation.backward(dA)
        dZ = self.batchnorm.backward(dZ)
        dZ = self.dense.backward(dZ)

        return dZ
    
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


def add_padding(X, padding):
    # X : (B, C, H, W)

    B, C, H, W = X.shape
    out = np.zeros((B, C, H + padding, W + padding), dtype=X.dtype)

    out[:, :, :H, :W] = X
    return out

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
    print(f"{input_size[1]}", end="")
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
    
    if fonction not in ["relu", "sigmoide", "max", "tanh", "leaky relu"]:
        show_information(dimensions, (nb_activation, input_size, input_size))
        raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu', 'leaky relu', 'sigmoide', 'max' ou 'tanh'.")

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
def display_kernel_and_biais(model):

    for i, block in enumerate(model.layers):
        
        if isinstance(block.dense, Convolution):

            K = block.dense.K
            b = block.dense.b

            display_kernel(K, "Conv", i)
            display_biais(b, "Biais", i)


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


def display_activation(X, y, model):

    # Affichage côte à côte
    plt.figure(figsize=(10, 5))

    # Afficher l'image X
    plt.subplot(1, 2, 1)
    X_reduced = np.sum(X[0], axis=0)
    plt.imshow(X_reduced, cmap='gray')
    plt.axis('off')
    plt.title("Image X")

    # Afficher l'image y
    plt.subplot(1, 2, 2)
    y_reduced = np.sum(y[0], axis=0)
    plt.imshow(y_reduced, cmap='gray')
    plt.axis('off')
    plt.title("Image y")

    plt.show()
    
    C = model.C_CNN
    for i in range(C):  
        A, Z =  model.get_activatoins(X, i)
        display_comparaison_layer(A, Z)
        

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

    def __init__(self, dimensions, input_shape, padding_mode, alpha, optimizer):

        self.dimensions = dimensions
        self.layers = []
        self.C_CNN = len(dimensions)
        self.logits = None
       
        self.initialisation (input_shape, padding_mode, alpha)

        self.optimizer = optimizer

    def initialisation(self, x_shape, padding_mode, alpha):
        
        dimensions = self.dimensions

        dimensions = initialisation_calcul(x_shape, dimensions, padding_mode)
        self.dimensions = dimensions
        self.initialisation_affectation(x_shape, alpha)
        self.show_information(x_shape)
   
    def initialisation_affectation(self, x_shape, alpha):

        nb_layer = x_shape[0]
        o_size = x_shape[1]
        C = self.C_CNN
        dimensions = self.dimensions

        for i in range(1, C + 1):
            k_size, stride, padding, nb_kernel, type_layer, activation_function, dropout_per = initialisation_extraction(dimensions, i)
            o_size = calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2])

            if type_layer == "conv":
                
                if (i < C):
                    o_size = o_size + padding

                corr =  Convolution(nb_kernel, nb_layer, k_size, stride, o_size, padding)

                #Batchnorm
                batchnorm = BatchNorm(nb_kernel)

                #Activation
                if activation_function == "sigmoide":
                    activation = Sigmoide()
                
                elif activation_function == "tanh":
                    activation = Tanh()
                
                elif activation_function == "relu":
                    activation = ReLU()

                elif activation_function == "leaky relu":
                    activation = LeakyReLU(alpha)

                #Droout
                dropout = Dropout(dropout_per)

            elif type_layer == "pool":
                corr = MaxPooling(k_size, stride, padding)

                batchnorm = Linear()
                activation = Linear()
                dropout = Linear()

            self.layers.append(Block(corr, batchnorm, activation, dropout))
            nb_layer = nb_kernel

    def forward_propagation(self, X, training):

        for block in self.layers:
            X = block.forward(X, training)

        self.logits = X

    def backward_propagation(self, dZ):
        
        for block in reversed(self.layers):
            dZ = block.backward(dZ)

    def update(self):
        params = self.get_parameters()
        self.optimizer.update(params)


    def get_parameters(self):
        params = []
        for block in self.layers:

            if isinstance(block.dense, Convolution):
                params += block.dense.get_params()
                params += block.batchnorm.get_params()

        return params
    
    def get_activatoins(self, X, i):
        
        c = 0
        for block in self.layers:

            Z1 = block.dense.forward(X)
            Z1_cpy = Z1.copy()
            Z2 = block.batchnorm.forward(Z1, False)
            Z3 = block.activation.forward(Z2)
            Z3_cpy = Z3.copy()
            Z4 = block.dropout.forward(Z3, False)

            if c >= i:
                break
        
            c += 1
            X = Z4
        
        if isinstance(block.dense, MaxPooling):
            return Z1_cpy, None
        return Z1_cpy, Z3_cpy
    
    def show_information(self, input_size):
        
        dimensions = self.dimensions

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
        
class Adam:

    def __init__(self, lr, beta1, beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.state = {}

    def update(self, params):
        self.t += 1

        for param, grad in params:
            key = id(param)

            if key not in self.state:
                self.state[key] = {
                    "m": np.zeros_like(param),
                    "v": np.zeros_like(param)
                }

            m = self.state[key]["m"]
            v = self.state[key]["v"]

            # update Adam
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            self.state[key]["m"] = m
            self.state[key]["v"] = v

class CrossEntropyLoss:

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true
        
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        return loss

    def backward(self):
        m = self.y_pred.shape[0]
        return -(self.y_true / self.y_pred) / m
    

class MSE:

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
    
def main():

    #Initialisation
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    alpha = 0.001
    nb_iteration = 1_000

    x_shape = 28
    input_shape = (1, x_shape, x_shape)

    X1 = np.random.rand(*input_shape)
    X2 = np.random.rand(*input_shape)
    X = np.stack([X1, X2], axis=0)  # batch=2

    #X = np.zeros((x_shape, x_shape))
    #X[:, 8:16] = 1
    #X[8:16, :] = 1


    dimensions = {}
    #Kernel size, stride, padding, nb_kernel, type layer, function, dropout
    dimensions = {
        "1": (5, 1, 0, 32, "conv", "sigmoide", 0.0),
        "2": (2, 2, 0, 1, "pool", "max", 0.0),
        "3": (3, 1, 0, 64, "conv", "sigmoide", 0.0),
        "4": (2, 2, 0, 1, "pool", "max", 0.0),
        "5": (3, 1, 0, 64, "conv", "sigmoide", 0.0)
    }
    
    padding_mode = "auto"
    loss = CrossEntropyLoss()
    output_layer = Softmax()    
    optimizer = Adam(learning_rate, beta1, beta2)
    model = CNN(dimensions, input_shape, padding_mode, alpha, optimizer)
    
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

    for j in tqdm(range(nb_iteration)):
    
        #Foreward propagation
        model.forward_propagation(X, True)
        res = model.logits

        #Backpropagation
        if isinstance(output_layer, Softmax) and isinstance(loss, CrossEntropyLoss):
            dZ = res - y

        else:
            output_layer.forward(res)
            loss.forward(res, y)
            dA = loss.backward()
            dZ = output_layer.backward(dA)

        model.backward_propagation(dZ)
        model.update()

        model.forward_propagation(X, False)
        res = model.logits
        
        l_array = np.append(l_array, log_loss(res, y))
        a_array = np.append(a_array, accuracy_score(res.flatten(), y.flatten()))
        d_array = np.append(d_array, dx_log_loss(res, y))

    print("Final accuracy ", a_array[-1])
    print("Final loss ", l_array[-1])

    #Display info of during the learing
    display_info_learning(l_array, a_array, d_array)

    #Display kernel & biais
    #display_kernel_and_biais(model)

    #Display target vs prediction
    model.forward_propagation(X, False)
    y_pred = model.logits

    display_comparaison_layer(y, y_pred, label_A="Y", label_Z="Y Pred")

    #display_activation(X, y, model)

main()