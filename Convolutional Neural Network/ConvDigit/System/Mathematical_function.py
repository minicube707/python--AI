
import numpy as np
from scipy.signal import correlate2d

def tanh(X):
    return np.tanh(X)

def dx_tanh(X):
    return (1 - X**2)
            
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


"""
max_pooling:
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
softmax:
=========DESCRIPTION=========
Apply the softmax function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def softmax(X):

    X = np.clip(X, -64, 64)
    X_max = np.max(X, axis=1, keepdims=True)
    e_x = np.exp(X - X_max)
    
    return e_x / np.sum(e_x, axis=1, keepdims=True)

"""
correlate
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